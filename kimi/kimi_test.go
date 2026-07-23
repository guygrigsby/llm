package kimi

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

func TestGenerateText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header = %q, want Bearer test-key", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, err := New(Config{APIKey: "test-key", Model: "kimi-k3", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hey")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.TextContent(); got != "hi there" {
		t.Errorf("text = %q, want %q", got, "hi there")
	}
	if resp.Message.StopReason != ac.StopReasonStop {
		t.Errorf("stop = %q, want stop", resp.Message.StopReason)
	}
}

// TestGenerateToolCall proves a tool_calls response round-trips to an agentcore
// tool call with a toolUse stop reason.
func TestGenerateToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Denver\"}"}}]},"finish_reason":"tool_calls"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	tools := []ac.ToolSpec{{Name: "get_weather", Description: "weather", Parameters: map[string]any{"type": "object"}}}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("weather?")}, tools)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "get_weather" {
		t.Errorf("call = %+v", calls[0])
	}
	if string(calls[0].Args) != `{"city":"Denver"}` {
		t.Errorf("args = %s", calls[0].Args)
	}
	if resp.Message.StopReason != ac.StopReasonToolUse {
		t.Errorf("stop = %q, want toolUse", resp.Message.StopReason)
	}
}

// TestGenerateReasoning proves reasoning_content becomes a thinking block.
func TestGenerateReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"42","reasoning_content":"let me think"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.ThinkingContent(); got != "let me think" {
		t.Errorf("thinking = %q", got)
	}
	if got := resp.Message.TextContent(); got != "42" {
		t.Errorf("text = %q", got)
	}
}

// TestReasoningEffortSent proves a thinking level is mapped to reasoning_effort
// on the wire.
func TestReasoningEffortSent(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithThinking(ac.ThinkingXHigh))
	if err != nil {
		t.Fatal(err)
	}
	if body.ReasoningEffort != "max" {
		t.Errorf("reasoning_effort = %q, want max", body.ReasoningEffort)
	}
}

// TestPartialMode proves the partial/name metadata on an assistant message
// surfaces as partial:true on the wire (Kimi prefix continuation).
func TestPartialMode(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"...continued"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	prefix := ac.Message{
		Role:     ac.RoleAssistant,
		Content:  []ac.ContentBlock{ac.TextBlock("Once upon")},
		Metadata: map[string]any{"partial": true, "name": "narrator"},
	}
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("story"), prefix}, nil)
	if err != nil {
		t.Fatal(err)
	}
	last := body.Messages[len(body.Messages)-1]
	if !last.Partial || last.Name != "narrator" || last.Content != "Once upon" {
		t.Errorf("partial message = %+v", last)
	}
}

// TestGenerateMeters proves usage is reported and cached tokens are split out of
// prompt tokens so llm.Usage stays additive.
func TestGenerateMeters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":30,"completion_tokens":5,"total_tokens":35,"cached_tokens":10}}`))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{
		APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL,
		Meter: llm.MeterFunc(func(u llm.Usage) { got = u }),
	})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if got.PromptTokens != 20 || got.CacheReadTokens != 10 || got.CompletionTokens != 5 || got.TotalTokens != 35 {
		t.Errorf("usage = %+v, want prompt 20 / cache 10 / completion 5 / total 35", got)
	}
	if got.Provider != "kimi" || got.Model != "kimi-k3" {
		t.Errorf("provider/model = %q/%q", got.Provider, got.Model)
	}
	if got.Latency <= 0 {
		t.Error("latency not measured")
	}
}

// TestEstimateTokens proves the token-count endpoint is called and parsed.
func TestEstimateTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenizers/estimate-token-count" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var req struct {
			Model    string        `json:"model"`
			Messages []wireMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "kimi-k3" || len(req.Messages) != 1 {
			t.Errorf("estimate req = %+v", req)
		}
		_, _ = w.Write([]byte(`{"data":{"total_tokens":45}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	n, err := a.EstimateTokens(context.Background(), []ac.Message{ac.UserMsg("hello, what is 1+1?")})
	if err != nil {
		t.Fatal(err)
	}
	if n != 45 {
		t.Errorf("tokens = %d, want 45", n)
	}
}

// TestGenerateHTTPError proves a non-200 surfaces the response text as an error.
func TestGenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"invalid api key"}}`)
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "bad", Model: "kimi-k3", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("x")}, nil); err == nil {
		t.Fatal("want error on 401")
	}
}

func TestNewValidates(t *testing.T) {
	if _, err := New(Config{Model: "kimi-k3"}); err == nil {
		t.Error("want error without APIKey")
	}
	if _, err := New(Config{APIKey: "k"}); err == nil {
		t.Error("want error without Model")
	}
}
