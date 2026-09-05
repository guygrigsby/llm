package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

func TestNewValidates(t *testing.T) {
	if _, err := New(Config{Model: "m"}); err == nil {
		t.Error("missing APIKey should fail")
	}
	if _, err := New(Config{APIKey: "k"}); err == nil {
		t.Error("missing Model should fail")
	}
	a, err := New(Config{APIKey: "k", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if a.client.BaseURL != "https://openrouter.ai/api/v1" {
		t.Errorf("baseURL = %q", a.client.BaseURL)
	}
	if a.ProviderName() != "openrouter" || !a.SupportsTools() {
		t.Error("provider name or tools flag wrong")
	}
}

func TestGenerateTextAndHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth = %q", got)
		}
		if got := r.Header.Get("HTTP-Referer"); got != "https://example.test" {
			t.Errorf("referer = %q", got)
		}
		if got := r.Header.Get("X-Title"); got != "autophage" {
			t.Errorf("title = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, err := New(Config{APIKey: "test-key", Model: "moonshotai/kimi-k3", BaseURL: srv.URL, Referer: "https://example.test", Title: "autophage"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hey")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.TextContent(); got != "hi there" {
		t.Errorf("text = %q", got)
	}
	if resp.Message.StopReason != ac.StopReasonStop {
		t.Errorf("stop = %q", resp.Message.StopReason)
	}
}

func TestGenerateToolCall(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Denver\"}"}}]},"finish_reason":"tool_calls"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	tools := []ac.ToolSpec{{Name: "get_weather", Description: "weather", Parameters: map[string]any{"type": "object"}}}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("weather?")}, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(body.Tools) != 1 || body.Tools[0].Function.Name != "get_weather" {
		t.Errorf("tools on wire = %+v", body.Tools)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "get_weather" || string(calls[0].Args) != `{"city":"Denver"}` {
		t.Errorf("calls = %+v", calls)
	}
	if resp.Message.StopReason != ac.StopReasonToolUse {
		t.Errorf("stop = %q", resp.Message.StopReason)
	}
}

func TestGenerateReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"42","reasoning":"let me think","reasoning_details":[{"type":"reasoning.text","text":"let me think"}]},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
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
	details, _ := resp.Message.Metadata["reasoning_details"].(json.RawMessage)
	if string(details) != `[{"type":"reasoning.text","text":"let me think"}]` {
		t.Errorf("reasoning_details metadata = %s", details)
	}
}

func TestReasoningEffortSent(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithThinking(ac.ThinkingXHigh)); err != nil {
		t.Fatal(err)
	}
	if body.Reasoning == nil || body.Reasoning.Effort != "xhigh" {
		t.Errorf("reasoning = %+v, want effort xhigh", body.Reasoning)
	}
	body = chatRequest{}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
	if body.Reasoning != nil {
		t.Errorf("reasoning should be omitted without a thinking level, got %+v", body.Reasoning)
	}
}

func TestReasoningDetailsEchoed(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	prior := ac.Message{
		Role:     ac.RoleAssistant,
		Content:  []ac.ContentBlock{ac.ThinkingBlock("thought"), ac.ToolCallBlock(ac.ToolCall{ID: "c1", Name: "t", Args: json.RawMessage(`{}`)})},
		Metadata: map[string]any{"reasoning_details": json.RawMessage(`[{"type":"reasoning.encrypted","data":"xyz"}]`)},
	}
	toolResult := ac.Message{Role: ac.RoleTool, Content: []ac.ContentBlock{ac.TextBlock("ok")}, Metadata: map[string]any{"tool_call_id": "c1"}}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q"), prior, toolResult}, nil); err != nil {
		t.Fatal(err)
	}
	if len(body.Messages) != 3 {
		t.Fatalf("messages = %d", len(body.Messages))
	}
	if string(body.Messages[1].ReasoningDetails) != `[{"type":"reasoning.encrypted","data":"xyz"}]` {
		t.Errorf("reasoning_details not echoed: %s", body.Messages[1].ReasoningDetails)
	}
	if body.Messages[1].Reasoning != "" {
		t.Errorf("plaintext reasoning must not be sent when details are present, got %q", body.Messages[1].Reasoning)
	}
	if body.Messages[2].Role != "tool" || body.Messages[2].ToolCallID != "c1" {
		t.Errorf("tool message = %+v", body.Messages[2])
	}
}

func TestPlaintextReasoningEchoedWithoutDetails(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	prior := ac.Message{Role: ac.RoleAssistant, Content: []ac.ContentBlock{ac.ThinkingBlock("thought"), ac.TextBlock("partial")}}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q"), prior}, nil); err != nil {
		t.Fatal(err)
	}
	if body.Messages[1].Reasoning != "thought" {
		t.Errorf("reasoning = %q, want thought", body.Messages[1].Reasoning)
	}
}

func TestGenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"error":{"code":402,"message":"insufficient credits"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil || !strings.Contains(err.Error(), "insufficient credits") {
		t.Errorf("err = %v", err)
	}
}

func TestGenerateBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"error":{"code":400,"message":"model not found"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("err = %v", err)
	}
}

func TestSamplerParamsSent(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	temp, topP := 0.2, 0.9
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Temperature: &temp, TopP: &topP})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithMaxTokens(77)); err != nil {
		t.Fatal(err)
	}
	if body.Temperature == nil || *body.Temperature != 0.2 || body.TopP == nil || *body.TopP != 0.9 {
		t.Errorf("sampler = %+v %+v", body.Temperature, body.TopP)
	}
	if body.MaxTokens != 77 {
		t.Errorf("max_tokens = %d", body.MaxTokens)
	}
}

func TestGenerateMeters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150,"cost":0.00042,"prompt_tokens_details":{"cached_tokens":100},"completion_tokens_details":{"reasoning_tokens":10}}}`))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "moonshotai/kimi-k3", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { got = u })})
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Provider != "openrouter" || got.Model != "moonshotai/kimi-k3" {
		t.Errorf("identity = %+v", got)
	}
	if got.PromptTokens != 20 || got.CacheReadTokens != 100 || got.CompletionTokens != 30 || got.TotalTokens != 150 {
		t.Errorf("tokens = %+v", got)
	}
	if got.ReasoningTokens != 10 {
		t.Errorf("reasoning tokens = %d, want 10", got.ReasoningTokens)
	}
	if got.Cost != 0.00042 {
		t.Errorf("cost = %v", got.Cost)
	}
	if got.Latency <= 0 {
		t.Error("latency not recorded")
	}
	if resp.Message.Usage == nil || resp.Message.Usage.Input != 120 || resp.Message.Usage.CacheRead != 100 {
		t.Errorf("agent usage = %+v", resp.Message.Usage)
	}
	if resp.Message.Usage.Cost == nil || resp.Message.Usage.Cost.Total != 0.00042 {
		t.Errorf("agent usage cost = %+v", resp.Message.Usage.Cost)
	}
}

// TestReasoningDetailsEchoedAfterRoundTrip proves the reasoning_details echo
// survives a JSON round-trip of history: agentcore metadata that has been
// marshalled and unmarshalled comes back as []any, not json.RawMessage.
func TestReasoningDetailsEchoedAfterRoundTrip(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	var roundTripped any
	if err := json.Unmarshal([]byte(`[{"type":"reasoning.encrypted","data":"xyz"}]`), &roundTripped); err != nil {
		t.Fatal(err)
	}
	if _, ok := roundTripped.([]any); !ok {
		t.Fatalf("fixture is %T, want []any", roundTripped)
	}

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	prior := ac.Message{
		Role:     ac.RoleAssistant,
		Content:  []ac.ContentBlock{ac.ThinkingBlock("thought")},
		Metadata: map[string]any{"reasoning_details": roundTripped},
	}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q"), prior}, nil); err != nil {
		t.Fatal(err)
	}
	if string(body.Messages[1].ReasoningDetails) != `[{"data":"xyz","type":"reasoning.encrypted"}]` {
		t.Errorf("reasoning_details = %s, want the round-tripped echo", body.Messages[1].ReasoningDetails)
	}
	if body.Messages[1].Reasoning != "" {
		t.Errorf("plaintext reasoning must not be sent when details are present, got %q", body.Messages[1].Reasoning)
	}
}

// TestReasoningEffortOff proves ThinkingOff maps to OpenRouter's "none", which
// actually disables reasoning; only an unspecified level omits the object.
func TestReasoningEffortOff(t *testing.T) {
	if got := reasoningEffort(ac.ThinkingOff); got != "none" {
		t.Errorf("reasoningEffort(off) = %q, want none", got)
	}
	if got := reasoningEffort(""); got != "" {
		t.Errorf("reasoningEffort(unspecified) = %q, want empty", got)
	}
}

// TestResponseFormatSent proves a call's response format reaches the wire.
func TestResponseFormatSent(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithJSONMode()); err != nil {
		t.Fatal(err)
	}
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format on wire = %#v", body["response_format"])
	}
	if rf["type"] != "json_object" {
		t.Errorf("response_format = %#v, want type json_object", rf)
	}

	body = nil
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, present := body["response_format"]; present {
		t.Errorf("response_format must be omitted when unset, got %#v", body["response_format"])
	}
}

// TestGenerateObservesUsageOnErrorResponse proves billed usage on an errored
// response still reaches the Meter: the provider charged for the call.
func TestGenerateObservesUsageOnErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"error":{"code":502,"message":"upstream failed"},"usage":{"prompt_tokens":11,"completion_tokens":0,"total_tokens":11,"cost":0.0003}}`))
	}))
	defer srv.Close()

	var metered llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { metered = u })})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err == nil {
		t.Fatal("expected an error")
	}
	if metered.TotalTokens != 11 {
		t.Errorf("usage not metered on an errored response: %+v", metered)
	}
}
