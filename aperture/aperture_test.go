package aperture

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

func TestNewValidatesAndDefaults(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Error("missing Model should fail")
	}
	// No APIKey: the tailnet authenticates the host, so none is required.
	a, err := New(Config{Model: "anthropic/claude-sonnet-5.5"})
	if err != nil {
		t.Fatal(err)
	}
	if a.client.BaseURL != "https://ai.guy.ts.net/v1" {
		t.Errorf("baseURL = %q", a.client.BaseURL)
	}
	if a.ProviderName() != "aperture" || !a.SupportsTools() {
		t.Error("provider name or tools flag wrong")
	}
}

// TestNoKeyNoAuthorizationHeader proves the header is omitted, not sent
// empty: a strict frontend would read "Bearer " as a malformed credential.
func TestNoKeyNoAuthorizationHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := r.Header["Authorization"]; ok {
			t.Errorf("Authorization = %q, want none", r.Header.Get("Authorization"))
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{Model: "qwen-27b", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
}

// TestKeySendsBearer proves an explicit key still reaches the wire for routes
// that ask for one.
func TestKeySendsBearer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Errorf("auth = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "qwen-27b", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
}

// TestGenerateMeters proves usage reaches the Meter labelled aperture, with
// the dialect's cost and cached/reasoning token accounting.
func TestGenerateMeters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150,"cost":0.00042,"prompt_tokens_details":{"cached_tokens":100},"completion_tokens_details":{"reasoning_tokens":10}}}`))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{Model: "qwen-27b", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { got = u })})
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Provider != "aperture" || got.Model != "qwen-27b" {
		t.Errorf("identity = %+v", got)
	}
	if got.PromptTokens != 20 || got.CacheReadTokens != 100 || got.CompletionTokens != 30 || got.TotalTokens != 150 {
		t.Errorf("tokens = %+v", got)
	}
	if got.ReasoningTokens != 10 || got.Cost != 0.00042 || got.Latency <= 0 {
		t.Errorf("cost/reasoning/latency = %+v", got)
	}
	if resp.Message.Usage == nil || resp.Message.Usage.Input != 120 {
		t.Errorf("agent usage = %+v", resp.Message.Usage)
	}
}

// TestErrorMessageWithoutCode proves Aperture's error shape (a message, no
// numeric code, as its "no route found" body carries) renders without a
// phantom zero.
func TestErrorMessageWithoutCode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Some routes answer 200 with an error body rather than an HTTP
		// error status; the parsed block must render without a phantom zero.
		_, _ = w.Write([]byte(`{"error":{"message":"no route found for model \"x\""},"source":"aperture"}`))
	}))
	defer srv.Close()

	a, _ := New(Config{Model: "x", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil || !strings.Contains(err.Error(), "aperture: no route found") || strings.Contains(err.Error(), "0:") {
		t.Errorf("err = %v", err)
	}
}

// TestStreamAssembles proves a streamed turn arrives as one done event with
// the assembled text, stop reason and usage.
func TestStreamAssembles(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"he\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7,\"cost\":0.0001}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{Model: "qwen-27b", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { got = u })})
	out, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var done *ac.StreamEvent
	for ev := range out {
		if ev.Type == ac.StreamEventDone {
			e := ev
			done = &e
		}
		if ev.Type == ac.StreamEventError {
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	if done == nil {
		t.Fatal("no done event")
	}
	if done.Message.TextContent() != "hello" {
		t.Errorf("text = %q", done.Message.TextContent())
	}
	if done.Message.StopReason != "stop" {
		t.Errorf("stop = %q", done.Message.StopReason)
	}
	if got.Provider != "aperture" || got.TotalTokens != 7 || got.Cost != 0.0001 {
		t.Errorf("metered = %+v", got)
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		fmt.Fprint(w, `{"data":[{"id":"anthropic/claude-sonnet-5.5"},{"id":"qwen-27b"},{"id":""}]}`)
	}))
	defer srv.Close()
	got, err := ListModels(t.Context(), srv.URL+"/v1", "k")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"anthropic/claude-sonnet-5.5", "qwen-27b"}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestListModelsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()
	if _, err := ListModels(t.Context(), srv.URL, ""); err == nil {
		t.Fatal("want error on 401")
	}
}
