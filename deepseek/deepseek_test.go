package deepseek

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

func TestGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header = %q, want Bearer test-key", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"a short summary"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, err := New(Config{APIKey: "test-key", Model: "deepseek-chat", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("summarize")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.TextContent(); got != "a short summary" {
		t.Errorf("text = %q, want %q", got, "a short summary")
	}
}

func TestGenerateMeters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}`))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{
		APIKey: "x", Model: "deepseek-chat", BaseURL: srv.URL,
		Meter: llm.MeterFunc(func(u llm.Usage) { got = u }),
	})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if got.PromptTokens != 11 || got.CompletionTokens != 3 || got.TotalTokens != 14 {
		t.Errorf("usage tokens = %+v, want 11/3/14", got)
	}
	if got.Provider != "deepseek" || got.Model != "deepseek-chat" {
		t.Errorf("provider/model = %q/%q", got.Provider, got.Model)
	}
	if got.Latency <= 0 {
		t.Error("latency not measured")
	}
}

func TestGenerateErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "x", Model: "deepseek-chat", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil); err == nil {
		t.Fatal("expected an error on 401")
	}
}

func TestGenerateStreamEmitsDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "x", Model: "deepseek-chat", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var last ac.StreamEvent
	n := 0
	for ev := range ch {
		last = ev
		n++
	}
	if n != 1 || last.Type != ac.StreamEventDone {
		t.Errorf("got %d events, last type %v; want 1 StreamEventDone", n, last.Type)
	}
}
