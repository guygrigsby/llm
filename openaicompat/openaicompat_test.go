package openaicompat

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

const okBody = `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

func TestNewValidates(t *testing.T) {
	for _, cfg := range []Config{
		{BaseURL: "https://x/v1"},
		{Model: "m"},
		{Model: "m", BaseURL: "ftp://x"},
		{Model: "m", BaseURL: "https://"},
	} {
		if _, err := New(cfg); err == nil {
			t.Errorf("New(%+v): want error", cfg)
		}
	}
	a, err := New(Config{Model: "m", BaseURL: "https://x/v1/"})
	if err != nil {
		t.Fatal(err)
	}
	if a.client.BaseURL != "https://x/v1" || a.ProviderName() != "openaicompat" || !a.SupportsTools() {
		t.Errorf("baseURL %q, provider %q", a.client.BaseURL, a.ProviderName())
	}
}

// TestPlainOpenAIRequest proves the body is the plain OpenAI shape every
// compatible server accepts: max_tokens, reasoning_effort only when a thinking
// level is set, and no OpenRouter reasoning object or reasoning echo on
// assistant history.
func TestPlainOpenAIRequest(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		body = nil
		_ = json.Unmarshal(b, &body)
		_, _ = w.Write([]byte(okBody))
	}))
	defer srv.Close()

	a, _ := New(Config{Model: "m", BaseURL: srv.URL + "/v1"})
	history := []ac.Message{
		ac.UserMsg("hi"),
		{Role: ac.RoleAssistant, Content: []ac.ContentBlock{ac.ThinkingBlock("hmm"), ac.TextBlock("hello")}},
		ac.UserMsg("again"),
	}
	if _, err := a.Generate(context.Background(), history, nil, ac.WithMaxTokens(50), ac.WithThinking(ac.ThinkingLow)); err != nil {
		t.Fatal(err)
	}
	if body["max_tokens"] != float64(50) || body["max_completion_tokens"] != nil {
		t.Errorf("max tokens: %v / %v", body["max_tokens"], body["max_completion_tokens"])
	}
	if body["reasoning_effort"] != "low" || body["reasoning"] != nil {
		t.Errorf("reasoning: effort %v, object %v", body["reasoning_effort"], body["reasoning"])
	}
	asst := body["messages"].([]any)[1].(map[string]any)
	if asst["content"] != "hello" || asst["reasoning"] != nil || asst["reasoning_details"] != nil {
		t.Errorf("assistant message = %v", asst)
	}

	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Error("reasoning_effort sent with no thinking level")
	}
}

// TestKeys proves no key sends no Authorization (local servers often have
// none) and a per-call key overrides the configured one, so one adapter can
// serve many users' own keys.
func TestKeys(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Header.Get("Authorization"))
		_, _ = w.Write([]byte(okBody))
	}))
	defer srv.Close()

	bare, _ := New(Config{Model: "m", BaseURL: srv.URL})
	keyed, _ := New(Config{Model: "m", BaseURL: srv.URL, APIKey: "cfg"})
	msgs := []ac.Message{ac.UserMsg("q")}
	_, _ = bare.Generate(context.Background(), msgs, nil)
	_, _ = keyed.Generate(context.Background(), msgs, nil)
	_, _ = keyed.Generate(context.Background(), msgs, nil, ac.WithAPIKey("user"))
	want := []string{"", "Bearer cfg", "Bearer user"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("auth = %q, want %q", got, want)
	}
}

type countingTransport struct{ n atomic.Int32 }

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	c.n.Add(1)
	return http.DefaultTransport.RoundTrip(r)
}

// TestTransportUsed proves both one-shot and streaming calls go through the
// caller's transport, where a host restricting user-supplied URLs (SSRF
// guard) installs its dialer.
func TestTransportUsed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(okBody))
	}))
	defer srv.Close()

	tr := &countingTransport{}
	a, _ := New(Config{Model: "m", BaseURL: srv.URL, Transport: tr})
	_, _ = a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	out, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil {
		for range out {
		}
	}
	if n := tr.n.Load(); n != 2 {
		t.Errorf("transport saw %d requests, want 2", n)
	}
}

func TestStreamAssembles(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"he\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{Model: "m", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { got = u })})
	out, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for ev := range out {
		switch ev.Type {
		case ac.StreamEventDone:
			text = ev.Message.TextContent()
		case ac.StreamEventError:
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	if text != "hello" {
		t.Errorf("text = %q", text)
	}
	if got.Provider != "openaicompat" || got.Model != "m" || got.TotalTokens != 7 {
		t.Errorf("metered = %+v", got)
	}
}
