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

type llmUsage = llm.Usage

func sseServer(t *testing.T, frames ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("accept = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(": OPENROUTER PROCESSING\n\n"))
		for _, f := range frames {
			_, _ = w.Write([]byte("data: " + f + "\n\n"))
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
}

func collect(t *testing.T, ch <-chan ac.StreamEvent) []ac.StreamEvent {
	t.Helper()
	var evs []ac.StreamEvent
	for ev := range ch {
		evs = append(evs, ev)
	}
	return evs
}

func TestStreamText(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"reasoning":"hmm "},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning":"ok"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"cost":0.0001}}`,
	)
	defer srv.Close()

	var metered bool
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: meterFunc(func() { metered = true })})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	var text, thinking strings.Builder
	var done *ac.StreamEvent
	for i := range evs {
		switch evs[i].Type {
		case ac.StreamEventTextDelta:
			text.WriteString(evs[i].Delta)
		case ac.StreamEventThinkingDelta:
			thinking.WriteString(evs[i].Delta)
		case ac.StreamEventDone:
			done = &evs[i]
		case ac.StreamEventError:
			t.Fatalf("unexpected error event: %v", evs[i].Err)
		}
	}
	if text.String() != "Hello" || thinking.String() != "hmm ok" {
		t.Errorf("text = %q thinking = %q", text.String(), thinking.String())
	}
	if done == nil {
		t.Fatal("no done event")
	}
	if done.Message.TextContent() != "Hello" || done.Message.ThinkingContent() != "hmm ok" || done.StopReason != ac.StopReasonStop {
		t.Errorf("final = %+v", done.Message)
	}
	if done.Message.Usage == nil || done.Message.Usage.Input != 5 {
		t.Errorf("usage = %+v", done.Message.Usage)
	}
	if !metered {
		t.Error("meter not called")
	}
}

func TestStreamToolCallAndReasoningDetails(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"a"}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"b"}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":"tool_calls"}]}`,
	)
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	var ends int
	var done *ac.StreamEvent
	for i := range evs {
		switch evs[i].Type {
		case ac.StreamEventToolCallEnd:
			ends++
			if evs[i].CompletedToolCall == nil || evs[i].CompletedToolCall.Name != "lookup" || string(evs[i].CompletedToolCall.Args) != `{"q":"x"}` {
				t.Errorf("completed call = %+v", evs[i].CompletedToolCall)
			}
		case ac.StreamEventDone:
			done = &evs[i]
		}
	}
	if ends != 1 {
		t.Errorf("tool call ends = %d", ends)
	}
	if done == nil || done.StopReason != ac.StopReasonToolUse {
		t.Fatalf("done = %+v", done)
	}
	details, _ := done.Message.Metadata["reasoning_details"].(json.RawMessage)
	if string(details) != `[{"type":"reasoning.text","text":"a"},{"type":"reasoning.encrypted","data":"b"}]` {
		t.Errorf("reasoning_details = %s", details)
	}
}

func TestStreamErrorChunk(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"content":"par"},"finish_reason":null}]}`,
		`{"error":{"code":502,"message":"upstream failed"}}`,
	)
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	last := evs[len(evs)-1]
	if last.Type != ac.StreamEventError || last.Err == nil || !strings.Contains(last.Err.Error(), "upstream failed") {
		t.Errorf("last event = %+v", last)
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"rate limited"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	if _, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil); err == nil || !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("err = %v", err)
	}
}

type meterFunc func()

func (f meterFunc) Observe(_ llmUsage) { f() }
