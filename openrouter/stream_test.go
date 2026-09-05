package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

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
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: llm.MeterFunc(func(llm.Usage) { metered = true })})
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

// TestStreamTruncatedMidToolCall proves a stream that ends at EOF without
// [DONE] and without a finish_reason emits no done event and no tool-call-end,
// so the agent loop builds a PartialStreamError and retries rather than
// persisting a half-written tool call. Usage that already arrived is still
// metered.
func TestStreamTruncatedMidToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a"}}]},"finish_reason":null}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"cost":0.0001}}` + "\n\n"))
	}))
	defer srv.Close()

	var metered llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { metered = u })})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, ev := range collect(t, ch) {
		if ev.Type == ac.StreamEventDone {
			t.Errorf("truncated stream emitted a done event: %+v", ev.Message)
		}
		if ev.Type == ac.StreamEventToolCallEnd {
			t.Errorf("truncated stream emitted a tool-call-end: %+v", ev.CompletedToolCall)
		}
	}
	if metered.TotalTokens != 7 {
		t.Errorf("usage not metered on a truncated stream: %+v", metered)
	}
}

// TestStreamErrorChunkStillMeters proves usage that arrived before a
// mid-stream error chunk still reaches the Meter: the call was billed.
func TestStreamErrorChunkStillMeters(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"content":"par"},"finish_reason":null}],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10,"cost":0.0002}}`,
		`{"error":{"code":502,"message":"upstream failed"}}`,
	)
	defer srv.Close()

	var metered llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { metered = u })})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	if last := evs[len(evs)-1]; last.Type != ac.StreamEventError {
		t.Fatalf("last event = %+v, want error", last)
	}
	if metered.TotalTokens != 10 {
		t.Errorf("usage not metered on an aborted stream: %+v", metered)
	}
}

// TestStreamContextCancelReleasesProducer proves the producer goroutine exits
// when the caller cancels while nothing is reading the channel: every send is
// guarded by the request context, so a stalled consumer cannot pin the
// goroutine and the HTTP body forever.
func TestStreamContextCancelReleasesProducer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for range 500 {
			if _, err := w.Write([]byte(`data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}` + "\n\n")); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})

	// Warm the transport so the baseline includes its connection goroutines.
	warm, cancelWarm := context.WithCancel(context.Background())
	if ch, err := a.GenerateStream(warm, []ac.Message{ac.UserMsg("hi")}, nil); err == nil {
		go func() {
			for range ch {
			}
		}()
	}
	time.Sleep(200 * time.Millisecond)
	cancelWarm()
	time.Sleep(200 * time.Millisecond)
	baseline := runtime.NumGoroutine()

	ctx, cancel := context.WithCancel(context.Background())
	if _, err := a.GenerateStream(ctx, []ac.Message{ac.UserMsg("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	// Nothing reads the channel: the producer fills the buffer and blocks.
	time.Sleep(300 * time.Millisecond)
	cancel()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= baseline {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("goroutines = %d after cancel, baseline %d: producer did not exit", runtime.NumGoroutine(), baseline)
}

// TestStreamIdleTimeout proves a stream that stalls mid-flight aborts on the
// idle deadline with an error the agent loop classifies as stream-idle,
// instead of hanging until the request timeout.
func TestStreamIdleTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"one"},"finish_reason":null}]}` + "\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, StreamIdleTimeout: 200 * time.Millisecond})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var last ac.StreamEvent
	for ev := range ch {
		last = ev
		if ev.Type == ac.StreamEventDone {
			t.Errorf("stalled stream emitted a done event: %+v", ev.Message)
		}
	}
	if last.Type != ac.StreamEventError || last.Err == nil {
		t.Fatalf("last event = %+v, want an error", last)
	}
	if !errors.Is(last.Err, ac.ErrProviderStreamIdle) {
		t.Errorf("err = %v, want it to wrap ErrProviderStreamIdle", last.Err)
	}
}
