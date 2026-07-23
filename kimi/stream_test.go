package kimi

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

// sseServer streams the given raw SSE payload with the event-stream content
// type, so the adapter reads it as a live stream.
func sseServer(t *testing.T, payload string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(payload))
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

// TestStreamText proves reasoning then text stream as thinking/text framing and
// the terminal done event carries the assembled message + usage.
func TestStreamText(t *testing.T) {
	payload := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"cached_tokens\":4}}\n\n" +
		"data: [DONE]\n\n"
	srv := sseServer(t, payload)
	defer srv.Close()

	var metered llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { metered = u })})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)

	// Ordered framing: thinking start/delta/end, text start/delta*, text end, done.
	want := []ac.StreamEventType{
		ac.StreamEventThinkingStart, ac.StreamEventThinkingDelta, ac.StreamEventThinkingEnd,
		ac.StreamEventTextStart, ac.StreamEventTextDelta, ac.StreamEventTextDelta,
		ac.StreamEventTextEnd, ac.StreamEventDone,
	}
	if len(evs) != len(want) {
		t.Fatalf("events = %d %v, want %d", len(evs), types(evs), len(want))
	}
	for i := range want {
		if evs[i].Type != want[i] {
			t.Errorf("event[%d] = %q, want %q", i, evs[i].Type, want[i])
		}
	}
	done := evs[len(evs)-1]
	if got := done.Message.TextContent(); got != "hello" {
		t.Errorf("done text = %q", got)
	}
	if got := done.Message.ThinkingContent(); got != "hmm" {
		t.Errorf("done thinking = %q", got)
	}
	if done.StopReason != ac.StopReasonStop {
		t.Errorf("done stop = %q", done.StopReason)
	}
	if metered.PromptTokens != 6 || metered.CacheReadTokens != 4 {
		t.Errorf("metered = %+v, want prompt 6 / cache 4", metered)
	}
}

// TestStreamToolCall proves a tool call streamed across fragments assembles into
// one completed tool call on the toolcall-end event and the done message.
func TestStreamToolCall(t *testing.T) {
	payload := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"42}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	srv := sseServer(t, payload)
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "kimi-k3", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("go")}, []ac.ToolSpec{{Name: "lookup"}})
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)

	var end *ac.StreamEvent
	for i := range evs {
		if evs[i].Type == ac.StreamEventToolCallEnd {
			end = &evs[i]
		}
	}
	if end == nil {
		t.Fatalf("no toolcall_end event in %v", types(evs))
	}
	if end.CompletedToolCall == nil {
		t.Fatal("toolcall_end missing completed call")
	}
	call := end.CompletedToolCall
	if call.ID != "call_9" || call.Name != "lookup" || string(call.Args) != `{"q":42}` {
		t.Errorf("completed call = %+v", call)
	}

	done := evs[len(evs)-1]
	if done.Type != ac.StreamEventDone {
		t.Fatalf("last event = %q, want done", done.Type)
	}
	calls := done.Message.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":42}` {
		t.Errorf("done tool calls = %+v", calls)
	}
	if done.StopReason != ac.StopReasonToolUse {
		t.Errorf("done stop = %q, want toolUse", done.StopReason)
	}
}

func types(evs []ac.StreamEvent) []ac.StreamEventType {
	out := make([]ac.StreamEventType, len(evs))
	for i, e := range evs {
		out[i] = e.Type
	}
	return out
}
