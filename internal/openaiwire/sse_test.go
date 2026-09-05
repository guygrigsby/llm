package openaiwire

import (
	"context"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

// TestEmitterSequence drives Emitter with a delta sequence (reasoning, text,
// then a tool call streamed across two argument fragments) and asserts both
// the emitted event sequence and the flushed terminal message.
func TestEmitterSequence(t *testing.T) {
	out := make(chan ac.StreamEvent, 64)
	em := NewEmitter(context.Background(), out)

	em.Handle(Delta{Reasoning: "hmm"}, "")
	em.Handle(Delta{Content: "hel"}, "")
	em.Handle(Delta{Content: "lo"}, "")
	em.Handle(Delta{ToolCalls: []StreamToolCall{{Index: 0, ID: "call_1", Type: "function", Function: Func{Name: "lookup"}}}}, "")
	em.Handle(Delta{ToolCalls: []StreamToolCall{{Index: 0, Function: Func{Arguments: `{"q":`}}}}, "")
	em.Handle(Delta{ToolCalls: []StreamToolCall{{Index: 0, Function: Func{Arguments: `42}`}}}}, "tool_calls")

	final := em.Flush(func(reasoning, content string, calls []ToolCall, finish string) ac.Message {
		return AssembleMessage(reasoning, content, calls, finish)
	})
	close(out)

	var evs []ac.StreamEvent
	for ev := range out {
		evs = append(evs, ev)
	}

	want := []ac.StreamEventType{
		ac.StreamEventThinkingStart, ac.StreamEventThinkingDelta, ac.StreamEventThinkingEnd,
		ac.StreamEventTextStart, ac.StreamEventTextDelta, ac.StreamEventTextDelta, ac.StreamEventTextEnd,
		ac.StreamEventToolCallStart, ac.StreamEventToolCallDelta, ac.StreamEventToolCallDelta,
		ac.StreamEventToolCallEnd,
	}
	if len(evs) != len(want) {
		t.Fatalf("events = %d, want %d: %v", len(evs), len(want), eventTypes(evs))
	}
	for i := range want {
		if evs[i].Type != want[i] {
			t.Errorf("event[%d] = %q, want %q (%v)", i, evs[i].Type, want[i], eventTypes(evs))
		}
	}

	end := evs[len(evs)-1]
	if end.CompletedToolCall == nil {
		t.Fatal("toolcall_end missing completed call")
	}
	if end.CompletedToolCall.ID != "call_1" || end.CompletedToolCall.Name != "lookup" || string(end.CompletedToolCall.Args) != `{"q":42}` {
		t.Errorf("completed call = %+v", end.CompletedToolCall)
	}

	if final.TextContent() != "hello" {
		t.Errorf("final text = %q, want hello", final.TextContent())
	}
	if final.ThinkingContent() != "hmm" {
		t.Errorf("final thinking = %q, want hmm", final.ThinkingContent())
	}
	if final.StopReason != ac.StopReasonToolUse {
		t.Errorf("final stop = %q, want toolUse", final.StopReason)
	}
	calls := final.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":42}` {
		t.Errorf("final tool calls = %+v", calls)
	}
}

func eventTypes(evs []ac.StreamEvent) []ac.StreamEventType {
	out := make([]ac.StreamEventType, len(evs))
	for i, e := range evs {
		out[i] = e.Type
	}
	return out
}

// TestEmitterAbortsOnContextCancel proves a send that no consumer is reading
// releases when the request context is cancelled, so the producer goroutine
// (and the HTTP body it holds) cannot be pinned by a stalled consumer.
func TestEmitterAbortsOnContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan ac.StreamEvent) // unbuffered: nobody reads
	em := NewEmitter(ctx, out)

	returned := make(chan struct{})
	go func() {
		em.Handle(Delta{Content: "hi"}, "")
		close(returned)
	}()
	cancel()
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("Handle did not return after the context was cancelled")
	}
	if !em.Aborted() {
		t.Error("emitter not marked aborted after cancel")
	}
}
