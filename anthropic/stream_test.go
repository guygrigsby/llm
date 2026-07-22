package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	agentcore "github.com/voocel/agentcore"
)

// fakeStream replays a fixed list of SDK stream events, mimicking the SDK
// stream's Next/Current/Err surface so emitStream can be exercised offline.
type fakeStream struct {
	events []anthropic.MessageStreamEventUnion
	pos    int
	err    error
}

func (f *fakeStream) Next() bool {
	if f.pos >= len(f.events) {
		return false
	}
	f.pos++
	return true
}

func (f *fakeStream) Current() anthropic.MessageStreamEventUnion { return f.events[f.pos-1] }
func (f *fakeStream) Err() error                                 { return f.err }

// event decodes a stream event from raw JSON. The SDK's Accumulate and AsAny
// both read the union's raw JSON, so events must be built by unmarshalling
// rather than setting struct fields.
func event(t *testing.T, raw string) anthropic.MessageStreamEventUnion {
	t.Helper()
	var ev anthropic.MessageStreamEventUnion
	if err := ev.UnmarshalJSON([]byte(raw)); err != nil {
		t.Fatalf("decode event %q: %v", raw, err)
	}
	return ev
}

func collect(out <-chan agentcore.StreamEvent) []agentcore.StreamEvent {
	var events []agentcore.StreamEvent
	for e := range out {
		events = append(events, e)
	}
	return events
}

func TestEmitStream_TextThenToolCall(t *testing.T) {
	// A turn that streams a text block, then a tool_use block whose arguments
	// arrive across two partial_json deltas, then stops with tool_use.
	fs := &fakeStream{events: []anthropic.MessageStreamEventUnion{
		event(t, `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}`),
		event(t, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		event(t, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"check"}}`),
		event(t, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ing"}}`),
		event(t, `{"type":"content_block_stop","index":0}`),
		event(t, `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_7","name":"service_status","input":{}}}`),
		event(t, `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"name\":"}}`),
		event(t, `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"nginx\"}"}}`),
		event(t, `{"type":"content_block_stop","index":1}`),
		event(t, `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":12}}`),
		event(t, `{"type":"message_stop"}`),
	}}

	out := make(chan agentcore.StreamEvent, 64)
	go func() {
		defer close(out)
		emitStream(fs, out, nil)
	}()
	events := collect(out)

	// Expected framing: text start/delta/delta/end, toolcall start/delta/delta/end, done.
	want := []agentcore.StreamEventType{
		agentcore.StreamEventTextStart,
		agentcore.StreamEventTextDelta,
		agentcore.StreamEventTextDelta,
		agentcore.StreamEventTextEnd,
		agentcore.StreamEventToolCallStart,
		agentcore.StreamEventToolCallDelta,
		agentcore.StreamEventToolCallDelta,
		agentcore.StreamEventToolCallEnd,
		agentcore.StreamEventDone,
	}
	if len(events) != len(want) {
		t.Fatalf("got %d events, want %d: %v", len(events), len(want), types(events))
	}
	for i, w := range want {
		if events[i].Type != w {
			t.Errorf("event[%d] = %q, want %q (seq %v)", i, events[i].Type, w, types(events))
		}
	}

	// The tool-call-end event must carry the completed call with parsed args.
	end := events[7]
	if end.CompletedToolCall == nil {
		t.Fatal("toolcall_end missing CompletedToolCall")
	}
	if end.CompletedToolCall.ID != "toolu_7" || end.CompletedToolCall.Name != "service_status" {
		t.Errorf("completed call id/name = %q/%q", end.CompletedToolCall.ID, end.CompletedToolCall.Name)
	}
	if string(end.CompletedToolCall.Args) != `{"name":"nginx"}` {
		t.Errorf("completed call args = %s", end.CompletedToolCall.Args)
	}

	// LOAD-BEARING: the done message must carry the tool call and stop reason.
	done := events[8]
	calls := done.Message.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu_7" {
		t.Fatalf("done message tool calls = %v, want one toolu_7", calls)
	}
	if string(calls[0].Args) != `{"name":"nginx"}` {
		t.Errorf("done tool call args = %s", calls[0].Args)
	}
	if done.Message.TextContent() != "checking" {
		t.Errorf("done text = %q, want 'checking'", done.Message.TextContent())
	}
	if done.StopReason != agentcore.StopReasonToolUse {
		t.Errorf("done stop reason = %q, want toolUse", done.StopReason)
	}
}

func TestEmitStream_StreamError(t *testing.T) {
	fs := &fakeStream{
		events: []anthropic.MessageStreamEventUnion{
			event(t, `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`),
		},
		err: errNoModel, // any non-nil error
	}
	out := make(chan agentcore.StreamEvent, 16)
	go func() {
		defer close(out)
		emitStream(fs, out, nil)
	}()
	events := collect(out)
	last := events[len(events)-1]
	if last.Type != agentcore.StreamEventError {
		t.Fatalf("last event = %q, want error", last.Type)
	}
	if last.Err == nil {
		t.Error("error event has nil Err")
	}
	// No done event should follow an error.
	for _, e := range events {
		if e.Type == agentcore.StreamEventDone {
			t.Error("done event emitted despite stream error")
		}
	}
}

func types(events []agentcore.StreamEvent) []agentcore.StreamEventType {
	out := make([]agentcore.StreamEventType, len(events))
	for i, e := range events {
		out[i] = e.Type
	}
	return out
}
