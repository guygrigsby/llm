package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go"
	agentcore "github.com/voocel/agentcore"
)

// emitter translates the Anthropic incremental stream-event sequence into the
// agentcore start/delta/end framing the agent loop consumes. It tracks the kind
// of the currently-open content block so deltas and the closing event map to
// the right event type.
//
// The fully assembled message (accumulated separately) is what the terminal
// done event carries, so this framing is for live progress; the exact tool-call
// set is guaranteed by convertResponse on the done event, not by these deltas.
type emitter struct {
	out     chan<- agentcore.StreamEvent
	kind    blockKind // kind of the currently-open block
	index   int       // agentcore content index of the open block
	partial agentcore.Message
}

type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockThinking
	blockToolCall
)

// handle processes one SDK stream event, emitting the corresponding agentcore
// events. acc is the message accumulated so far; it is attached to each emitted
// event as the partial snapshot.
func (e *emitter) handle(event anthropic.MessageStreamEventUnion, acc *anthropic.Message) {
	e.partial = convertResponse(acc)
	switch event.Type {
	case "content_block_start":
		e.start(event)
	case "content_block_delta":
		e.delta(event)
	case "content_block_stop":
		e.stop()
	}
}

// start opens a new content block, emitting the matching *Start event.
func (e *emitter) start(event anthropic.MessageStreamEventUnion) {
	e.index = int(event.Index)
	switch event.ContentBlock.Type {
	case "text":
		e.kind = blockText
		e.send(agentcore.StreamEventTextStart)
	case "thinking", "redacted_thinking":
		e.kind = blockThinking
		e.send(agentcore.StreamEventThinkingStart)
	case "tool_use", "server_tool_use":
		e.kind = blockToolCall
		e.send(agentcore.StreamEventToolCallStart)
	default:
		e.kind = blockNone
	}
}

// delta forwards an incremental delta as the matching *Delta event.
func (e *emitter) delta(event anthropic.MessageStreamEventUnion) {
	switch e.kind {
	case blockText:
		if event.Delta.Text != "" {
			e.sendDelta(agentcore.StreamEventTextDelta, event.Delta.Text)
		}
	case blockThinking:
		if event.Delta.Thinking != "" {
			e.sendDelta(agentcore.StreamEventThinkingDelta, event.Delta.Thinking)
		}
	case blockToolCall:
		if event.Delta.PartialJSON != "" {
			e.sendDelta(agentcore.StreamEventToolCallDelta, event.Delta.PartialJSON)
		}
	}
}

// stop closes the currently-open block, emitting the matching *End event. For a
// tool call it also attaches the completed call (read from the accumulated
// partial) so the loop can begin execution without re-parsing.
func (e *emitter) stop() {
	switch e.kind {
	case blockText:
		e.send(agentcore.StreamEventTextEnd)
	case blockThinking:
		e.send(agentcore.StreamEventThinkingEnd)
	case blockToolCall:
		ev := agentcore.StreamEvent{
			Type:         agentcore.StreamEventToolCallEnd,
			ContentIndex: e.index,
			Message:      e.partial,
		}
		if call := e.completedToolCall(); call != nil {
			ev.CompletedToolCall = call
		}
		e.out <- ev
	}
	e.kind = blockNone
}

// completedToolCall returns the tool call at the open block index from the
// accumulated partial message, if present.
func (e *emitter) completedToolCall() *agentcore.ToolCall {
	if e.index < 0 || e.index >= len(e.partial.Content) {
		return nil
	}
	block := e.partial.Content[e.index]
	if block.Type == agentcore.ContentToolCall && block.ToolCall != nil {
		call := *block.ToolCall
		return &call
	}
	return nil
}

func (e *emitter) send(t agentcore.StreamEventType) {
	e.out <- agentcore.StreamEvent{Type: t, ContentIndex: e.index, Message: e.partial}
}

func (e *emitter) sendDelta(t agentcore.StreamEventType, delta string) {
	e.out <- agentcore.StreamEvent{Type: t, ContentIndex: e.index, Delta: delta, Message: e.partial}
}
