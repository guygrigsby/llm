package openaiwire

import (
	"bufio"
	"context"
	"io"
	"strings"

	ac "github.com/voocel/agentcore"
)

// This file drives an OpenAI-format SSE stream, translating the incremental
// deltas an adapter extracts from its own chunk type into the agentcore
// start/delta/end framing the agent loop consumes. The terminal message is
// assembled once at Flush; live deltas are for progress only. Usage
// accounting and mid-stream error chunks stay the adapter's concern: this
// package only knows content, reasoning and tool-call framing.

// ReadSSE reads r line by line to completion, extracting each "data:"
// frame's payload and invoking onData with it. It stops at a data frame of
// "[DONE]" or when onData reports stop=true, and returns any error onData
// reports or any read error other than a clean EOF (an SSE body ends without
// a trailing frame, so that is not itself an error).
//
// done reports whether the stream terminated on an explicit "[DONE]" frame.
// A false done with a nil error means the body simply ended: the response may
// be truncated mid message, so the caller must not treat what it accumulated
// as a complete turn unless a finish_reason arrived.
func ReadSSE(r io.Reader, onData func(data string) (stop bool, err error)) (done bool, err error) {
	br := bufio.NewReader(r)
	for {
		line, readErr := br.ReadString('\n')
		if data, ok := sseData(line); ok {
			if data == "[DONE]" {
				return true, nil
			}
			stop, onErr := onData(data)
			if onErr != nil {
				return false, onErr
			}
			if stop {
				return false, nil
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				return false, readErr
			}
			return false, nil
		}
	}
}

// sseData extracts the payload of a `data:` SSE line, or ok=false for blank
// lines and other fields (event:, id:, comments).
func sseData(line string) (string, bool) {
	line = strings.TrimRight(line, "\r\n")
	if !strings.HasPrefix(line, "data:") {
		return "", false
	}
	return strings.TrimSpace(line[len("data:"):]), true
}

// Send delivers ev on out unless ctx is already done, reporting whether the
// send happened. Every producer send goes through this: a consumer that stops
// reading must not pin the producer goroutine or the HTTP body it holds for
// the life of the process.
func Send(ctx context.Context, out chan<- ac.StreamEvent, ev ac.StreamEvent) bool {
	select {
	case out <- ev:
		return true
	case <-ctx.Done():
		return false
	}
}

// Emitter accumulates streamed content and frames it as agentcore
// StreamEvents. It tracks whether the thinking/text blocks are open and
// builds tool calls by provider index, assigning each block a stable
// agentcore content index. The adapter decodes its own chunk type and feeds
// this Emitter one Delta per choice via Handle.
type Emitter struct {
	Out chan<- ac.StreamEvent

	ctx     context.Context
	aborted bool

	nextIndex int

	thinkingOpen bool
	thinkingIdx  int
	thinking     strings.Builder

	textOpen bool
	textIdx  int
	text     strings.Builder

	builders map[int]*callBuilder // provider tool-call index -> builder
	order    []int                // provider indices, first-seen order

	finish string
}

// callBuilder accumulates one streamed tool call.
type callBuilder struct {
	contentIdx int
	id         string
	name       string
	args       strings.Builder
	started    bool
}

// NewEmitter builds an Emitter that sends its framed StreamEvents on out. ctx
// is the request context: once it is done the Emitter stops sending and
// reports Aborted, so the adapter can drop the rest of the stream.
func NewEmitter(ctx context.Context, out chan<- ac.StreamEvent) *Emitter {
	return &Emitter{Out: out, ctx: ctx, builders: map[int]*callBuilder{}}
}

// Aborted reports whether a send was abandoned because the request context
// ended. The adapter checks it before sending its own terminal events.
func (e *Emitter) Aborted() bool { return e.aborted }

// Finish returns the finish_reason seen so far, empty when the provider has
// not reported one. An empty Finish once the body has ended means the stream
// was cut short, not that the model stopped.
func (e *Emitter) Finish() string { return e.finish }

// Handle processes one delta: reasoning first (thinking), then text, then
// tool calls, recording the finish reason when present. The adapter calls
// Handle once per choice in its chunk (and, when a provider can attach more
// than one reasoning fragment to a single chunk, once more per fragment).
func (e *Emitter) Handle(d Delta, finish string) {
	if finish != "" {
		e.finish = finish
	}
	if d.Reasoning != "" {
		e.onReasoning(d.Reasoning)
	}
	if d.Content != "" {
		e.onContent(d.Content)
	}
	for _, tc := range d.ToolCalls {
		e.onToolCall(tc)
	}
}

func (e *Emitter) onReasoning(s string) {
	if !e.thinkingOpen {
		e.thinkingOpen = true
		e.thinkingIdx = e.nextIndex
		e.nextIndex++
		e.send(ac.StreamEventThinkingStart, e.thinkingIdx)
	}
	e.thinking.WriteString(s)
	e.sendDelta(ac.StreamEventThinkingDelta, e.thinkingIdx, s)
}

func (e *Emitter) onContent(s string) {
	e.closeThinking()
	if !e.textOpen {
		e.textOpen = true
		e.textIdx = e.nextIndex
		e.nextIndex++
		e.send(ac.StreamEventTextStart, e.textIdx)
	}
	e.text.WriteString(s)
	e.sendDelta(ac.StreamEventTextDelta, e.textIdx, s)
}

func (e *Emitter) onToolCall(tc StreamToolCall) {
	e.closeThinking()
	e.closeText()
	b := e.builders[tc.Index]
	if b == nil {
		b = &callBuilder{contentIdx: e.nextIndex}
		e.nextIndex++
		e.builders[tc.Index] = b
		e.order = append(e.order, tc.Index)
	}
	if tc.ID != "" {
		b.id = tc.ID
	}
	if tc.Function.Name != "" {
		b.name = tc.Function.Name
	}
	if !b.started {
		b.started = true
		e.send(ac.StreamEventToolCallStart, b.contentIdx)
	}
	if tc.Function.Arguments != "" {
		b.args.WriteString(tc.Function.Arguments)
		e.sendDelta(ac.StreamEventToolCallDelta, b.contentIdx, tc.Function.Arguments)
	}
}

func (e *Emitter) closeThinking() {
	if e.thinkingOpen {
		e.send(ac.StreamEventThinkingEnd, e.thinkingIdx)
		e.thinkingOpen = false
	}
}

func (e *Emitter) closeText() {
	if e.textOpen {
		e.send(ac.StreamEventTextEnd, e.textIdx)
		e.textOpen = false
	}
}

// Flush closes any open blocks, emits a tool-call-end (with the completed
// call) for each streamed tool call and returns the terminal message built
// by final from the accumulated reasoning, text, tool calls and finish
// reason. The caller decorates the result (Usage, provider-specific
// metadata) and sends it as the terminal Done event; usage and mid-stream
// errors are the adapter's concern, not this package's. Only a stream that
// actually finished may be flushed: a truncated one has no complete tool call
// or message to report.
func (e *Emitter) Flush(final func(reasoning, content string, calls []ToolCall, finish string) ac.Message) ac.Message {
	e.closeThinking()
	e.closeText()
	for _, idx := range e.order {
		b := e.builders[idx]
		call := ac.ToolCall{ID: b.id, Name: b.name, Args: RawArgs(b.args.String())}
		e.sendEvent(ac.StreamEvent{
			Type:              ac.StreamEventToolCallEnd,
			ContentIndex:      b.contentIdx,
			Message:           e.snapshot(),
			CompletedToolCall: &call,
		})
	}
	return final(e.thinking.String(), e.text.String(), e.wireCalls(), e.finish)
}

// snapshot builds the partial message from accumulated state (no stop
// reason), attached to each live event so a consumer can render progress.
func (e *Emitter) snapshot() ac.Message {
	return AssembleMessage(e.thinking.String(), e.text.String(), e.wireCalls(), "")
}

// wireCalls renders the accumulated tool-call builders as wire tool calls,
// in first-seen order, so AssembleMessage can build the content blocks
// uniformly.
func (e *Emitter) wireCalls() []ToolCall {
	if len(e.order) == 0 {
		return nil
	}
	calls := make([]ToolCall, 0, len(e.order))
	for _, idx := range e.order {
		b := e.builders[idx]
		calls = append(calls, ToolCall{
			ID:       b.id,
			Type:     "function",
			Function: Func{Name: b.name, Arguments: b.args.String()},
		})
	}
	return calls
}

func (e *Emitter) send(t ac.StreamEventType, idx int) {
	e.sendEvent(ac.StreamEvent{Type: t, ContentIndex: idx, Message: e.snapshot()})
}

func (e *Emitter) sendDelta(t ac.StreamEventType, idx int, delta string) {
	e.sendEvent(ac.StreamEvent{Type: t, ContentIndex: idx, Delta: delta, Message: e.snapshot()})
}

// sendEvent delivers one event, latching Aborted the first time the request
// context ends so later sends become no-ops and the producer unwinds.
func (e *Emitter) sendEvent(ev ac.StreamEvent) {
	if e.aborted {
		return
	}
	if !Send(e.ctx, e.Out, ev) {
		e.aborted = true
	}
}
