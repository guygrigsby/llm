package kimi

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ac "github.com/voocel/agentcore"
)

// This file drives Kimi's OpenAI-format SSE stream, translating incremental
// chunks into the agentcore start/delta/end framing the agent loop consumes.
// The terminal done event carries the fully assembled message, so live deltas
// are for progress; the exact content is assembled once at flush.

// chatChunk is one SSE `data:` frame from the streaming chat-completions API.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content          string           `json:"content"`
			ReasoningContent string           `json:"reasoning_content"`
			ToolCalls        []streamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
}

// streamToolCall is a tool_call delta. Index identifies which call the fragment
// belongs to (id and name arrive on the first fragment; arguments stream across
// later ones).
type streamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// emitSSE reads the SSE body to completion, emitting agentcore stream events via
// an sseEmitter and reporting usage once through onUsage. A read error (other
// than clean EOF) becomes a terminal error event.
func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {
	br := bufio.NewReader(r)
	em := &sseEmitter{out: out, builders: map[int]*callBuilder{}}
	for {
		line, err := br.ReadString('\n')
		if data, ok := sseData(line); ok {
			if data == "[DONE]" {
				break
			}
			var chunk chatChunk
			if json.Unmarshal([]byte(data), &chunk) == nil {
				em.handle(&chunk)
			}
		}
		if err != nil {
			if err != io.EOF {
				out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("kimi: stream: %w", err)}
				return
			}
			break
		}
	}
	em.flush(onUsage)
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

// sseEmitter accumulates streamed content and frames it as agentcore events. It
// tracks whether the thinking/text blocks are open and builds tool calls by
// provider index, assigning each block a stable agentcore content index.
type sseEmitter struct {
	out chan<- ac.StreamEvent

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
	usage  *wireUsage
}

// callBuilder accumulates one streamed tool call.
type callBuilder struct {
	contentIdx int
	id         string
	name       string
	args       strings.Builder
	started    bool
}

// handle processes one chunk: reasoning first (thinking), then text, then tool
// calls, recording finish reason and usage when present.
func (e *sseEmitter) handle(c *chatChunk) {
	if c.Usage != nil {
		e.usage = c.Usage
	}
	for _, ch := range c.Choices {
		if ch.FinishReason != "" {
			e.finish = ch.FinishReason
		}
		if ch.Delta.ReasoningContent != "" {
			e.onReasoning(ch.Delta.ReasoningContent)
		}
		if ch.Delta.Content != "" {
			e.onContent(ch.Delta.Content)
		}
		for _, tc := range ch.Delta.ToolCalls {
			e.onToolCall(tc)
		}
	}
}

func (e *sseEmitter) onReasoning(s string) {
	if !e.thinkingOpen {
		e.thinkingOpen = true
		e.thinkingIdx = e.nextIndex
		e.nextIndex++
		e.send(ac.StreamEventThinkingStart, e.thinkingIdx)
	}
	e.thinking.WriteString(s)
	e.sendDelta(ac.StreamEventThinkingDelta, e.thinkingIdx, s)
}

func (e *sseEmitter) onContent(s string) {
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

func (e *sseEmitter) onToolCall(tc streamToolCall) {
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

func (e *sseEmitter) closeThinking() {
	if e.thinkingOpen {
		e.send(ac.StreamEventThinkingEnd, e.thinkingIdx)
		e.thinkingOpen = false
	}
}

func (e *sseEmitter) closeText() {
	if e.textOpen {
		e.send(ac.StreamEventTextEnd, e.textIdx)
		e.textOpen = false
	}
}

// flush closes any open blocks, emits a tool-call-end (with the completed call)
// for each streamed tool call, reports usage, and sends the terminal done event
// carrying the fully assembled message.
func (e *sseEmitter) flush(onUsage func(*wireUsage)) {
	e.closeThinking()
	e.closeText()
	for _, idx := range e.order {
		b := e.builders[idx]
		call := ac.ToolCall{ID: b.id, Name: b.name, Args: rawArgs(b.args.String())}
		e.out <- ac.StreamEvent{
			Type:              ac.StreamEventToolCallEnd,
			ContentIndex:      b.contentIdx,
			Message:           e.snapshot(),
			CompletedToolCall: &call,
		}
	}
	if e.usage != nil && onUsage != nil {
		onUsage(e.usage)
	}
	final := assembleMessage(e.thinking.String(), e.text.String(), e.wireCalls(), e.finish)
	final.Usage = toAgentUsage(e.usage)
	e.out <- ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason}
}

// snapshot builds the partial message from accumulated state (no stop/usage),
// attached to each live event so a consumer can render progress.
func (e *sseEmitter) snapshot() ac.Message {
	return assembleMessage(e.thinking.String(), e.text.String(), e.wireCalls(), "")
}

// wireCalls renders the accumulated tool-call builders as wire tool calls, in
// first-seen order, so assembleMessage can build the content blocks uniformly.
func (e *sseEmitter) wireCalls() []wireToolCall {
	if len(e.order) == 0 {
		return nil
	}
	calls := make([]wireToolCall, 0, len(e.order))
	for _, idx := range e.order {
		b := e.builders[idx]
		calls = append(calls, wireToolCall{
			ID:       b.id,
			Type:     "function",
			Function: wireFunc{Name: b.name, Arguments: b.args.String()},
		})
	}
	return calls
}

func (e *sseEmitter) send(t ac.StreamEventType, idx int) {
	e.out <- ac.StreamEvent{Type: t, ContentIndex: idx, Message: e.snapshot()}
}

func (e *sseEmitter) sendDelta(t ac.StreamEventType, idx int, delta string) {
	e.out <- ac.StreamEvent{Type: t, ContentIndex: idx, Delta: delta, Message: e.snapshot()}
}
