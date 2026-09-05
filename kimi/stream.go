package kimi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// This file drives Kimi's OpenAI-format SSE stream on top of the shared
// openaiwire.Emitter, translating incremental chunks (with Kimi's
// reasoning_content field) into the Delta the emitter expects. The terminal
// done event carries the fully assembled message, so live deltas are for
// progress; the exact content is assembled once at flush.

// chatChunk is one SSE `data:` frame from the streaming chat-completions API.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content          string                      `json:"content"`
			ReasoningContent string                      `json:"reasoning_content"`
			ToolCalls        []openaiwire.StreamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
}

// emitSSE reads the SSE body to completion, emitting agentcore stream events
// via a shared openaiwire.Emitter and reporting usage once through onUsage. A
// read error (other than clean EOF) becomes a terminal error event.
//
// A body that just ends, with no "[DONE]" frame and no finish_reason, is a
// truncated response: the done event and the tool-call-end events are skipped
// so the agent loop raises its own PartialStreamError and retries the turn,
// instead of persisting a half message with cut-off tool-call arguments.
//
// ctx is the request context. Every send is guarded by it, so a consumer that
// stops reading releases this goroutine and the HTTP body rather than pinning
// both for the life of the process.
func emitSSE(ctx context.Context, r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {
	em := openaiwire.NewEmitter(ctx, out)
	var usage *wireUsage

	done, err := openaiwire.ReadSSE(r, func(data string) (bool, error) {
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			return false, nil
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, ch := range chunk.Choices {
			em.Handle(openaiwire.Delta{
				Content:   ch.Delta.Content,
				Reasoning: ch.Delta.ReasoningContent,
				ToolCalls: ch.Delta.ToolCalls,
			}, ch.FinishReason)
		}
		return em.Aborted(), nil
	})
	if err != nil {
		reportUsage(usage, onUsage)
		openaiwire.Send(ctx, out, ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("kimi: stream: %w", err)})
		return
	}
	if em.Aborted() {
		return
	}
	if !done && em.Finish() == "" {
		reportUsage(usage, onUsage)
		return
	}

	final := em.Flush(func(reasoning, content string, calls []openaiwire.ToolCall, finish string) ac.Message {
		return openaiwire.AssembleMessage(reasoning, content, calls, finish)
	})
	reportUsage(usage, onUsage)
	final.Usage = toAgentUsage(usage)
	if em.Aborted() {
		return
	}
	openaiwire.Send(ctx, out, ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason})
}

// reportUsage hands whatever usage arrived to the meter. Every path out of
// emitSSE calls it, complete turn or not: the provider billed for the tokens
// either way, and a cost that never reaches the Meter is a cost nobody sees.
func reportUsage(u *wireUsage, onUsage func(*wireUsage)) {
	if u != nil && onUsage != nil {
		onUsage(u)
	}
}
