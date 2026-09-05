package kimi

import (
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
func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {
	em := openaiwire.NewEmitter(out)
	var usage *wireUsage

	err := openaiwire.ReadSSE(r, func(data string) (bool, error) {
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
		return false, nil
	})
	if err != nil {
		out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("kimi: stream: %w", err)}
		return
	}

	final := em.Flush(func(reasoning, content string, calls []openaiwire.ToolCall, finish string) ac.Message {
		return openaiwire.AssembleMessage(reasoning, content, calls, finish)
	})
	if usage != nil && onUsage != nil {
		onUsage(usage)
	}
	final.Usage = toAgentUsage(usage)
	out <- ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason}
}
