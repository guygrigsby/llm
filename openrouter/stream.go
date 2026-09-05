package openrouter

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// This file drives OpenRouter's OpenAI-format SSE stream on top of the
// shared openaiwire.Emitter, translating OpenRouter's own chunk quirks (the
// reasoning_details array, cost usage, mid-stream error chunks) into the
// Delta the emitter expects. OpenRouter sends comment lines
// (": OPENROUTER PROCESSING") as keepalives; openaiwire.ReadSSE skips any
// line that isn't a "data:" frame.

// chatChunk is one SSE data frame from the streaming chat-completions API.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content          string                      `json:"content"`
			Reasoning        string                      `json:"reasoning"`
			ReasoningDetails []json.RawMessage           `json:"reasoning_details"`
			ToolCalls        []openaiwire.StreamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

// emitSSE reads the SSE body to completion, emitting agentcore stream events
// via a shared openaiwire.Emitter and reporting usage once through onUsage.
// A read error (other than clean EOF) or an error chunk becomes a terminal
// error event, and skips the terminal done event (there is nothing complete
// to report).
func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {
	em := openaiwire.NewEmitter(out)
	var usage *wireUsage
	var details []json.RawMessage
	aborted := false

	err := openaiwire.ReadSSE(r, func(data string) (bool, error) {
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			return false, nil
		}
		if chunk.Error != nil {
			out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("openrouter: %d: %s", chunk.Error.Code, chunk.Error.Message)}
			aborted = true
			return true, nil
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, ch := range chunk.Choices {
			handleReasoningDetails(em, ch.Delta.Reasoning, ch.Delta.ReasoningDetails, &details)
			em.Handle(openaiwire.Delta{Content: ch.Delta.Content, ToolCalls: ch.Delta.ToolCalls}, ch.FinishReason)
		}
		return false, nil
	})
	if err != nil {
		out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("openrouter: stream: %w", err)}
		return
	}
	if aborted {
		return
	}

	final := em.Flush(func(reasoning, content string, calls []openaiwire.ToolCall, finish string) ac.Message {
		msg := openaiwire.AssembleMessage(reasoning, content, calls, finish)
		if len(details) > 0 {
			if b, err := json.Marshal(details); err == nil {
				msg.Metadata = map[string]any{"reasoning_details": json.RawMessage(b)}
			}
		}
		return msg
	})
	if usage != nil && onUsage != nil {
		onUsage(usage)
	}
	final.Usage = toAgentUsage(usage)
	out <- ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason}
}

// handleReasoningDetails feeds a chunk's plaintext reasoning to the emitter
// and collects every reasoning_details object verbatim into details for the
// echo on the final message. When plaintext reasoning is absent, a details
// object of type reasoning.text also feeds the emitter, so a provider that
// sends details without a plaintext reasoning delta still shows thinking.
func handleReasoningDetails(em *openaiwire.Emitter, reasoning string, ds []json.RawMessage, details *[]json.RawMessage) {
	if reasoning != "" {
		em.Handle(openaiwire.Delta{Reasoning: reasoning}, "")
	}
	for _, d := range ds {
		*details = append(*details, d)
		if reasoning == "" {
			var td struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(d, &td) == nil && td.Type == "reasoning.text" && td.Text != "" {
				em.Handle(openaiwire.Delta{Reasoning: td.Text}, "")
			}
		}
	}
}
