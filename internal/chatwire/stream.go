package chatwire

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// This file drives the dialect's OpenAI-format SSE stream on top of the
// shared openaiwire.Emitter, translating the dialect's own chunk quirks (the
// reasoning_details array, cost usage, mid-stream error chunks) into the
// Delta the emitter expects. Gateways send comment lines as keepalives;
// openaiwire.ReadSSE skips any line that isn't a "data:" frame.

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
	Usage *WireUsage `json:"usage"`
	Error *WireError `json:"error"`
}

// EmitSSE reads the SSE body to completion, emitting agentcore stream events
// via a shared openaiwire.Emitter and reporting usage once through onUsage.
// A read error (other than clean EOF) or an error chunk becomes a terminal
// error event, and skips the terminal done event (there is nothing complete
// to report).
//
// A body that just ends, with no "[DONE]" frame and no finish_reason, is a
// truncated response: the done event and the tool-call-end events are skipped
// so the agent loop raises its own PartialStreamError and retries the turn,
// instead of persisting a half message with cut-off tool-call arguments.
//
// ctx is the request context. Every send is guarded by it, so a consumer that
// stops reading releases this goroutine and the HTTP body rather than pinning
// both for the life of the process.
func EmitSSE(ctx context.Context, r io.Reader, out chan<- ac.StreamEvent, provider string, onUsage func(*WireUsage)) {
	em := openaiwire.NewEmitter(ctx, out)
	var usage *WireUsage
	var details []json.RawMessage
	aborted := false

	done, err := openaiwire.ReadSSE(r, func(data string) (bool, error) {
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			return false, nil
		}
		if chunk.Error != nil {
			// Any thinking or text block still open is left unclosed on
			// purpose: the agent loop discards the partial message when it
			// sees an error event, so closing frames would be noise.
			openaiwire.Send(ctx, out, ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("%s", chunk.Error.Error(provider))})
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
		return em.Aborted(), nil
	})
	if err != nil {
		reportUsage(usage, onUsage)
		openaiwire.Send(ctx, out, ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("%s: stream: %w", provider, err)})
		return
	}
	if aborted {
		// The turn failed, but the tokens it burned were billed: report them.
		reportUsage(usage, onUsage)
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
		msg := openaiwire.AssembleMessage(reasoning, content, calls, finish)
		if len(details) > 0 {
			if b, err := json.Marshal(details); err == nil {
				msg.Metadata = map[string]any{"reasoning_details": json.RawMessage(b)}
			}
		}
		return msg
	})
	reportUsage(usage, onUsage)
	final.Usage = ToAgentUsage(usage)
	if em.Aborted() {
		return
	}
	openaiwire.Send(ctx, out, ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason})
}

// reportUsage hands whatever usage arrived to the meter. Every path out of
// EmitSSE calls it, complete turn or not: the provider billed for the tokens
// either way, and a cost that never reaches the Meter is a cost nobody sees.
func reportUsage(u *WireUsage, onUsage func(*WireUsage)) {
	if u != nil && onUsage != nil {
		onUsage(u)
	}
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
