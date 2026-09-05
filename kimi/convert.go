package kimi

import (
	"time"

	"github.com/guygrigsby/llm"
	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// This file holds Kimi's own wire shapes and its extensions over the shared
// OpenAI-format chat-completions wire format in internal/openaiwire: the
// reasoning_effort field, the partial/name assistant-message fields (Kimi's
// prefix continuation) and reasoning_content on the response.

const providerName = "kimi"

// wire request -------------------------------------------------------------

// chatRequest embeds the shared request shape (its max_completion_tokens tag
// matches Kimi's own wire name, so it needs no override) and adds Kimi's
// stream_options and reasoning_effort.
type chatRequest struct {
	openaiwire.Request[wireMessage]
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// wireMessage is one OpenAI-format chat message. Partial rides on an assistant
// message to trigger Kimi's prefix-continuation (partial mode): the model
// continues the assistant content rather than starting a fresh turn. Name is
// the optional speaker label partial mode uses for role-play prefixes.
type wireMessage struct {
	openaiwire.Message
	Name    string `json:"name,omitempty"`
	Partial bool   `json:"partial,omitempty"`
}

// wire response ------------------------------------------------------------

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role             string                `json:"role"`
			Content          string                `json:"content"`
			ReasoningContent string                `json:"reasoning_content"`
			ToolCalls        []openaiwire.ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// wireUsage is Kimi's usage block. prompt_tokens INCLUDES cached_tokens (OpenAI
// convention), so toLLMUsage subtracts to keep llm.Usage additive.
type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CachedTokens     int `json:"cached_tokens"`
}

// genParams carries the adapter-level sampler settings (from the model profile)
// into a request. Pointer fields mean "unset, do not send".
type genParams struct {
	Temperature *float64
	TopP        *float64
}

// buildRequest assembles a chatRequest from agentcore inputs and the resolved
// per-call config. reasoning_effort is set for kimi-k3-style models when the
// call carries a thinking level; tool_choice and max_completion_tokens pass
// through when set. streaming adds stream_options so the terminal chunk carries
// usage.
func buildRequest(model string, msgs []ac.Message, tools []ac.ToolSpec, cfg ac.CallConfig, gen genParams, stream bool) chatRequest {
	req := chatRequest{
		Request: openaiwire.Request[wireMessage]{
			Model:       model,
			Messages:    convertMessages(msgs),
			Tools:       openaiwire.ConvertTools(tools),
			ToolChoice:  cfg.ToolChoice,
			Stream:      stream,
			MaxTokens:   cfg.MaxTokens,
			Temperature: gen.Temperature,
			TopP:        gen.TopP,
		},
		ReasoningEffort: reasoningEffort(cfg.ThinkingLevel),
	}
	if stream {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return req
}

// convertMessages maps agentcore messages to OpenAI-format wire messages.
// Unlike Anthropic, the system prompt stays inline as a system-role message and
// tool results are their own "tool" role (not a user message), matching the
// OpenAI shape Kimi speaks.
func convertMessages(msgs []ac.Message) []wireMessage {
	return openaiwire.ConvertMessages(msgs, assistantMessage, func(role, content, toolCallID string) wireMessage {
		return wireMessage{Message: openaiwire.Message{Role: role, Content: content, ToolCallID: toolCallID}}
	})
}

// assistantMessage builds the wire assistant message: text content plus any
// tool calls, and the partial/name hints when metadata requests prefix
// continuation. Every agentcore tool call round-trips to a wire tool_call so a
// replayed transcript keeps an exact tool-call set.
func assistantMessage(m ac.Message) wireMessage {
	wm := wireMessage{Message: openaiwire.Message{
		Role:      "assistant",
		Content:   m.TextContent(),
		ToolCalls: openaiwire.ToolCallsFrom(m),
	}}
	if p, _ := m.Metadata["partial"].(bool); p {
		wm.Partial = true
		if name, _ := m.Metadata["name"].(string); name != "" {
			wm.Name = name
		}
	}
	return wm
}

// reasoningEffort maps an agentcore thinking level to Kimi's kimi-k3
// reasoning_effort scale (low/high/max). Off and unspecified return "" so the
// field is omitted and the model uses its default (k3 defaults to max). Kimi's
// k2.5/k2.6 use a different `thinking` object shape instead; wire that when a
// k2 model is actually targeted.
func reasoningEffort(level ac.ThinkingLevel) string {
	switch level {
	case ac.ThinkingMinimal, ac.ThinkingLow:
		return "low"
	case ac.ThinkingMedium, ac.ThinkingHigh:
		return "high"
	case ac.ThinkingXHigh:
		return "max"
	default:
		return ""
	}
}

// convertResponse maps a non-stream chat response choice to an agentcore
// message: reasoning first (a thinking block), then text, then every tool call.
func convertResponse(cr *chatResponse) ac.Message {
	c := cr.Choices[0]
	msg := openaiwire.AssembleMessage(c.Message.ReasoningContent, c.Message.Content, c.Message.ToolCalls, c.FinishReason)
	msg.Usage = toAgentUsage(cr.Usage)
	return msg
}

// toAgentUsage maps wire usage to agentcore Usage (Input keeps the OpenAI
// convention of including cached tokens). Returns nil when nothing was reported.
func toAgentUsage(u *wireUsage) *ac.Usage {
	if u == nil || (u.PromptTokens == 0 && u.CompletionTokens == 0) {
		return nil
	}
	return &ac.Usage{
		Input:       u.PromptTokens,
		Output:      u.CompletionTokens,
		CacheRead:   u.CachedTokens,
		TotalTokens: u.TotalTokens,
	}
}

// toLLMUsage maps wire usage to llm.Usage for the Meter. prompt_tokens includes
// cached tokens on the wire, so PromptTokens is the uncached remainder and the
// three input fields stay additive (PromptTokens + CacheRead = total input).
func toLLMUsage(model string, u *wireUsage, latency time.Duration) llm.Usage {
	prompt := u.PromptTokens - u.CachedTokens
	if prompt < 0 {
		prompt = 0
	}
	return llm.Usage{
		Provider:         providerName,
		Model:            model,
		PromptTokens:     prompt,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		Latency:          latency,
		CacheReadTokens:  u.CachedTokens,
	}
}
