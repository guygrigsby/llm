package openrouter

import (
	"encoding/json"
	"time"

	"github.com/guygrigsby/llm"
	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// This file holds OpenRouter's own wire shapes and its extensions over the
// shared OpenAI-format chat-completions wire format in internal/openaiwire:
// the reasoning request object, the reasoning_details echo on messages and
// cost-aware usage accounting.

const providerName = "openrouter"

// wire request -------------------------------------------------------------

// chatRequest embeds the shared request shape and adds OpenRouter's own
// max_tokens (openaiwire.Request tags this max_completion_tokens for kimi;
// redeclaring the field here shadows that promoted one, so only "max_tokens"
// reaches the wire) and reasoning object.
type chatRequest struct {
	openaiwire.Request[wireMessage]
	MaxTokens int             `json:"max_tokens,omitempty"`
	Reasoning *reasoningParam `json:"reasoning,omitempty"`
}

// reasoningParam is OpenRouter's unified reasoning control. Only effort is
// wired; max_tokens, exclude and enabled are left to the model's defaults.
type reasoningParam struct {
	Effort string `json:"effort"`
}

// wireMessage is one OpenAI-format chat message. On assistant messages,
// ReasoningDetails echoes the provider's structured reasoning back unmodified
// so a model resumes its chain of thought across tool calls; Reasoning is the
// plaintext fallback when no details were returned.
type wireMessage struct {
	openaiwire.Message
	Reasoning        string          `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
}

// wire response ------------------------------------------------------------

type wireError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role             string                `json:"role"`
			Content          string                `json:"content"`
			Reasoning        string                `json:"reasoning"`
			ReasoningDetails json.RawMessage       `json:"reasoning_details"`
			ToolCalls        []openaiwire.ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

// wireUsage is OpenRouter's usage block. prompt_tokens INCLUDES cached tokens
// (OpenAI convention), so toLLMUsage subtracts to keep llm.Usage additive.
// Cost is in USD credits, always reported by OpenRouter.
type wireUsage struct {
	PromptTokens        int     `json:"prompt_tokens"`
	CompletionTokens    int     `json:"completion_tokens"`
	TotalTokens         int     `json:"total_tokens"`
	Cost                float64 `json:"cost"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

// genParams carries the adapter-level sampler settings into a request.
// Pointer fields mean "unset, do not send".
type genParams struct {
	Temperature *float64
	TopP        *float64
}

// buildRequest assembles a chatRequest from agentcore inputs and the resolved
// per-call config. The reasoning object is set only when the call carries a
// thinking level. OpenRouter always returns usage, in the last chunk when
// streaming, so no stream_options are sent.
func buildRequest(model string, msgs []ac.Message, tools []ac.ToolSpec, cfg ac.CallConfig, gen genParams, stream bool) chatRequest {
	req := chatRequest{
		Request: openaiwire.Request[wireMessage]{
			Model:       model,
			Messages:    convertMessages(msgs),
			Tools:       openaiwire.ConvertTools(tools),
			ToolChoice:  cfg.ToolChoice,
			Stream:      stream,
			Temperature: gen.Temperature,
			TopP:        gen.TopP,
		},
		MaxTokens: cfg.MaxTokens,
	}
	if effort := reasoningEffort(cfg.ThinkingLevel); effort != "" {
		req.Reasoning = &reasoningParam{Effort: effort}
	}
	return req
}

// convertMessages maps agentcore messages to OpenAI-format wire messages. The
// system prompt stays inline as a system-role message and tool results are
// their own "tool" role.
func convertMessages(msgs []ac.Message) []wireMessage {
	return openaiwire.ConvertMessages(msgs, assistantMessage, func(role, content, toolCallID string) wireMessage {
		return wireMessage{Message: openaiwire.Message{Role: role, Content: content, ToolCallID: toolCallID}}
	})
}

// assistantMessage builds the wire assistant message: text, tool calls, and
// the reasoning echo. reasoning_details from metadata wins; otherwise the
// thinking block text is sent as plaintext reasoning.
func assistantMessage(m ac.Message) wireMessage {
	wm := wireMessage{Message: openaiwire.Message{
		Role:      "assistant",
		Content:   m.TextContent(),
		ToolCalls: openaiwire.ToolCallsFrom(m),
	}}
	if d, ok := m.Metadata["reasoning_details"].(json.RawMessage); ok && len(d) > 0 {
		wm.ReasoningDetails = d
	} else if r := m.ThinkingContent(); r != "" {
		wm.Reasoning = r
	}
	return wm
}

// reasoningEffort maps an agentcore thinking level to OpenRouter's effort
// enum. Off and unspecified return "" so the reasoning object is omitted.
func reasoningEffort(level ac.ThinkingLevel) string {
	switch level {
	case ac.ThinkingMinimal:
		return "minimal"
	case ac.ThinkingLow:
		return "low"
	case ac.ThinkingMedium:
		return "medium"
	case ac.ThinkingHigh:
		return "high"
	case ac.ThinkingXHigh:
		return "xhigh"
	default:
		return ""
	}
}

// convertResponse maps a non-stream chat response choice to an agentcore
// message: reasoning first (a thinking block), then text, then every tool
// call, then the reasoning_details echo in metadata when the response
// carried one.
func convertResponse(cr *chatResponse) ac.Message {
	c := cr.Choices[0]
	msg := openaiwire.AssembleMessage(c.Message.Reasoning, c.Message.Content, c.Message.ToolCalls, c.FinishReason)
	if d := c.Message.ReasoningDetails; len(d) > 0 && string(d) != "null" {
		msg.Metadata = map[string]any{"reasoning_details": d}
	}
	msg.Usage = toAgentUsage(cr.Usage)
	return msg
}

// toAgentUsage maps wire usage to agentcore Usage (Input keeps the OpenAI
// convention of including cached tokens). nil when nothing was reported.
// Cost only carries a Total: OpenRouter reports one call-level cost, not a
// per-token-class breakdown, so Input/Output/CacheRead/CacheWrite on ac.Cost
// stay zero rather than claim a precision the wire data doesn't have.
func toAgentUsage(u *wireUsage) *ac.Usage {
	if u == nil || (u.PromptTokens == 0 && u.CompletionTokens == 0) {
		return nil
	}
	return &ac.Usage{
		Input:       u.PromptTokens,
		Output:      u.CompletionTokens,
		CacheRead:   u.PromptTokensDetails.CachedTokens,
		TotalTokens: u.TotalTokens,
		Cost:        &ac.Cost{Total: u.Cost},
	}
}

// toLLMUsage maps wire usage to llm.Usage for the Meter. prompt_tokens includes
// cached tokens on the wire, so PromptTokens is the uncached remainder and the
// input fields stay additive.
func toLLMUsage(model string, u *wireUsage, latency time.Duration) llm.Usage {
	cached := u.PromptTokensDetails.CachedTokens
	prompt := u.PromptTokens - cached
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
		CacheReadTokens:  cached,
		Cost:             u.Cost,
	}
}
