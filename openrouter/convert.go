package openrouter

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

// This file holds the pure (network-free) translation between agentcore types
// and OpenRouter's OpenAI-shaped chat-completions wire format, plus the wire
// structs. Request building and response conversion are standalone functions
// so they are unit-testable without a live API.

const providerName = "openrouter"

// wire request -------------------------------------------------------------

type chatRequest struct {
	Model       string          `json:"model"`
	Messages    []wireMessage   `json:"messages"`
	Tools       []wireTool      `json:"tools,omitempty"`
	ToolChoice  any             `json:"tool_choice,omitempty"`
	Stream      bool            `json:"stream"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Reasoning   *reasoningParam `json:"reasoning,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
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
	Role             string          `json:"role"`
	Content          string          `json:"content,omitempty"`
	ToolCalls        []wireToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	Reasoning        string          `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
}

type wireToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function wireFunc `json:"function"`
}

type wireFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireToolFunc `json:"function"`
}

type wireToolFunc struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// wire response ------------------------------------------------------------

type wireError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role             string          `json:"role"`
			Content          string          `json:"content"`
			Reasoning        string          `json:"reasoning"`
			ReasoningDetails json.RawMessage `json:"reasoning_details"`
			ToolCalls        []wireToolCall  `json:"tool_calls"`
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
		Model:       model,
		Messages:    convertMessages(msgs),
		Tools:       convertTools(tools),
		ToolChoice:  cfg.ToolChoice,
		Stream:      stream,
		MaxTokens:   cfg.MaxTokens,
		Temperature: gen.Temperature,
		TopP:        gen.TopP,
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
	out := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case ac.RoleSystem:
			out = append(out, wireMessage{Role: "system", Content: m.TextContent()})
		case ac.RoleUser:
			out = append(out, wireMessage{Role: "user", Content: m.TextContent()})
		case ac.RoleTool:
			id, _ := m.Metadata["tool_call_id"].(string)
			out = append(out, wireMessage{Role: "tool", ToolCallID: id, Content: m.TextContent()})
		case ac.RoleAssistant:
			out = append(out, assistantMessage(m))
		}
	}
	return out
}

// assistantMessage builds the wire assistant message: text, tool calls, and
// the reasoning echo. reasoning_details from metadata wins; otherwise the
// thinking block text is sent as plaintext reasoning.
func assistantMessage(m ac.Message) wireMessage {
	wm := wireMessage{Role: "assistant", Content: m.TextContent()}
	for _, c := range m.ToolCalls() {
		wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
			ID:       c.ID,
			Type:     "function",
			Function: wireFunc{Name: c.Name, Arguments: argsString(c.Args)},
		})
	}
	if d, ok := m.Metadata["reasoning_details"].(json.RawMessage); ok && len(d) > 0 {
		wm.ReasoningDetails = d
	} else if r := m.ThinkingContent(); r != "" {
		wm.Reasoning = r
	}
	return wm
}

// convertTools maps agentcore tool specs to OpenAI-format function tools.
func convertTools(tools []ac.ToolSpec) []wireTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		out = append(out, wireTool{
			Type: "function",
			Function: wireToolFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
				Strict:      t.Strict,
			},
		})
	}
	return out
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
// message: reasoning first (a thinking block), then text, then every tool call.
func convertResponse(cr *chatResponse) ac.Message {
	c := cr.Choices[0]
	msg := assembleMessage(c.Message.Reasoning, c.Message.Content, c.Message.ToolCalls, c.FinishReason, c.Message.ReasoningDetails)
	msg.Usage = toAgentUsage(cr.Usage)
	return msg
}

// assembleMessage builds an assistant message from its parts, in the order the
// agent loop expects (thinking, text, tool calls), and stashes the provider's
// reasoning_details in metadata so assistantMessage can echo it later. Shared
// by the non-stream path and the stream's terminal message.
func assembleMessage(reasoning, content string, calls []wireToolCall, finish string, details json.RawMessage) ac.Message {
	var blocks []ac.ContentBlock
	if reasoning != "" {
		blocks = append(blocks, ac.ThinkingBlock(reasoning))
	}
	if content != "" {
		blocks = append(blocks, ac.TextBlock(content))
	}
	for _, c := range calls {
		blocks = append(blocks, ac.ToolCallBlock(ac.ToolCall{
			ID:   c.ID,
			Name: c.Function.Name,
			Args: rawArgs(c.Function.Arguments),
		}))
	}
	msg := ac.Message{Role: ac.RoleAssistant, Content: blocks, StopReason: mapStopReason(finish)}
	if len(details) > 0 && string(details) != "null" {
		msg.Metadata = map[string]any{"reasoning_details": details}
	}
	return msg
}

// mapStopReason maps an OpenAI finish_reason to an agentcore StopReason.
func mapStopReason(reason string) ac.StopReason {
	switch reason {
	case "tool_calls":
		return ac.StopReasonToolUse
	case "length":
		return ac.StopReasonLength
	case "stop", "":
		return ac.StopReasonStop
	default:
		return ac.StopReason(reason)
	}
}

// toAgentUsage maps wire usage to agentcore Usage (Input keeps the OpenAI
// convention of including cached tokens). nil when nothing was reported.
func toAgentUsage(u *wireUsage) *ac.Usage {
	if u == nil || (u.PromptTokens == 0 && u.CompletionTokens == 0) {
		return nil
	}
	return &ac.Usage{
		Input:       u.PromptTokens,
		Output:      u.CompletionTokens,
		CacheRead:   u.PromptTokensDetails.CachedTokens,
		TotalTokens: u.TotalTokens,
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

// argsString renders tool-call args as the OpenAI arguments string. Empty
// args become "{}" so the wire message stays valid.
func argsString(args json.RawMessage) string {
	if len(strings.TrimSpace(string(args))) == 0 {
		return "{}"
	}
	return string(args)
}

// rawArgs is the inverse: a wire arguments string to json.RawMessage,
// defaulting empty to "{}".
func rawArgs(s string) json.RawMessage {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}
