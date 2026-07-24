package kimi

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

// This file holds the pure (network-free) translation between agentcore types
// and Kimi's OpenAI-shaped chat-completions wire format, plus the wire structs
// themselves. Keeping request building and response conversion as standalone
// functions makes them unit-testable without a live API.

const providerName = "kimi"

// wire request -------------------------------------------------------------

type chatRequest struct {
	Model           string         `json:"model"`
	Messages        []wireMessage  `json:"messages"`
	Tools           []wireTool     `json:"tools,omitempty"`
	ToolChoice      any            `json:"tool_choice,omitempty"`
	Stream          bool           `json:"stream"`
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
	MaxTokens       int            `json:"max_completion_tokens,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	Temperature     *float64       `json:"temperature,omitempty"`
	TopP            *float64       `json:"top_p,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// wireMessage is one OpenAI-format chat message. Partial rides on an assistant
// message to trigger Kimi's prefix-continuation (partial mode): the model
// continues the assistant content rather than starting a fresh turn. Name is
// the optional speaker label partial mode uses for role-play prefixes.
type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Partial    bool           `json:"partial,omitempty"`
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

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role             string         `json:"role"`
			Content          string         `json:"content"`
			ReasoningContent string         `json:"reasoning_content"`
			ToolCalls        []wireToolCall `json:"tool_calls"`
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
		Model:           model,
		Messages:        convertMessages(msgs),
		Tools:           convertTools(tools),
		ToolChoice:      cfg.ToolChoice,
		Stream:          stream,
		MaxTokens:       cfg.MaxTokens,
		ReasoningEffort: reasoningEffort(cfg.ThinkingLevel),
		Temperature:     gen.Temperature,
		TopP:            gen.TopP,
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

// assistantMessage builds the wire assistant message: text content plus any
// tool calls, and the partial/name hints when metadata requests prefix
// continuation. Every agentcore tool call round-trips to a wire tool_call so a
// replayed transcript keeps an exact tool-call set.
func assistantMessage(m ac.Message) wireMessage {
	wm := wireMessage{Role: "assistant", Content: m.TextContent()}
	for _, c := range m.ToolCalls() {
		wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
			ID:       c.ID,
			Type:     "function",
			Function: wireFunc{Name: c.Name, Arguments: argsString(c.Args)},
		})
	}
	if p, _ := m.Metadata["partial"].(bool); p {
		wm.Partial = true
		if name, _ := m.Metadata["name"].(string); name != "" {
			wm.Name = name
		}
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
	msg := assembleMessage(c.Message.ReasoningContent, c.Message.Content, c.Message.ToolCalls, c.FinishReason)
	msg.Usage = toAgentUsage(cr.Usage)
	return msg
}

// assembleMessage builds an assistant message from its parts, in the order the
// agent loop expects (thinking, text, tool calls). Shared by the non-stream
// path and the stream's terminal message so both produce identical structure.
func assembleMessage(reasoning, content string, calls []wireToolCall, finish string) ac.Message {
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
	return ac.Message{Role: ac.RoleAssistant, Content: blocks, StopReason: mapStopReason(finish)}
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

// argsString renders tool-call args (json.RawMessage) as the OpenAI arguments
// string. Empty args become "{}" so the wire message stays valid.
func argsString(args json.RawMessage) string {
	if len(strings.TrimSpace(string(args))) == 0 {
		return "{}"
	}
	return string(args)
}

// rawArgs is the inverse: a wire arguments string to json.RawMessage, defaulting
// empty to "{}" so the enclosing message stays JSON-serializable.
func rawArgs(s string) json.RawMessage {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}
