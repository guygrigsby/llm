package openaiwire

import (
	"encoding/json"
	"strings"

	ac "github.com/voocel/agentcore"
)

// This file holds the pure (network-free) translation between agentcore
// types and the OpenAI-shaped chat-completions wire format that kimi and
// openrouter both speak. Keeping request building and response conversion as
// standalone functions makes them unit-testable without a live API.

// ConvertMessages maps agentcore messages to the provider's wire message
// type M, delegating construction to two adapter-supplied constructors:
// assistant builds the (provider-specific) assistant message, and plain
// builds every other role from a role, its text content and, for a tool
// result, the tool_call_id it answers. The system prompt stays inline as a
// system-role message and tool results are their own "tool" role (not a
// user message), matching the OpenAI shape every such provider speaks.
func ConvertMessages[M any](msgs []ac.Message, assistant func(ac.Message) M, plain func(role, content, toolCallID string) M) []M {
	out := make([]M, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case ac.RoleSystem:
			out = append(out, plain("system", m.TextContent(), ""))
		case ac.RoleUser:
			out = append(out, plain("user", m.TextContent(), ""))
		case ac.RoleTool:
			id, _ := m.Metadata["tool_call_id"].(string)
			out = append(out, plain("tool", m.TextContent(), id))
		case ac.RoleAssistant:
			out = append(out, assistant(m))
		}
	}
	return out
}

// ToolCallsFrom renders an agentcore message's tool calls as wire tool
// calls, so every agentcore tool call round-trips to a wire tool_call and a
// replayed transcript keeps an exact tool-call set. Both adapters call this
// from their own assistant-message constructor.
func ToolCallsFrom(m ac.Message) []ToolCall {
	var calls []ToolCall
	for _, c := range m.ToolCalls() {
		calls = append(calls, ToolCall{
			ID:       c.ID,
			Type:     "function",
			Function: Func{Name: c.Name, Arguments: ArgsString(c.Args)},
		})
	}
	return calls
}

// ConvertTools maps agentcore tool specs to OpenAI-format function tools,
// skipping any tool with an empty name.
func ConvertTools(tools []ac.ToolSpec) []Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]Tool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		out = append(out, Tool{
			Type: "function",
			Function: ToolFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
				Strict:      t.Strict,
			},
		})
	}
	return out
}

// MapStopReason maps an OpenAI finish_reason to an agentcore StopReason.
func MapStopReason(reason string) ac.StopReason {
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

// AssembleMessage builds an assistant message from its parts, in the order
// the agent loop expects (thinking, text, tool calls). Shared by the
// non-stream path and the stream's terminal message so both produce
// identical structure. The adapter adds any extra metadata (such as
// openrouter's reasoning_details echo) to the returned message afterwards.
func AssembleMessage(reasoning, content string, calls []ToolCall, finish string) ac.Message {
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
			Args: RawArgs(c.Function.Arguments),
		}))
	}
	return ac.Message{Role: ac.RoleAssistant, Content: blocks, StopReason: MapStopReason(finish)}
}

// ArgsString renders tool-call args (json.RawMessage) as the OpenAI
// arguments string. Empty args become "{}" so the wire message stays valid.
func ArgsString(args json.RawMessage) string {
	if len(strings.TrimSpace(string(args))) == 0 {
		return "{}"
	}
	return string(args)
}

// RawArgs is the inverse: a wire arguments string to json.RawMessage,
// defaulting empty to "{}" so the enclosing message stays JSON-serializable.
func RawArgs(s string) json.RawMessage {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}
