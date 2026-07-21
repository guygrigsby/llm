package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	agentcore "github.com/voocel/agentcore"
)

// This file holds the pure (network-free) translation between agentcore types
// and anthropic-sdk-go types. Keeping request building and response conversion
// as standalone functions makes them unit-testable without a live API.

// defaultMaxTokens is used when a call does not specify MaxTokens. Anthropic
// requires max_tokens on every request.
const (
	defaultMaxTokens   = 8192
	streamingMaxTokens = 32000
)

// buildParams assembles MessageNewParams from agentcore inputs and the resolved
// per-call config. It enforces the adapter invariants:
//   - max_tokens is always set,
//   - adaptive thinking is set when the thinking level is on (never the
//     deprecated enabled/budget_tokens shape),
//   - temperature / top_p / top_k are never set.
//
// streaming raises the default max_tokens ceiling (large output is streamed).
func buildParams(modelID string, messages []agentcore.Message, tools []agentcore.ToolSpec, cfg agentcore.CallConfig, streaming bool) anthropic.MessageNewParams {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(modelID),
		Messages:  convertMessages(messages),
		MaxTokens: resolveMaxTokens(cfg.MaxTokens, streaming),
	}

	if sys := systemPrompt(messages); sys != "" {
		params.System = []anthropic.TextBlockParam{{Text: sys}}
	}

	if toolParams := convertTools(tools); len(toolParams) > 0 {
		params.Tools = toolParams
	}

	applyThinking(&params, cfg.ThinkingLevel)

	return params
}

// resolveMaxTokens picks the per-call max_tokens, falling back to a sane default
// that is higher for streaming responses.
func resolveMaxTokens(callMax int, streaming bool) int64 {
	if callMax > 0 {
		return int64(callMax)
	}
	if streaming {
		return streamingMaxTokens
	}
	return defaultMaxTokens
}

// applyThinking sets adaptive thinking + effort, or disables thinking, per the
// agentcore thinking level. It NEVER sets temperature / top_p / top_k.
func applyThinking(params *anthropic.MessageNewParams, level agentcore.ThinkingLevel) {
	switch level {
	case "":
		// Unspecified: leave thinking to the provider/model default.
	case agentcore.ThinkingOff:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{},
		}
	default:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}
		if effort, ok := mapEffort(level); ok {
			params.OutputConfig = anthropic.OutputConfigParam{Effort: effort}
		}
	}
}

// mapEffort maps an agentcore thinking level to an Anthropic effort level.
// Returns ok=false for levels with no effort equivalent (off / minimal /
// unspecified), in which case effort is left unset.
func mapEffort(level agentcore.ThinkingLevel) (anthropic.OutputConfigEffort, bool) {
	switch level {
	case agentcore.ThinkingLow:
		return anthropic.OutputConfigEffortLow, true
	case agentcore.ThinkingMedium:
		return anthropic.OutputConfigEffortMedium, true
	case agentcore.ThinkingHigh:
		return anthropic.OutputConfigEffortHigh, true
	case agentcore.ThinkingXHigh:
		return anthropic.OutputConfigEffortXhigh, true
	default:
		return "", false
	}
}

// systemPrompt extracts the concatenated text of any leading system messages.
// Anthropic carries the system prompt out-of-band (the top-level System field),
// not as a message in the messages array.
func systemPrompt(messages []agentcore.Message) string {
	var out string
	for _, msg := range messages {
		if msg.Role == agentcore.RoleSystem {
			out += msg.TextContent()
		}
	}
	return out
}

// convertMessages maps agentcore messages to Anthropic message params. System
// messages are skipped (handled by systemPrompt). Tool-result (RoleTool)
// messages become tool_result content blocks carried on a user message; the
// tool_use id is read from the agentcore message metadata, matching how the
// litellm adapter pairs them.
func convertMessages(messages []agentcore.Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case agentcore.RoleSystem:
			// Carried via params.System.
		case agentcore.RoleUser:
			if blocks := userContentBlocks(msg); len(blocks) > 0 {
				out = append(out, anthropic.NewUserMessage(blocks...))
			}
		case agentcore.RoleTool:
			if block, ok := toolResultBlock(msg); ok {
				out = append(out, anthropic.NewUserMessage(block))
			}
		case agentcore.RoleAssistant:
			if blocks := assistantContentBlocks(msg); len(blocks) > 0 {
				out = append(out, anthropic.NewAssistantMessage(blocks...))
			}
		}
	}
	return out
}

// userContentBlocks builds the content blocks for a user message. v1 handles
// text only; images can be added at the boundary later without leaking SDK
// types upward.
func userContentBlocks(msg agentcore.Message) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	if text := msg.TextContent(); text != "" {
		blocks = append(blocks, anthropic.NewTextBlock(text))
	}
	return blocks
}

// assistantContentBlocks builds the content blocks for an assistant turn:
// text first, then every tool call as a tool_use block. Every agentcore tool
// call MUST round-trip to a tool_use block so the transcript replays with an
// exact tool-call set.
func assistantContentBlocks(msg agentcore.Message) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	if text := msg.TextContent(); text != "" {
		blocks = append(blocks, anthropic.NewTextBlock(text))
	}
	for _, call := range msg.ToolCalls() {
		blocks = append(blocks, anthropic.NewToolUseBlock(call.ID, toolUseInput(call), call.Name))
	}
	return blocks
}

// toolUseInput returns the tool-call arguments as a value the SDK can marshal.
// Args is JSON; decode it so the SDK re-encodes a real object rather than a
// double-encoded string. Falls back to an empty object on malformed args.
func toolUseInput(call agentcore.ToolCall) any {
	if len(call.Args) == 0 {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal(call.Args, &v); err != nil {
		return map[string]any{}
	}
	return v
}

// toolResultBlock builds a tool_result content block from a RoleTool message.
// The tool_use id comes from the message metadata (set by agentcore's
// ToolResultMsg); without it the result can't be paired and is dropped.
func toolResultBlock(msg agentcore.Message) (anthropic.ContentBlockParamUnion, bool) {
	id, _ := msg.Metadata["tool_call_id"].(string)
	if id == "" {
		return anthropic.ContentBlockParamUnion{}, false
	}
	isErr, _ := msg.Metadata["is_error"].(bool)
	return anthropic.NewToolResultBlock(id, msg.TextContent(), isErr), true
}

// convertTools maps agentcore tool specs to Anthropic tool params. The JSON
// schema in ToolSpec.Parameters is normalised into the SDK input-schema shape
// (properties + required) by remarshalling.
func convertTools(tools []agentcore.ToolSpec) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		tool := anthropic.ToolParam{
			Name:        t.Name,
			InputSchema: toolInputSchema(t.Parameters),
		}
		if t.Description != "" {
			tool.Description = anthropic.String(t.Description)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out
}

// toolInputSchema converts an arbitrary JSON-schema object (as held in
// ToolSpec.Parameters) into the SDK's ToolInputSchemaParam. It pulls out
// properties and required; the type is always "object" (the SDK default).
func toolInputSchema(params any) anthropic.ToolInputSchemaParam {
	schema := anthropic.ToolInputSchemaParam{}
	if params == nil {
		return schema
	}
	// Round-trip through JSON to normalise from map[string]any, a typed struct,
	// or json.RawMessage into a uniform map.
	raw, err := json.Marshal(params)
	if err != nil {
		return schema
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return schema
	}
	if props, ok := m["properties"]; ok {
		schema.Properties = props
	}
	if req, ok := m["required"].([]any); ok {
		required := make([]string, 0, len(req))
		for _, r := range req {
			if s, ok := r.(string); ok {
				required = append(required, s)
			}
		}
		schema.Required = required
	}
	return schema
}

// convertResponse maps an Anthropic Message (the non-stream response) to an
// agentcore Message. Every tool_use block becomes a ToolCall; thinking and text
// blocks become their agentcore equivalents; the stop reason and usage are
// mapped across.
func convertResponse(resp *anthropic.Message) agentcore.Message {
	msg := agentcore.Message{
		Role:       agentcore.RoleAssistant,
		Content:    convertContentBlocks(resp.Content),
		StopReason: mapStopReason(resp.StopReason),
		Usage:      mapUsage(resp.Usage),
	}
	return msg
}

// convertContentBlocks maps Anthropic response content blocks to agentcore
// content blocks, preserving order. Unknown block types are skipped.
func convertContentBlocks(blocks []anthropic.ContentBlockUnion) []agentcore.ContentBlock {
	out := make([]agentcore.ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			out = append(out, agentcore.TextBlock(variant.Text))
		case anthropic.ThinkingBlock:
			out = append(out, agentcore.ThinkingBlock(variant.Thinking))
		case anthropic.ToolUseBlock:
			out = append(out, agentcore.ToolCallBlock(buildToolCall(variant)))
		}
	}
	return out
}

// buildToolCall constructs an agentcore ToolCall from an Anthropic tool_use
// block. Input is already valid JSON (json.RawMessage); an empty input becomes
// "{}" so the surrounding message stays JSON-serializable.
func buildToolCall(block anthropic.ToolUseBlock) agentcore.ToolCall {
	args := json.RawMessage(block.Input)
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	return agentcore.ToolCall{
		ID:   block.ID,
		Name: block.Name,
		Args: args,
	}
}

// mapStopReason maps an Anthropic stop reason to an agentcore StopReason.
func mapStopReason(reason anthropic.StopReason) agentcore.StopReason {
	switch reason {
	case anthropic.StopReasonToolUse:
		return agentcore.StopReasonToolUse
	case anthropic.StopReasonMaxTokens:
		return agentcore.StopReasonLength
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonPauseTurn, "":
		return agentcore.StopReasonStop
	case anthropic.StopReasonRefusal:
		return agentcore.StopReasonError
	default:
		return agentcore.StopReason(reason)
	}
}

// mapUsage maps Anthropic token usage to agentcore Usage. Returns nil when no
// tokens were reported.
func mapUsage(u anthropic.Usage) *agentcore.Usage {
	if u.InputTokens == 0 && u.OutputTokens == 0 {
		return nil
	}
	return &agentcore.Usage{
		Input:       int(u.InputTokens),
		Output:      int(u.OutputTokens),
		CacheRead:   int(u.CacheReadInputTokens),
		CacheWrite:  int(u.CacheCreationInputTokens),
		TotalTokens: int(u.InputTokens + u.OutputTokens),
	}
}

// providerName is the constant provider identity for this adapter.
const providerName = "anthropic"

// errNoModel guards against an empty model id.
var errNoModel = fmt.Errorf("anthropic: model id is required")
