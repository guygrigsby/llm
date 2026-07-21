package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	agentcore "github.com/voocel/agentcore"
)

// marshalParams renders MessageNewParams to a generic JSON map so tests can
// assert on the wire shape (presence/absence of fields).
func marshalParams(t *testing.T, p anthropic.MessageNewParams) map[string]any {
	t.Helper()
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return m
}

func TestBuildParams_MaxTokensAlwaysSet(t *testing.T) {
	// No per-call max: default for non-stream and stream differ but both > 0.
	non := buildParams("claude-sonnet-4-6", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{}, false)
	if non.MaxTokens != defaultMaxTokens {
		t.Errorf("non-stream default max_tokens = %d, want %d", non.MaxTokens, defaultMaxTokens)
	}
	str := buildParams("claude-sonnet-4-6", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{}, true)
	if str.MaxTokens != streamingMaxTokens {
		t.Errorf("stream default max_tokens = %d, want %d", str.MaxTokens, streamingMaxTokens)
	}
	// Per-call override wins.
	over := buildParams("claude-sonnet-4-6", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{MaxTokens: 1234}, false)
	if over.MaxTokens != 1234 {
		t.Errorf("override max_tokens = %d, want 1234", over.MaxTokens)
	}
}

func TestBuildParams_NoSamplingParams(t *testing.T) {
	// Across every thinking level, temperature/top_p/top_k must never appear.
	for _, level := range []agentcore.ThinkingLevel{
		"", agentcore.ThinkingOff, agentcore.ThinkingLow, agentcore.ThinkingMedium,
		agentcore.ThinkingHigh, agentcore.ThinkingXHigh,
	} {
		p := buildParams("claude-opus-4-8", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{ThinkingLevel: level}, false)
		m := marshalParams(t, p)
		for _, banned := range []string{"temperature", "top_p", "top_k"} {
			if _, ok := m[banned]; ok {
				t.Errorf("level %q: %q present in params, must never be set", level, banned)
			}
		}
	}
}

func TestBuildParams_AdaptiveThinkingWhenOn(t *testing.T) {
	p := buildParams("claude-opus-4-8", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{ThinkingLevel: agentcore.ThinkingHigh}, false)
	m := marshalParams(t, p)
	thinking, ok := m["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking not set when level is high: %v", m["thinking"])
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
	}
	// Must NOT be the deprecated enabled/budget_tokens shape.
	if _, bad := thinking["budget_tokens"]; bad {
		t.Error("budget_tokens present; adaptive thinking must not carry a budget")
	}
	oc, ok := m["output_config"].(map[string]any)
	if !ok || oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want high", m["output_config"])
	}
}

func TestBuildParams_ThinkingOffDisables(t *testing.T) {
	p := buildParams("claude-opus-4-8", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{ThinkingLevel: agentcore.ThinkingOff}, false)
	m := marshalParams(t, p)
	thinking, ok := m["thinking"].(map[string]any)
	if !ok || thinking["type"] != "disabled" {
		t.Errorf("thinking = %v, want type disabled", m["thinking"])
	}
}

func TestBuildParams_ThinkingUnspecifiedOmitted(t *testing.T) {
	p := buildParams("claude-opus-4-8", []agentcore.Message{agentcore.UserMsg("hi")}, nil, agentcore.CallConfig{}, false)
	m := marshalParams(t, p)
	if _, ok := m["thinking"]; ok {
		t.Errorf("thinking should be omitted when level unspecified, got %v", m["thinking"])
	}
}

func TestMapEffort(t *testing.T) {
	cases := []struct {
		level agentcore.ThinkingLevel
		want  anthropic.OutputConfigEffort
		ok    bool
	}{
		{agentcore.ThinkingLow, anthropic.OutputConfigEffortLow, true},
		{agentcore.ThinkingMedium, anthropic.OutputConfigEffortMedium, true},
		{agentcore.ThinkingHigh, anthropic.OutputConfigEffortHigh, true},
		{agentcore.ThinkingXHigh, anthropic.OutputConfigEffortXhigh, true},
		{agentcore.ThinkingMinimal, "", false},
		{agentcore.ThinkingOff, "", false},
		{"", "", false},
	}
	for _, c := range cases {
		got, ok := mapEffort(c.level)
		if ok != c.ok || got != c.want {
			t.Errorf("mapEffort(%q) = (%q, %v), want (%q, %v)", c.level, got, ok, c.want, c.ok)
		}
	}
}

func TestConvertTools(t *testing.T) {
	tools := []agentcore.ToolSpec{
		{
			Name:        "restart_service",
			Description: "Restart a systemd service",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
				"required": []any{"name"},
			},
		},
		{Name: ""}, // skipped
	}
	got := convertTools(tools)
	if len(got) != 1 {
		t.Fatalf("convertTools len = %d, want 1 (empty-name tool skipped)", len(got))
	}
	tool := got[0].OfTool
	if tool == nil {
		t.Fatal("OfTool nil")
	}
	if tool.Name != "restart_service" {
		t.Errorf("name = %q, want restart_service", tool.Name)
	}
	if tool.Description.Value != "Restart a systemd service" {
		t.Errorf("description = %q", tool.Description.Value)
	}
	if len(tool.InputSchema.Required) != 1 || tool.InputSchema.Required[0] != "name" {
		t.Errorf("required = %v, want [name]", tool.InputSchema.Required)
	}
	if tool.InputSchema.Properties == nil {
		t.Error("properties not carried into input schema")
	}
}

func TestConvertMessages_ToolUseAndResultPairing(t *testing.T) {
	// Assistant turn with a tool call, then the tool result paired by id.
	assistant := agentcore.Message{
		Role: agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{
			agentcore.TextBlock("calling"),
			agentcore.ToolCallBlock(agentcore.ToolCall{
				ID:   "toolu_1",
				Name: "service_status",
				Args: json.RawMessage(`{"name":"nginx"}`),
			}),
		},
	}
	result := agentcore.ToolResultMsg("toolu_1", json.RawMessage(`"active"`), false)
	msgs := []agentcore.Message{
		agentcore.SystemMsg("you are gyr"),
		agentcore.UserMsg("status?"),
		assistant,
		result,
	}

	out := convertMessages(msgs)
	// system is carried out-of-band, so 3 messages remain: user, assistant, tool-result(user)
	if len(out) != 3 {
		t.Fatalf("convertMessages len = %d, want 3", len(out))
	}

	// Assistant message must contain a tool_use block with the right id/name.
	asst := marshalMessage(t, out[1])
	if asst["role"] != "assistant" {
		t.Fatalf("out[1] role = %v", asst["role"])
	}
	blocks := asst["content"].([]any)
	var foundToolUse bool
	for _, b := range blocks {
		bm := b.(map[string]any)
		if bm["type"] == "tool_use" {
			foundToolUse = true
			if bm["id"] != "toolu_1" || bm["name"] != "service_status" {
				t.Errorf("tool_use block id/name = %v/%v", bm["id"], bm["name"])
			}
			input := bm["input"].(map[string]any)
			if input["name"] != "nginx" {
				t.Errorf("tool_use input = %v, want name=nginx", input)
			}
		}
	}
	if !foundToolUse {
		t.Error("no tool_use block in assistant message")
	}

	// Tool result must be a user message carrying a tool_result block keyed by id.
	tr := marshalMessage(t, out[2])
	if tr["role"] != "user" {
		t.Fatalf("tool result role = %v, want user", tr["role"])
	}
	trBlocks := tr["content"].([]any)
	trBlock := trBlocks[0].(map[string]any)
	if trBlock["type"] != "tool_result" {
		t.Fatalf("block type = %v, want tool_result", trBlock["type"])
	}
	if trBlock["tool_use_id"] != "toolu_1" {
		t.Errorf("tool_use_id = %v, want toolu_1", trBlock["tool_use_id"])
	}
}

func TestConvertMessages_SystemExtractedToTopLevel(t *testing.T) {
	p := buildParams("claude-opus-4-8", []agentcore.Message{
		agentcore.SystemMsg("be terse"),
		agentcore.UserMsg("hi"),
	}, nil, agentcore.CallConfig{}, false)
	if len(p.System) != 1 || p.System[0].Text != "be terse" {
		t.Errorf("system = %v, want one block 'be terse'", p.System)
	}
}

func marshalMessage(t *testing.T, m anthropic.MessageParam) map[string]any {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return out
}

// decodeMessage builds an anthropic.Message from raw JSON. The content-block
// union's AsAny reads the union's raw JSON, so a response must be decoded rather
// than built by setting struct fields.
func decodeMessage(t *testing.T, raw string) *anthropic.Message {
	t.Helper()
	var m anthropic.Message
	if err := m.UnmarshalJSON([]byte(raw)); err != nil {
		t.Fatalf("decode message: %v", err)
	}
	return &m
}

func TestConvertResponse_ToolUseBecomesToolCall(t *testing.T) {
	resp := decodeMessage(t, `{
		"id":"m","type":"message","role":"assistant","model":"claude",
		"stop_reason":"tool_use","stop_sequence":null,
		"content":[
			{"type":"text","text":"let me check"},
			{"type":"tool_use","id":"toolu_9","name":"bash","input":{"cmd":"ls"}}
		],
		"usage":{"input_tokens":10,"output_tokens":5}
	}`)
	msg := convertResponse(resp)
	if msg.Role != agentcore.RoleAssistant {
		t.Errorf("role = %q", msg.Role)
	}
	if msg.TextContent() != "let me check" {
		t.Errorf("text = %q", msg.TextContent())
	}
	calls := msg.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].ID != "toolu_9" || calls[0].Name != "bash" {
		t.Errorf("tool call id/name = %q/%q", calls[0].ID, calls[0].Name)
	}
	if string(calls[0].Args) != `{"cmd":"ls"}` {
		t.Errorf("tool call args = %s", calls[0].Args)
	}
	if msg.StopReason != agentcore.StopReasonToolUse {
		t.Errorf("stop reason = %q, want toolUse", msg.StopReason)
	}
	if msg.Usage == nil || msg.Usage.Input != 10 || msg.Usage.Output != 5 || msg.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", msg.Usage)
	}
}

func TestConvertResponse_TextConcatenation(t *testing.T) {
	resp := decodeMessage(t, `{
		"id":"m","type":"message","role":"assistant","model":"claude",
		"stop_reason":"end_turn","stop_sequence":null,
		"content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}],
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	msg := convertResponse(resp)
	if msg.TextContent() != "hello world" {
		t.Errorf("text = %q, want 'hello world'", msg.TextContent())
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[anthropic.StopReason]agentcore.StopReason{
		anthropic.StopReasonToolUse:      agentcore.StopReasonToolUse,
		anthropic.StopReasonMaxTokens:    agentcore.StopReasonLength,
		anthropic.StopReasonEndTurn:      agentcore.StopReasonStop,
		anthropic.StopReasonStopSequence: agentcore.StopReasonStop,
		anthropic.StopReasonPauseTurn:    agentcore.StopReasonStop,
		anthropic.StopReasonRefusal:      agentcore.StopReasonError,
		"":                               agentcore.StopReasonStop,
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBuildToolCall_EmptyInput(t *testing.T) {
	got := buildToolCall(anthropic.ToolUseBlock{ID: "x", Name: "y", Input: nil})
	if string(got.Args) != "{}" {
		t.Errorf("empty input args = %s, want {}", got.Args)
	}
}
