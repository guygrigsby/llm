package openaiwire

import (
	"testing"

	ac "github.com/voocel/agentcore"
)

// TestConvertToolsSkipsEmptyNames proves a tool with no name is dropped and
// no-tools maps to a nil slice (so the wire request omits "tools" entirely).
func TestConvertToolsSkipsEmptyNames(t *testing.T) {
	if got := ConvertTools(nil); got != nil {
		t.Errorf("ConvertTools(nil) = %v, want nil", got)
	}
	tools := []ac.ToolSpec{
		{Name: "get_weather", Description: "weather", Parameters: map[string]any{"type": "object"}},
		{Name: ""},
	}
	out := ConvertTools(tools)
	if len(out) != 1 {
		t.Fatalf("tools = %d, want 1: %+v", len(out), out)
	}
	if out[0].Type != "function" || out[0].Function.Name != "get_weather" || out[0].Function.Description != "weather" {
		t.Errorf("tool = %+v", out[0])
	}
}

// TestMapStopReason proves every OpenAI finish_reason maps to the expected
// agentcore StopReason, including an unrecognized reason passing through.
func TestMapStopReason(t *testing.T) {
	cases := []struct {
		reason string
		want   ac.StopReason
	}{
		{"tool_calls", ac.StopReasonToolUse},
		{"length", ac.StopReasonLength},
		{"stop", ac.StopReasonStop},
		{"", ac.StopReasonStop},
		{"content_filter", ac.StopReason("content_filter")},
	}
	for _, c := range cases {
		if got := MapStopReason(c.reason); got != c.want {
			t.Errorf("MapStopReason(%q) = %q, want %q", c.reason, got, c.want)
		}
	}
}

// TestArgsString proves empty or blank args render as the valid empty object.
func TestArgsString(t *testing.T) {
	if got := ArgsString(nil); got != "{}" {
		t.Errorf("ArgsString(nil) = %q, want {}", got)
	}
	if got := ArgsString([]byte("   ")); got != "{}" {
		t.Errorf("ArgsString(blank) = %q, want {}", got)
	}
	if got := ArgsString([]byte(`{"city":"Denver"}`)); got != `{"city":"Denver"}` {
		t.Errorf("ArgsString = %q", got)
	}
}

// TestRawArgs proves the inverse: an empty or blank arguments string decodes
// to the valid empty object, preserving whatever else round-trips.
func TestRawArgs(t *testing.T) {
	if got := RawArgs(""); string(got) != "{}" {
		t.Errorf("RawArgs(empty) = %q, want {}", got)
	}
	if got := RawArgs("   "); string(got) != "{}" {
		t.Errorf("RawArgs(blank) = %q, want {}", got)
	}
	if got := RawArgs(`{"q":42}`); string(got) != `{"q":42}` {
		t.Errorf("RawArgs = %q", got)
	}
}
