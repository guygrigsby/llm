//go:build e2e

// These tests hit the real Anthropic API and cost a (tiny) amount of money.
// They are off by default; run them with:
//
//	ANTHROPIC_API_KEY=... go test -tags e2e ./internal/model/anthropic/ -v
//
// They use the cheapest current model and skip when no key is present.
package anthropic

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	agentcore "github.com/voocel/agentcore"
)

const e2eModel = "claude-haiku-4-5"

func e2eAdapter(t *testing.T) *Adapter {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping live e2e test")
	}
	a, err := New(Config{APIKey: key, Model: e2eModel})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func userMsg(text string) []agentcore.Message {
	return []agentcore.Message{{
		Role:    agentcore.RoleUser,
		Content: []agentcore.ContentBlock{agentcore.TextBlock(text)},
	}}
}

// Generate against the live API returns a non-empty reply. Exercises the real
// request shape: disabled thinking (cheap, Haiku has no effort), no sampling
// params, max_tokens set.
func TestE2E_Generate(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := a.Generate(ctx, userMsg("Reply with the single word: pong"), nil,
		agentcore.WithThinking(agentcore.ThinkingOff), agentcore.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if strings.TrimSpace(resp.Message.TextContent()) == "" {
		t.Fatalf("empty reply; stop=%s", resp.Message.StopReason)
	}
	t.Logf("reply=%q stop=%s", resp.Message.TextContent(), resp.Message.StopReason)
}

// GenerateStream against the live API streams to a non-empty final message and
// never emits an error event.
func TestE2E_GenerateStream(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ch, err := a.GenerateStream(ctx, userMsg("Count from 1 to 3, comma-separated."), nil,
		agentcore.WithThinking(agentcore.ThinkingOff), agentcore.WithMaxTokens(128))
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var final string
	for ev := range ch {
		if ev.Type == agentcore.StreamEventError {
			t.Fatalf("stream error event: %v", ev.Err)
		}
		if txt := ev.Message.TextContent(); txt != "" {
			final = txt
		}
	}
	if strings.TrimSpace(final) == "" {
		t.Fatal("empty streamed reply")
	}
	t.Logf("streamed=%q", final)
}

// A tool offered to the live model produces a real tool call, converted to an
// agentcore ToolCall. This is the load-bearing invariant (the confirm gate and
// ledger depend on tool calls surviving translation) checked end to end.
func TestE2E_ToolCall(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	tools := []agentcore.ToolSpec{{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "city name"},
			},
			"required": []string{"city"},
		},
	}}

	resp, err := a.Generate(ctx, userMsg("What is the weather in Paris? Use the get_weather tool."), tools,
		agentcore.WithThinking(agentcore.ThinkingOff), agentcore.WithMaxTokens(256))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) == 0 {
		t.Fatalf("expected a tool call, got none; stop=%s text=%q", resp.Message.StopReason, resp.Message.TextContent())
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", calls[0].Name)
	}
	if len(calls[0].Args) == 0 {
		t.Errorf("tool args empty")
	}
	t.Logf("tool=%s args=%s stop=%s", calls[0].Name, calls[0].Args, resp.Message.StopReason)
}
