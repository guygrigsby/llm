//go:build e2e

// These tests hit the real Kimi (Moonshot) API and cost a (tiny) amount of
// money. They are off by default; run them with:
//
//	JOHNNY_KIMI_TOKEN=... go test -tags e2e ./kimi/ -v
//
// They skip when no key is present.
package kimi

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

const e2eModel = "kimi-k3"

func e2eAdapter(t *testing.T) *Adapter {
	t.Helper()
	key := os.Getenv("JOHNNY_KIMI_TOKEN")
	if key == "" {
		t.Skip("JOHNNY_KIMI_TOKEN not set; skipping live e2e test")
	}
	a, err := New(Config{APIKey: key, Model: e2eModel})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func userMsg(text string) []ac.Message {
	return []ac.Message{{Role: ac.RoleUser, Content: []ac.ContentBlock{ac.TextBlock(text)}}}
}

// Generate against the live API returns a non-empty reply. Confirms the real
// request/response wire shape matches the docs this adapter was built from.
func TestE2E_Generate(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := a.Generate(ctx, userMsg("Reply with the single word: pong"), nil, ac.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if strings.TrimSpace(resp.Message.TextContent()) == "" {
		t.Fatalf("empty reply; stop=%s", resp.Message.StopReason)
	}
	t.Logf("reply=%q stop=%s usage=%+v", resp.Message.TextContent(), resp.Message.StopReason, resp.Message.Usage)
}

// GenerateStream against the live API streams to a non-empty final message and
// never emits an error event. Exercises the real SSE framing.
func TestE2E_GenerateStream(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	ch, err := a.GenerateStream(ctx, userMsg("Count from 1 to 3, comma-separated."), nil, ac.WithMaxTokens(128))
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var final string
	var sawDone bool
	for ev := range ch {
		if ev.Type == ac.StreamEventError {
			t.Fatalf("stream error event: %v", ev.Err)
		}
		if ev.Type == ac.StreamEventDone {
			sawDone = true
		}
		if txt := ev.Message.TextContent(); txt != "" {
			final = txt
		}
	}
	if !sawDone {
		t.Error("no done event")
	}
	if strings.TrimSpace(final) == "" {
		t.Fatal("empty streamed reply")
	}
	t.Logf("streamed=%q", final)
}

// A tool offered to the live model produces a real tool call, converted to an
// agentcore ToolCall: the load-bearing invariant, checked end to end.
func TestE2E_ToolCall(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	tools := []ac.ToolSpec{{
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

	resp, err := a.Generate(ctx, userMsg("What is the weather in Paris? Use the get_weather tool."), tools, ac.WithMaxTokens(256))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) == 0 {
		t.Fatalf("expected a tool call, got none; stop=%s text=%q", resp.Message.StopReason, resp.Message.TextContent())
	}
	if calls[0].Name != "get_weather" || len(calls[0].Args) == 0 {
		t.Errorf("tool call = %+v", calls[0])
	}
	t.Logf("tool=%s args=%s stop=%s", calls[0].Name, calls[0].Args, resp.Message.StopReason)
}

// EstimateTokens against the live endpoint returns a positive count.
func TestE2E_EstimateTokens(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	n, err := a.EstimateTokens(ctx, userMsg("Hello, what is 1+1?"))
	if err != nil {
		t.Fatalf("EstimateTokens: %v", err)
	}
	if n <= 0 {
		t.Fatalf("token estimate = %d, want > 0", n)
	}
	t.Logf("estimated tokens=%d", n)
}
