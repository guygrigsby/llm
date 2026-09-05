//go:build e2e

// These tests hit the real OpenRouter API and cost a tiny amount of money.
// They are off by default; run them with:
//
//	OPENROUTER_API_KEY=... go test -tags e2e ./openrouter/ -v
//
// They skip when no key is present.
package openrouter

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

// e2eModel is a cheap tool-capable model; the adapter is model-agnostic.
const e2eModel = "moonshotai/kimi-k2.5"

func e2eAdapter(t *testing.T) *Adapter {
	t.Helper()
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("OPENROUTER_API_KEY not set; skipping live e2e test")
	}
	a, err := New(Config{APIKey: key, Model: e2eModel, Title: "llm e2e"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func userMsg(text string) []ac.Message {
	return []ac.Message{{Role: ac.RoleUser, Content: []ac.ContentBlock{ac.TextBlock(text)}}}
}

func TestE2E_Generate(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	resp, err := a.Generate(ctx, userMsg("Reply with the single word: pong"), nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Message.TextContent()), "pong") {
		t.Errorf("text = %q", resp.Message.TextContent())
	}
	if resp.Message.Usage == nil || resp.Message.Usage.Output == 0 {
		t.Errorf("usage = %+v", resp.Message.Usage)
	}
}

func TestE2E_GenerateStream(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	ch, err := a.GenerateStream(ctx, userMsg("Count from 1 to 5, digits separated by spaces."), nil)
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var deltas int
	var final *ac.Message
	for ev := range ch {
		switch ev.Type {
		case ac.StreamEventTextDelta:
			deltas++
		case ac.StreamEventDone:
			m := ev.Message
			final = &m
		case ac.StreamEventError:
			t.Fatalf("stream error: %v", ev.Err)
		}
	}
	if deltas == 0 || final == nil || !strings.Contains(final.TextContent(), "5") {
		t.Errorf("deltas = %d final = %+v", deltas, final)
	}
}

func TestE2E_ToolCall(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	tools := []ac.ToolSpec{{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"}},
	}}
	resp, err := a.Generate(ctx, userMsg("Use the get_weather tool to check the weather in Denver."), tools)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "get_weather" || !strings.Contains(string(calls[0].Args), "Denver") {
		t.Errorf("calls = %+v", calls)
	}
}
