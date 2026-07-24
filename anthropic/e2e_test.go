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

	"github.com/guygrigsby/llm"
)

const e2eModel = "claude-haiku-4-5"

// e2eCacheModel is used by the prompt-caching test instead of e2eModel: the
// minimum cacheable prefix is 4096 tokens on Haiku 4.5 but 1024 on Sonnet 5, and
// below the minimum Anthropic silently ignores a breakpoint, so the cheaper model
// would need a far larger filler prompt to prove anything. Sonnet 5 is also what
// the companion actually runs.
const e2eCacheModel = "claude-sonnet-5"

// recordingMeter captures the Usage the adapter reports for each call.
type recordingMeter struct{ usage []llm.Usage }

func (m *recordingMeter) Observe(u llm.Usage) { m.usage = append(m.usage, u) }

// TestE2E_PromptCachingReadsGrowWithHistory proves both breakpoints against the
// live API. Turn 1 writes. Turn 2 reads, since the system prefix is unchanged.
// Turn 3 must read strictly more than turn 2: the system block is byte-identical
// across all three, so the only way the read can grow is if the conversation
// history is cached too, which is the top-level breakpoint doing its job.
func TestE2E_PromptCachingReadsGrowWithHistory(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping live e2e test")
	}
	// Over the 1024-token minimum for Sonnet 5, and byte-identical every turn.
	system := strings.Repeat("You are a terse assistant working from a long stable briefing. ", 200)

	var meter recordingMeter
	a, err := New(Config{APIKey: key, Model: e2eCacheModel, Meter: &meter})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	msgs := []agentcore.Message{agentcore.SystemMsg(system)}
	for i, ask := range []string{"say the word one", "say the word two", "say the word three"} {
		msgs = append(msgs, agentcore.UserMsg(ask))
		resp, err := a.Generate(ctx, msgs, nil, agentcore.WithMaxTokens(64), agentcore.WithThinking(agentcore.ThinkingOff))
		if err != nil {
			t.Fatalf("turn %d: %v", i+1, err)
		}
		msgs = append(msgs, resp.Message)
	}
	if len(meter.usage) != 3 {
		t.Fatalf("metered calls = %d, want 3", len(meter.usage))
	}
	for i, u := range meter.usage {
		t.Logf("turn %d: prompt=%d cache_read=%d cache_write=%d", i+1, u.PromptTokens, u.CacheReadTokens, u.CacheWriteTokens)
	}

	if meter.usage[0].CacheWriteTokens == 0 {
		t.Errorf("turn 1 wrote nothing to cache: %+v", meter.usage[0])
	}
	if meter.usage[1].CacheReadTokens == 0 {
		t.Errorf("turn 2 read nothing from cache: %+v", meter.usage[1])
	}
	if meter.usage[2].CacheReadTokens <= meter.usage[1].CacheReadTokens {
		t.Errorf("cache read did not grow with history: turn 2 = %d, turn 3 = %d; history is not being cached",
			meter.usage[1].CacheReadTokens, meter.usage[2].CacheReadTokens)
	}
}

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
