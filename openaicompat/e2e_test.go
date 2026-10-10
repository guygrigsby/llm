//go:build e2e

// These tests hit a real OpenAI-compatible server. They are off by default;
// run them with:
//
//	OPENAICOMPAT_BASE_URL=https://host/v1 OPENAICOMPAT_MODEL=m \
//	  go test -tags e2e ./openaicompat/ -v
//
// OPENAICOMPAT_API_KEY is sent when set.
package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

func e2eAdapter(t *testing.T) *Adapter {
	t.Helper()
	base, model := os.Getenv("OPENAICOMPAT_BASE_URL"), os.Getenv("OPENAICOMPAT_MODEL")
	if base == "" || model == "" {
		t.Skip("OPENAICOMPAT_BASE_URL and OPENAICOMPAT_MODEL not set")
	}
	a, err := New(Config{BaseURL: base, Model: model, APIKey: os.Getenv("OPENAICOMPAT_API_KEY")})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func TestE2E_Generate(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	resp, err := a.Generate(ctx, []ac.Message{ac.UserMsg("Reply with the single word: pong")}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Message.TextContent()), "pong") {
		t.Errorf("text = %q", resp.Message.TextContent())
	}
}

func TestE2E_GenerateStream(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	out, err := a.GenerateStream(ctx, []ac.Message{ac.UserMsg("Count from one to five in words.")}, nil)
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var deltas int
	var text string
	for ev := range out {
		switch ev.Type {
		case ac.StreamEventTextDelta:
			deltas++
		case ac.StreamEventDone:
			text = ev.Message.TextContent()
		case ac.StreamEventError:
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	if !strings.Contains(strings.ToLower(text), "five") {
		t.Errorf("text = %q", text)
	}
	if deltas < 2 {
		t.Errorf("%d text deltas: not streamed", deltas)
	}
}
