// Package anthropic is the provider-native Anthropic model adapter: the
// anti-corruption layer that is the ONLY package allowed to import
// github.com/anthropics/anthropic-sdk-go. It translates that SDK to and from
// agentcore types, so consumers (gyr, johnny, ...) never see a vendor type.
//
// The adapter satisfies the llm.LLM port, so it plugs straight into
// jess.WithModel. It uses adaptive thinking + effort, and never sends
// temperature / top_p / top_k.
package anthropic

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	agentcore "github.com/voocel/agentcore"

	"github.com/guygrigsby/llm"
)

// Adapter wraps the Anthropic SDK client as an agentcore.ChatModel.
type Adapter struct {
	client anthropic.Client
	model  string
	meter  llm.Meter
}

// Config configures the adapter. APIKey and Model are required; BaseURL is
// optional (e.g. for a gateway).
type Config struct {
	APIKey  string
	Model   string
	BaseURL string
	// Meter, if set, receives token/latency Usage for every call.
	Meter llm.Meter
}

// compile-time check: Adapter satisfies the llm.LLM port (structurally the
// agentcore.ChatModel contract), so it plugs into jess.WithModel.
var _ llm.LLM = (*Adapter)(nil)

// New builds an Adapter from config. It returns an error when the model id is
// empty (Anthropic requires it on every request).
func New(cfg Config) (*Adapter, error) {
	if cfg.Model == "" {
		return nil, errNoModel
	}
	opts := []option.RequestOption{}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Adapter{
		client: anthropic.NewClient(opts...),
		model:  cfg.Model,
		meter:  cfg.Meter,
	}, nil
}

// SupportsTools reports that this provider supports tool calling.
func (a *Adapter) SupportsTools() bool { return true }

// observe reports token/latency usage to the configured Meter, if any.
func (a *Adapter) observe(u anthropic.Usage, latency time.Duration) {
	if a.meter == nil {
		return
	}
	a.meter.Observe(llm.Usage{
		Provider:         providerName,
		Model:            a.model,
		PromptTokens:     int(u.InputTokens),
		CompletionTokens: int(u.OutputTokens),
		TotalTokens:      int(u.InputTokens + u.OutputTokens),
		Latency:          latency,
		CacheReadTokens:  int(u.CacheReadInputTokens),
		CacheWriteTokens: int(u.CacheCreationInputTokens),
	})
}

// ProviderName implements agentcore.ProviderNamer.
func (a *Adapter) ProviderName() string { return providerName }

// Generate produces a synchronous response.
func (a *Adapter) Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error) {
	cfg := agentcore.ResolveCallConfig(opts)
	params := buildParams(a.model, messages, tools, cfg, false)

	start := time.Now()
	resp, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic: generate failed: %w", err)
	}
	a.observe(resp.Usage, time.Since(start))
	return &agentcore.LLMResponse{Message: convertResponse(resp)}, nil
}

// GenerateStream produces a streaming response, emitting the agentcore
// StreamEvent sequence the agent loop expects: text/thinking start-delta-end,
// tool-call start/delta/end with the completed call, a terminal done event
// carrying the fully assembled message + stop reason, and an error event on
// provider failure.
//
// The Anthropic Go stream has no GetFinalMessage; the message is reconstructed
// with Message.Accumulate. The fully assembled message is converted once at the
// end (via convertResponse), guaranteeing every tool_use block becomes an
// agentcore tool call in the done event — the load-bearing invariant.
func (a *Adapter) GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error) {
	cfg := agentcore.ResolveCallConfig(opts)
	params := buildParams(a.model, messages, tools, cfg, true)

	start := time.Now()
	stream := a.client.Messages.NewStreaming(ctx, params)

	out := make(chan agentcore.StreamEvent, 64)
	go func() {
		defer close(out)
		emitStream(stream, out, func(u anthropic.Usage) { a.observe(u, time.Since(start)) })
	}()
	return out, nil
}

// streamReader is the subset of the SDK stream this package consumes. Defining
// it lets the emission logic be exercised against a fake in tests.
type streamReader interface {
	Next() bool
	Current() anthropic.MessageStreamEventUnion
	Err() error
}

// emitStream drives a stream to completion, translating SDK stream events into
// agentcore StreamEvents and accumulating the final message. It uses an emitter
// to track block boundaries so the start/delta/end framing matches the loop's
// expectations.
func emitStream(stream streamReader, out chan<- agentcore.StreamEvent, onComplete func(anthropic.Usage)) {
	var (
		acc anthropic.Message
		em  = &emitter{out: out}
	)
	for stream.Next() {
		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: fmt.Errorf("anthropic: accumulate failed: %w", err)}
			return
		}
		em.handle(event, &acc)
	}
	if err := stream.Err(); err != nil {
		out <- agentcore.StreamEvent{Type: agentcore.StreamEventError, Err: fmt.Errorf("anthropic: stream failed: %w", err)}
		return
	}

	// Report token/latency usage once the message is fully accumulated.
	if onComplete != nil {
		onComplete(acc.Usage)
	}

	// Terminal done event: the fully assembled message, converted once so the
	// tool-call set is exact.
	final := convertResponse(&acc)
	out <- agentcore.StreamEvent{
		Type:       agentcore.StreamEventDone,
		Message:    final,
		StopReason: final.StopReason,
	}
}
