// Package aperture is the adapter for Aperture, the model gateway on the
// tailnet: an agentcore.ChatModel (llm.LLM) backed by its chat-completions
// API. Aperture speaks the OpenRouter dialect (internal/chatwire), so like
// openrouter it needs no SDK — and it differs from OpenRouter in exactly the
// parts this adapter owns: the endpoint is on the tailnet, reached by host
// rather than by key, so no API key is required (one may still be set for a
// route that asks), and there are no app-attribution headers.
//
// Model ids are Aperture's own, as its /v1/models lists them, e.g.
// "anthropic/claude-sonnet-5.5" or "qwen-27b".
package aperture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	"github.com/guygrigsby/llm/internal/chatwire"
	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// defaultBaseURL is Aperture's endpoint on the tailnet. /v1 is part of the
// base, so the chat path is baseURL + "/chat/completions".
const defaultBaseURL = "https://ai.guy.ts.net/v1"

// defaultStreamIdle is how long a stream may produce nothing before the
// adapter gives up on it. Long thinking pauses are normal, a two minute
// silence is a stuck connection.
const defaultStreamIdle = 2 * time.Minute

// providerName labels this adapter's usage and errors.
const providerName = "aperture"

// Config configures the adapter. Model is required; it is an Aperture model
// id such as "anthropic/claude-sonnet-5.5". BaseURL defaults to the tailnet
// endpoint. APIKey is optional: the tailnet authenticates the host, not a
// bearer, so the header is left off entirely when no key is set.
type Config struct {
	APIKey  string
	Model   string
	BaseURL string
	// Meter, if set, receives token, cost and latency Usage for every call.
	Meter llm.Meter
	// Temperature and TopP are optional sampler params. nil leaves them off the
	// wire (the model's own default).
	Temperature *float64
	TopP        *float64
	// StreamIdleTimeout aborts a stream that has produced nothing for this
	// long, surfacing agentcore.ErrProviderStreamIdle so the agent loop can
	// fail over instead of waiting on a dead connection. Zero uses two
	// minutes; negative disables the watchdog. Streaming has no overall
	// deadline: a legitimately long generation must not be cut off.
	StreamIdleTimeout time.Duration
}

// Adapter wraps Aperture's chat-completions API as an llm.LLM.
type Adapter struct {
	model       string
	client      openaiwire.Client
	meter       llm.Meter
	temperature *float64
	topP        *float64
}

// compile-time check: Adapter satisfies the llm.LLM port (agentcore.ChatModel).
var _ llm.LLM = (*Adapter)(nil)

// New builds an Adapter from config.
func New(cfg Config) (*Adapter, error) {
	if cfg.Model == "" {
		return nil, errors.New("aperture: Model is required")
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	idle := cfg.StreamIdleTimeout
	if idle == 0 {
		idle = defaultStreamIdle
	}
	return &Adapter{
		model: cfg.Model,
		client: openaiwire.Client{
			HTTP: &http.Client{Timeout: 10 * time.Minute},
			// Streaming gets its own client with no overall timeout: the idle
			// watchdog catches a stalled stream in seconds, where a deadline
			// either fires ten minutes late or kills a long healthy one.
			StreamHTTP: &http.Client{},
			StreamIdle: idle,
			BaseURL:    strings.TrimRight(base, "/"),
			APIKey:     cfg.APIKey,
			Provider:   providerName,
		},
		meter:       cfg.Meter,
		temperature: cfg.Temperature,
		topP:        cfg.TopP,
	}, nil
}

// SupportsTools reports true: this adapter wires OpenAI-format tool calling.
func (a *Adapter) SupportsTools() bool { return true }

// ProviderName implements agentcore.ProviderNamer.
func (a *Adapter) ProviderName() string { return providerName }

// observe reports usage to the configured Meter, if any.
func (a *Adapter) observe(u *chatwire.WireUsage, latency time.Duration) {
	if a.meter == nil || u == nil {
		return
	}
	a.meter.Observe(chatwire.ToLLMUsage(providerName, a.model, u, latency))
}

// Generate runs a one-shot chat completion.
func (a *Adapter) Generate(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (*ac.LLMResponse, error) {
	cfg := ac.ResolveCallConfig(opts)
	req := chatwire.BuildRequest(a.model, msgs, tools, cfg, chatwire.GenParams{Temperature: a.temperature, TopP: a.topP}, false)

	start := time.Now()
	data, err := a.client.Post(ctx, "/chat/completions", req, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	var cr chatwire.ChatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("aperture: decode: %w", err)
	}
	// Meter first: a response that carries an error or no choices was still
	// billed for whatever it consumed, and that cost belongs in the Meter.
	a.observe(cr.Usage, time.Since(start))
	if cr.Error != nil {
		return nil, fmt.Errorf("%s", cr.Error.Error(providerName))
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("aperture: response had no choices")
	}
	return &ac.LLMResponse{Message: chatwire.ConvertResponse(&cr)}, nil
}

// GenerateStream runs a streaming chat completion, emitting the agentcore
// StreamEvent sequence the loop expects: thinking/text start-delta-end,
// tool-call start/delta/end with the completed call, a terminal done event
// carrying the fully assembled message with its stop reason and an error
// event on failure.
func (a *Adapter) GenerateStream(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (<-chan ac.StreamEvent, error) {
	cfg := ac.ResolveCallConfig(opts)
	req := chatwire.BuildRequest(a.model, msgs, tools, cfg, chatwire.GenParams{Temperature: a.temperature, TopP: a.topP}, true)

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("aperture: encode: %w", err)
	}
	start := time.Now()
	sse, err := a.client.Stream(ctx, "/chat/completions", body, cfg.APIKey)
	if err != nil {
		return nil, err
	}

	out := make(chan ac.StreamEvent, 64)
	go func() {
		defer close(out)
		defer func() { _ = sse.Close() }()
		chatwire.EmitSSE(ctx, sse, out, providerName, func(u *chatwire.WireUsage) { a.observe(u, time.Since(start)) })
	}()
	return out, nil
}
