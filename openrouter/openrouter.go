// Package openrouter is the provider-native OpenRouter model adapter: an
// agentcore.ChatModel (llm.LLM) backed by OpenRouter's chat-completions API.
// OpenRouter's own API is OpenAI-shaped, so a plain net/http client is native
// here and needs no SDK.
//
// It is the anti-corruption layer between OpenRouter's wire format and
// agentcore types. Full scope: SSE token streaming and tool calling, so any
// OpenRouter model id works as a primary conversational model. OpenRouter
// extensions beyond standard OpenAI are wired:
//   - the reasoning request object (effort mapped from the call's thinking
//     level), with the model's reasoning surfaced as a thinking block and its
//     reasoning_details echoed back on assistant messages so a model keeps its
//     chain of thought across tool calls,
//   - usage accounting: cost and cached prompt tokens reach the Meter.
//
// The wire dialect itself lives in internal/chatwire, shared with the
// aperture adapter: OpenRouter is the reference speaker of that dialect, and
// gateways that speak it differ only in endpoint, auth and attribution
// headers, which are this package's.
package openrouter

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

// defaultBaseURL is OpenRouter's public endpoint. /api/v1 is part of the base,
// so the chat path is baseURL + "/chat/completions".
const defaultBaseURL = "https://openrouter.ai/api/v1"

// defaultStreamIdle is how long a stream may produce nothing before the
// adapter gives up on it. Long thinking pauses are normal, a two minute
// silence is a stuck connection.
const defaultStreamIdle = 2 * time.Minute

// providerName labels this adapter's usage and errors.
const providerName = "openrouter"

// Config configures the adapter. APIKey and Model are required; Model is an
// OpenRouter model id such as "moonshotai/kimi-k3". BaseURL defaults to
// OpenRouter's public endpoint. Referer and Title are OpenRouter's optional
// app attribution headers (HTTP-Referer, X-Title).
type Config struct {
	APIKey  string
	Model   string
	BaseURL string
	Referer string
	Title   string
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

// Adapter wraps OpenRouter's chat-completions API as an llm.LLM.
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
	if cfg.APIKey == "" {
		return nil, errors.New("openrouter: APIKey is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("openrouter: Model is required")
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	headers := map[string]string{}
	if cfg.Referer != "" {
		headers["HTTP-Referer"] = cfg.Referer
	}
	if cfg.Title != "" {
		headers["X-Title"] = cfg.Title
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
			Headers:    headers,
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
		return nil, fmt.Errorf("openrouter: decode: %w", err)
	}
	// Meter first: a response that carries an error or no choices was still
	// billed for whatever it consumed, and that cost belongs in the Meter.
	a.observe(cr.Usage, time.Since(start))
	if cr.Error != nil {
		return nil, fmt.Errorf("%s", cr.Error.Error(providerName))
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("openrouter: response had no choices")
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
		return nil, fmt.Errorf("openrouter: encode: %w", err)
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
