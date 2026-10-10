// Package openaicompat is the adapter for any server that speaks the plain
// OpenAI chat-completions API: a user's own endpoint (vLLM, llama.cpp,
// Ollama, LM Studio, a hosted provider) whose real identity this code never
// knows. That API is the provider's native one here, so this is not a shim
// over a different provider.
//
// It sends only the shape every compatible server shares: max_tokens rather
// than max_completion_tokens, reasoning_effort only when the call sets a
// thinking level, and no reasoning echo on assistant history. Responses are
// read with the OpenRouter-dialect parser (internal/chatwire), a superset of
// the plain shape: fields a plain server never sends are simply absent.
package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	"github.com/guygrigsby/llm/internal/chatwire"
	"github.com/guygrigsby/llm/internal/openaiwire"
	ac "github.com/voocel/agentcore"
)

// defaultStreamIdle is how long a stream may produce nothing before the
// adapter gives up on it.
const defaultStreamIdle = 2 * time.Minute

// providerName labels this adapter's usage and errors.
const providerName = "openaicompat"

// Config configures the adapter. Model and BaseURL are required; BaseURL
// includes the version path, e.g. "https://host/v1".
type Config struct {
	Model   string
	BaseURL string
	// APIKey is optional: local servers often take none, and the header is
	// left off when it is empty. agentcore.WithAPIKey overrides it per call,
	// so one adapter can carry many users' keys. Leave it empty on an adapter
	// that serves user endpoints: a call with no per-call key falls back to
	// it, which would send this key to that user's endpoint.
	APIKey string
	// Transport, if set, carries every request. A host calling a
	// user-supplied URL installs its SSRF-restricting dialer here.
	Transport http.RoundTripper
	// Meter, if set, receives token and latency Usage for every call.
	// Streaming usage arrives only from servers that send it unasked.
	Meter llm.Meter
	// Temperature and TopP are optional sampler params. nil leaves them off
	// the wire.
	Temperature *float64
	TopP        *float64
	// StreamIdleTimeout aborts a stream that has produced nothing for this
	// long. Zero uses two minutes; negative disables the watchdog.
	StreamIdleTimeout time.Duration
}

// Adapter wraps an OpenAI-compatible chat-completions API as an llm.LLM.
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
		return nil, errors.New("openaicompat: Model is required")
	}
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("openaicompat: BaseURL %q must be an absolute http(s) URL", cfg.BaseURL)
	}
	idle := cfg.StreamIdleTimeout
	if idle == 0 {
		idle = defaultStreamIdle
	}
	return &Adapter{
		model: cfg.Model,
		client: openaiwire.Client{
			HTTP: &http.Client{Timeout: 10 * time.Minute, Transport: cfg.Transport},
			// Streaming has no overall deadline; the idle watchdog catches a
			// stalled stream.
			StreamHTTP: &http.Client{Transport: cfg.Transport},
			StreamIdle: idle,
			BaseURL:    strings.TrimRight(cfg.BaseURL, "/"),
			APIKey:     cfg.APIKey,
			Provider:   providerName,
		},
		meter:       cfg.Meter,
		temperature: cfg.Temperature,
		topP:        cfg.TopP,
	}, nil
}

// SupportsTools reports true: tool calling is part of the shared shape. A
// server that lacks it answers with an error rather than this adapter
// guessing.
func (a *Adapter) SupportsTools() bool { return true }

// ProviderName implements agentcore.ProviderNamer.
func (a *Adapter) ProviderName() string { return providerName }

// chatRequest is the plain OpenAI request. MaxTokens shadows the promoted
// max_completion_tokens field (see openaiwire.Request), which many
// compatible servers don't accept.
type chatRequest struct {
	openaiwire.Request[openaiwire.Message]
	MaxTokens       int    `json:"max_tokens,omitempty"`
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

func (a *Adapter) buildRequest(msgs []ac.Message, tools []ac.ToolSpec, cfg ac.CallConfig, stream bool) chatRequest {
	req := chatRequest{
		Request: openaiwire.Request[openaiwire.Message]{
			Model:       a.model,
			Messages:    convertMessages(msgs),
			Tools:       openaiwire.ConvertTools(tools),
			ToolChoice:  cfg.ToolChoice,
			Stream:      stream,
			Temperature: a.temperature,
			TopP:        a.topP,
		},
		MaxTokens:       cfg.MaxTokens,
		ReasoningEffort: chatwire.ReasoningEffort(cfg.ThinkingLevel),
	}
	// A typed nil pointer in the any field would reach the wire as null.
	if cfg.ResponseFormat != nil {
		req.ResponseFormat = cfg.ResponseFormat
	}
	return req
}

// convertMessages maps agentcore messages to plain OpenAI messages. Thinking
// blocks are dropped from assistant history: the plain shape has no field
// for them.
func convertMessages(msgs []ac.Message) []openaiwire.Message {
	return openaiwire.ConvertMessages(msgs,
		func(m ac.Message) openaiwire.Message {
			return openaiwire.Message{Role: "assistant", Content: m.TextContent(), ToolCalls: openaiwire.ToolCallsFrom(m)}
		},
		func(role, content, toolCallID string) openaiwire.Message {
			return openaiwire.Message{Role: role, Content: content, ToolCallID: toolCallID}
		})
}

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
	req := a.buildRequest(msgs, tools, cfg, false)

	start := time.Now()
	data, err := a.client.Post(ctx, "/chat/completions", req, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	var cr chatwire.ChatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("openaicompat: decode: %w", err)
	}
	a.observe(cr.Usage, time.Since(start))
	if cr.Error != nil {
		return nil, fmt.Errorf("%s", cr.Error.Error(providerName))
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("openaicompat: response had no choices")
	}
	return &ac.LLMResponse{Message: chatwire.ConvertResponse(&cr)}, nil
}

// GenerateStream runs a streaming chat completion, emitting the agentcore
// StreamEvent sequence the loop expects.
func (a *Adapter) GenerateStream(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (<-chan ac.StreamEvent, error) {
	cfg := ac.ResolveCallConfig(opts)
	body, err := json.Marshal(a.buildRequest(msgs, tools, cfg, true))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: encode: %w", err)
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
