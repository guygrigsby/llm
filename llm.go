// Package llm is the LLM port for jess/agentcore-based agents, plus a home for
// native, per-provider adapters that satisfy it.
//
// LLM is the ubiquitous term for "the model the agent talks to." It mirrors the
// agentcore ChatModel contract, so anything satisfying LLM plugs directly into
// jess.WithModel — but naming it here keeps the vendor type out of consumers'
// domain language.
//
// Each provider adapter lives in its own subpackage (llm/anthropic, and later
// llm/openai, ...) and is native: it speaks that provider's own API/SDK, not an
// OpenAI-compatible translation. An adapter is an anti-corruption layer — the
// only package allowed to import its provider's SDK — translating that SDK to
// and from agentcore message / tool / stream-event types.
package llm

import (
	"context"

	agentcore "github.com/voocel/agentcore"
)

// LLM is the model port the agent consumes. It is the agentcore ChatModel
// contract under a domain name: any LLM is usable directly as an agent's model.
// agentcore types cross this boundary freely — they are the ubiquitous language
// jess is built on, not an isolated vendor. The isolated vendor is each
// provider's SDK, which only that provider's adapter imports.
type LLM interface {
	Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error)
	GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error)
	SupportsTools() bool
}
