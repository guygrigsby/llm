# llm

Small, native, per-provider model adapters for
[agentcore](https://github.com/voocel/agentcore) /
[jess](https://github.com/guygrigsby/jess) agents, behind one domain-named port:
`llm.LLM`.

## Why

`agentcore.ChatModel` is the interface an agent talks to. `llm.LLM` is that same
contract under a name that belongs to *your* domain, so the rest of your code
never imports a vendor SDK. Each provider adapter is:

- **Native** — it speaks the provider's own API, not an OpenAI-compatibility
  shim layered over a different provider.
- **An anti-corruption layer** — the only package allowed to import that
  provider's SDK, translating it to and from agentcore message / tool /
  stream-event types.

Add a provider by adding a subpackage; the rest of your agent doesn't change.

## What it's not

Not a gateway. If you want one process that fronts every provider behind an
OpenAI-compatible endpoint, with routing, fallbacks, load balancing, budgets, and
a proxy server, that's [LiteLLM](https://github.com/BerriAI/litellm) or
[OpenRouter](https://openrouter.ai). `llm` is the opposite shape:

- **A library, not a server.** No proxy, no gateway, no daemon. You import it;
  calls go straight from your process to the provider.
- **Native, not OpenAI-flattened.** Each adapter speaks its provider's real API
  and can surface provider-specific behavior (Anthropic's thinking/effort),
  instead of collapsing everything to a lowest-common-denominator shape.
- **No orchestration baked in.** Routing, fallback, retries, load balancing,
  response caching, cost and budget tracking are the agent's or caller's job,
  not this layer's.
- **No giant model registry.** You add the one or two adapters you actually use;
  nothing else ships.

One port (`llm.LLM`), native adapters behind it. That's the whole scope.

## Install

```sh
go get github.com/guygrigsby/llm
```

Requires Go 1.26+.

## The port

```go
type LLM interface {
	Generate(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (*agentcore.LLMResponse, error)
	GenerateStream(ctx context.Context, messages []agentcore.Message, tools []agentcore.ToolSpec, opts ...agentcore.CallOption) (<-chan agentcore.StreamEvent, error)
	SupportsTools() bool
}
```

Anything satisfying `llm.LLM` is an `agentcore.ChatModel`, so it drops straight
into `jess.WithModel`. agentcore types cross the boundary freely — they are the
ubiquitous language jess is built on, not an isolated vendor. The isolated vendor
is each provider's SDK, which only that provider's adapter imports.

## Adapters

### anthropic

`github.com/guygrigsby/llm/anthropic` — native Anthropic adapter on the official
[`anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go). Adaptive
thinking + effort; never sends temperature/top_p/top_k. Full streaming and tool
support.

```go
m, err := anthropic.New(anthropic.Config{APIKey: key, Model: "claude-sonnet-5"})
if err != nil {
	return err
}
agent := jess.New(jess.WithModel(m) /* , ... */)
```

### deepseek

`github.com/guygrigsby/llm/deepseek` — native DeepSeek adapter over its
OpenAI-format chat-completions API, using only `net/http` (no SDK dependency).
Built for cheap one-shot work such as summaries and extraction; tool-calling and
true token streaming are not wired yet (`GenerateStream` returns the whole result
as one terminal event). Wire those before using it as a primary conversational
model.

```go
m, err := deepseek.New(deepseek.Config{APIKey: key, Model: "deepseek-chat"})
```

### kimi

`github.com/guygrigsby/llm/kimi` — native Kimi (Moonshot) adapter over its
OpenAI-format chat-completions API, using only `net/http` (no SDK dependency).
Full scope: real SSE token streaming and tool calling, so `kimi-k3` works as a
primary conversational model. Kimi-specific extensions beyond standard OpenAI
are wired too: `reasoning_effort` (mapped from the call's thinking level, with
the model's `reasoning_content` surfaced as a thinking block), partial mode
(prefix continuation, triggered by `"partial": true` metadata on a trailing
assistant message), and an `EstimateTokens` helper over the
`/tokenizers/estimate-token-count` endpoint.

```go
m, err := kimi.New(kimi.Config{APIKey: key, Model: "kimi-k3"})
if err != nil {
	return err
}
agent := jess.New(jess.WithModel(m) /* , ... */)
```

### openrouter

`github.com/guygrigsby/llm/openrouter`, a native OpenRouter adapter over its
OpenAI-format chat-completions API, using only `net/http` (no SDK dependency).
One key, any OpenRouter model id per adapter (`moonshotai/kimi-k3`,
`anthropic/claude-opus-5`, ...). Full scope: SSE token streaming and tool
calling. OpenRouter extensions are wired: the `reasoning` request object with
effort mapped from the call's thinking level; the model's reasoning surfaced as
a thinking block; `reasoning_details` echoed back on assistant messages so the
model keeps its chain of thought across tool calls. Usage accounting reaches
the `Meter` with `cost` and cached prompt tokens.

```go
m, err := openrouter.New(openrouter.Config{APIKey: key, Model: "moonshotai/kimi-k3", Title: "my-app"})
if err != nil {
	return err
}
agent := jess.New(jess.WithModel(m) /* , ... */)
```

### aperture

`github.com/guygrigsby/llm/aperture`, a native adapter for Aperture, the
Tailscale model gateway on the tailnet, over its OpenAI-format chat-completions API (the OpenRouter dialect,
shared with `openrouter` through `internal/chatwire`). The tailnet
authenticates the host rather than a bearer, so no API key is required (one
can still be set for a route that asks); the endpoint defaults to
`https://ai.guy.ts.net/v1` and `BaseURL` overrides it. Model ids are
Aperture's own, as its `/v1/models` lists them (`anthropic/claude-sonnet-5.5`,
`moonshotai/kimi-k3`, `qwen-27b`, ...). Full scope matches `openrouter`: SSE
token streaming, tool calling, the `reasoning` request object and
`reasoning_details` echo, and usage accounting (cost included where the route
prices) reaching the `Meter` labelled `aperture`.

```go
m, err := aperture.New(aperture.Config{Model: "anthropic/claude-sonnet-5.5"})
if err != nil {
	return err
}
agent := jess.New(jess.WithModel(m) /* , ... */)
```

### openaicompat

`github.com/guygrigsby/llm/openaicompat`, for any server whose native API is
plain OpenAI chat completions: a user's own endpoint (vLLM, llama.cpp, Ollama,
LM Studio, a hosted provider) that the caller can't know in advance. It sends
only the shape every compatible server shares: `max_tokens`,
`reasoning_effort` only when the call sets a thinking level, and no reasoning
echo on assistant history. `BaseURL` and `Model` are required; `APIKey` is
optional and `agentcore.WithAPIKey` overrides it per call, so one adapter can
carry many users' keys. `Transport` carries every request, for a host that
must restrict where user-supplied URLs can reach. SSE token streaming and tool
calling; usage reaches the `Meter` labelled `openaicompat` when the server
reports it.

```go
m, err := openaicompat.New(openaicompat.Config{BaseURL: userURL, Model: userModel, Transport: restricted})
if err != nil {
	return err
}
resp, err := m.Generate(ctx, msgs, nil, agentcore.WithAPIKey(userKey))
```

## License

MIT. See [LICENSE](LICENSE).
