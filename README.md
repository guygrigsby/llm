# llm

The `LLM` port for [jess](https://github.com/guygrigsby/jess) /
[agentcore](https://github.com/voocel/agentcore) agents, plus native,
per-provider adapters that satisfy it.

`llm.LLM` names "the model the agent talks to" in domain language — it mirrors
the agentcore `ChatModel` contract, so anything satisfying it plugs straight into
`jess.WithModel`, without the rest of the codebase importing a vendor SDK.

Each adapter is **native and provider-specific**: it speaks that provider's own
API/SDK, not an OpenAI-compatible translation layer. An adapter is an
anti-corruption layer — the only package allowed to import its provider's SDK —
translating it to and from agentcore message / tool / stream-event types.

## anthropic

`github.com/guygrigsby/llm/anthropic` wraps `anthropic-sdk-go` as an `llm.LLM`.
Adaptive thinking + effort; never sends temperature/top_p/top_k.

```go
m, err := anthropic.New(anthropic.Config{APIKey: key, Model: "claude-sonnet-5"})
if err != nil {
    return err
}
agent := jess.New(jess.WithModel(m), /* ... */)
```

Extracted from gyr's `internal/model/anthropic` so gyr, johnny, and future
services share one tested adapter.
