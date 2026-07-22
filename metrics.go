package llm

import "time"

// Usage reports the token counts and wall-clock latency of a single model call.
// Adapters populate it from the provider response and hand it to a Meter (when
// one is configured), so a consumer can record cost and latency without the
// adapter knowing anything about pricing.
type Usage struct {
	Provider         string        `json:"provider"`
	Model            string        `json:"model"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	TotalTokens      int           `json:"total_tokens"`
	Latency          time.Duration `json:"latency"`
	// CacheReadTokens and CacheWriteTokens split out prompt-caching input tokens
	// so a Meter can price the cache tiers (read ~0.1x, write ~1.25x-2x base
	// input) and measure hit rate. Zero when the provider reports no caching or
	// none was requested. PromptTokens is the uncached-input remainder, so the
	// three are additive: total input = PromptTokens + CacheReadTokens + CacheWriteTokens.
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// Meter observes per-call Usage. Set one on an adapter's Config to capture
// tokens and latency for every generation. Implementations must be safe for
// concurrent use — adapters may call Observe from multiple goroutines.
type Meter interface {
	Observe(Usage)
}

// MeterFunc adapts a plain function to a Meter.
type MeterFunc func(Usage)

// Observe calls f.
func (f MeterFunc) Observe(u Usage) { f(u) }
