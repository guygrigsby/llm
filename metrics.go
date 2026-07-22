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
