package openrouter

import (
	"io"

	ac "github.com/voocel/agentcore"
)

// emitSSE is a placeholder until real SSE streaming lands: it reads nothing and
// emits no events, so GenerateStream compiles but does not yet stream tokens.
func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {}
