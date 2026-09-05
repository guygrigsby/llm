// Package openaiwire holds the OpenAI-shaped chat-completions wire format
// shared by every native adapter whose provider speaks that dialect (kimi,
// openrouter): the request and response structs, message and tool
// conversion, stop-reason mapping, the SSE emitter and the authenticated
// HTTP client. Provider-specific extensions (kimi's partial/name and
// reasoning_effort, openrouter's reasoning object and reasoning_details)
// stay in the adapter that needs them; this package only knows the shape
// every OpenAI-compatible provider shares.
package openaiwire

// Request is the shared OpenAI-format chat-completions request body. M is
// the provider's own message type, which embeds Message and adds its own
// fields, so Messages can carry provider extensions without this package
// knowing them. An adapter that needs a different wire name for a shared
// field (kimi sends max_completion_tokens, openrouter sends max_tokens)
// redeclares that field in its own request struct: the redeclaration shadows
// the promoted one for both Go field access and JSON encoding, so only the
// adapter's own tag reaches the wire.
type Request[M any] struct {
	Model       string   `json:"model"`
	Messages    []M      `json:"messages"`
	Tools       []Tool   `json:"tools,omitempty"`
	ToolChoice  any      `json:"tool_choice,omitempty"`
	Stream      bool     `json:"stream"`
	MaxTokens   int      `json:"max_completion_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
}

// Message is the OpenAI-format chat message shape common to every adapter: a
// role, its content, any tool calls an assistant message carries and an
// optional tool_call_id for tool-role messages. Provider extensions (kimi's
// partial/name, openrouter's reasoning/reasoning_details) live in the
// adapter's own message type, which embeds this one.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is a completed OpenAI-format tool call, on an assistant message or
// a non-stream response.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function Func   `json:"function"`
}

// Func is a tool call's function name and arguments (a JSON-encoded string,
// not a raw object, per the OpenAI wire format).
type Func struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool is an OpenAI-format function tool declaration.
type Tool struct {
	Type     string   `json:"type"`
	Function ToolFunc `json:"function"`
}

// ToolFunc is a tool declaration's function shape.
type ToolFunc struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// Delta is one incremental update extracted from a provider's own SSE chunk
// type. The adapter decodes its chunk (with its own quirks: kimi's
// reasoning_content, openrouter's reasoning/reasoning_details) and hands
// Emitter.Handle a Delta so accumulation and event framing stay in one
// place.
type Delta struct {
	Content   string
	Reasoning string
	ToolCalls []StreamToolCall
}

// StreamToolCall is one tool_call delta fragment from an SSE chunk. Index
// identifies which call the fragment belongs to: id and name arrive on the
// first fragment for that index, arguments stream across later ones.
type StreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function Func   `json:"function"`
}
