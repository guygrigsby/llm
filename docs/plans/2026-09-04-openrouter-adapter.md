# OpenRouter adapter implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `github.com/guygrigsby/llm/openrouter`, a native adapter for OpenRouter's chat-completions API, so any OpenRouter model id (Kimi K3, Claude, DeepSeek, ...) plugs into `jess.WithModel`.

**Architecture:** One subpackage, no SDK, `net/http` only, structured exactly like `llm/kimi`: `openrouter.go` (Config, New, Generate, GenerateStream, HTTP), `convert.go` (pure wire translation), `stream.go` (SSE framing). OpenRouter's API is OpenAI-shaped with three extensions this adapter wires: the `reasoning` request object, `reasoning` / `reasoning_details` on responses (echoed back on assistant messages for tool-call continuity), and usage with `cost` and `prompt_tokens_details.cached_tokens`.

**Tech Stack:** Go 1.26, `github.com/voocel/agentcore` types, `net/http/httptest` for tests. Live tests behind the `e2e` build tag keyed by `OPENROUTER_API_KEY`.

**Spec:** `/Users/guygrigsby/projects/autophage/docs/adr/0006-openrouter-as-model-provider.md` (the decision) and the OpenRouter docs: parameters (`reasoning` object with `effort` in `xhigh, high, medium, low, minimal, none`), reasoning tokens (`message.reasoning` string, `message.reasoning_details` array, streamed as `delta.reasoning` and `delta.reasoning_details`; echo `reasoning_details` unmodified on assistant messages), usage accounting (`usage` is always present, in the last SSE chunk when streaming; `usage.cost`, `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens_details.reasoning_tokens`; `stream_options` and `usage.include` are deprecated no-ops).

## Global Constraints

- Repo: `/Users/guygrigsby/projects/llm`, module `github.com/guygrigsby/llm`, Go 1.26. Work on `main`.
- Only this package may know OpenRouter's wire format. No vendor types leave it. The port is `llm.LLM`.
- No em or en dashes anywhere. No Oxford commas. Commit messages terse and verb-first, prefixed `openrouter:` (or `llm:` for the port change). No Claude or Anthropic attribution, no `Co-Authored-By` trailers.
- `go vet ./... && go test -race ./...` green before every commit.
- Base URL `https://openrouter.ai/api/v1`; chat path `/chat/completions`. Auth header `Authorization: Bearer <key>`. Optional attribution headers `HTTP-Referer` and `X-Title`.
- Never send `stream_options` or `usage.include`.
- Do not push and do not tag beyond what Task 6 says.

---

### Task 1: Package skeleton, config and non-streaming Generate

**Files:**
- Create: `openrouter/openrouter.go`
- Create: `openrouter/convert.go`
- Test: `openrouter/openrouter_test.go`

**Interfaces:**
- Consumes: `llm.LLM`, `llm.Meter`, `llm.Usage` from the module root; agentcore `Message`, `ToolSpec`, `CallOption`, `ResolveCallConfig`, `ThinkingLevel`.
- Produces: `openrouter.Config{APIKey, Model, BaseURL, Referer, Title string; Meter llm.Meter; Temperature, TopP *float64}`, `openrouter.New(Config) (*Adapter, error)`, `(*Adapter).Generate`, `(*Adapter).SupportsTools() bool`, `(*Adapter).ProviderName() string` returning `"openrouter"`. Unexported: `chatRequest`, `chatResponse`, `wireMessage`, `wireToolCall`, `wireTool`, `wireUsage`, `reasoningParam`, `buildRequest`, `convertMessages`, `convertTools`, `assembleMessage`, `mapStopReason`, `toAgentUsage`, `toLLMUsage`, `reasoningEffort`, `argsString`, `rawArgs`.

- [ ] **Step 1: Write the failing tests**

```go
package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	ac "github.com/voocel/agentcore"
)

func TestNewValidates(t *testing.T) {
	if _, err := New(Config{Model: "m"}); err == nil {
		t.Error("missing APIKey should fail")
	}
	if _, err := New(Config{APIKey: "k"}); err == nil {
		t.Error("missing Model should fail")
	}
	a, err := New(Config{APIKey: "k", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if a.baseURL != "https://openrouter.ai/api/v1" {
		t.Errorf("baseURL = %q", a.baseURL)
	}
	if a.ProviderName() != "openrouter" || !a.SupportsTools() {
		t.Error("provider name or tools flag wrong")
	}
}

func TestGenerateTextAndHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth = %q", got)
		}
		if got := r.Header.Get("HTTP-Referer"); got != "https://example.test" {
			t.Errorf("referer = %q", got)
		}
		if got := r.Header.Get("X-Title"); got != "autophage" {
			t.Errorf("title = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, err := New(Config{APIKey: "test-key", Model: "moonshotai/kimi-k3", BaseURL: srv.URL, Referer: "https://example.test", Title: "autophage"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("hey")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.TextContent(); got != "hi there" {
		t.Errorf("text = %q", got)
	}
	if resp.Message.StopReason != ac.StopReasonStop {
		t.Errorf("stop = %q", resp.Message.StopReason)
	}
}

func TestGenerateToolCall(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Denver\"}"}}]},"finish_reason":"tool_calls"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	tools := []ac.ToolSpec{{Name: "get_weather", Description: "weather", Parameters: map[string]any{"type": "object"}}}
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("weather?")}, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(body.Tools) != 1 || body.Tools[0].Function.Name != "get_weather" {
		t.Errorf("tools on wire = %+v", body.Tools)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "get_weather" || string(calls[0].Args) != `{"city":"Denver"}` {
		t.Errorf("calls = %+v", calls)
	}
	if resp.Message.StopReason != ac.StopReasonToolUse {
		t.Errorf("stop = %q", resp.Message.StopReason)
	}
}

func TestGenerateReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"42","reasoning":"let me think","reasoning_details":[{"type":"reasoning.text","text":"let me think"}]},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.ThinkingContent(); got != "let me think" {
		t.Errorf("thinking = %q", got)
	}
	if got := resp.Message.TextContent(); got != "42" {
		t.Errorf("text = %q", got)
	}
	details, _ := resp.Message.Metadata["reasoning_details"].(json.RawMessage)
	if string(details) != `[{"type":"reasoning.text","text":"let me think"}]` {
		t.Errorf("reasoning_details metadata = %s", details)
	}
}

func TestReasoningEffortSent(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithThinking(ac.ThinkingXHigh)); err != nil {
		t.Fatal(err)
	}
	if body.Reasoning == nil || body.Reasoning.Effort != "xhigh" {
		t.Errorf("reasoning = %+v, want effort xhigh", body.Reasoning)
	}
	body = chatRequest{}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil); err != nil {
		t.Fatal(err)
	}
	if body.Reasoning != nil {
		t.Errorf("reasoning should be omitted without a thinking level, got %+v", body.Reasoning)
	}
}

func TestReasoningDetailsEchoed(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	prior := ac.Message{
		Role:     ac.RoleAssistant,
		Content:  []ac.ContentBlock{ac.ThinkingBlock("thought"), ac.ToolCallBlock(ac.ToolCall{ID: "c1", Name: "t", Args: json.RawMessage(`{}`)})},
		Metadata: map[string]any{"reasoning_details": json.RawMessage(`[{"type":"reasoning.encrypted","data":"xyz"}]`)},
	}
	toolResult := ac.Message{Role: ac.RoleTool, Content: []ac.ContentBlock{ac.TextBlock("ok")}, Metadata: map[string]any{"tool_call_id": "c1"}}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q"), prior, toolResult}, nil); err != nil {
		t.Fatal(err)
	}
	if len(body.Messages) != 3 {
		t.Fatalf("messages = %d", len(body.Messages))
	}
	if string(body.Messages[1].ReasoningDetails) != `[{"type":"reasoning.encrypted","data":"xyz"}]` {
		t.Errorf("reasoning_details not echoed: %s", body.Messages[1].ReasoningDetails)
	}
	if body.Messages[1].Reasoning != "" {
		t.Errorf("plaintext reasoning must not be sent when details are present, got %q", body.Messages[1].Reasoning)
	}
	if body.Messages[2].Role != "tool" || body.Messages[2].ToolCallID != "c1" {
		t.Errorf("tool message = %+v", body.Messages[2])
	}
}

func TestPlaintextReasoningEchoedWithoutDetails(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	prior := ac.Message{Role: ac.RoleAssistant, Content: []ac.ContentBlock{ac.ThinkingBlock("thought"), ac.TextBlock("partial")}}
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q"), prior}, nil); err != nil {
		t.Fatal(err)
	}
	if body.Messages[1].Reasoning != "thought" {
		t.Errorf("reasoning = %q, want thought", body.Messages[1].Reasoning)
	}
}

func TestGenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"error":{"code":402,"message":"insufficient credits"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil || !contains(err.Error(), "insufficient credits") {
		t.Errorf("err = %v", err)
	}
}

func TestGenerateBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"error":{"code":400,"message":"model not found"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	_, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err == nil || !contains(err.Error(), "model not found") {
		t.Errorf("err = %v", err)
	}
}

func TestSamplerParamsSent(t *testing.T) {
	var body chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	temp, topP := 0.2, 0.9
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Temperature: &temp, TopP: &topP})
	if _, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil, ac.WithMaxTokens(77)); err != nil {
		t.Fatal(err)
	}
	if body.Temperature == nil || *body.Temperature != 0.2 || body.TopP == nil || *body.TopP != 0.9 {
		t.Errorf("sampler = %+v %+v", body.Temperature, body.TopP)
	}
	if body.MaxTokens != 77 {
		t.Errorf("max_tokens = %d", body.MaxTokens)
	}
}

func contains(s, sub string) bool { return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0) }

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
```

If `ac.WithMaxTokens` does not exist under that name, read `$(go env GOMODCACHE)/github.com/voocel/agentcore@v1.6.9/model.go` for the `CallOption` that sets `CallConfig.MaxTokens` and use it; the kimi adapter's `buildRequest` reads `cfg.MaxTokens`, so the option exists.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/guygrigsby/projects/llm && go test ./openrouter/ 2>&1 | head -20`
Expected: build failure, `undefined: New`, `undefined: chatRequest`.

- [ ] **Step 3: Write convert.go**

```go
package openrouter

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

// This file holds the pure (network-free) translation between agentcore types
// and OpenRouter's OpenAI-shaped chat-completions wire format, plus the wire
// structs. Request building and response conversion are standalone functions
// so they are unit-testable without a live API.

const providerName = "openrouter"

// wire request -------------------------------------------------------------

type chatRequest struct {
	Model       string          `json:"model"`
	Messages    []wireMessage   `json:"messages"`
	Tools       []wireTool      `json:"tools,omitempty"`
	ToolChoice  any             `json:"tool_choice,omitempty"`
	Stream      bool            `json:"stream"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Reasoning   *reasoningParam `json:"reasoning,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
}

// reasoningParam is OpenRouter's unified reasoning control. Only effort is
// wired; max_tokens, exclude and enabled are left to the model's defaults.
type reasoningParam struct {
	Effort string `json:"effort"`
}

// wireMessage is one OpenAI-format chat message. On assistant messages,
// ReasoningDetails echoes the provider's structured reasoning back unmodified
// so a model resumes its chain of thought across tool calls; Reasoning is the
// plaintext fallback when no details were returned.
type wireMessage struct {
	Role             string          `json:"role"`
	Content          string          `json:"content,omitempty"`
	ToolCalls        []wireToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	Reasoning        string          `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
}

type wireToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function wireFunc `json:"function"`
}

type wireFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireToolFunc `json:"function"`
}

type wireToolFunc struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// wire response ------------------------------------------------------------

type wireError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role             string          `json:"role"`
			Content          string          `json:"content"`
			Reasoning        string          `json:"reasoning"`
			ReasoningDetails json.RawMessage `json:"reasoning_details"`
			ToolCalls        []wireToolCall  `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

// wireUsage is OpenRouter's usage block. prompt_tokens INCLUDES cached tokens
// (OpenAI convention), so toLLMUsage subtracts to keep llm.Usage additive.
// Cost is in USD credits, always reported by OpenRouter.
type wireUsage struct {
	PromptTokens        int     `json:"prompt_tokens"`
	CompletionTokens    int     `json:"completion_tokens"`
	TotalTokens         int     `json:"total_tokens"`
	Cost                float64 `json:"cost"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

// genParams carries the adapter-level sampler settings into a request.
// Pointer fields mean "unset, do not send".
type genParams struct {
	Temperature *float64
	TopP        *float64
}

// buildRequest assembles a chatRequest from agentcore inputs and the resolved
// per-call config. The reasoning object is set only when the call carries a
// thinking level. OpenRouter always returns usage, in the last chunk when
// streaming, so no stream_options are sent.
func buildRequest(model string, msgs []ac.Message, tools []ac.ToolSpec, cfg ac.CallConfig, gen genParams, stream bool) chatRequest {
	req := chatRequest{
		Model:       model,
		Messages:    convertMessages(msgs),
		Tools:       convertTools(tools),
		ToolChoice:  cfg.ToolChoice,
		Stream:      stream,
		MaxTokens:   cfg.MaxTokens,
		Temperature: gen.Temperature,
		TopP:        gen.TopP,
	}
	if effort := reasoningEffort(cfg.ThinkingLevel); effort != "" {
		req.Reasoning = &reasoningParam{Effort: effort}
	}
	return req
}

// convertMessages maps agentcore messages to OpenAI-format wire messages. The
// system prompt stays inline as a system-role message and tool results are
// their own "tool" role.
func convertMessages(msgs []ac.Message) []wireMessage {
	out := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case ac.RoleSystem:
			out = append(out, wireMessage{Role: "system", Content: m.TextContent()})
		case ac.RoleUser:
			out = append(out, wireMessage{Role: "user", Content: m.TextContent()})
		case ac.RoleTool:
			id, _ := m.Metadata["tool_call_id"].(string)
			out = append(out, wireMessage{Role: "tool", ToolCallID: id, Content: m.TextContent()})
		case ac.RoleAssistant:
			out = append(out, assistantMessage(m))
		}
	}
	return out
}

// assistantMessage builds the wire assistant message: text, tool calls, and
// the reasoning echo. reasoning_details from metadata wins; otherwise the
// thinking block text is sent as plaintext reasoning.
func assistantMessage(m ac.Message) wireMessage {
	wm := wireMessage{Role: "assistant", Content: m.TextContent()}
	for _, c := range m.ToolCalls() {
		wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
			ID:       c.ID,
			Type:     "function",
			Function: wireFunc{Name: c.Name, Arguments: argsString(c.Args)},
		})
	}
	if d, ok := m.Metadata["reasoning_details"].(json.RawMessage); ok && len(d) > 0 {
		wm.ReasoningDetails = d
	} else if r := m.ThinkingContent(); r != "" {
		wm.Reasoning = r
	}
	return wm
}

// convertTools maps agentcore tool specs to OpenAI-format function tools.
func convertTools(tools []ac.ToolSpec) []wireTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]wireTool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			continue
		}
		out = append(out, wireTool{
			Type: "function",
			Function: wireToolFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
				Strict:      t.Strict,
			},
		})
	}
	return out
}

// reasoningEffort maps an agentcore thinking level to OpenRouter's effort
// enum. Off and unspecified return "" so the reasoning object is omitted.
func reasoningEffort(level ac.ThinkingLevel) string {
	switch level {
	case ac.ThinkingMinimal:
		return "minimal"
	case ac.ThinkingLow:
		return "low"
	case ac.ThinkingMedium:
		return "medium"
	case ac.ThinkingHigh:
		return "high"
	case ac.ThinkingXHigh:
		return "xhigh"
	default:
		return ""
	}
}

// convertResponse maps a non-stream chat response choice to an agentcore
// message: reasoning first (a thinking block), then text, then every tool call.
func convertResponse(cr *chatResponse) ac.Message {
	c := cr.Choices[0]
	msg := assembleMessage(c.Message.Reasoning, c.Message.Content, c.Message.ToolCalls, c.FinishReason, c.Message.ReasoningDetails)
	msg.Usage = toAgentUsage(cr.Usage)
	return msg
}

// assembleMessage builds an assistant message from its parts, in the order the
// agent loop expects (thinking, text, tool calls), and stashes the provider's
// reasoning_details in metadata so assistantMessage can echo it later. Shared
// by the non-stream path and the stream's terminal message.
func assembleMessage(reasoning, content string, calls []wireToolCall, finish string, details json.RawMessage) ac.Message {
	var blocks []ac.ContentBlock
	if reasoning != "" {
		blocks = append(blocks, ac.ThinkingBlock(reasoning))
	}
	if content != "" {
		blocks = append(blocks, ac.TextBlock(content))
	}
	for _, c := range calls {
		blocks = append(blocks, ac.ToolCallBlock(ac.ToolCall{
			ID:   c.ID,
			Name: c.Function.Name,
			Args: rawArgs(c.Function.Arguments),
		}))
	}
	msg := ac.Message{Role: ac.RoleAssistant, Content: blocks, StopReason: mapStopReason(finish)}
	if len(details) > 0 && string(details) != "null" {
		msg.Metadata = map[string]any{"reasoning_details": details}
	}
	return msg
}

// mapStopReason maps an OpenAI finish_reason to an agentcore StopReason.
func mapStopReason(reason string) ac.StopReason {
	switch reason {
	case "tool_calls":
		return ac.StopReasonToolUse
	case "length":
		return ac.StopReasonLength
	case "stop", "":
		return ac.StopReasonStop
	default:
		return ac.StopReason(reason)
	}
}

// toAgentUsage maps wire usage to agentcore Usage (Input keeps the OpenAI
// convention of including cached tokens). nil when nothing was reported.
func toAgentUsage(u *wireUsage) *ac.Usage {
	if u == nil || (u.PromptTokens == 0 && u.CompletionTokens == 0) {
		return nil
	}
	return &ac.Usage{
		Input:       u.PromptTokens,
		Output:      u.CompletionTokens,
		CacheRead:   u.PromptTokensDetails.CachedTokens,
		TotalTokens: u.TotalTokens,
	}
}

// toLLMUsage maps wire usage to llm.Usage for the Meter. prompt_tokens includes
// cached tokens on the wire, so PromptTokens is the uncached remainder and the
// input fields stay additive.
func toLLMUsage(model string, u *wireUsage, latency time.Duration) llm.Usage {
	cached := u.PromptTokensDetails.CachedTokens
	prompt := u.PromptTokens - cached
	if prompt < 0 {
		prompt = 0
	}
	return llm.Usage{
		Provider:         providerName,
		Model:            model,
		PromptTokens:     prompt,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		Latency:          latency,
		CacheReadTokens:  cached,
		Cost:             u.Cost,
	}
}

// argsString renders tool-call args as the OpenAI arguments string. Empty
// args become "{}" so the wire message stays valid.
func argsString(args json.RawMessage) string {
	if len(strings.TrimSpace(string(args))) == 0 {
		return "{}"
	}
	return string(args)
}

// rawArgs is the inverse: a wire arguments string to json.RawMessage,
// defaulting empty to "{}".
func rawArgs(s string) json.RawMessage {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}
```

`llm.Usage.Cost` does not exist yet; Task 2 adds it. Write convert.go as above and expect that one compile error until Task 2, or do Task 2's one-line port change first (either order; keep the commits separate).

- [ ] **Step 4: Write openrouter.go**

```go
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
package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/guygrigsby/llm"
	ac "github.com/voocel/agentcore"
)

// defaultBaseURL is OpenRouter's public endpoint. /api/v1 is part of the base,
// so the chat path is baseURL + "/chat/completions".
const defaultBaseURL = "https://openrouter.ai/api/v1"

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
}

// Adapter wraps OpenRouter's chat-completions API as an llm.LLM.
type Adapter struct {
	apiKey      string
	model       string
	baseURL     string
	referer     string
	title       string
	http        *http.Client
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
	return &Adapter{
		apiKey:      cfg.APIKey,
		model:       cfg.Model,
		baseURL:     strings.TrimRight(base, "/"),
		referer:     cfg.Referer,
		title:       cfg.Title,
		http:        &http.Client{Timeout: 10 * time.Minute},
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
func (a *Adapter) observe(u *wireUsage, latency time.Duration) {
	if a.meter == nil || u == nil {
		return
	}
	a.meter.Observe(toLLMUsage(a.model, u, latency))
}

// Generate runs a one-shot chat completion.
func (a *Adapter) Generate(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (*ac.LLMResponse, error) {
	cfg := ac.ResolveCallConfig(opts)
	req := buildRequest(a.model, msgs, tools, cfg, genParams{Temperature: a.temperature, TopP: a.topP}, false)

	start := time.Now()
	data, err := a.post(ctx, "/chat/completions", req, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("openrouter: decode: %w", err)
	}
	if cr.Error != nil {
		return nil, fmt.Errorf("openrouter: %d: %s", cr.Error.Code, cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("openrouter: response had no choices")
	}
	a.observe(cr.Usage, time.Since(start))
	return &ac.LLMResponse{Message: convertResponse(&cr)}, nil
}

// GenerateStream runs a streaming chat completion, emitting the agentcore
// StreamEvent sequence the loop expects: thinking/text start-delta-end,
// tool-call start/delta/end with the completed call, a terminal done event
// carrying the fully assembled message and stop reason, and an error event on
// failure.
func (a *Adapter) GenerateStream(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (<-chan ac.StreamEvent, error) {
	cfg := ac.ResolveCallConfig(opts)
	req := buildRequest(a.model, msgs, tools, cfg, genParams{Temperature: a.temperature, TopP: a.topP}, true)

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("openrouter: encode: %w", err)
	}
	httpReq, err := a.newRequest(ctx, "/chat/completions", body, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	start := time.Now()
	resp, err := a.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, fmt.Errorf("openrouter: %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}

	out := make(chan ac.StreamEvent, 64)
	go func() {
		defer close(out)
		defer func() { _ = resp.Body.Close() }()
		emitSSE(resp.Body, out, func(u *wireUsage) { a.observe(u, time.Since(start)) })
	}()
	return out, nil
}

// post marshals body, sends it to path, and returns the response bytes,
// failing on a non-200 with the response text. apiKey overrides the adapter
// key when set (per-call key rotation via WithAPIKey).
func (a *Adapter) post(ctx context.Context, path string, body any, apiKey string) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openrouter: encode: %w", err)
	}
	req, err := a.newRequest(ctx, path, raw, apiKey)
	if err != nil {
		return nil, err
	}
	resp, err := a.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openrouter: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openrouter: %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	return data, nil
}

// newRequest builds an authenticated JSON POST with OpenRouter's optional
// attribution headers. apiKey overrides the adapter's key when non-empty.
func (a *Adapter) newRequest(ctx context.Context, path string, body []byte, apiKey string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	key := a.apiKey
	if apiKey != "" {
		key = apiKey
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	if a.referer != "" {
		req.Header.Set("HTTP-Referer", a.referer)
	}
	if a.title != "" {
		req.Header.Set("X-Title", a.title)
	}
	return req, nil
}
```

`emitSSE` is written in Task 3. Until then add a one-line stub in `stream.go` so the package compiles: `func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {}` with the needed imports. Task 3 replaces it.

- [ ] **Step 5: Run the tests**

Run: `cd /Users/guygrigsby/projects/llm && go test ./openrouter/ -run 'TestNew|TestGenerate|TestReasoning|TestPlaintext|TestSampler' -v 2>&1 | tail -30`
Expected: all PASS once Task 2's `Cost` field exists (do Task 2 Step 3 first if the build fails on `Cost`).

- [ ] **Step 6: Commit**

```bash
cd /Users/guygrigsby/projects/llm && go vet ./... && gofmt -l . && git add openrouter/openrouter.go openrouter/convert.go openrouter/stream.go openrouter/openrouter_test.go && git commit -m "openrouter: native adapter, non-stream generate with reasoning and tool calls"
```

---

### Task 2: Cost on the port's Usage

**Files:**
- Modify: `metrics.go` (the `Usage` struct)
- Test: `openrouter/openrouter_test.go` (append)

**Interfaces:**
- Produces: `llm.Usage.Cost float64` (`json:"cost,omitempty"`), provider-reported cost in USD credits, zero when the provider does not report one.

- [ ] **Step 1: Write the failing test**

Append to `openrouter/openrouter_test.go`:

```go
func TestGenerateMeters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150,"cost":0.00042,"prompt_tokens_details":{"cached_tokens":100},"completion_tokens_details":{"reasoning_tokens":10}}}`))
	}))
	defer srv.Close()

	var got llm.Usage
	a, _ := New(Config{APIKey: "k", Model: "moonshotai/kimi-k3", BaseURL: srv.URL, Meter: llm.MeterFunc(func(u llm.Usage) { got = u })})
	resp, err := a.Generate(context.Background(), []ac.Message{ac.UserMsg("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Provider != "openrouter" || got.Model != "moonshotai/kimi-k3" {
		t.Errorf("identity = %+v", got)
	}
	if got.PromptTokens != 20 || got.CacheReadTokens != 100 || got.CompletionTokens != 30 || got.TotalTokens != 150 {
		t.Errorf("tokens = %+v", got)
	}
	if got.Cost != 0.00042 {
		t.Errorf("cost = %v", got.Cost)
	}
	if got.Latency <= 0 {
		t.Error("latency not recorded")
	}
	if resp.Message.Usage == nil || resp.Message.Usage.Input != 120 || resp.Message.Usage.CacheRead != 100 {
		t.Errorf("agent usage = %+v", resp.Message.Usage)
	}
}
```

Add `"github.com/guygrigsby/llm"` to the test file's imports.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/guygrigsby/projects/llm && go test ./openrouter/ -run TestGenerateMeters 2>&1 | head`
Expected: compile error `got.Cost undefined` (or `Cost` unknown field in convert.go).

- [ ] **Step 3: Add the field**

In `metrics.go`, inside `type Usage struct` after `CacheWriteTokens`:

```go
	// Cost is the provider-reported cost of the call in USD credits, for
	// providers that price per call (OpenRouter). Zero when not reported; a
	// Meter that prices from tokens ignores it.
	Cost float64 `json:"cost,omitempty"`
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/guygrigsby/projects/llm && go test -race ./... 2>&1 | tail -8`
Expected: PASS for every package.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/llm && git add metrics.go openrouter/openrouter_test.go && git commit -m "llm: report provider cost on Usage"
```

---

### Task 3: SSE streaming

**Files:**
- Create: `openrouter/stream.go` (replace the stub)
- Test: `openrouter/stream_test.go`

**Interfaces:**
- Consumes: `assembleMessage`, `toAgentUsage`, `rawArgs`, `wireUsage`, `wireToolCall`, `wireFunc`, `wireError` from Task 1.
- Produces: `emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage))`.

- [ ] **Step 1: Write the failing tests**

```go
package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ac "github.com/voocel/agentcore"
)

func sseServer(t *testing.T, frames ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("accept = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(": OPENROUTER PROCESSING\n\n"))
		for _, f := range frames {
			_, _ = w.Write([]byte("data: " + f + "\n\n"))
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
}

func collect(t *testing.T, ch <-chan ac.StreamEvent) []ac.StreamEvent {
	t.Helper()
	var evs []ac.StreamEvent
	for ev := range ch {
		evs = append(evs, ev)
	}
	return evs
}

func TestStreamText(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"reasoning":"hmm "},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning":"ok"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"cost":0.0001}}`,
	)
	defer srv.Close()

	var metered bool
	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Meter: meterFunc(func() { metered = true })})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	var text, thinking strings.Builder
	var done *ac.StreamEvent
	for i := range evs {
		switch evs[i].Type {
		case ac.StreamEventTextDelta:
			text.WriteString(evs[i].Delta)
		case ac.StreamEventThinkingDelta:
			thinking.WriteString(evs[i].Delta)
		case ac.StreamEventDone:
			done = &evs[i]
		case ac.StreamEventError:
			t.Fatalf("unexpected error event: %v", evs[i].Err)
		}
	}
	if text.String() != "Hello" || thinking.String() != "hmm ok" {
		t.Errorf("text = %q thinking = %q", text.String(), thinking.String())
	}
	if done == nil {
		t.Fatal("no done event")
	}
	if done.Message.TextContent() != "Hello" || done.Message.ThinkingContent() != "hmm ok" || done.StopReason != ac.StopReasonStop {
		t.Errorf("final = %+v", done.Message)
	}
	if done.Message.Usage == nil || done.Message.Usage.Input != 5 {
		t.Errorf("usage = %+v", done.Message.Usage)
	}
	if !metered {
		t.Error("meter not called")
	}
}

func TestStreamToolCallAndReasoningDetails(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"a"}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"b"}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":"tool_calls"}]}`,
	)
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	var ends int
	var done *ac.StreamEvent
	for i := range evs {
		switch evs[i].Type {
		case ac.StreamEventToolCallEnd:
			ends++
			if evs[i].CompletedToolCall == nil || evs[i].CompletedToolCall.Name != "lookup" || string(evs[i].CompletedToolCall.Args) != `{"q":"x"}` {
				t.Errorf("completed call = %+v", evs[i].CompletedToolCall)
			}
		case ac.StreamEventDone:
			done = &evs[i]
		}
	}
	if ends != 1 {
		t.Errorf("tool call ends = %d", ends)
	}
	if done == nil || done.StopReason != ac.StopReasonToolUse {
		t.Fatalf("done = %+v", done)
	}
	details, _ := done.Message.Metadata["reasoning_details"].(json.RawMessage)
	if string(details) != `[{"type":"reasoning.text","text":"a"},{"type":"reasoning.encrypted","data":"b"}]` {
		t.Errorf("reasoning_details = %s", details)
	}
}

func TestStreamErrorChunk(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"content":"par"},"finish_reason":null}]}`,
		`{"error":{"code":502,"message":"upstream failed"}}`,
	)
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	ch, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evs := collect(t, ch)
	last := evs[len(evs)-1]
	if last.Type != ac.StreamEventError || last.Err == nil || !strings.Contains(last.Err.Error(), "upstream failed") {
		t.Errorf("last event = %+v", last)
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"rate limited"}}`))
	}))
	defer srv.Close()

	a, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL})
	if _, err := a.GenerateStream(context.Background(), []ac.Message{ac.UserMsg("hi")}, nil); err == nil || !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("err = %v", err)
	}
}

type meterFunc func()

func (f meterFunc) Observe(_ llmUsage) { f() }
```

`llmUsage` is a type alias to keep the fake meter short: add `type llmUsage = llm.Usage` and import `"github.com/guygrigsby/llm"` at the top of `stream_test.go`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/guygrigsby/projects/llm && go test ./openrouter/ -run TestStream -v 2>&1 | tail -20`
Expected: FAIL (the stub emits nothing, so `no done event` and friends).

- [ ] **Step 3: Write stream.go**

```go
package openrouter

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ac "github.com/voocel/agentcore"
)

// This file drives OpenRouter's OpenAI-format SSE stream, translating chunks
// into the agentcore start/delta/end framing the agent loop consumes. The
// terminal done event carries the fully assembled message; live deltas are
// for progress. OpenRouter sends comment lines (": OPENROUTER PROCESSING")
// as keepalives, usage in the last chunk, and mid-stream errors as a chunk
// with an error object.

// chatChunk is one SSE data frame from the streaming chat-completions API.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content          string            `json:"content"`
			Reasoning        string            `json:"reasoning"`
			ReasoningDetails []json.RawMessage `json:"reasoning_details"`
			ToolCalls        []streamToolCall  `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

// streamToolCall is a tool_call delta. Index identifies which call the
// fragment belongs to (id and name arrive on the first fragment; arguments
// stream across later ones).
type streamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// emitSSE reads the SSE body to completion, emitting agentcore stream events
// via an sseEmitter and reporting usage once through onUsage. A read error
// (other than clean EOF) or an error chunk becomes a terminal error event.
func emitSSE(r io.Reader, out chan<- ac.StreamEvent, onUsage func(*wireUsage)) {
	br := bufio.NewReader(r)
	em := &sseEmitter{out: out, builders: map[int]*callBuilder{}}
	for {
		line, err := br.ReadString('\n')
		if data, ok := sseData(line); ok {
			if data == "[DONE]" {
				break
			}
			var chunk chatChunk
			if json.Unmarshal([]byte(data), &chunk) == nil {
				if chunk.Error != nil {
					out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("openrouter: %d: %s", chunk.Error.Code, chunk.Error.Message)}
					return
				}
				em.handle(&chunk)
			}
		}
		if err != nil {
			if err != io.EOF {
				out <- ac.StreamEvent{Type: ac.StreamEventError, Err: fmt.Errorf("openrouter: stream: %w", err)}
				return
			}
			break
		}
	}
	em.flush(onUsage)
}

// sseData extracts the payload of a data: line, or ok=false for blank lines,
// comments and other fields.
func sseData(line string) (string, bool) {
	line = strings.TrimRight(line, "\r\n")
	if !strings.HasPrefix(line, "data:") {
		return "", false
	}
	return strings.TrimSpace(line[len("data:"):]), true
}

// sseEmitter accumulates streamed content and frames it as agentcore events.
type sseEmitter struct {
	out chan<- ac.StreamEvent

	nextIndex int

	thinkingOpen bool
	thinkingIdx  int
	thinking     strings.Builder
	details      []json.RawMessage

	textOpen bool
	textIdx  int
	text     strings.Builder

	builders map[int]*callBuilder
	order    []int

	finish string
	usage  *wireUsage
}

// callBuilder accumulates one streamed tool call.
type callBuilder struct {
	contentIdx int
	id         string
	name       string
	args       strings.Builder
	started    bool
}

// handle processes one chunk: reasoning first (thinking), then text, then tool
// calls, recording finish reason and usage when present. reasoning_details
// objects are collected verbatim for the echo on the final message; when a
// details object is a reasoning.text it also feeds the thinking block so a
// provider that sends details without a plaintext reasoning delta still shows
// thinking.
func (e *sseEmitter) handle(c *chatChunk) {
	if c.Usage != nil {
		e.usage = c.Usage
	}
	for _, ch := range c.Choices {
		if ch.FinishReason != "" {
			e.finish = ch.FinishReason
		}
		if ch.Delta.Reasoning != "" {
			e.onReasoning(ch.Delta.Reasoning)
		}
		for _, d := range ch.Delta.ReasoningDetails {
			e.details = append(e.details, d)
			if ch.Delta.Reasoning == "" {
				var td struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}
				if json.Unmarshal(d, &td) == nil && td.Type == "reasoning.text" && td.Text != "" {
					e.onReasoning(td.Text)
				}
			}
		}
		if ch.Delta.Content != "" {
			e.onContent(ch.Delta.Content)
		}
		for _, tc := range ch.Delta.ToolCalls {
			e.onToolCall(tc)
		}
	}
}

func (e *sseEmitter) onReasoning(s string) {
	if !e.thinkingOpen {
		e.thinkingOpen = true
		e.thinkingIdx = e.nextIndex
		e.nextIndex++
		e.send(ac.StreamEventThinkingStart, e.thinkingIdx)
	}
	e.thinking.WriteString(s)
	e.sendDelta(ac.StreamEventThinkingDelta, e.thinkingIdx, s)
}

func (e *sseEmitter) onContent(s string) {
	e.closeThinking()
	if !e.textOpen {
		e.textOpen = true
		e.textIdx = e.nextIndex
		e.nextIndex++
		e.send(ac.StreamEventTextStart, e.textIdx)
	}
	e.text.WriteString(s)
	e.sendDelta(ac.StreamEventTextDelta, e.textIdx, s)
}

func (e *sseEmitter) onToolCall(tc streamToolCall) {
	e.closeThinking()
	e.closeText()
	b := e.builders[tc.Index]
	if b == nil {
		b = &callBuilder{contentIdx: e.nextIndex}
		e.nextIndex++
		e.builders[tc.Index] = b
		e.order = append(e.order, tc.Index)
	}
	if tc.ID != "" {
		b.id = tc.ID
	}
	if tc.Function.Name != "" {
		b.name = tc.Function.Name
	}
	if !b.started {
		b.started = true
		e.send(ac.StreamEventToolCallStart, b.contentIdx)
	}
	if tc.Function.Arguments != "" {
		b.args.WriteString(tc.Function.Arguments)
		e.sendDelta(ac.StreamEventToolCallDelta, b.contentIdx, tc.Function.Arguments)
	}
}

func (e *sseEmitter) closeThinking() {
	if e.thinkingOpen {
		e.send(ac.StreamEventThinkingEnd, e.thinkingIdx)
		e.thinkingOpen = false
	}
}

func (e *sseEmitter) closeText() {
	if e.textOpen {
		e.send(ac.StreamEventTextEnd, e.textIdx)
		e.textOpen = false
	}
}

// flush closes open blocks, emits a tool-call-end per streamed call, reports
// usage, and sends the terminal done event carrying the assembled message.
func (e *sseEmitter) flush(onUsage func(*wireUsage)) {
	e.closeThinking()
	e.closeText()
	for _, idx := range e.order {
		b := e.builders[idx]
		call := ac.ToolCall{ID: b.id, Name: b.name, Args: rawArgs(b.args.String())}
		e.out <- ac.StreamEvent{
			Type:              ac.StreamEventToolCallEnd,
			ContentIndex:      b.contentIdx,
			Message:           e.snapshot(),
			CompletedToolCall: &call,
		}
	}
	if e.usage != nil && onUsage != nil {
		onUsage(e.usage)
	}
	final := assembleMessage(e.thinking.String(), e.text.String(), e.wireCalls(), e.finish, e.detailsJSON())
	final.Usage = toAgentUsage(e.usage)
	e.out <- ac.StreamEvent{Type: ac.StreamEventDone, Message: final, StopReason: final.StopReason}
}

// detailsJSON joins the collected reasoning_details objects into one array,
// or nil when none arrived.
func (e *sseEmitter) detailsJSON() json.RawMessage {
	if len(e.details) == 0 {
		return nil
	}
	b, _ := json.Marshal(e.details)
	return b
}

// snapshot builds the partial message from accumulated state.
func (e *sseEmitter) snapshot() ac.Message {
	return assembleMessage(e.thinking.String(), e.text.String(), e.wireCalls(), "", nil)
}

// wireCalls renders the accumulated tool-call builders as wire tool calls in
// first-seen order.
func (e *sseEmitter) wireCalls() []wireToolCall {
	if len(e.order) == 0 {
		return nil
	}
	calls := make([]wireToolCall, 0, len(e.order))
	for _, idx := range e.order {
		b := e.builders[idx]
		calls = append(calls, wireToolCall{
			ID:       b.id,
			Type:     "function",
			Function: wireFunc{Name: b.name, Arguments: b.args.String()},
		})
	}
	return calls
}

func (e *sseEmitter) send(t ac.StreamEventType, idx int) {
	e.out <- ac.StreamEvent{Type: t, ContentIndex: idx, Message: e.snapshot()}
}

func (e *sseEmitter) sendDelta(t ac.StreamEventType, idx int, delta string) {
	e.out <- ac.StreamEvent{Type: t, ContentIndex: idx, Delta: delta, Message: e.snapshot()}
}
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/guygrigsby/projects/llm && go test -race ./openrouter/ -v 2>&1 | tail -30`
Expected: every test PASS, including the Task 1 and 2 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/llm && go vet ./... && git add openrouter/stream.go openrouter/stream_test.go && git commit -m "openrouter: SSE streaming with reasoning details and error chunks"
```

---

### Task 4: Live e2e test behind the e2e tag

**Files:**
- Create: `openrouter/e2e_test.go`

- [ ] **Step 1: Write the test**

```go
//go:build e2e

// These tests hit the real OpenRouter API and cost a tiny amount of money.
// They are off by default; run them with:
//
//	OPENROUTER_API_KEY=... go test -tags e2e ./openrouter/ -v
//
// They skip when no key is present.
package openrouter

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

// e2eModel is a cheap tool-capable model; the adapter is model-agnostic.
const e2eModel = "moonshotai/kimi-k2.5"

func e2eAdapter(t *testing.T) *Adapter {
	t.Helper()
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("OPENROUTER_API_KEY not set; skipping live e2e test")
	}
	a, err := New(Config{APIKey: key, Model: e2eModel, Title: "llm e2e"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func userMsg(text string) []ac.Message {
	return []ac.Message{{Role: ac.RoleUser, Content: []ac.ContentBlock{ac.TextBlock(text)}}}
}

func TestE2E_Generate(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	resp, err := a.Generate(ctx, userMsg("Reply with the single word: pong"), nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Message.TextContent()), "pong") {
		t.Errorf("text = %q", resp.Message.TextContent())
	}
	if resp.Message.Usage == nil || resp.Message.Usage.Output == 0 {
		t.Errorf("usage = %+v", resp.Message.Usage)
	}
}

func TestE2E_GenerateStream(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	ch, err := a.GenerateStream(ctx, userMsg("Count from 1 to 5, digits separated by spaces."), nil)
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var deltas int
	var final *ac.Message
	for ev := range ch {
		switch ev.Type {
		case ac.StreamEventTextDelta:
			deltas++
		case ac.StreamEventDone:
			m := ev.Message
			final = &m
		case ac.StreamEventError:
			t.Fatalf("stream error: %v", ev.Err)
		}
	}
	if deltas == 0 || final == nil || !strings.Contains(final.TextContent(), "5") {
		t.Errorf("deltas = %d final = %+v", deltas, final)
	}
}

func TestE2E_ToolCall(t *testing.T) {
	a := e2eAdapter(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	tools := []ac.ToolSpec{{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"}},
	}}
	resp, err := a.Generate(ctx, userMsg("Use the get_weather tool to check the weather in Denver."), tools)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	calls := resp.Message.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "get_weather" || !strings.Contains(string(calls[0].Args), "Denver") {
		t.Errorf("calls = %+v", calls)
	}
}
```

- [ ] **Step 2: Compile it and run it if a key is at hand**

Run: `cd /Users/guygrigsby/projects/llm && go vet -tags e2e ./openrouter/ && (grep -q '^export OPENROUTER_API_KEY=.\+' ~/Library/Caches/op-secrets.env && (set -a; . ~/Library/Caches/op-secrets.env; set +a; go test -tags e2e ./openrouter/ -run TestE2E -v 2>&1 | tail -15) || echo "no key in op cache; e2e compiled only")`
Expected: vet clean; if the key is present all three PASS. Never print the key. If the run fails because the model id is unavailable, pick another id from `curl -s https://openrouter.ai/api/v1/models | jq -r '.data[].id' | grep moonshotai` and update `e2eModel`.

- [ ] **Step 3: Commit**

```bash
cd /Users/guygrigsby/projects/llm && git add openrouter/e2e_test.go && git commit -m "openrouter: live e2e tests behind the e2e tag"
```

---

### Task 5: README and changelog

**Files:**
- Modify: `README.md` (the `## Adapters` section, after `### kimi`)
- Modify: `CHANGELOG.md` if the repo has one (it does not today; skip if absent)

- [ ] **Step 1: Add the README section**

Insert before `## License`:

~~~markdown
### openrouter

`github.com/guygrigsby/llm/openrouter`, a native OpenRouter adapter over its
OpenAI-format chat-completions API, using only `net/http` (no SDK dependency).
One key, any OpenRouter model id per adapter (`moonshotai/kimi-k3`,
`anthropic/claude-opus-5`, ...). Full scope: SSE token streaming and tool
calling. OpenRouter extensions are wired: the `reasoning` request object
(effort mapped from the call's thinking level, the model's reasoning surfaced as
a thinking block, and `reasoning_details` echoed back on assistant messages so
the model keeps its chain of thought across tool calls), and usage accounting
with `cost` and cached prompt tokens reaching the `Meter`.

```go
m, err := openrouter.New(openrouter.Config{APIKey: key, Model: "moonshotai/kimi-k3", Title: "my-app"})
if err != nil {
	return err
}
agent := jess.New(jess.WithModel(m) /* , ... */)
```
~~~

Match the surrounding README's tone. Do not introduce em dashes even though older sections have them.

- [ ] **Step 2: Full gate**

Run: `cd /Users/guygrigsby/projects/llm && gofmt -l . && go vet ./... && go test -race ./... 2>&1 | tail -8`
Expected: gofmt prints nothing, vet clean, every package PASS.

- [ ] **Step 3: Commit and tag locally**

```bash
cd /Users/guygrigsby/projects/llm && git add README.md && git commit -m "openrouter: document the adapter" && git tag v0.4.0
```

The tag is local. Do not push.

---

---

### Task 6: One OpenAI-shaped wire package for kimi and openrouter

Added after the Task 1 review: `openrouter` duplicates `kimi` almost line for line (wire structs, message and tool conversion, stop-reason mapping, args helpers, the SSE emitter, `post` and `newRequest`). The repository rule is one factored thing, not two copies, so the shared shape moves into an internal package both adapters use. Behavior of both adapters is unchanged; their existing tests are the proof.

**Files:**
- Create: `internal/openaiwire/wire.go` (shared request and response structs with provider extension points)
- Create: `internal/openaiwire/convert.go` (messages, tools, stop reasons, args helpers)
- Create: `internal/openaiwire/sse.go` (the emitter)
- Create: `internal/openaiwire/client.go` (post, newRequest, error shaping)
- Test: `internal/openaiwire/convert_test.go`, `internal/openaiwire/sse_test.go` (moved from the adapters where they tested shared behavior)
- Modify: `kimi/*.go` and `openrouter/*.go` to call the shared package; delete what moved.

**Interfaces:**
- `openaiwire.Message`, `openaiwire.ToolCall`, `openaiwire.Tool`, `openaiwire.Func`, `openaiwire.ToolFunc`: the OpenAI-format wire structs, exported. Provider-specific fields stay in the adapters: kimi keeps `partial` and `name` by embedding `openaiwire.Message` in its own `wireMessage`; openrouter keeps `reasoning` and `reasoning_details` the same way. Both adapters' `chatRequest` embed `openaiwire.Request{Model, Messages, Tools, ToolChoice, Stream, MaxTokens, Temperature, TopP}` and add their own fields (`reasoning_effort`, `stream_options` for kimi; `reasoning` for openrouter). Because `Messages` must carry the provider's message type, `Request` is generic: `type Request[M any] struct { ...; Messages []M ... }`.
- `openaiwire.ConvertMessages[M any](msgs []ac.Message, assistant func(ac.Message) M, plain func(role, content, toolCallID string) M) []M`: the role switch, with the adapter supplying its two constructors.
- `openaiwire.ConvertTools(tools []ac.ToolSpec) []Tool`, `openaiwire.MapStopReason(string) ac.StopReason`, `openaiwire.ArgsString(json.RawMessage) string`, `openaiwire.RawArgs(string) json.RawMessage`, `openaiwire.AssembleMessage(reasoning, content string, calls []ToolCall, finish string) ac.Message` (the adapter adds metadata such as reasoning_details afterwards).
- `openaiwire.Emitter`: the SSE state machine with `Handle(delta Delta, finish string)` where `Delta{Content, Reasoning string; ToolCalls []StreamToolCall}` is what the adapter extracts from its own chunk type, `Flush(final func(reasoning, content string, calls []ToolCall, finish string) ac.Message)` for the terminal message, and `openaiwire.ReadSSE(r io.Reader, onData func(data string) (stop bool, err error)) error` for the line loop. Each adapter keeps its own chunk struct and decides what is an error chunk and what is usage.
- `openaiwire.Client{HTTP *http.Client; BaseURL, APIKey, Provider string; Headers map[string]string}` with `Post(ctx, path string, body any, apiKeyOverride string) ([]byte, error)` and `NewRequest(ctx, path string, body []byte, apiKeyOverride string) (*http.Request, error)`; errors are prefixed with the provider name.

- [ ] **Step 1: Move the shared tests first**

The behaviors the shared package must keep are already pinned by `kimi/kimi_test.go`, `kimi/stream_test.go`, `openrouter/openrouter_test.go` and `openrouter/stream_test.go`. Those stay where they are and keep passing; that is the regression net. Add `internal/openaiwire/convert_test.go` with direct tests of `ConvertTools` (skips empty names), `MapStopReason` (tool_calls, length, stop, empty, other), `ArgsString`/`RawArgs` (empty to `{}`), and `internal/openaiwire/sse_test.go` driving `Emitter` with a delta sequence (reasoning, text, two tool-call fragments) and asserting the event sequence and the flushed message.

Run: `go test ./internal/openaiwire/` and expect a build failure.

- [ ] **Step 2: Write the package by lifting the code out of openrouter**

Take `openrouter/convert.go` and `openrouter/stream.go` as the source (they are the newer copy), generalize as the Interfaces block says, and keep every doc comment that still applies. Then rewrite `openrouter` and `kimi` on top of it: each adapter file shrinks to its provider extensions (kimi: `reasoning_effort`, `partial`, `EstimateTokens`, `reasoning_content` on the chunk; openrouter: the `reasoning` object, `reasoning_details` collection and echo, `cost`, error chunks). No behavior changes: run each adapter's full test file after each rewrite.

- [ ] **Step 3: Gate**

Run: `gofmt -l . && go vet ./... && go test -race ./... 2>&1 | tail -8`
Expected: clean, every package PASS, including deepseek which is untouched.

- [ ] **Step 4: Commit**

```bash
git add internal/openaiwire kimi openrouter && git commit -m "llm: share the OpenAI wire shape between kimi and openrouter"
```

Move the `v0.4.0` tag from Task 5 to this commit (`git tag -f v0.4.0`); the tag is local and unpushed.

## Self-review

Spec coverage: ADR 0006 asks for a native OpenRouter adapter with chat-completions wire format, OpenRouter headers, reasoning passthrough and usage accounting, contributed upstream. Tasks 1 to 3 cover the wire format, headers, reasoning in both directions and usage; Task 4 proves it live; Task 5 documents it. Provider preferences (`provider` object) and `reasoning.max_tokens` are deliberately not wired: nothing in autophage needs them.

Type consistency: `wireUsage` nested detail structs are referenced the same way in `toAgentUsage`, `toLLMUsage` and the tests; `assembleMessage` takes five arguments everywhere (convert.go, stream.go snapshot and flush); `emitSSE` signature matches the stub and openrouter.go.
