// Package deepseek is a native DeepSeek model adapter: an agentcore.ChatModel
// (llm.LLM) backed by DeepSeek's chat-completions API. DeepSeek's own API is
// OpenAI-shaped, so a plain HTTP client IS native here, not a compatibility
// shim, and it needs no heavyweight SDK.
//
// Scope: built for cheap one-shot work (summaries, extraction). Tool-calling is
// not advertised, and GenerateStream returns the full result as one terminal
// event (the Once pattern) rather than true token streaming. Wire real streaming
// and tools before using DeepSeek as a primary conversational model.
package deepseek

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

const defaultBaseURL = "https://api.deepseek.com"

// Config configures the adapter. APIKey and Model are required; BaseURL defaults
// to DeepSeek's public endpoint.
type Config struct {
	APIKey  string
	Model   string // e.g. "deepseek-chat"
	BaseURL string
}

// Adapter wraps DeepSeek's chat-completions API as an llm.LLM.
type Adapter struct {
	apiKey  string
	model   string
	baseURL string
	http    *http.Client
}

// compile-time check: Adapter satisfies the llm.LLM port (agentcore.ChatModel).
var _ llm.LLM = (*Adapter)(nil)

// New builds an Adapter from config.
func New(cfg Config) (*Adapter, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("deepseek: APIKey is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("deepseek: Model is required")
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	return &Adapter{
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		baseURL: strings.TrimRight(base, "/"),
		http:    &http.Client{Timeout: 60 * time.Second},
	}, nil
}

// SupportsTools reports false: this adapter does not wire tool-calling yet.
func (a *Adapter) SupportsTools() bool { return false }

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Generate runs a one-shot chat completion. Messages are flattened to their text
// content (this adapter is text-only for now); tools are ignored.
func (a *Adapter) Generate(ctx context.Context, msgs []ac.Message, _ []ac.ToolSpec, _ ...ac.CallOption) (*ac.LLMResponse, error) {
	reqBody := chatRequest{Model: a.model}
	for _, m := range msgs {
		reqBody.Messages = append(reqBody.Messages, chatMessage{Role: roleOf(m), Content: m.TextContent()})
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("deepseek: encode: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+a.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("deepseek: %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("deepseek: decode: %w", err)
	}
	if cr.Error != nil {
		return nil, fmt.Errorf("deepseek: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("deepseek: response had no choices")
	}
	return &ac.LLMResponse{Message: ac.Message{
		Role:       ac.RoleAssistant,
		Content:    []ac.ContentBlock{ac.TextBlock(cr.Choices[0].Message.Content)},
		StopReason: ac.StopReasonStop,
	}}, nil
}

// GenerateStream synthesizes a degenerate stream from the one-shot result: one
// terminal StreamEventDone with the full message (the Once pattern). Enough for
// the agent loop to advance; add real token streaming before conversational use.
func (a *Adapter) GenerateStream(ctx context.Context, msgs []ac.Message, tools []ac.ToolSpec, opts ...ac.CallOption) (<-chan ac.StreamEvent, error) {
	resp, err := a.Generate(ctx, msgs, tools, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan ac.StreamEvent, 1)
	ch <- ac.StreamEvent{Type: ac.StreamEventDone, Message: resp.Message, StopReason: resp.Message.StopReason}
	close(ch)
	return ch, nil
}

func roleOf(m ac.Message) string {
	switch m.GetRole() {
	case ac.RoleAssistant:
		return "assistant"
	case ac.RoleSystem:
		return "system"
	default:
		return "user"
	}
}
