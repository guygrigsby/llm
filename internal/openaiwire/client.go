package openaiwire

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Client is the shared authenticated JSON-over-HTTP client both adapters use
// to talk to their OpenAI-shaped chat-completions endpoint. Provider names
// the source in every error message; Headers carries any extra headers a
// provider always sends (OpenRouter's HTTP-Referer/X-Title attribution),
// applied to every request alongside the bearer token.
type Client struct {
	HTTP     *http.Client
	BaseURL  string
	APIKey   string
	Provider string
	Headers  map[string]string
}

// Post marshals body, sends it to path and returns the response bytes,
// failing on a non-200 with the response text. apiKeyOverride overrides the
// client's APIKey when non-empty (per-call key rotation).
func (c *Client) Post(ctx context.Context, path string, body any, apiKeyOverride string) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%s: encode: %w", c.Provider, err)
	}
	req, err := c.NewRequest(ctx, path, raw, apiKeyOverride)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: request: %w", c.Provider, err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s: %s", c.Provider, resp.Status, strings.TrimSpace(string(data)))
	}
	return data, nil
}

// NewRequest builds an authenticated JSON POST, with any configured extra
// headers set. apiKeyOverride overrides the client's APIKey when non-empty.
func (c *Client) NewRequest(ctx context.Context, path string, body []byte, apiKeyOverride string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	key := c.APIKey
	if apiKeyOverride != "" {
		key = apiKeyOverride
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.Headers {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
	return req, nil
}
