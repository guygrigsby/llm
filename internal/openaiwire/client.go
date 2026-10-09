package openaiwire

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	ac "github.com/voocel/agentcore"
)

// Client is the shared authenticated JSON-over-HTTP client both adapters use
// to talk to their OpenAI-shaped chat-completions endpoint. Provider names
// the source in every error message; Headers carries any extra headers a
// provider always sends (OpenRouter's HTTP-Referer/X-Title attribution),
// applied to every request alongside the bearer token.
//
// Every request retries the transient statuses (429 and 5xx) so a coding
// agent's long run is not ended by one throttled call, and every failure is
// classified: a provider message about the context window comes back as
// agentcore's ContextOverflowError so the agent loop compacts rather than
// aborts.
type Client struct {
	HTTP     *http.Client
	BaseURL  string
	APIKey   string
	Provider string
	Headers  map[string]string
	// StreamHTTP sends streaming requests. It carries no overall timeout: a
	// long generation is legitimate, and a stalled one is caught by StreamIdle
	// instead. Falls back to HTTP when nil.
	StreamHTTP *http.Client
	// StreamIdle aborts a stream whose body has produced nothing for this
	// long. Zero disables the watchdog.
	StreamIdle time.Duration
}

// maxAttempts caps how many times one request is sent, first try included.
const maxAttempts = 3

// maxRetryWait caps a single backoff wait, however long the provider's
// Retry-After asks for: an agent that sleeps for minutes has stalled.
const maxRetryWait = 60 * time.Second

// retryBase is the first backoff wait; each further attempt doubles it.
const retryBase = 500 * time.Millisecond

// contextOverflowMarkers are the substrings OpenAI-shaped providers use to
// report a request larger than the model's context window. Matched
// case-insensitively against the error body.
var contextOverflowMarkers = []string{"context length", "context_length", "maximum context", "too many tokens"}

// Post marshals body, sends it to path and returns the response bytes,
// failing on a non-200 with the response text. apiKeyOverride overrides the
// client's APIKey when non-empty (per-call key rotation).
func (c *Client) Post(ctx context.Context, path string, body any, apiKeyOverride string) ([]byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%s: encode: %w", c.Provider, err)
	}
	resp, err := c.send(ctx, c.HTTP, path, raw, apiKeyOverride, "")
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	return data, nil
}

// Stream sends body to path as an SSE request and returns the live body for
// the caller to read to completion and close. Retries apply to the initial
// status only: once a byte of the body has been consumed the turn is under
// way, and resending it would duplicate whatever the model already emitted.
// The returned reader enforces StreamIdle, so a body that goes quiet fails
// with an error wrapping agentcore.ErrProviderStreamIdle.
func (c *Client) Stream(ctx context.Context, path string, body []byte, apiKeyOverride string) (io.ReadCloser, error) {
	httpc := c.StreamHTTP
	if httpc == nil {
		httpc = c.HTTP
	}
	resp, err := c.send(ctx, httpc, path, body, apiKeyOverride, "text/event-stream")
	if err != nil {
		return nil, err
	}
	return newIdleReader(resp.Body, c.StreamIdle), nil
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
	// An endpoint with no key (a gateway on a private network) gets no
	// Authorization header at all, rather than an empty bearer that some
	// strict frontends reject as malformed.
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.Headers {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
	return req, nil
}

// send performs the request over httpc, retrying a throttled or failed
// upstream (429, 5xx) with exponential backoff and honoring Retry-After. It
// returns a 200 response whose body the caller owns; every other outcome is
// an error, classified as a context overflow when the body says so.
func (c *Client) send(ctx context.Context, httpc *http.Client, path string, body []byte, apiKeyOverride, accept string) (*http.Response, error) {
	var lastErr error
	for attempt := range maxAttempts {
		req, err := c.NewRequest(ctx, path, body, apiKeyOverride)
		if err != nil {
			return nil, err
		}
		if accept != "" {
			req.Header.Set("Accept", accept)
		}
		resp, err := httpc.Do(req)
		if err != nil {
			return nil, fmt.Errorf("%s: request: %w", c.Provider, err)
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		_ = resp.Body.Close()
		lastErr = classifyError(fmt.Errorf("%s: %s: %s", c.Provider, resp.Status, strings.TrimSpace(string(msg))), string(msg))
		if !retryableStatus(resp.StatusCode) || attempt == maxAttempts-1 {
			return nil, lastErr
		}
		if err := wait(ctx, backoff(attempt, resp.Header.Get("Retry-After"))); err != nil {
			return nil, fmt.Errorf("%s: %w", c.Provider, err)
		}
	}
	return nil, lastErr
}

// retryableStatus reports whether a status is worth resending: the provider
// is throttling us or its upstream broke, neither of which the request itself
// can fix.
func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}

// backoff returns how long to wait before the next attempt: the server's
// Retry-After when it sent a usable one, otherwise an exponential ramp from
// retryBase. Capped at maxRetryWait either way.
func backoff(attempt int, retryAfter string) time.Duration {
	if d, ok := parseRetryAfter(retryAfter, time.Now()); ok {
		return min(d, maxRetryWait)
	}
	return min(retryBase<<attempt, maxRetryWait)
}

// parseRetryAfter reads both Retry-After forms: a delay in seconds and an
// HTTP-date to wait until. A past date or unparsable value reports ok=false
// so the caller falls back to its own ramp.
func parseRetryAfter(v string, now time.Time) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0, false
		}
		return time.Duration(secs) * time.Second, true
	}
	if when, err := http.ParseTime(v); err == nil {
		if d := when.Sub(now); d > 0 {
			return d, true
		}
		return 0, true
	}
	return 0, false
}

// wait sleeps for d, or returns the context's error if the caller gave up
// first.
func wait(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// classifyError wraps err as agentcore's ContextOverflowError when the
// provider's message says the request exceeded the model's context window, so
// agentcore.IsContextOverflow holds and the agent loop compacts instead of
// failing the run.
func classifyError(err error, body string) error {
	low := strings.ToLower(body)
	for _, marker := range contextOverflowMarkers {
		if strings.Contains(low, marker) {
			return &ac.ContextOverflowError{Cause: err}
		}
	}
	return err
}

// idleReader aborts a stream whose body has gone quiet. Every Read re-arms a
// timer; when the window passes with no data the underlying body is closed,
// which unblocks the pending Read, and the failure surfaces as an error
// wrapping agentcore.ErrProviderStreamIdle rather than a hang until the
// request timeout.
type idleReader struct {
	rc      io.ReadCloser
	idle    time.Duration
	timer   *time.Timer
	tripped atomic.Bool
}

// newIdleReader wraps rc with an idle watchdog, or returns rc unchanged when
// no idle window is configured.
func newIdleReader(rc io.ReadCloser, idle time.Duration) io.ReadCloser {
	if idle <= 0 {
		return rc
	}
	r := &idleReader{rc: rc, idle: idle}
	r.timer = time.AfterFunc(idle, func() {
		r.tripped.Store(true)
		_ = rc.Close()
	})
	return r
}

func (r *idleReader) Read(p []byte) (int, error) {
	if r.tripped.Load() {
		return 0, r.idleErr()
	}
	r.timer.Reset(r.idle)
	n, err := r.rc.Read(p)
	if err != nil && r.tripped.Load() {
		return n, r.idleErr()
	}
	return n, err
}

func (r *idleReader) Close() error {
	r.timer.Stop()
	return r.rc.Close()
}

func (r *idleReader) idleErr() error {
	return fmt.Errorf("stream idle timeout after %s: %w", r.idle, ac.ErrProviderStreamIdle)
}
