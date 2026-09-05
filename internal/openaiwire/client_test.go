package openaiwire

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ac "github.com/voocel/agentcore"
)

// testClient builds a Client pointed at srv with the shared defaults.
func testClient(url string) *Client {
	return &Client{
		HTTP:     &http.Client{Timeout: 30 * time.Second},
		BaseURL:  url,
		APIKey:   "k",
		Provider: "testprov",
	}
}

// TestPostRetriesRateLimit proves a 429 is retried after the Retry-After wait
// and the second attempt's body is returned.
func TestPostRetriesRateLimit(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	start := time.Now()
	data, err := testClient(srv.URL).Post(context.Background(), "/chat/completions", map[string]any{"a": 1}, "")
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	if string(data) != `{"ok":true}` {
		t.Errorf("body = %s", data)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("attempts = %d, want 2", got)
	}
	if elapsed := time.Since(start); elapsed < time.Second {
		t.Errorf("elapsed = %v, want at least the Retry-After second", elapsed)
	}
}

// TestPostRetriesServerError proves 5xx responses are retried up to the
// attempt cap and a later success is returned.
func TestPostRetriesServerError(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"message":"overloaded"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	data, err := testClient(srv.URL).Post(context.Background(), "/chat/completions", map[string]any{"a": 1}, "")
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	if string(data) != `{"ok":true}` {
		t.Errorf("body = %s", data)
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("attempts = %d, want 3", got)
	}
}

// TestPostDoesNotRetryClientError proves a 400 is terminal: retrying a request
// the provider rejected only burns quota.
func TestPostDoesNotRetryClientError(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"model not found"}}`))
	}))
	defer srv.Close()

	_, err := testClient(srv.URL).Post(context.Background(), "/chat/completions", map[string]any{"a": 1}, "")
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("err = %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("attempts = %d, want 1", got)
	}
}

// TestPostContextOverflow proves a provider overflow message is classified as
// agentcore's ContextOverflowError, so the agent loop compacts instead of
// failing the run.
func TestPostContextOverflow(t *testing.T) {
	for _, msg := range []string{
		"This model's maximum context length is 200000 tokens",
		"context_length_exceeded",
		"Input is too many tokens for this model",
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"` + msg + `"}}`))
		}))
		_, err := testClient(srv.URL).Post(context.Background(), "/chat/completions", map[string]any{"a": 1}, "")
		srv.Close()
		if err == nil {
			t.Fatalf("%q: expected an error", msg)
		}
		if !ac.IsContextOverflow(err) {
			t.Errorf("%q: not classified as context overflow: %v", msg, err)
		}
	}
}
