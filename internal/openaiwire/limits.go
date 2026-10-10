package openaiwire

import (
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Limits on what an endpoint can make this process hold or echo. A
// user-supplied endpoint can send anything, and one tenant must not be able
// to exhaust a host shared with others.
const (
	// maxSSELine bounds one SSE line. Real chunks are a few KB.
	maxSSELine = 1 << 20
	// maxStreamBytes bounds a whole streamed response body.
	maxStreamBytes = 32 << 20
	// maxToolCalls bounds the distinct tool calls in one streamed turn.
	maxToolCalls = 128
	// maxErrorBody is how much of a failed response is read for its message.
	maxErrorBody = 4 << 10
	// MaxErrorText bounds endpoint-chosen text echoed into an error, which
	// may reach logs or clients.
	MaxErrorText = 512
)

var errStreamTooLarge = fmt.Errorf("stream exceeds %d bytes", maxStreamBytes)

// capReader fails a body once it has delivered more than its limit.
type capReader struct {
	io.ReadCloser
	left int64
}

func newCapReader(rc io.ReadCloser, n int64) io.ReadCloser {
	return &capReader{ReadCloser: rc, left: n}
}

func (r *capReader) Read(p []byte) (int, error) {
	if r.left <= 0 {
		return 0, errStreamTooLarge
	}
	if int64(len(p)) > r.left {
		p = p[:r.left]
	}
	n, err := r.ReadCloser.Read(p)
	r.left -= int64(n)
	return n, err
}

// Sanitize makes endpoint-chosen text safe to put in an error: newlines and
// tabs become spaces, other control characters (terminal escapes) are
// dropped, and the result is cut to max bytes on a rune boundary.
func Sanitize(s string, max int) string {
	s = strings.Map(func(r rune) rune {
		switch {
		case r == '\n' || r == '\r' || r == '\t':
			return ' '
		case unicode.IsControl(r):
			return -1
		}
		return r
	}, s)
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "..."
}
