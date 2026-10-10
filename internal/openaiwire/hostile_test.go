package openaiwire

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"testing"

	ac "github.com/voocel/agentcore"
)

// These tests stand in for a hostile endpoint: a user-supplied URL can send
// anything, and one tenant must not be able to exhaust the shared process.

func TestReadSSERejectsOverlongLine(t *testing.T) {
	r := strings.NewReader("data: " + strings.Repeat("a", maxSSELine+1))
	_, err := ReadSSE(r, func(string) (bool, error) { return false, nil })
	if err == nil {
		t.Fatal("want an error for a line past the cap")
	}
}

func TestStreamBytesCapped(t *testing.T) {
	r := newCapReader(io.NopCloser(strings.NewReader(strings.Repeat("x", 100))), 50)
	_, err := io.ReadAll(r)
	if !errors.Is(err, errStreamTooLarge) {
		t.Fatalf("err = %v, want errStreamTooLarge", err)
	}
}

// drain sends every event to a goroutine that discards it, so the emitter
// never blocks.
func drain() (chan ac.StreamEvent, func() []ac.StreamEvent) {
	out := make(chan ac.StreamEvent, 16)
	var evs []ac.StreamEvent
	done := make(chan struct{})
	go func() {
		for ev := range out {
			if ev.Type == ac.StreamEventError {
				evs = append(evs, ev)
			}
		}
		close(done)
	}()
	return out, func() []ac.StreamEvent { close(out); <-done; return evs }
}

// TestEmitterLinearInArgs proves streaming one tool call's arguments in many
// small deltas costs memory proportional to the arguments, not to their
// square: each live snapshot used to copy everything accumulated so far.
func TestEmitterLinearInArgs(t *testing.T) {
	out, wait := drain()
	em := NewEmitter(context.Background(), out)
	piece := strings.Repeat("a", 100)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	for range 20000 { // 2 MB of arguments
		em.Handle(Delta{ToolCalls: []StreamToolCall{{Index: 0, Function: Func{Arguments: piece}}}}, "")
	}
	runtime.ReadMemStats(&after)
	wait()
	if got := after.TotalAlloc - before.TotalAlloc; got > 200<<20 {
		t.Errorf("allocated %d MB for 2 MB of arguments", got>>20)
	}
}

func TestEmitterCapsToolCalls(t *testing.T) {
	out, wait := drain()
	em := NewEmitter(context.Background(), out)
	for i := range maxToolCalls + 1 {
		em.Handle(Delta{ToolCalls: []StreamToolCall{{Index: i, ID: strconv.Itoa(i), Function: Func{Name: "f"}}}}, "")
	}
	errs := wait()
	if !em.Aborted() || len(errs) != 1 {
		t.Errorf("aborted %v, %d error events; want the stream refused once", em.Aborted(), len(errs))
	}
}

func TestErrorBodyBoundedAndClean(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("\x1b[31mfake\nlog line " + strings.Repeat("z", 5<<20)))
	}))
	defer srv.Close()

	c := testClient(srv.URL)
	_, err := c.Post(context.Background(), "/x", map[string]string{}, "")
	if err == nil {
		t.Fatal("want error")
	}
	if s := err.Error(); len(s) > 1024 || strings.ContainsAny(s, "\x1b\n") {
		t.Errorf("error of %d bytes, control chars %v", len(s), strings.ContainsAny(s, "\x1b\n"))
	}
}

func TestSanitize(t *testing.T) {
	if got := Sanitize("a\nb\x1b[0mc", 100); got != "a b[0mc" {
		t.Errorf("got %q", got)
	}
	if got := Sanitize(strings.Repeat("é", 10), 5); got != "éé..." {
		t.Errorf("got %q", got)
	}
}
