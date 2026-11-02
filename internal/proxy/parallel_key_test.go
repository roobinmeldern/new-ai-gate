package proxy

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
)

// TestParallelBatch_HedgingRace verifies that in a multi-key pool, multiple keys
// are evaluated concurrently. If earlier keys return 402/depleted balance, the gateway
// does NOT prematurely abort the provider; the first successful key claims victory,
// cancels slower in-flight workers, and returns the response directly.
func TestParallelBatch_HedgingRace(t *testing.T) {
	setRetryRoundSettings(t, 1, 100*time.Millisecond)

	creds := []*credentials.RuntimeCredential{
		newRoundTestCred(1, "cleanapis", "key-depleted-1", "https://api.cleanapis.com/v1"),
		newRoundTestCred(2, "cleanapis", "key-depleted-2", "https://api.cleanapis.com/v1"),
		newRoundTestCred(3, "cleanapis", "key-success-3", "https://api.cleanapis.com/v1"),
		newRoundTestCred(4, "cleanapis", "key-slow-4", "https://api.cleanapis.com/v1"),
	}

	cleanPool := credentials.NewBalancedPool("claude-3-7-sonnet", "round-robin", creds, nil)

	var cred1Called, cred2Called, cred3Called atomic.Bool
	var cred4Canceled atomic.Bool

	roundTrip := func(req *http.Request) (*http.Response, error) {
		bodyBytes, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		auth := req.Header.Get("Authorization")

		switch {
		case strings.Contains(auth, "key-depleted-1"):
			cred1Called.Store(true)
			return roundTestJSONResponse(http.StatusPaymentRequired,
				`{"error": {"message": "insufficient_balance", "type": "payment_required"}}`), nil

		case strings.Contains(auth, "key-depleted-2"):
			cred2Called.Store(true)
			return roundTestJSONResponse(http.StatusPaymentRequired,
				`{"error": {"message": "insufficient_balance", "type": "payment_required"}}`), nil

		case strings.Contains(auth, "key-success-3"):
			cred3Called.Store(true)
			// Fast success
			time.Sleep(30 * time.Millisecond)
			return roundTestJSONResponse(http.StatusOK,
				`{"id":"ok-3","choices":[{"message":{"role":"assistant","content":"cleanapi key 3 success"}}]}`), nil

		case strings.Contains(auth, "key-slow-4"):
			// Slower worker — should be canceled by key 3 winning
			select {
			case <-time.After(500 * time.Millisecond):
				return roundTestJSONResponse(http.StatusOK,
					`{"id":"ok-4","choices":[{"message":{"role":"assistant","content":"slow key 4"}}]}`), nil
			case <-req.Context().Done():
				cred4Canceled.Store(true)
				return nil, req.Context().Err()
			}
		}

		return roundTestJSONResponse(http.StatusInternalServerError, `{"error":"unexpected"}`), nil
	}

	w := serveRetryRoundsChatModel(t, "claude-3-7-sonnet", roundTrip, cleanPool)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 from winning key 3, got %d: %s", w.Code, w.Body.String())
	}

	if !strings.Contains(w.Body.String(), "cleanapi key 3 success") {
		t.Fatalf("expected response from key 3, got: %s", w.Body.String())
	}

	if !cred1Called.Load() || !cred2Called.Load() || !cred3Called.Load() {
		t.Errorf("expected parallel wave to evaluate keys 1, 2, and 3; called: 1=%v, 2=%v, 3=%v",
			cred1Called.Load(), cred2Called.Load(), cred3Called.Load())
	}
}

// TestParallelBatch_AllProviderKeysFail_ThenFallsBack verifies that if ALL keys
// in a provider fail (e.g. 4 keys all depleted), the gateway does NOT stop after 2 keys,
// tests all keys, and then cleanly falls back to the exact same model on another provider.
func TestParallelBatch_AllProviderKeysFail_ThenFallsBack(t *testing.T) {
	setRetryRoundSettings(t, 1, 50*time.Millisecond)

	cleanCreds := []*credentials.RuntimeCredential{
		newRoundTestCred(1, "cleanapis", "clean-key-1", "https://api.cleanapis.com/v1"),
		newRoundTestCred(2, "cleanapis", "clean-key-2", "https://api.cleanapis.com/v1"),
		newRoundTestCred(3, "cleanapis", "clean-key-3", "https://api.cleanapis.com/v1"),
		newRoundTestCred(4, "cleanapis", "clean-key-4", "https://api.cleanapis.com/v1"),
	}
	routerCreds := []*credentials.RuntimeCredential{
		newRoundTestCred(10, "routerai", "router-key-1", "https://api.routerai.com/v1"),
	}
	routerPool := credentials.NewBalancedPool("routerai/claude-opus-5.5", "round-robin", routerCreds, nil)
	cleanPool := credentials.NewBalancedPool("cleanapis/claude-opus-5.5", "round-robin", cleanCreds, routerPool)

	var cleanKeysHit int32
	var routerHit atomic.Bool

	roundTrip := func(req *http.Request) (*http.Response, error) {
		bodyBytes, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		auth := req.Header.Get("Authorization")

		if strings.Contains(auth, "clean-key") {
			atomic.AddInt32(&cleanKeysHit, 1)
			return roundTestJSONResponse(http.StatusPaymentRequired,
				`{"error": {"message": "not_enough_balance", "type": "payment_required"}}`), nil
		}

		if strings.Contains(auth, "router-key-1") {
			routerHit.Store(true)
			return roundTestJSONResponse(http.StatusOK,
				`{"id":"ok-router","choices":[{"message":{"role":"assistant","content":"routerai answered opus 5.5"}}]}`), nil
		}

		return roundTestJSONResponse(http.StatusInternalServerError, `{"error":"unexpected host"}`), nil
	}

	w := serveRetryRoundsChatModel(t, "cleanapis/claude-opus-5.5", roundTrip, cleanPool, routerPool)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 from fallback provider routerai, got %d: %s", w.Code, w.Body.String())
	}

	if !strings.Contains(w.Body.String(), "routerai answered opus 5.5") {
		t.Fatalf("expected response from routerai, got: %s", w.Body.String())
	}

	// Verify that ALL 4 cleanapis keys were evaluated before falling back
	hitCount := atomic.LoadInt32(&cleanKeysHit)
	if hitCount != 4 {
		t.Errorf("expected all 4 cleanapis keys to be tested before switching provider, got %d", hitCount)
	}

	if !routerHit.Load() {
		t.Errorf("expected routerai fallback to be invoked")
	}
}
