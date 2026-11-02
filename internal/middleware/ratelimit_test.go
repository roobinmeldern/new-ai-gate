package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestRateLimiter_AllowsUnderLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(100)

	allowed := 0
	for i := 0; i < 50; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-1")
		c.Set("tenant_rate_limit", 100)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

		rl.Middleware()(c)
		if !c.IsAborted() {
			allowed++
		}
	}

	if allowed != 50 {
		t.Errorf("expected 50 allowed, got %d", allowed)
	}
}

func TestRateLimiter_BlocksOverLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(5)

	blocked := 0
	for i := 0; i < 20; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-block")
		c.Set("tenant_rate_limit", 5)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

		rl.Middleware()(c)
		if c.IsAborted() {
			blocked++
		}
	}

	if blocked != 15 {
		t.Errorf("expected 15 blocked, got %d", blocked)
	}
}

func TestRateLimiter_ZeroRPMSkipsRateLimiting(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(0)

	allowed := 0
	for i := 0; i < 100; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-unlimited")
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

		rl.Middleware()(c)
		if !c.IsAborted() {
			allowed++
		}
	}

	if allowed != 100 {
		t.Errorf("expected 100 allowed with zero RPM (no limit), got %d", allowed)
	}
}

func TestRateLimiter_BitPackedConcurrentWindowRotation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(10000)

	const goroutines = 200
	const requestsPerGoroutine = 50

	var wg sync.WaitGroup
	var allowed int64
	var mu sync.Mutex

	// Pre-create the limiter entry so all goroutines hit the same window boundary
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Set("tenant_id", "tenant-cas")
	c.Set("tenant_rate_limit", 10000)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rl.Middleware()(c)

	// Force the window to be expired by backdating the minute epoch.
	// Pack an old minute (2 minutes ago) into the high 32 bits, with count=0
	// in the low 32 bits. The concurrent goroutines will all see this stale
	// minute and race to rotate the window via CAS.
	limiterVal, _ := rl.limiters.Load("tenant-cas")
	limiter := limiterVal.(*tenantLimiter)
	oldMinute := uint64(time.Now().Add(-2*time.Minute).Unix() / 60)
	atomic.StoreUint64(&limiter.state, oldMinute<<32)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				w := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(w)
				c.Set("tenant_id", "tenant-cas")
				c.Set("tenant_rate_limit", 10000)
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

				rl.Middleware()(c)
				if !c.IsAborted() {
					mu.Lock()
					allowed++
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()

	totalRequests := goroutines * requestsPerGoroutine

	// With the bit-packed CAS approach, the window rotation and counter
	// increment happen as a single atomic transaction. No goroutine can
	// observe a stale high count from the previous window. All requests
	// should be allowed since the limit (10000) >= total requests.
	if allowed != int64(totalRequests) {
		t.Errorf("expected %d allowed, got %d (false 429 from micro-gap race)", totalRequests, allowed)
	}

	// Verify the counter is exactly totalRequests — no double-reset, no lost increments
	finalState := atomic.LoadUint64(&limiter.state)
	finalCount := finalState & 0xFFFFFFFF
	if finalCount != uint64(totalRequests) {
		t.Errorf("expected counter %d, got %d (window was reset by multiple goroutines)", totalRequests, finalCount)
	}
}

func TestRateLimiter_RetryAfterHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(2)

	// Make 2 allowed requests (at the limit)
	for i := 0; i < 2; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-retry")
		c.Set("tenant_rate_limit", 2)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		rl.Middleware()(c)
	}

	// 3rd request should be blocked with a Retry-After header
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Set("tenant_id", "tenant-retry")
	c.Set("tenant_rate_limit", 2)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rl.Middleware()(c)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("expected status 429, got %d", w.Code)
	}
	if retryAfter := w.Header().Get("Retry-After"); retryAfter == "" {
		t.Error("expected non-empty Retry-After header")
	}
	if remaining := w.Header().Get("X-RateLimit-Remaining"); remaining != "0" {
		t.Errorf("expected X-RateLimit-Remaining '0', got %q", remaining)
	}
}

// TestRateLimiter_AnthropicRateLimitEnvelope verifies that 429 responses for
// Anthropic requests (/v1/messages) use the nested rate_limit_error envelope
// while preserving the Retry-After and X-RateLimit-* headers.
func TestRateLimiter_AnthropicRateLimitEnvelope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(2)

	request := func() *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-anthropic-rl")
		c.Set("tenant_rate_limit", 2)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
		rl.Middleware()(c)
		return w
	}

	// Two allowed requests reach the limit.
	for i := 0; i < 2; i++ {
		if w := request(); w.Code != http.StatusOK {
			t.Fatalf("request %d: expected status 200, got %d", i+1, w.Code)
		}
	}

	// 3rd request is blocked with the Anthropic envelope.
	w := request()
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("expected status 429, got %d", w.Code)
	}

	body := w.Body.String()
	for _, want := range []string{
		`"type":"error"`,
		`"type":"rate_limit_error"`,
		`"message":"rate limit exceeded"`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected Anthropic envelope containing %s, got %s", want, body)
		}
	}
	if retryAfter := w.Header().Get("Retry-After"); retryAfter == "" {
		t.Error("expected non-empty Retry-After header on Anthropic 429")
	}
	if remaining := w.Header().Get("X-RateLimit-Remaining"); remaining != "0" {
		t.Errorf("expected X-RateLimit-Remaining '0', got %q", remaining)
	}
}

// TestRateLimiter_Flat429BodyPreservedOnOpenAIPath verifies that OpenAI
// endpoints keep the original flat 429 body {"error","limit","retry_after"}
// — no Anthropic envelope may leak into the existing protocol.
func TestRateLimiter_Flat429BodyPreservedOnOpenAIPath(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rl := NewRateLimiter(2)

	var blockedBody string
	var blockedCode int
	for i := 0; i < 3; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Set("tenant_id", "tenant-flat-rl")
		c.Set("tenant_rate_limit", 2)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		rl.Middleware()(c)

		if i < 2 {
			if w.Code != http.StatusOK {
				t.Fatalf("request %d: expected status 200, got %d", i+1, w.Code)
			}
			continue
		}
		blockedCode = w.Code
		blockedBody = w.Body.String()
	}

	if blockedCode != http.StatusTooManyRequests {
		t.Fatalf("expected status 429, got %d", blockedCode)
	}
	for _, want := range []string{
		`"error":"rate limit exceeded"`,
		`"limit":2`,
		`"retry_after":`,
	} {
		if !strings.Contains(blockedBody, want) {
			t.Fatalf("expected flat 429 body to contain %s, got %s", want, blockedBody)
		}
	}
	if strings.Contains(blockedBody, `"type":"error"`) || strings.Contains(blockedBody, "rate_limit_error") {
		t.Fatalf("OpenAI path must keep the flat error format, got %s", blockedBody)
	}
}
