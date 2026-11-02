package middleware

import (
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

// RateLimiter implements per-tenant rate limiting using a bit-packed atomic
// state variable for lock-free, zero-contention synchronization on the hot-path.
//
// Design: Each tenant gets an entry with a single uint64 state that packs the
// minute epoch (high 32 bits) and the request counter (low 32 bits). Window
// rotation and counter increment execute as a single atomic CAS transaction,
// eliminating the micro-gap race between separate atomic stores.
type RateLimiter struct {
	limiters   sync.Map // map[string]*tenantLimiter
	defaultRPM int
}

// tenantLimiter packs the minute epoch and request counter into a single
// uint64 so that window rotation and counter increment can be performed
// atomically via a single CompareAndSwapUint64.
//
//   High 32 bits: Minute epoch (unix seconds / 60)
//   Low 32 bits:  Request counter for the current minute
type tenantLimiter struct {
	state uint64
}

// NewRateLimiter creates a new rate limiter with the given default RPM.
func NewRateLimiter(defaultRPM int) *RateLimiter {
	return &RateLimiter{
		defaultRPM: defaultRPM,
	}
}

// Middleware returns a Gin middleware that enforces rate limits with
// zero-gap coordination via a bit-packed atomic CAS loop.
func (rl *RateLimiter) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tenantID, exists := c.Get("tenant_id")
		if !exists {
			c.Next()
			return
		}

		// Get tenant-specific rate limit (set by auth middleware)
		rpmLimit := rl.defaultRPM
		if limit, ok := c.Get("tenant_rate_limit"); ok {
			rpmLimit = limit.(int)
		}

		// No rate limit configured — allow all requests
		if rpmLimit <= 0 {
			c.Next()
			return
		}

		key := tenantID.(string)
		currentMinute := uint64(time.Now().Unix() / 60)
		limit := uint64(rpmLimit)

		// Get or create limiter for this tenant
		limiterVal, _ := rl.limiters.LoadOrStore(key, &tenantLimiter{})
		limiter := limiterVal.(*tenantLimiter)

		var count uint64

		// Lock-free atomic state transition loop.
		// The CAS loop guarantees that window rotation and counter increment
		// happen as a single atomic operation — no concurrent goroutine can
		// observe a stale high count from the previous window.
		for {
			oldState := atomic.LoadUint64(&limiter.state)
			oldMinute := oldState >> 32
			oldCount := oldState & 0xFFFFFFFF

			var newState uint64
			if oldMinute == currentMinute {
				// Same window — increment counter
				count = oldCount + 1
				if count > limit {
					// Hard quota exceeded — reject without mutating state
					now := time.Now()
					retryAfter := 60 - int(now.Unix()%60)
					if retryAfter <= 0 {
						retryAfter = 1
					}
					c.Header("Retry-After", fmt.Sprintf("%d", retryAfter))
					c.Header("X-RateLimit-Limit", fmt.Sprintf("%d", limit))
					c.Header("X-RateLimit-Remaining", "0")
					// Anthropic clients (/v1/messages) receive the nested error
					// envelope with type "rate_limit_error"; OpenAI-compatible
					// clients keep the flat {"error":..., "limit":..., "retry_after":...}.
					abortWithProtocolError(c, http.StatusTooManyRequests, "rate_limit_error", "rate limit exceeded",
						gin.H{
							"limit":       limit,
							"retry_after": retryAfter,
						})
					return
				}
				newState = (currentMinute << 32) | count
			} else {
				// Window boundary crossed — reset counter to 1
				count = 1
				newState = (currentMinute << 32) | count
			}

			// Attempt to commit the state transition. If another goroutine
			// modified state in the meantime, CAS fails and we retry.
			if atomic.CompareAndSwapUint64(&limiter.state, oldState, newState) {
				break
			}
		}

		// Set rate limit headers
		remaining := limit - count
		c.Header("X-RateLimit-Limit", fmt.Sprintf("%d", limit))
		c.Header("X-RateLimit-Remaining", fmt.Sprintf("%d", remaining))

		c.Next()
	}
}

// Cleanup removes expired limiters to prevent memory leaks.
// Should be called periodically (e.g., every 5 minutes).
func (rl *RateLimiter) Cleanup() {
	currentMinute := uint64(time.Now().Unix() / 60)

	rl.limiters.Range(func(key, value interface{}) bool {
		limiter := value.(*tenantLimiter)
		oldState := atomic.LoadUint64(&limiter.state)
		oldMinute := oldState >> 32

		// Delete limiters that haven't been used in the last 5 minutes
		if currentMinute-oldMinute > 5 {
			rl.limiters.Delete(key)
		}
		return true
	})
}
