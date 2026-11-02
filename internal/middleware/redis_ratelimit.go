package middleware

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// slidingWindowLua implements a precise per-tenant sliding window rate limiter
// entirely inside Redis using an atomic Lua script. This ensures exact enforcement
// across all horizontal gateway nodes without any inter-node coordination overhead.
//
// The script operates on a Redis Sorted Set where each request is stored as a
// member with its timestamp as the score. Old entries outside the window are
// pruned atomically before checking the count.
//
// KEYS[1] = rate limit key (e.g. "rl:tenant123")
// ARGV[1] = window duration in milliseconds
// ARGV[2] = max requests allowed in window
// ARGV[3] = current timestamp in milliseconds
// Returns: {allowed (0|1), current_count}
var slidingWindowLua = redis.NewScript(`
local key      = KEYS[1]
local window   = tonumber(ARGV[1])
local limit    = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cutoff   = now - window

-- Remove timestamps older than the window
redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)

-- Count current requests in window
local count = redis.call('ZCARD', key)

if count >= limit then
    return {0, count}
end

-- Record this request with a unique member (timestamp + sequence counter)
local seq_key = key .. ':seq'
local member = now .. '-' .. redis.call('INCR', seq_key)
redis.call('ZADD', key, now, member)

-- Set TTL on BOTH keys — :seq left without TTL causes a permanent memory leak
-- for every unique tenant that ever hits the gateway.
redis.call('PEXPIRE', key, window + 1000)
redis.call('PEXPIRE', seq_key, window + 1000)

return {1, count + 1}
`)

// RedisRateLimiter enforces per-tenant rate limits using a Redis Lua script.
// Falls back to local in-memory limiter if Redis is unavailable.
type RedisRateLimiter struct {
	rdb          *redis.Client
	localFallback *RateLimiter // in-process fallback for Redis outages
	defaultRPM   int
}

// NewRedisRateLimiter creates a Redis-backed rate limiter.
// localFallback is used when Redis is unavailable.
func NewRedisRateLimiter(rdb *redis.Client, defaultRPM int) *RedisRateLimiter {
	return &RedisRateLimiter{
		rdb:          rdb,
		localFallback: NewRateLimiter(defaultRPM),
		defaultRPM:   defaultRPM,
	}
}

// Middleware returns a Gin handler that enforces Redis sliding-window rate limits.
func (rl *RedisRateLimiter) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tenantID, exists := c.Get("tenant_id")
		if !exists {
			c.Next()
			return
		}

		rpmLimit := rl.defaultRPM
		if limit, ok := c.Get("tenant_rate_limit"); ok {
			rpmLimit = limit.(int)
		}

		// No rate limit configured — allow all requests
		if rpmLimit <= 0 {
			c.Next()
			return
		}

		tenantKey := fmt.Sprintf("rl:%s", tenantID.(string))
		nowMs := time.Now().UnixMilli()
		windowMs := int64(60_000) // 1 minute in milliseconds

		// Use Redis Lua script — single round-trip, atomic
		ctx, cancel := context.WithTimeout(c.Request.Context(), 200*time.Millisecond)
		result, err := slidingWindowLua.Run(ctx, rl.rdb,
			[]string{tenantKey},
			windowMs,
			int64(rpmLimit),
			nowMs,
		).Int64Slice()
		cancel()

		if err != nil {
			// Redis unavailable — degrade gracefully to local limiter
			rl.localFallback.Middleware()(c)
			return
		}

		allowed := result[0] == 1
		count := result[1]

		c.Header("X-RateLimit-Limit", fmt.Sprintf("%d", rpmLimit))
		c.Header("X-RateLimit-Remaining", fmt.Sprintf("%d", max64(int64(rpmLimit)-count, 0)))

		if !allowed {
			c.Header("Retry-After", "60")
			// Anthropic clients (/v1/messages) receive the nested error
			// envelope with type "rate_limit_error"; OpenAI-compatible
			// clients keep the flat {"error":..., "limit":..., "retry_after":...}.
			abortWithProtocolError(c, http.StatusTooManyRequests, "rate_limit_error", "rate limit exceeded",
				gin.H{
					"limit":       rpmLimit,
					"retry_after": 60,
				})
			return
		}

		c.Next()
	}
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
