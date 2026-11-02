package middleware

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/database"
)

// tenantLookup abstracts single and two-layer tenant lookup.
type tenantLookup interface {
	Get(ctx context.Context, apiKey string) (*database.TenantRow, bool)
}

// ristrettoLookup wraps bare *cache.Store to satisfy tenantLookup.
type ristrettoLookup struct{ store *cache.Store }

func (r *ristrettoLookup) Get(_ context.Context, apiKey string) (*database.TenantRow, bool) {
	val, ok := r.store.Get(cache.TenantKey(apiKey))
	if !ok {
		return nil, false
	}
	return val.(*database.TenantRow), true
}

// ProxyAuth extracts and validates the virtual API key from the Authorization header.
// Uses RedisTenantCache (L1 Ristretto + L2 Redis) when available, falling back to
// Ristretto-only — zero database calls on every hot-path request.
// Accepts optional master admin API key to allow admin users to query proxy routes.
//
// Header format: Authorization: Bearer <virtual-api-key>
func ProxyAuth(cacheStore *cache.Store, adminAPIKey ...string) gin.HandlerFunc {
	var adminKey string
	if len(adminAPIKey) > 0 {
		adminKey = adminAPIKey[0]
	}
	return proxyAuthWith(&ristrettoLookup{cacheStore}, adminKey)
}

// ProxyAuthWithRedis uses the two-layer Ristretto+Redis tenant cache.
// Accepts optional master admin API key to allow admin users to query proxy routes.
func ProxyAuthWithRedis(tenantCache *cache.RedisTenantCache, adminAPIKey ...string) gin.HandlerFunc {
	var adminKey string
	if len(adminAPIKey) > 0 {
		adminKey = adminAPIKey[0]
	}
	return proxyAuthWith(tenantCache, adminKey)
}

// isAnthropicRequest returns true when the request targets the Anthropic Messages API
// (/v1/messages or /v1/messages/*) or carries an explicit anthropic-version header.
// Used to select the error envelope format: Anthropic clients expect
// {"type":"error","error":{...}} while OpenAI-compatible clients expect flat
// {"error": ...} objects.
func isAnthropicRequest(c *gin.Context) bool {
	p := c.Request.URL.Path
	for strings.HasPrefix(p, "/v1") {
		p = strings.TrimPrefix(p, "/v1")
	}
	return strings.HasPrefix(p, "/messages") || c.GetHeader("anthropic-version") != ""
}

// abortWithProtocolError writes an Anthropic-formatted error envelope for Anthropic
// requests, or the existing flat JSON error object for OpenAI-compatible requests.
//
// Anthropic envelope:
//
//	{"type":"error","error":{"type":"<anthropicErrType>","message":"<message>"}}
//
// OpenAI envelope (unchanged behavior): {"error":"<message>"} plus any extra
// fields supplied in extraFields[0] (e.g. "limit", "retry_after" for 429s).
func abortWithProtocolError(c *gin.Context, status int, anthropicErrType, message string, extraFields ...gin.H) {
	if isAnthropicRequest(c) {
		c.AbortWithStatusJSON(status, gin.H{
			"type": "error",
			"error": gin.H{
				"type":    anthropicErrType,
				"message": message,
			},
		})
		return
	}
	if len(extraFields) > 0 {
		payload := extraFields[0]
		payload["error"] = message
		c.AbortWithStatusJSON(status, payload)
		return
	}
	c.AbortWithStatusJSON(status, gin.H{"error": message})
}

func proxyAuthWith(lookup tenantLookup, adminAPIKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var apiKey string

		// Dual-header key extraction:
		//   1. Authorization: Bearer <key>   (OpenAI-compatible clients, admin tools)
		//   2. x-api-key: <key>              (Anthropic SDKs, Claude Code CLI, Cursor, Cline)
		// Authorization wins when both are present; a malformed Authorization
		// header is rejected even when x-api-key is set (explicit-but-broken
		// credentials must never silently fall through to another source).
		authHeader := c.GetHeader("Authorization")
		if authHeader != "" {
			if !strings.HasPrefix(authHeader, "Bearer ") {
				abortWithProtocolError(c, http.StatusUnauthorized, "authentication_error",
					"invalid Authorization format, expected 'Bearer <key>'")
				return
			}
			apiKey = strings.TrimSpace(authHeader[7:])
			if apiKey == "" {
				abortWithProtocolError(c, http.StatusUnauthorized, "authentication_error",
					"empty API key")
				return
			}
		} else if xApiKey := strings.TrimSpace(c.GetHeader("x-api-key")); xApiKey != "" {
			apiKey = xApiKey
		} else {
			abortWithProtocolError(c, http.StatusUnauthorized, "authentication_error",
				"missing Authorization header")
			return
		}

		// Master Admin key bypass for playground and admin tools
		if adminAPIKey != "" && constantTimeEqual(apiKey, adminAPIKey) {
			c.Set("tenant_id", "admin")
			c.Set("tenant_name", "Master Admin")
			c.Set("tenant_rate_limit", 100000)
			c.Set("tenant_balance", int64(1000000000000))
			c.Next()
			return
		}

		// L1 Ristretto lookup (~200ns) — L2 Redis on miss (~1ms) — never hits DB
		ctx, cancel := context.WithTimeout(c.Request.Context(), 200*time.Millisecond)
		tenant, found := lookup.Get(ctx, apiKey)
		cancel()

		if !found {
			abortWithProtocolError(c, http.StatusUnauthorized, "authentication_error", "invalid API key")
			return
		}

		if !tenant.IsActive {
			abortWithProtocolError(c, http.StatusForbidden, "permission_error", "account is suspended")
			return
		}

		// Attach tenant context for downstream handlers
		c.Set("tenant_id", tenant.ID)
		c.Set("tenant_name", tenant.Name)
		c.Set("tenant_rate_limit", tenant.RateLimitRPM)
		c.Set("tenant_balance", tenant.TokenBalance)

		c.Next()
	}
}

// AdminAuth validates the master admin API key for management endpoints.
// Uses constant-time comparison to prevent timing attacks.
func AdminAuth(adminAPIKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "missing Authorization header",
			})
			return
		}

		if !strings.HasPrefix(authHeader, "Bearer ") {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "invalid Authorization format",
			})
			return
		}
		token := authHeader[7:]

		if !constantTimeEqual(token, adminAPIKey) {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "invalid admin API key",
			})
			return
		}

		c.Next()
	}
}

// constantTimeEqual performs a constant-time string comparison.
func constantTimeEqual(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	var result byte
	for i := 0; i < len(a); i++ {
		result |= a[i] ^ b[i]
	}
	return result == 0
}
