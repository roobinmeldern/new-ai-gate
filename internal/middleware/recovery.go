package middleware

import (
	"fmt"
	"net/http"
	"runtime/debug"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Recovery returns a middleware that recovers from panics and logs the stack trace.
// Applied ONLY to admin API routes — the proxy hot-path runs without recovery
// middleware to avoid the overhead of defer/recover on every request.
func Recovery(logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if r := recover(); r != nil {
				logger.Error("panic recovered",
					zap.Any("error", r),
					zap.ByteString("stack", debug.Stack()),
					zap.String("path", c.Request.URL.Path),
					zap.String("method", c.Request.Method),
				)

				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
					"error": "internal server error",
				})
			}
		}()
		c.Next()
	}
}

// CORS returns a middleware that handles Cross-Origin Resource Sharing.
// Required for the admin UI panel to communicate with the API, and for
// browser-based Anthropic SDK clients that target /v1/messages with
// x-api-key / anthropic-version / anthropic-beta headers (which trigger
// CORS preflights and need the headers echoed on actual responses).
func CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers",
			"Authorization, Content-Type, Accept, x-api-key, anthropic-version, anthropic-beta, X-Request-ID")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// RequestID adds a unique request ID to each request for tracing.
func RequestID() gin.HandlerFunc {
	var counter uint64

	return func(c *gin.Context) {
		// Check if client provided a request ID
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			// Fast, non-cryptographic ID for request tracing
			id := atomic.AddUint64(&counter, 1)
			requestID = fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), id)
		}
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)
		c.Next()
	}
}
