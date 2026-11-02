package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func init() {
	// Sanitize GIN_MODE before Gin's init() reads it.
	// Some deployment platforms (Clever Cloud) may include inline
	// comments in env vars, e.g. 'debug  # description'.
	if mode := os.Getenv("GIN_MODE"); mode != "" {
		os.Setenv("GIN_MODE", stripInlineComment(mode))
	}
}

// Config holds all application configuration loaded from environment variables.
// All fields are resolved at startup — no reflection or lazy loading on the hot-path.
type Config struct {
	// Server
	Port    int
	GinMode string

	// Database
	DatabaseURL string
	RedisURL    string

	// Redis connection pool tuning
	RedisPoolSize    int
	RedisMinIdle     int
	RedisReadTimeout time.Duration
	RedisWriteTimeout time.Duration
	RedisDialTimeout  time.Duration

	// Security
	MasterEncryptionKey string // AES-256-GCM key (32 bytes hex-encoded)
	AdminAPIKey         string // Master admin API key for management endpoints

	// Cache
	CacheMaxSizeMB   int64
	CacheNumCounters int64 // Ristretto: 10x expected item count

	// Telemetry Pipeline
	TelemetryFlushInterval time.Duration
	TelemetryBatchSize     int
	TelemetryQueueSize     int

	// Transport
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
	DialTimeout         time.Duration
	KeepAlive           time.Duration

	// Edge Probing — comma-separated list of upstream hosts to probe for
	// fastest Anycast CDN edge IP (e.g. "api.cloudflare.com,api-inference.huggingface.co")
	EdgeProbeHosts string

	// Rate Limiting
	DefaultRateLimitRPM int

	// Logging
	LogLevel string

	// Playground Authentication
	PlaygroundUser string
	PlaygroundPass string
}

// Load reads configuration from environment variables.
// It panics if any required configuration is missing, ensuring
// the application never starts in an invalid state.
func Load() *Config {
	cfg := &Config{
		// Server defaults
		Port:    envInt("PORT", 8080),
		GinMode: envStr("GIN_MODE", "release"),

		// Database (required)
		DatabaseURL: envFirst("DATABASE_URL", "POSTGRESQL_ADDON_URI"),
		RedisURL:    envStr("REDIS_URL", envStr("REDIS_DIRECT_URI", "")),

		// Redis pool tuning — tuned for Clever Cloud edge
		RedisPoolSize:     envInt("REDIS_POOL_SIZE", 200),
		RedisMinIdle:      envInt("REDIS_MIN_IDLE", 20),
		RedisReadTimeout:  envDuration("REDIS_READ_TIMEOUT", 200*time.Millisecond),
		RedisWriteTimeout: envDuration("REDIS_WRITE_TIMEOUT", 200*time.Millisecond),
		RedisDialTimeout:  envDuration("REDIS_DIAL_TIMEOUT", 500*time.Millisecond),

		// Security (required)
		MasterEncryptionKey: envRequired("MASTER_ENCRYPTION_KEY"),
		AdminAPIKey:         envRequired("ADMIN_API_KEY"),

		// Cache defaults
		CacheMaxSizeMB:   envInt64("CACHE_MAX_SIZE_MB", 256),
		CacheNumCounters: envInt64("CACHE_NUM_COUNTERS", 1_000_000),

		// Telemetry defaults
		TelemetryFlushInterval: envDuration("TELEMETRY_FLUSH_INTERVAL", 2*time.Second),
		TelemetryBatchSize:     envInt("TELEMETRY_BATCH_SIZE", 500),
		TelemetryQueueSize:     envInt("TELEMETRY_QUEUE_SIZE", 10_000),

		// Transport defaults — tuned for high-throughput proxy
		MaxIdleConns:        envInt("MAX_IDLE_CONNS", 5000),
		MaxIdleConnsPerHost: envInt("MAX_IDLE_CONNS_PER_HOST", 500),
		IdleConnTimeout:     envDuration("IDLE_CONN_TIMEOUT", 120*time.Second),
		DialTimeout:         envDuration("DIAL_TIMEOUT", 2*time.Second),
		KeepAlive:           envDuration("KEEP_ALIVE", 90*time.Second),

		// Edge probing — resolves Anycast IPs and selects the fastest edge node
		EdgeProbeHosts: envStr("EDGE_PROBE_HOSTS", "api.cloudflare.com,api-inference.huggingface.co"),

		// Rate limiting defaults
		DefaultRateLimitRPM: envInt("DEFAULT_RATE_LIMIT_RPM", 60),

		// Logging defaults
		LogLevel: envStr("LOG_LEVEL", "info"),

		// Playground authentication defaults
		PlaygroundUser: envStr("PLAYGROUND_USER", "salman"),
		PlaygroundPass: envStr("PLAYGROUND_PASS", "136517"),
	}

	cfg.validate()
	return cfg
}

// validate checks configuration invariants.
func (c *Config) validate() {
	// Master encryption key must be exactly 64 hex chars (32 bytes)
	key := strings.TrimSpace(c.MasterEncryptionKey)
	if len(key) != 64 {
		panic(fmt.Sprintf("MASTER_ENCRYPTION_KEY must be 64 hex characters (32 bytes), got %d", len(key)))
	}
	for _, ch := range key {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')) {
			panic("MASTER_ENCRYPTION_KEY must contain only hexadecimal characters")
		}
	}

	if c.AdminAPIKey == "" || len(c.AdminAPIKey) < 16 {
		panic("ADMIN_API_KEY must be at least 16 characters")
	}

	if c.Port < 1 || c.Port > 65535 {
		panic(fmt.Sprintf("PORT must be between 1 and 65535, got %d", c.Port))
	}
}

// --- Helper functions: direct os.Getenv with typed conversion, zero reflection ---

// stripInlineComment removes shell-style inline comments from env values.
// e.g. "debug  # description" → "debug"
func stripInlineComment(val string) string {
	if idx := strings.Index(val, "#"); idx > 0 {
		val = val[:idx]
	}
	return strings.TrimSpace(val)
}

func envFirst(keys ...string) string {
	for _, key := range keys {
		if val := stripInlineComment(os.Getenv(key)); val != "" {
			return val
		}
	}
	panic(fmt.Sprintf("one of required environment variables (%s) is not set", strings.Join(keys, ", ")))
}

func envRequired(key string) string {
	val := stripInlineComment(os.Getenv(key))
	if val == "" {
		panic(fmt.Sprintf("required environment variable %s is not set", key))
	}
	return val
}

func envStr(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return stripInlineComment(val)
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if val := os.Getenv(key); val != "" {
		n, err := strconv.Atoi(val)
		if err != nil {
			panic(fmt.Sprintf("environment variable %s must be an integer: %v", key, err))
		}
		return n
	}
	return fallback
}

func envInt64(key string, fallback int64) int64 {
	if val := os.Getenv(key); val != "" {
		n, err := strconv.ParseInt(val, 10, 64)
		if err != nil {
			panic(fmt.Sprintf("environment variable %s must be an int64: %v", key, err))
		}
		return n
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if val := os.Getenv(key); val != "" {
		d, err := time.ParseDuration(val)
		if err != nil {
			panic(fmt.Sprintf("environment variable %s must be a valid duration: %v", key, err))
		}
		return d
	}
	return fallback
}
