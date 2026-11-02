package jobs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/database"
	"go.uber.org/zap"
)

// RegisterBuiltinExecutors registers all built-in job types into the registry.
// Call this once at startup before calling Scheduler.Start().
func RegisterBuiltinExecutors(reg *Registry, db *pgxpool.Pool, rdb *redis.Client, vault *credentials.Vault, logger *zap.Logger) {
	reg.Register("telemetry_cleanup", newTelemetryCleanupExecutor(db, logger))
	reg.Register("credential_health_check", newCredentialHealthCheckExecutor(db, logger))
	reg.Register("log_rotation", newLogRotationExecutor(logger))
	reg.Register("cache_warmup", newCacheWarmupExecutor(db, logger))
	reg.Register("job_log_cleanup", newJobLogCleanupExecutor(db, logger))
	reg.Register("noop", newNoopExecutor(logger))
	reg.Register("bulk_pool_health_check", newBulkPoolHealthCheckExecutor(db, rdb, vault, logger))
	reg.Register("exhaustive_pool_health_check", newExhaustivePoolHealthCheckExecutor(db, vault, logger))
	reg.Register("provider_rediscovery", newProviderRediscoveryExecutor(db, vault, logger))

	logger.Info("built-in job executors registered",
		zap.Strings("types", reg.ListTypes()),
	)
}

// --- Built-in Executor: exhaustive_pool_health_check ---
// Probes every (pool × credential) combination, records sessions and granular results to DB,
// and streams live progress via SSE.

func newExhaustivePoolHealthCheckExecutor(db *pgxpool.Pool, vault *credentials.Vault, logger *zap.Logger) ExecutorFunc {
	return func(execCtx *ExecutionContext) (string, error) {
		ctx := execCtx.Context
		if ctx == nil {
			ctx = context.Background()
		}

		job := NewExhaustiveHealthCheckJob(db, vault, logger, nil)
		summary, err := job.Run(ctx, "scheduled")
		if err != nil {
			return "", err
		}

		return fmt.Sprintf("Exhaustive health check session %s completed: %d total tasks, %d passed, %d failed, avg latency %.1fms",
			summary.ID, summary.TotalTasks, summary.PassedCount, summary.FailedCount, summary.AvgLatencyMS), nil
	}
}

// --- Built-in Executor: provider_rediscovery ---
// Scans all registered provider endpoints for newly available models and
// auto-registers them into the pool system. Produces a structured JSON report
// with counts of truly new models (diff-based) and per-provider breakdown.

func newProviderRediscoveryExecutor(db *pgxpool.Pool, vault *credentials.Vault, logger *zap.Logger) ExecutorFunc {
	return func(execCtx *ExecutionContext) (string, error) {
		// Use the scheduler's timeout-aware context, NOT context.Background().
		// context.Background() would detach from the scheduler timeout, leaving
		// goroutines running indefinitely after the job is marked as timed out.
		ctx := execCtx.Context
		if ctx == nil {
			ctx = context.Background() // safe fallback for tests / manual calls
		}

		// Allow per-provider timeout to be configured via job payload.
		// Default is 15 seconds (applied inside RunReDiscovery).
		perProviderTimeout := 0
		if v, ok := execCtx.Payload["per_provider_timeout_seconds"]; ok {
			if d, ok := v.(float64); ok && d > 0 {
				perProviderTimeout = int(d)
			}
		}

		logger.Info("provider_rediscovery job started",
			zap.String("run_id", execCtx.RunID),
		)

		report, err := credentials.RunReDiscovery(ctx, db, vault, logger, perProviderTimeout)
		if err != nil {
			return "", fmt.Errorf("re-discovery failed: %w", err)
		}

		// Broadcast reload to all cluster nodes so new pools are active immediately
		_, _ = db.Exec(ctx, "NOTIFY config_change, 'model_pools:reload'")

		logger.Info("provider_rediscovery job finished",
			zap.String("run_id", execCtx.RunID),
			zap.Int("new_models", report.NewModelsAdded),
			zap.Int("total_synced", report.TotalModelsSynced),
			zap.Int("workers_used", report.WorkerCount),
			zap.Int64("duration_ms", report.DurationMs),
		)

		return credentials.MarshalReport(report), nil
	}
}

// --- Built-in Executor: telemetry_cleanup ---
// Deletes request_logs older than the configured retention window.

func newTelemetryCleanupExecutor(db *pgxpool.Pool, logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		retentionDays := 30
		if v, ok := ctx.Payload["retention_days"]; ok {
			if d, ok := v.(float64); ok && d > 0 {
				retentionDays = int(d)
			}
		}

		c := context.Background()
		result, err := db.Exec(c, `
			DELETE FROM request_logs
			WHERE created_at < NOW() - ($1 || ' days')::INTERVAL
		`, retentionDays)
		if err != nil {
			return "", fmt.Errorf("telemetry cleanup failed: %w", err)
		}

		deleted := result.RowsAffected()
		logger.Info("telemetry_cleanup complete",
			zap.Int64("deleted_rows", deleted),
			zap.Int("retention_days", retentionDays),
		)
		return fmt.Sprintf("Deleted %d telemetry records older than %d days", deleted, retentionDays), nil
	}
}

// --- Built-in Executor: credential_health_check ---
// Read-only audit of credential error state. is_healthy is exclusively
// controlled by admin action — this job no longer mutates it automatically.

func newCredentialHealthCheckExecutor(db *pgxpool.Pool, logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		c := context.Background()

		// Count credentials that have a recorded error (read-only — no state mutation).
		var withErrors int64
		if err := db.QueryRow(c, `
			SELECT COUNT(*) FROM credentials
			WHERE last_error IS NOT NULL AND last_error != ''
		`).Scan(&withErrors); err != nil {
			return "", fmt.Errorf("credential health audit query failed: %w", err)
		}

		// Count credentials currently flagged healthy by admin.
		var healthy int64
		_ = db.QueryRow(c, `SELECT COUNT(*) FROM credentials WHERE is_healthy = TRUE`).Scan(&healthy)

		var total int64
		_ = db.QueryRow(c, `SELECT COUNT(*) FROM credentials`).Scan(&total)

		logger.Info("credential_health_check audit complete (read-only)",
			zap.Int64("total", total),
			zap.Int64("healthy", healthy),
			zap.Int64("with_errors", withErrors),
		)
		return fmt.Sprintf("Health audit: %d/%d healthy, %d with recorded errors (no state changes — admin controls is_healthy)",
			healthy, total, withErrors), nil
	}
}

// --- Built-in Executor: log_rotation ---
// Archives application log files.

func newLogRotationExecutor(logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		// Log rotation is typically handled by the OS/logrotate daemon.
		// This executor signals any custom rotation logic.
		logger.Info("log_rotation job triggered")
		return "Log rotation signal sent. OS-level log rotation handles actual file management.", nil
	}
}

// --- Built-in Executor: cache_warmup ---
// Pre-warms the tenant cache from the database.

func newCacheWarmupExecutor(db *pgxpool.Pool, logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		c := context.Background()

		var count int
		err := db.QueryRow(c, `SELECT COUNT(*) FROM tenants WHERE is_active = TRUE`).Scan(&count)
		if err != nil {
			return "", fmt.Errorf("cache warmup count query failed: %w", err)
		}

		logger.Info("cache_warmup triggered",
			zap.Int("active_tenants", count),
		)
		return fmt.Sprintf("Cache warmup triggered for %d active tenants (cache warming handled by sync manager)", count), nil
	}
}

// --- Built-in Executor: job_log_cleanup ---
// Removes old job run records from the database.

func newJobLogCleanupExecutor(db *pgxpool.Pool, logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		retentionDays := 30
		if v, ok := ctx.Payload["retention_days"]; ok {
			if d, ok := v.(float64); ok && d > 0 {
				retentionDays = int(d)
			}
		}

		c := context.Background()
		cutoff := time.Now().AddDate(0, 0, -retentionDays)

		result, err := db.Exec(c, `
			DELETE FROM job_runs WHERE created_at < $1
		`, cutoff)
		if err != nil {
			return "", fmt.Errorf("job log cleanup failed: %w", err)
		}

		deleted := result.RowsAffected()
		logger.Info("job_log_cleanup complete",
			zap.Int64("deleted_runs", deleted),
			zap.Int("retention_days", retentionDays),
		)
		return fmt.Sprintf("Deleted %d job run records older than %d days", deleted, retentionDays), nil
	}
}

// --- Built-in Executor: noop ---
// Does nothing — useful for testing the scheduler pipeline.

func newNoopExecutor(logger *zap.Logger) ExecutorFunc {
	return func(ctx *ExecutionContext) (string, error) {
		logger.Debug("noop job executed", zap.String("job_id", ctx.JobID), zap.String("run_id", ctx.RunID))
		return "No operation performed (noop)", nil
	}
}

// HostRateLimiter caps maximum concurrent active health check probes per domain host
// to prevent 503 ResourceExhausted rate limit bursts on upstream endpoints.
type HostRateLimiter struct {
	mu     sync.Mutex
	limits map[string]chan struct{}
}

func NewHostRateLimiter() *HostRateLimiter {
	return &HostRateLimiter{
		limits: make(map[string]chan struct{}),
	}
}

func (h *HostRateLimiter) Acquire(host string, maxConcurrent int) func() {
	h.mu.Lock()
	ch, exists := h.limits[host]
	if !exists {
		ch = make(chan struct{}, maxConcurrent)
		h.limits[host] = ch
	}
	h.mu.Unlock()

	ch <- struct{}{}
	return func() { <-ch }
}

// --- Built-in Executor: bulk_pool_health_check ---
// Probes all unique API keys (deduplicated from the N×M credential×pool join)
// to detect genuine quota/balance failures. Uses CPU-parallel, adaptive
// capability-aware HTTP probes.
//
// Architecture:
//   - ListAllCredentials returns N×M rows (e.g. 42,281 rows for 437 keys × ~97 pools).
//   - We deduplicate by encrypted_key → probe only ~437 unique keys (one probe per key).
//   - Strict error classification: only 401/402/403/429 + balance-body keywords
//     count as failures. 400/404/422/timeouts are structural probe errors — ignored.
//
// Observability-only: no DB writes. is_healthy is exclusively admin-controlled.

func newBulkPoolHealthCheckExecutor(db *pgxpool.Pool, rdb *redis.Client, vault *credentials.Vault, logger *zap.Logger) ExecutorFunc {
	return func(execCtx *ExecutionContext) (string, error) {
		ctx := execCtx.Context
		if ctx == nil {
			ctx = context.Background()
		}

		// 1. Pull all credentials joined with pool model_pattern and capabilities.
		// Returns N×M rows (credential × pool join — e.g. 42,281 rows).
		creds, err := database.ListAllCredentials(ctx, db)
		if err != nil {
			return "", fmt.Errorf("bulk health check: failed to list credentials: %w", err)
		}
		if len(creds) == 0 {
			return "No credentials found in database to evaluate.", nil
		}

		// 2. DEDUPLICATION: group by unique encrypted key.
		// We only need to probe each unique API key once — if the key is
		// valid/invalid it will be the same regardless of which model pool is tested.
		// This reduces 42,281 targets → ~437 unique API keys.
		type keyGroup struct {
			representative *database.CredentialWithPool
			decryptedKey   string
		}
		seenKeys := make(map[string]*keyGroup, 512)
		for _, c := range creds {
			if _, exists := seenKeys[c.EncryptedKey]; !exists {
				apiKey, decErr := vault.Decrypt(c.EncryptedKey)
				if decErr != nil {
					logger.Error("bulk health: key decryption failed (skipping)",
						zap.Int("cred_id", c.ID), zap.Error(decErr))
					continue
				}
				seenKeys[c.EncryptedKey] = &keyGroup{
					representative: c,
					decryptedKey:   apiKey,
				}
			}
		}

		uniqueKeys := make([]*keyGroup, 0, len(seenKeys))
		for _, kg := range seenKeys {
			uniqueKeys = append(uniqueKeys, kg)
		}

		workerCount := runtime.NumCPU() * 8
		if workerCount < 32 {
			workerCount = 32
		}

		logger.Info("bulk_pool_health_check started",
			zap.String("run_id", execCtx.RunID),
			zap.Int("total_credential_rows", len(creds)),
			zap.Int("unique_keys_to_probe", len(uniqueKeys)),
			zap.Int("cpu_workers", workerCount),
		)

		jobsChan := make(chan *keyGroup, len(uniqueKeys))
		for _, kg := range uniqueKeys {
			jobsChan <- kg
		}
		close(jobsChan)

		var successCount atomic.Int64
		var failureCount atomic.Int64
		var structuralCount atomic.Int64
		var wg sync.WaitGroup

		hostLimiter := NewHostRateLimiter()

		// Tuned outbound HTTP client: short timeout, high idle conn limits for throughput.
		client := &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        1000,
				MaxIdleConnsPerHost: 50,
				IdleConnTimeout:     30 * time.Second,
				DisableKeepAlives:   false,
			},
		}

		for i := 0; i < workerCount; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				for kg := range jobsChan {
					c := kg.representative
					apiKey := kg.decryptedKey

					// Acquire host concurrency lease (max 15 concurrent requests per host domain)
					hostKey := "default"
					if parsedURL, err := url.Parse(c.BaseURL); err == nil && parsedURL.Host != "" {
						hostKey = parsedURL.Host
					}
					release := hostLimiter.Acquire(hostKey, 15)

					isHealthy, statusCode, errStr := probeCredential(ctx, client, c, apiKey)
					release()

					if isHealthy {
						successCount.Add(1)
					} else if errStr != nil {
						// STRICT ERROR CLASSIFICATION:
						// Only warn on genuine quota/auth failures (401/402/403/429 + balance keywords).
						// Structural errors (400/404/422/timeout) mean our probe payload/endpoint is
						// wrong — NOT that the key is bad or out of credits.
						bodyForClassify := *errStr
						if IsQuotaOrBalanceError(statusCode, bodyForClassify) {
							logger.Warn("bulk health: quota/balance failure detected (admin should review)",
								zap.Int("cred_id", c.ID),
								zap.String("provider", c.Provider),
								zap.String("base_url", c.BaseURL),
								zap.String("model_pattern", c.ModelPattern),
								zap.Int("status_code", statusCode),
								zap.String("error", bodyForClassify),
							)
							failureCount.Add(1)
						} else {
							// Structural/connectivity error — not a credential problem.
							// Log at debug level only to avoid log noise.
							logger.Debug("bulk health: structural probe error (key likely still valid)",
								zap.Int("cred_id", c.ID),
								zap.String("provider", c.Provider),
								zap.String("model_pattern", c.ModelPattern),
								zap.Int("status_code", statusCode),
								zap.String("error", bodyForClassify),
							)
							structuralCount.Add(1)
							successCount.Add(1) // Don't penalise credential for our structural probe error
						}
					} else {
						successCount.Add(1)
					}
				}
			}()
		}

		wg.Wait()

		summary := fmt.Sprintf(
			"Bulk health probe complete — healthy keys: %d, quota/balance failures: %d, structural probe errors (ignored): %d, unique keys probed: %d (from %d total credential rows). No state changes made.",
			successCount.Load(), failureCount.Load(), structuralCount.Load(), len(uniqueKeys), len(creds),
		)
		logger.Info("bulk_pool_health_check finished",
			zap.String("run_id", execCtx.RunID),
			zap.Int("total_credential_rows", len(creds)),
			zap.Int("unique_keys_probed", len(uniqueKeys)),
			zap.Int64("healthy_keys", successCount.Load()),
			zap.Int64("quota_balance_failures", failureCount.Load()),
			zap.Int64("structural_probe_errors", structuralCount.Load()),
		)
		return summary, nil
	}
}

// probeCredential sends a capability- and provider-adaptive HTTP health probe
// for a single credential and returns (isHealthy, httpStatusCode, *errorString).
// The statusCode is returned separately so callers can apply IsQuotaOrBalanceError
// to distinguish genuine quota/auth failures from structural probe errors.
func probeCredential(ctx context.Context, client *http.Client, c *database.CredentialWithPool, apiKey string) (bool, int, *string) {
	var req *http.Request
	var buildErr error

	if c.Provider == "ollama" {
		// Ollama: use /api/tags endpoint (lightweight, no model inference needed)
		urlStr := strings.TrimRight(c.BaseURL, "/") + "/api/tags"
		req, buildErr = http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
		if buildErr == nil {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}
	} else {
		// Parse capabilities JSON if present
		var caps map[string]bool
		if len(c.Capabilities) > 0 {
			_ = json.Unmarshal(c.Capabilities, &caps)
		}

		// Resolve wildcard patterns to a concrete model name
		testModel := c.ModelPattern
		if strings.Contains(testModel, "*") {
			switch {
			case strings.Contains(testModel, "gpt"):
				testModel = "gpt-4o-mini"
			case strings.Contains(testModel, "claude"):
				testModel = "claude-3-5-haiku-20241022"
			case strings.Contains(testModel, "nvidia"):
				testModel = "nvidia/llama-3.1-nemotron-70b-instruct"
			default:
				testModel = strings.ReplaceAll(testModel, "*", "latest")
			}
		}

		probeReq, err := BuildAdaptiveProbeRequest(c.BaseURL, apiKey, c.Provider, testModel, caps)
		if err != nil {
			errStr := fmt.Sprintf("failed to build adaptive probe request: %v", err)
			return false, 0, &errStr
		}

		req, buildErr = http.NewRequestWithContext(ctx, probeReq.Method, probeReq.URL, bytes.NewReader(probeReq.Body))
		if buildErr == nil {
			for k, v := range probeReq.Headers {
				req.Header.Set(k, v)
			}
		}
	}

	if buildErr != nil {
		errStr := fmt.Sprintf("failed to build HTTP request: %v", buildErr)
		return false, 0, &errStr
	}

	resp, doErr := client.Do(req)
	if doErr != nil {
		// Network/timeout errors: return status 0 so IsQuotaOrBalanceError returns false.
		// These are connectivity issues, not credential failures.
		errStr := fmt.Sprintf("connection error: %v", doErr)
		return false, 0, &errStr
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, resp.StatusCode, nil
	}

	limitReader := io.LimitReader(resp.Body, 512)
	respBytes, _ := io.ReadAll(limitReader)
	errStr := fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(respBytes)))
	return false, resp.StatusCode, &errStr
}
