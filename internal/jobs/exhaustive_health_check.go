package jobs

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"go.uber.org/zap"
)

// ExhaustiveHealthCheckJob probes every (pool × credential) combination in the
// database, rather than the deduped-key approach of bulk_pool_health_check.
//
// This produces a complete session record with:
//   - One health_check_sessions row per job run
//   - One health_check_results row per (pool × credential) probe
//   - Real-time SSE events broadcast via the broadcaster channel
type ExhaustiveHealthCheckJob struct {
	db          *pgxpool.Pool
	vault       *credentials.Vault
	logger      *zap.Logger
	broadcaster chan<- HealthCheckSSEEvent
}

// NewExhaustiveHealthCheckJob creates a new exhaustive health check job.
// broadcaster is a send-only channel; nil disables SSE broadcasting (safe for tests).
func NewExhaustiveHealthCheckJob(db *pgxpool.Pool, vault *credentials.Vault, logger *zap.Logger, broadcaster chan<- HealthCheckSSEEvent) *ExhaustiveHealthCheckJob {
	return &ExhaustiveHealthCheckJob{
		db:          db,
		vault:       vault,
		logger:      logger,
		broadcaster: broadcaster,
	}
}

// probeTask is one unit of work: a specific credential probing a specific pool's model.
type probeTask struct {
	poolID       int64
	poolName     string
	modelPattern string
	providerID   string
	baseURL      string
	apiKey       string
	credentialID int64
}

// Run executes the exhaustive health check session.
// triggerType should be one of: "manual_full", "scheduled", "manual_pool".
func (j *ExhaustiveHealthCheckJob) Run(ctx context.Context, triggerType string) (*HealthCheckSessionSummary, error) {
	// ── Step 1: Create session record ──────────────────────────────────────────
	var sessionID string
	err := j.db.QueryRow(ctx, `
		INSERT INTO health_check_sessions (trigger_type, status)
		VALUES ($1, 'running') RETURNING id::text
	`, triggerType).Scan(&sessionID)
	if err != nil {
		return nil, fmt.Errorf("exhaustive health check: failed to create session: %w", err)
	}

	j.logger.Info("exhaustive_pool_health_check session created",
		zap.String("session_id", sessionID),
		zap.String("trigger_type", triggerType),
	)

	// ── Step 2: Build (pool × credential) probe matrix ────────────────────────
	tasks, totalPools, err := j.buildProbeMatrix(ctx)
	if err != nil {
		// Mark session failed and return
		_, _ = j.db.Exec(ctx, `UPDATE health_check_sessions SET status = 'failed', completed_at = NOW() WHERE id = $1::uuid`, sessionID)
		return nil, fmt.Errorf("exhaustive health check: build matrix failed: %w", err)
	}

	totalTasks := len(tasks)
	now := time.Now()
	summary := &HealthCheckSessionSummary{
		ID:          sessionID,
		TriggerType: triggerType,
		Status:      "running",
		TotalPools:  totalPools,
		TotalTasks:  totalTasks,
		StartedAt:   now,
	}

	// Update total counts in DB
	_, _ = j.db.Exec(ctx, `
		UPDATE health_check_sessions SET total_pools = $1, total_tasks = $2 WHERE id = $3::uuid
	`, totalPools, totalTasks, sessionID)

	// Broadcast session start event
	j.emit(HealthCheckSSEEvent{
		EventType: "start",
		SessionID: sessionID,
		Progress:  0,
		Summary:   summary,
	})

	j.logger.Info("exhaustive_pool_health_check probing started",
		zap.String("session_id", sessionID),
		zap.Int("total_pools", totalPools),
		zap.Int("total_tasks", totalTasks),
	)

	if totalTasks == 0 {
		completedAt := time.Now()
		summary.Status = "completed"
		summary.CompletedAt = &completedAt
		_, _ = j.db.Exec(ctx, `UPDATE health_check_sessions SET status = 'completed', completed_at = NOW() WHERE id = $1::uuid`, sessionID)
		j.emit(HealthCheckSSEEvent{EventType: "complete", SessionID: sessionID, Progress: 100, Summary: summary})
		return summary, nil
	}

	// ── Step 3: Worker pool concurrent probe ───────────────────────────────────
	taskChan := make(chan probeTask, totalTasks)
	for _, t := range tasks {
		taskChan <- t
	}
	close(taskChan)

	var (
		passedCount   int32
		failedCount   int32
		totalLatency  int64
		completedWork int32
	)

	client := &http.Client{
		Timeout: 15 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        500,
			MaxIdleConnsPerHost: 30,
			IdleConnTimeout:     30 * time.Second,
		},
	}

	workers := 10
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				if ctx.Err() != nil {
					return // context cancelled (scheduler timeout)
				}

				result := j.runProbe(ctx, client, sessionID, task)

				// Accumulate counters atomically
				atomic.AddInt64(&totalLatency, int64(result.LatencyMS))
				if result.IsHealthy {
					atomic.AddInt32(&passedCount, 1)
				} else {
					atomic.AddInt32(&failedCount, 1)
				}

				done := atomic.AddInt32(&completedWork, 1)
				progress := (float64(done) / float64(totalTasks)) * 100.0

				j.emit(HealthCheckSSEEvent{
					EventType: "progress",
					SessionID: sessionID,
					Progress:  progress,
					Item:      result,
				})
			}
		}()
	}

	wg.Wait()

	// ── Step 4: Finalize session record ───────────────────────────────────────
	completedAt := time.Now()
	passed := int(passedCount)
	failed := int(failedCount)
	avgLatency := float64(0)
	if totalTasks > 0 {
		avgLatency = float64(atomic.LoadInt64(&totalLatency)) / float64(totalTasks)
	}

	_, _ = j.db.Exec(ctx, `
		UPDATE health_check_sessions
		SET status = 'completed',
		    passed_count = $1,
		    failed_count = $2,
		    avg_latency_ms = $3,
		    completed_at = $4
		WHERE id = $5::uuid
	`, passed, failed, avgLatency, completedAt, sessionID)

	summary.Status = "completed"
	summary.PassedCount = passed
	summary.FailedCount = failed
	summary.AvgLatencyMS = avgLatency
	summary.CompletedAt = &completedAt

	j.emit(HealthCheckSSEEvent{
		EventType: "complete",
		SessionID: sessionID,
		Progress:  100,
		Summary:   summary,
	})

	j.logger.Info("exhaustive_pool_health_check session completed",
		zap.String("session_id", sessionID),
		zap.Int("total_tasks", totalTasks),
		zap.Int("passed", passed),
		zap.Int("failed", failed),
		zap.Float64("avg_latency_ms", avgLatency),
	)

	return summary, nil
}

// buildProbeMatrix queries all active (pool × credential) pairs and returns them
// as a flat task slice. Each pair is one HTTP probe target.
func (j *ExhaustiveHealthCheckJob) buildProbeMatrix(ctx context.Context) ([]probeTask, int, error) {
	rows, err := j.db.Query(ctx, `
		SELECT
			p.id            AS pool_id,
			p.model_pattern AS pool_name,
			p.model_pattern AS model_pattern,
			c.provider      AS provider_id,
			c.base_url      AS base_url,
			c.encrypted_key AS encrypted_key,
			c.id            AS credential_id
		FROM model_pools p
		JOIN credentials c ON c.pool_id = p.id
		WHERE p.is_active = true
		  AND c.is_healthy = true
		ORDER BY p.id, c.id
	`)
	if err != nil {
		return nil, 0, fmt.Errorf("buildProbeMatrix query failed: %w", err)
	}
	defer rows.Close()

	seenPools := make(map[int64]struct{})
	var tasks []probeTask

	for rows.Next() {
		var t probeTask
		var encryptedKey string
		if err := rows.Scan(
			&t.poolID, &t.poolName, &t.modelPattern,
			&t.providerID, &t.baseURL, &encryptedKey, &t.credentialID,
		); err != nil {
			j.logger.Warn("buildProbeMatrix: scan error (skipping row)", zap.Error(err))
			continue
		}

		var apiKey string
		if j.vault != nil {
			decrypted, decErr := j.vault.Decrypt(encryptedKey)
			if decErr != nil {
				j.logger.Warn("buildProbeMatrix: key decryption failed (skipping row)",
					zap.Int64("credential_id", t.credentialID), zap.Error(decErr))
				continue
			}
			apiKey = decrypted
		} else {
			apiKey = encryptedKey
		}

		t.apiKey = apiKey
		seenPools[t.poolID] = struct{}{}
		tasks = append(tasks, t)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("buildProbeMatrix rows iteration error: %w", err)
	}

	return tasks, len(seenPools), nil
}

// runProbe executes one HTTP probe for a single (pool × credential) task,
// persists the result to health_check_results, and returns the result item.
func (j *ExhaustiveHealthCheckJob) runProbe(ctx context.Context, client *http.Client, sessionID string, task probeTask) *HealthCheckResultItem {
	start := time.Now()

	cleanModel := CleanUpstreamModel(task.modelPattern, task.providerID)

	probeReq, buildErr := BuildAdaptiveProbeRequest(task.baseURL, task.apiKey, task.providerID, cleanModel, nil)

	latencyMS := int(time.Since(start).Milliseconds())

	result := &HealthCheckResultItem{
		SessionID:    sessionID,
		PoolID:       task.poolID,
		PoolName:     task.poolName,
		ModelPattern: task.modelPattern,
		ProviderID:   task.providerID,
		CredentialID: task.credentialID,
		CheckedAt:    time.Now(),
	}

	if buildErr != nil {
		result.StatusCode = 0
		result.IsHealthy = false
		result.ErrorMessage = fmt.Sprintf("probe build error: %v", buildErr)
		result.LatencyMS = latencyMS
		j.persistResult(ctx, result)
		return result
	}

	req, reqErr := http.NewRequestWithContext(ctx, probeReq.Method, probeReq.URL, bytes.NewReader(probeReq.Body))
	if reqErr != nil {
		result.StatusCode = 0
		result.IsHealthy = false
		result.ErrorMessage = fmt.Sprintf("request build error: %v", reqErr)
		result.LatencyMS = int(time.Since(start).Milliseconds())
		j.persistResult(ctx, result)
		return result
	}
	for k, v := range probeReq.Headers {
		req.Header.Set(k, v)
	}

	resp, doErr := client.Do(req)
	result.LatencyMS = int(time.Since(start).Milliseconds())

	if doErr != nil {
		result.StatusCode = 0
		result.IsHealthy = false
		result.ErrorMessage = fmt.Sprintf("connection error: %v", doErr)
		j.persistResult(ctx, result)
		return result
	}
	defer resp.Body.Close()

	limitReader := io.LimitReader(resp.Body, 512)
	respBytes, _ := io.ReadAll(limitReader)

	result.StatusCode = resp.StatusCode
	result.IsHealthy = resp.StatusCode >= 200 && resp.StatusCode < 300

	if !result.IsHealthy {
		result.ErrorMessage = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBytes))
	}

	j.persistResult(ctx, result)
	return result
}

// persistResult inserts one result row into health_check_results and sets result.ID.
func (j *ExhaustiveHealthCheckJob) persistResult(ctx context.Context, r *HealthCheckResultItem) {
	err := j.db.QueryRow(ctx, `
		INSERT INTO health_check_results
		  (session_id, pool_id, pool_name, model_pattern, provider_id, credential_id,
		   status_code, is_healthy, latency_ms, error_message, checked_at)
		VALUES
		  ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		RETURNING id
	`,
		r.SessionID, r.PoolID, r.PoolName, r.ModelPattern, r.ProviderID, r.CredentialID,
		r.StatusCode, r.IsHealthy, r.LatencyMS, r.ErrorMessage, r.CheckedAt,
	).Scan(&r.ID)
	if err != nil {
		j.logger.Warn("exhaustive health check: failed to persist result",
			zap.String("session_id", r.SessionID),
			zap.String("model_pattern", r.ModelPattern),
			zap.Error(err),
		)
	}
}

// emit sends an SSE event to the broadcaster channel without blocking.
// If the channel is full or nil the event is silently dropped.
func (j *ExhaustiveHealthCheckJob) emit(event HealthCheckSSEEvent) {
	if j.broadcaster == nil {
		return
	}
	select {
	case j.broadcaster <- event:
	default:
		// Channel full — drop event rather than block the worker
	}
}
