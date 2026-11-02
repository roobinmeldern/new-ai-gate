package admin

import (
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/skadraneshghn/clever-ai-gate/api/dto"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/jobs"
	"go.uber.org/zap"
)

// HealthCheckHandler handles admin endpoints for exhaustive model health check sessions,
// real-time SSE streaming, and session history/details retrieval.
type HealthCheckHandler struct {
	db          *pgxpool.Pool
	vault       *credentials.Vault
	logger      *zap.Logger
	broadcaster chan jobs.HealthCheckSSEEvent
}

// NewHealthCheckHandler creates a new admin health check handler.
func NewHealthCheckHandler(db *pgxpool.Pool, vault *credentials.Vault, logger *zap.Logger, b chan jobs.HealthCheckSSEEvent) *HealthCheckHandler {
	return &HealthCheckHandler{
		db:          db,
		vault:       vault,
		logger:      logger,
		broadcaster: b,
	}
}

// TriggerHealthCheck initiates an instant background exhaustive health check session.
//
// @Summary      Trigger exhaustive model pool health check
// @Description  Starts a new background session that probes all (pool × credential) targets and streams progress via SSE.
// @Tags         HealthCheck
// @Produce      json
// @Security     BearerAuth
// @Success      202  {object}  map[string]string
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/health-check/trigger [post]
func (h *HealthCheckHandler) TriggerHealthCheck(c *gin.Context) {
	job := jobs.NewExhaustiveHealthCheckJob(h.db, h.vault, h.logger, h.broadcaster)

	go func() {
		// Run in background with context.Background() since HTTP request context will be closed when handler returns
		ctx := c.Request.Context()
		if ctx == nil {
			ctx = c
		}
		// Create detached context with timeout to ensure job completes even after request finishes
		bgCtx := context.WithoutCancel(ctx)
		_, err := job.Run(bgCtx, "manual_full")
		if err != nil {
			h.logger.Error("manual exhaustive health check job failed", zap.Error(err))
		}
	}()

	c.JSON(http.StatusAccepted, gin.H{
		"message": "Exhaustive health check session initiated",
	})
}

// StreamHealthCheckSSE provides a real-time SSE stream of health check events (start, progress, complete).
//
// @Summary      Stream real-time health check events
// @Description  Streams progress updates (Server-Sent Events) during active health check sessions.
// @Tags         HealthCheck
// @Produce      text/event-stream
// @Security     BearerAuth
// @Router       /api/v1/admin/health-check/stream [get]
func (h *HealthCheckHandler) StreamHealthCheckSSE(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("X-Accel-Buffering", "no")

	if h.broadcaster == nil {
		c.SSEvent("message", `{"event_type":"error","error":"broadcaster not initialized"}`)
		return
	}

	c.Stream(func(w io.Writer) bool {
		select {
		case <-c.Request.Context().Done():
			return false
		case event, ok := <-h.broadcaster:
			if !ok {
				return false
			}
			data, err := json.Marshal(event)
			if err != nil {
				return true
			}
			c.SSEvent("message", string(data))
			return true
		}
	})
}

// GetSessions lists past health check sessions ordered by start time descending.
//
// @Summary      List health check sessions
// @Description  Returns historical health check sessions with aggregate metrics.
// @Tags         HealthCheck
// @Produce      json
// @Security     BearerAuth
// @Success      200  {array}   dto.HealthCheckSessionSummary
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/health-check/sessions [get]
func (h *HealthCheckHandler) GetSessions(c *gin.Context) {
	rows, err := h.db.Query(c.Request.Context(), `
		SELECT id::text, trigger_type, status, total_pools, total_tasks, passed_count, failed_count, avg_latency_ms, started_at, completed_at
		FROM health_check_sessions
		ORDER BY started_at DESC
		LIMIT 50
	`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to list sessions", Details: err.Error()})
		return
	}
	defer rows.Close()

	var sessions []jobs.HealthCheckSessionSummary
	for rows.Next() {
		var s jobs.HealthCheckSessionSummary
		if err := rows.Scan(
			&s.ID, &s.TriggerType, &s.Status, &s.TotalPools, &s.TotalTasks,
			&s.PassedCount, &s.FailedCount, &s.AvgLatencyMS, &s.StartedAt, &s.CompletedAt,
		); err == nil {
			sessions = append(sessions, s)
		}
	}
	if sessions == nil {
		sessions = []jobs.HealthCheckSessionSummary{}
	}

	c.JSON(http.StatusOK, sessions)
}

// GetSessionDetails fetches all individual probe results for a given session.
//
// @Summary      Get health check session details
// @Description  Returns itemized model-by-model probe results for a specific session ID.
// @Tags         HealthCheck
// @Produce      json
// @Security     BearerAuth
// @Param        id   path      string  true  "Session UUID"
// @Success      200  {array}   jobs.HealthCheckResultItem
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/health-check/sessions/{id} [get]
func (h *HealthCheckHandler) GetSessionDetails(c *gin.Context) {
	sessionID := c.Param("id")
	rows, err := h.db.Query(c.Request.Context(), `
		SELECT id, session_id::text, pool_id, pool_name, model_pattern, provider_id, credential_id, status_code, is_healthy, latency_ms, error_message, checked_at
		FROM health_check_results
		WHERE session_id = $1::uuid
		ORDER BY id ASC
	`, sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to fetch session details", Details: err.Error()})
		return
	}
	defer rows.Close()

	var results []jobs.HealthCheckResultItem
	for rows.Next() {
		var r jobs.HealthCheckResultItem
		if err := rows.Scan(
			&r.ID, &r.SessionID, &r.PoolID, &r.PoolName, &r.ModelPattern,
			&r.ProviderID, &r.CredentialID, &r.StatusCode, &r.IsHealthy,
			&r.LatencyMS, &r.ErrorMessage, &r.CheckedAt,
		); err == nil {
			results = append(results, r)
		}
	}
	if results == nil {
		results = []jobs.HealthCheckResultItem{}
	}

	c.JSON(http.StatusOK, results)
}
