// Package telemetry provides the async write-behind pipeline for request logs.
// All hot-path operations are non-blocking: json.Marshal and Redis I/O happen
// in background goroutines, never on the request execution thread.
package telemetry

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"github.com/skadraneshghn/clever-ai-gate/internal/database"
	"go.uber.org/zap"
)

// LogEntry represents a single request telemetry record.
// Instances are pooled via logEntryPool to eliminate hot-path allocations.
type LogEntry struct {
	TenantID         string    `json:"tenant_id,omitempty"`
	Model            string    `json:"model"`
	Provider         string    `json:"provider"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	LatencyMs        int       `json:"latency_ms"`
	StatusCode       int       `json:"status_code"`
	ErrorMessage     string    `json:"error_message,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	Prompt           string    `json:"prompt,omitempty"`
	Response         string    `json:"response,omitempty"`
}

// logEntryPool provides zero-allocation LogEntry reuse across requests.
// The proxy handler acquires an entry, fills it, hands it to Emit, and the
// background worker returns it to the pool after committing to Redis/DB.
var logEntryPool = sync.Pool{
	New: func() interface{} { return new(LogEntry) },
}

// AcquireEntry returns a zeroed LogEntry from the pool.
// The caller MUST call pipeline.Emit(entry) — do NOT return it manually.
func AcquireEntry() *LogEntry {
	e := logEntryPool.Get().(*LogEntry)
	// Zero all fields to prevent stale data from previous use
	*e = LogEntry{}
	return e
}

// releaseEntry returns a LogEntry to the pool after the background worker
// has finished using it. Never called by the proxy handler.
func releaseEntry(e *LogEntry) {
	logEntryPool.Put(e)
}

// Pipeline is the async write-behind telemetry system.
// Decouples database + Redis writes from the request hot-path using a buffered
// channel. json.Marshal is never called on the request goroutine.
type Pipeline struct {
	// internalQueue receives *LogEntry pointers directly — no marshal on emit
	internalQueue chan *LogEntry

	// redisBatch accumulates entries for bulk LPUSH (reduces round-trips)
	pool          *pgxpool.Pool
	redisClient   *redis.Client
	logger        *zap.Logger
	batchSize     int
	flushInterval time.Duration
	ctx           context.Context
	cancel        context.CancelFunc

	// redisBatchMu protects redisBatch accumulation between timer ticks
	redisBatchMu sync.Mutex
	redisBatch   []interface{} // pre-serialized JSON byte slices
}

// NewPipeline creates a new async telemetry pipeline.
// redisClient may be nil — the pipeline falls back to the local channel.
func NewPipeline(pool *pgxpool.Pool, logger *zap.Logger, queueSize, batchSize int, flushInterval time.Duration, redisClient *redis.Client) *Pipeline {
	ctx, cancel := context.WithCancel(context.Background())
	return &Pipeline{
		internalQueue: make(chan *LogEntry, queueSize),
		pool:          pool,
		redisClient:   redisClient,
		logger:        logger,
		batchSize:     batchSize,
		flushInterval: flushInterval,
		ctx:           ctx,
		cancel:        cancel,
		redisBatch:    make([]interface{}, 0, 50),
	}
}

// Start launches the background worker goroutines.
func (p *Pipeline) Start() {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				p.logger.Error("recovered from telemetry worker panic", zap.Any("panic", r))
				go p.worker()
			}
		}()
		p.worker()
	}()

	p.logger.Info("telemetry pipeline started",
		zap.Int("queue_size", cap(p.internalQueue)),
		zap.Int("batch_size", p.batchSize),
		zap.Duration("flush_interval", p.flushInterval),
		zap.Bool("using_redis", p.redisClient != nil),
	)
}

// Emit queues a log entry for async processing.
// This is ZERO-BLOCKING: it does a non-blocking channel send and returns
// immediately. json.Marshal happens in the background worker.
// The caller must use AcquireEntry() to get the entry, and must NOT access
// the entry after calling Emit (the pipeline owns it from this point).
func (p *Pipeline) Emit(entry *LogEntry) {
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now()
	}
	select {
	case p.internalQueue <- entry:
		// Sent successfully to background worker
	default:
		// Queue full — drop entry rather than blocking proxy
		p.logger.Debug("telemetry queue full, dropping entry")
		releaseEntry(entry)
	}
}

// Stop gracefully shuts down the pipeline, flushing remaining entries.
func (p *Pipeline) Stop() {
	p.cancel()
	close(p.internalQueue)
}

// worker is the background goroutine. It receives *LogEntry pointers,
// marshals to JSON (off hot-path), batches for Redis LPUSH, and flushes to DB.
func (p *Pipeline) worker() {
	dbBatch := make([]*LogEntry, 0, p.batchSize)
	ticker := time.NewTicker(p.flushInterval)
	defer ticker.Stop()

	// Redis batch flush ticker (more frequent to keep Redis queue fresh)
	redisTicker := time.NewTicker(5 * time.Millisecond)
	defer redisTicker.Stop()

	for {
		select {
		case entry, ok := <-p.internalQueue:
			if !ok {
				// Channel closed — flush remaining and exit
				if len(dbBatch) > 0 {
					p.flush(dbBatch)
				}
				p.flushRedisBatch()
				return
			}

			// Serialize and push to Redis batch buffer (off hot-path)
			if p.redisClient != nil {
				data, err := json.Marshal(entry)
				if err == nil {
					p.redisBatchMu.Lock()
					p.redisBatch = append(p.redisBatch, data)
					shouldFlush := len(p.redisBatch) >= 50
					p.redisBatchMu.Unlock()
					if shouldFlush {
						p.flushRedisBatch()
					}
				}
				// Release entry immediately after marshal — pipeline is done with it
				releaseEntry(entry)
			} else {
				// No Redis: accumulate for direct DB flush
				dbBatch = append(dbBatch, entry)
				if len(dbBatch) >= p.batchSize {
					p.flush(dbBatch)
					dbBatch = dbBatch[:0]
				}
			}

		case <-redisTicker.C:
			// Periodic micro-batch flush for Redis
			p.redisBatchMu.Lock()
			if len(p.redisBatch) > 0 {
				p.redisBatchMu.Unlock()
				p.flushRedisBatch()
			} else {
				p.redisBatchMu.Unlock()
			}

		case <-ticker.C:
			// Periodic DB batch flush (for non-Redis path)
			if p.redisClient == nil && len(dbBatch) > 0 {
				p.flush(dbBatch)
				dbBatch = dbBatch[:0]
			}

		case <-p.ctx.Done():
			if len(dbBatch) > 0 {
				p.flush(dbBatch)
			}
			p.flushRedisBatch()
			return
		}
	}
}

// flushRedisBatch sends accumulated entries to Redis in a single LPUSH call.
// One network round-trip for up to 50 entries instead of 50 separate calls.
func (p *Pipeline) flushRedisBatch() {
	p.redisBatchMu.Lock()
	if len(p.redisBatch) == 0 {
		p.redisBatchMu.Unlock()
		return
	}
	// Swap batch under lock, release lock before network I/O
	batch := p.redisBatch
	p.redisBatch = make([]interface{}, 0, 50)
	p.redisBatchMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	if err := p.redisClient.LPush(ctx, "clever_ai_gate:telemetry_queue", batch...).Err(); err != nil {
		p.logger.Warn("redis batch lpush failed", zap.Error(err), zap.Int("count", len(batch)))
	}
}

// redisWorker pulls entries from Redis and flushes them in batches to Postgres.
// Started as a separate goroutine when Redis is available.
func (p *Pipeline) StartRedisConsumer() {
	if p.redisClient == nil {
		return
	}
	go func() {
		defer func() {
			if r := recover(); r != nil {
				p.logger.Error("recovered from redis consumer panic", zap.Any("panic", r))
				go p.StartRedisConsumer()
			}
		}()
		p.redisConsumer()
	}()
	p.logger.Info("redis telemetry consumer started")
}

func (p *Pipeline) redisConsumer() {
	batch := make([]*LogEntry, 0, p.batchSize)
	ticker := time.NewTicker(p.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			if len(batch) > 0 {
				p.flush(batch)
				for _, e := range batch {
					releaseEntry(e)
				}
			}
			return
		default:
		}

		// BRPop blocks up to 1s waiting for entries
		res, err := p.redisClient.BRPop(p.ctx, 1*time.Second, "clever_ai_gate:telemetry_queue").Result()
		if err == nil && len(res) == 2 {
			entry := AcquireEntry()
			if err := json.Unmarshal([]byte(res[1]), entry); err == nil {
				batch = append(batch, entry)
				if len(batch) >= p.batchSize {
					p.flush(batch)
					for _, e := range batch {
						releaseEntry(e)
					}
					batch = batch[:0]
				}
			} else {
				releaseEntry(entry)
			}
		}

		select {
		case <-ticker.C:
			if len(batch) > 0 {
				p.flush(batch)
				for _, e := range batch {
					releaseEntry(e)
				}
				batch = batch[:0]
			}
		default:
		}
	}
}

// flush performs a bulk INSERT of the accumulated batch into request_logs.
func (p *Pipeline) flush(batch []*LogEntry) {
	if len(batch) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	tx, err := p.pool.Begin(ctx)
	if err != nil {
		p.logger.Error("failed to begin telemetry transaction", zap.Error(err))
		return
	}
	defer tx.Rollback(ctx)

	for _, entry := range batch {
		var logID int64
		var tenantID interface{}
		if entry.TenantID != "" {
			if _, err := uuid.Parse(entry.TenantID); err == nil {
				tenantID = entry.TenantID
			}
		}

		var errMsg interface{}
		if entry.ErrorMessage != "" {
			errMsg = entry.ErrorMessage
		}

		err := tx.QueryRow(ctx, `
			INSERT INTO request_logs (tenant_id, model, provider, prompt_tokens, completion_tokens, latency_ms, status_code, error_message, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
			RETURNING id
		`, tenantID, entry.Model, entry.Provider, entry.PromptTokens, entry.CompletionTokens, entry.LatencyMs, entry.StatusCode, errMsg, entry.CreatedAt).Scan(&logID)
		if err != nil {
			p.logger.Error("failed to insert request log", zap.Error(err))
			continue
		}

		// Calculate deterministic embedding and persist vector log
		embedding := database.GenerateEmbedding(entry.Prompt)

		// Convert []float32 to vector format for postgres pgvector (e.g. "[0.12,0.34,...]")
		var sb strings.Builder
		sb.WriteString("[")
		for i, val := range embedding {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(strconv.FormatFloat(float64(val), 'f', -1, 32))
		}
		sb.WriteString("]")
		vectorStr := sb.String()

		_, err = tx.Exec(ctx, `
			INSERT INTO request_vector_logs (log_id, prompt_text, response_text, prompt_embedding)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (log_id) DO NOTHING
		`, logID, entry.Prompt, entry.Response, vectorStr)
		if err != nil {
			p.logger.Error("failed to insert vector log", zap.Error(err))
		}
	}

	if err := tx.Commit(ctx); err != nil {
		p.logger.Error("failed to commit telemetry transaction", zap.Error(err))
		return
	}

	p.logger.Debug("telemetry batch flushed", zap.Int("count", len(batch)))
}
