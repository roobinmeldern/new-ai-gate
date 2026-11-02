package proxy

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/cluster"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/telemetry"
	"go.uber.org/zap"
)

// metadataScanLimit defines the maximum number of bytes to scan for routing
// metadata (model, stream flag) in the request body. Bodies larger than this
// are assumed to contain heavy multi-modal payloads (Base64 images from IDE
// extensions like Cline). Only the leading segment is parsed for routing;
// the full body is piped directly to the upstream without being scanned.
const metadataScanLimit = 2 * 1024 * 1024 // 2MB (initial bounded scan for routing fields)

// Handler is the main proxy handler for AI provider requests.
// It sits on the hot-path and is designed for zero heap allocations
// under normal operation using sync.Pool for buffer reuse.
type Handler struct {
	client        *http.Client
	cache         *cache.Store
	redisCacheMgr *cache.RedisCacheManager // nil when Redis not configured; all methods nil-safe
	pipeline      *telemetry.Pipeline
	broadcaster   *cluster.Broadcaster // nil when Redis not configured; safe no-op
	logger        *zap.Logger
	bufPool       sync.Pool
	rewriter      *Rewriter
	stream        *StreamProxy
	AlertManager  *telemetry.AlertManager
	db            *pgxpool.Pool
	syncManager   *credentials.SyncManager
}

// SetDB attaches the database pool for fallback queries (e.g. ListModels on cache miss).
func (h *Handler) SetDB(db *pgxpool.Pool) {
	h.db = db
}

// SetSyncManager attaches the sync manager for dynamic cross-provider fallback discovery.
func (h *Handler) SetSyncManager(sm *credentials.SyncManager) {
	h.syncManager = sm
}

// NewHandler creates the proxy handler with all its dependencies.
// broadcaster and redisCacheMgr may be nil — all methods are nil-safe no-ops.
func NewHandler(client *http.Client, cacheStore *cache.Store, redisCacheMgr *cache.RedisCacheManager, logger *zap.Logger, pipeline *telemetry.Pipeline, broadcaster *cluster.Broadcaster, alertManager *telemetry.AlertManager) *Handler {
	h := &Handler{
		client:        client,
		cache:         cacheStore,
		redisCacheMgr: redisCacheMgr,
		pipeline:      pipeline,
		broadcaster:   broadcaster,
		logger:        logger,
		bufPool: sync.Pool{
			New: func() interface{} {
				return bytes.NewBuffer(make([]byte, 0, 32*1024)) // 32KB initial scratch
			},
		},
		rewriter:     NewRewriter(),
		AlertManager: alertManager,
	}
	h.stream = NewStreamProxy(client, logger)
	return h
}

// proxyContext carries extracted fields through the hot-path without allocations.
type proxyContext struct {
	model          string
	isStream       bool
	isNvidia       bool   // True when model uses nvidia/ prefix — triggers reasoning injection
	isOneMinAI     bool   // True when model uses 1min/ prefix — triggers body/response translation
	isCloudflare   bool   // True when model uses cloudflare/ prefix — triggers prefix stripping
	isSarvam       bool   // True when model uses sarvam/ prefix — triggers prefix stripping
	isPuter        bool   // True when model uses puter/ prefix — triggers prefix stripping
	isAgentRouter  bool   // True when model uses agentrouter/ prefix — triggers sanitization
	isZenMux       bool   // True when model uses zenmux/ prefix — triggers prefix stripping
	isGemini       bool   // True when model routes to the Gemini AI Studio pipeline (gemini/ prefix or standalone gemini-* family) — triggers full body transpilation
	studioProvider string // Resolved routing label (ProviderGeminiStudio) for gemini requests; "" otherwise. Diagnostic only — the transpiler gate is cred.Provider == ProviderGemini
	requestedModel string // The original model requested by the client before prefix stripping
	isJiekou       bool   // True when model uses jiekou/ prefix — triggers body model rewrite + parameter clamping
	rawBody        []byte // Pristine request body as received before provider-specific modifications
	body           []byte
	credential     *credentials.AcquireResult
	pool           *credentials.BalancedChannelPool
	tenantID       string

	// Anthropic front door state (Phase 2). anthropicFront is set when the
	// request entered through /v1/messages, which changes two things:
	//   - the client expects Anthropic-shaped errors and responses;
	//   - native Anthropic-protocol credentials (anthropic/agentrouter) can
	//     receive the request with zero translation.
	anthropicFront       bool   // request came in through the Anthropic front door
	anthropicRawBody     []byte // original Anthropic body (input to native prep)
	anthropicTranslated  []byte // cached canonical OpenAI body for non-native providers (nil until needed)
	anthropicInputTokens int    // estimated input tokens for stream encoder and response translation
	promptText           string // pre-extracted prompt text for telemetry ("" = extract lazily)

	// Parallel hedging execution fields:
	coordinator    *parallelCoordinator
	workerIndex    int
	requestContext context.Context
}

// Handle processes incoming AI requests on the hot-path.
//
// Hot-path sequence (zero database calls, zero mutex locks):
//  1. Read body into sync.Pool scratch buffer
//  2. jsonparser.Get — extract model identifier (zero alloc)
//  3. jsonparser.GetBoolean — detect streaming mode
//  4. Cache lookup: model → BalancedChannelPool
//  5. AcquireActiveToken — lock-free atomic credential selection
//  6. Rewrite request headers + URL for target provider
//  7. Forward to upstream (stream or direct)
//  8. On failure: penalize credential, retry with next
//
// @Summary      Proxy AI request
// @Description  Routes AI requests to upstream providers with automatic failover
// @Tags         Proxy
// @Accept       json
// @Produce      json
// @Param        Authorization  header  string  true  "Bearer virtual-api-key"
// @Param        body           body    object  true  "OpenAI-compatible request body"
// @Router       /v1/{path} [post]
func (h *Handler) Handle(c *gin.Context) {
	requestStart := time.Now()
	// Step 1: Read body into pooled buffer — zero heap allocation
	buf := h.bufPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		h.bufPool.Put(buf)
	}()

	// Anthropic Messages API front door: requests to /v1/messages or
	// /v1/messages/count_tokens are served by the Anthropic-compatible front
	// API (anthropic_front.go). Every other path continues through the
	// OpenAI-compatible pipeline below, completely untouched.
	anthropicFront := isAnthropicFrontPath(c.Request.URL.Path)

	if _, err := io.Copy(buf, c.Request.Body); err != nil {
		if anthropicFront {
			AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "failed to read request body")
		} else {
			c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		}
		return
	}
	body := buf.Bytes()
	rawBodyCopy := make([]byte, len(body))
	copy(rawBodyCopy, body)

	if anthropicFront {
		if strings.HasSuffix(c.Request.URL.Path, "count_tokens") {
			h.handleAnthropicCountTokens(c, rawBodyCopy)
		} else {
			h.handleAnthropicMessages(c, rawBodyCopy)
		}
		return
	}

	// Step 2: Extract the "model" field for routing.
	//
	// Two code paths depending on Content-Type:
	//   - JSON (default): zero-alloc bounded jsonparser scan
	//   - multipart/form-data: byte-level search for the model form field
	//     (used by /v1/audio/transcriptions, /v1/images/edits, /v1/files, etc.)
	contentType := c.Request.Header.Get("Content-Type")
	isMultipart := strings.HasPrefix(contentType, "multipart/")

	var model string
	var scanSlice []byte
	var isStream bool

	if isMultipart {
		// Multipart model extraction: search the raw body for the form field
		// named "model". In multipart encoding this appears as:
		//   Content-Disposition: form-data; name="model"\r\n\r\nmodel-value\r\n
		// We do a fast byte scan to locate the value without parsing the
		// entire multipart structure (which would require heap allocations).
		model = extractModelFromMultipart(body)
		if model == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing 'model' field in multipart form data"})
			return
		}
		// Multipart requests are never streaming (audio, image, file uploads)
		isStream = false
	} else {
		// JSON path: bounded metadata scan (existing zero-alloc logic)
		//
		// Gap 2 Fix — Bounded Metadata Scanner:
		// When the body exceeds metadataScanLimit (256KB), the payload likely contains
		// heavy Base64 image data from IDE extensions (Cline sending screenshots, etc.).
		// Scanning the entire multi-megabyte body for the "model" and "stream" fields
		// would trigger heap allocations and GC pauses.
		//
		// Instead, we parse only the leading metadata segment. JSON objects place
		// structural keys (model, stream, messages) before binary content payloads,
		// so the routing fields are always in the first few hundred bytes.
		scanSlice = body
		if len(body) > metadataScanLimit {
			scanSlice = body[:metadataScanLimit]
			h.logger.Debug("large payload detected, using bounded metadata scan first",
				zap.Int("body_size", len(body)),
				zap.Int("scan_limit", metadataScanLimit),
			)
		}

		modelBytes, _, _, err := jsonparser.Get(scanSlice, "model")
		if (err != nil || len(modelBytes) == 0) && len(body) > len(scanSlice) {
			// Fallback: search the entire body. Long chat conversations or clients that serialize
			// "messages" before "model" will push "model" beyond the initial scan limit.
			modelBytes, _, _, err = jsonparser.Get(body, "model")
		}
		if err != nil || len(modelBytes) == 0 {
			h.logger.Warn("request rejected: missing or invalid 'model' field",
				zap.Int("body_size", len(body)),
				zap.String("path", c.Request.URL.Path),
				zap.String("client_ip", c.ClientIP()),
				zap.Error(err),
			)
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing or invalid 'model' field"})
			return
		}
		model = string(modelBytes)

		// Step 3: Detect streaming mode (with full body fallback if not found in leading segment)
		var streamErr error
		isStream, streamErr = jsonparser.GetBoolean(scanSlice, "stream")
		if streamErr != nil && len(body) > len(scanSlice) {
			isStream, _ = jsonparser.GetBoolean(body, "stream")
		}
	}

	requestedModel := model

	// --- Endpoint vs Model Modality Guard ---
	// Prevent sending embedding models to /v1/chat/completions (which causes upstream 404
	// and falsely penalizes all healthy pool credentials for 25 minutes).
	reqPath := c.Request.URL.Path
	if strings.Contains(reqPath, "/chat/completions") && isEmbeddingModel(model) {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": gin.H{
				"message": fmt.Sprintf("Model %q is an embedding model and must be called via /v1/embeddings, not /v1/chat/completions", model),
				"type":    "invalid_request_error",
				"code":    "endpoint_mismatch",
			},
		})
		return
	}

	// --- NVIDIA Prefix Detection ---
	// Models prefixed with "nvidia/" are routed through the NVIDIA NIM pipeline.
	// The prefix is stripped for cache lookup but the isNvidia flag triggers
	// reasoning parameter injection before forwarding to the upstream.
	isNvidia := false
	if strings.HasPrefix(model, "nvidia/") {
		isNvidia = true
		// Don't strip the prefix — the model_pattern in the pool already
		// includes "nvidia/" (e.g., pool pattern = "nvidia/nvidia/nemotron-3-super-120b-a12b")
		h.logger.Debug("nvidia model detected",
			zap.String("model", model),
		)
	}

	// --- Ollama Prefix Detection ---
	// Models prefixed with "ollama/" are routed to Ollama instances.
	// The prefix is stripped from the JSON body before forwarding so the
	// upstream Ollama server receives the clean model name (e.g., "llama3:8b").
	isOllama := false
	if strings.HasPrefix(model, "ollama/") {
		isOllama = true
		h.logger.Debug("ollama model detected",
			zap.String("model", model),
		)
	}

	// --- 1min.ai Prefix Detection ---
	// Models prefixed with "1min/" are routed through the 1min.ai Feature API
	// translation engine. The proxy body translator converts the OpenAI-compatible
	// request into 1min.ai's type/model/promptObject format, and the response
	// translator converts the aiRecord envelope back to OpenAI format.
	isOneMinAI := false
	if strings.HasPrefix(model, "1min/") {
		isOneMinAI = true
		h.logger.Debug("1min.ai model detected",
			zap.String("model", model),
		)
	}

	// --- Cloudflare Workers AI Prefix Detection ---
	// Models prefixed with "cloudflare/" are routed to Cloudflare Workers AI.
	// The prefix is stripped from the JSON body before forwarding so the upstream
	// receives the clean model ID (e.g. "@cf/meta/llama-3.1-8b-instruct").
	isCloudflare := false
	if strings.HasPrefix(model, "cloudflare/") {
		isCloudflare = true
		h.logger.Debug("cloudflare workers ai model detected",
			zap.String("model", model),
		)
	}

	// --- Sarvam AI Prefix Detection ---
	// Models prefixed with "sarvam/" are routed to Sarvam AI. The prefix is
	// stripped from the JSON body before forwarding so the upstream Sarvam API
	// receives the clean model name (e.g. "sarvam-105b"). Sarvam is natively
	// OpenAI-compatible, so no body/response translation or transmuxer is needed.
	isSarvam := false
	if strings.HasPrefix(model, "sarvam/") {
		isSarvam = true
		h.logger.Debug("sarvam ai model detected",
			zap.String("model", model),
		)
	}

	// --- Puter Prefix Detection ---
	isPuter := false
	if strings.HasPrefix(model, "puter/") {
		isPuter = true
		h.logger.Debug("puter model detected",
			zap.String("model", model),
		)
	}

	// --- AgentRouter Prefix Detection ---
	// Models prefixed with "agentrouter/" are routed to AgentRouter.org.
	// The prefix is stripped from the JSON body before forwarding so the
	// upstream AgentRouter receives the clean model ID (e.g. "gpt-4o").
	// The request sanitizer (OmniRoute Issue #1921 fix) strips unsupported
	// OpenAI fields that AgentRouter's diverse upstream backends reject.
	isAgentRouter := false
	if strings.HasPrefix(model, "agentrouter/") {
		isAgentRouter = true
		h.logger.Debug("agentrouter model detected",
			zap.String("model", model),
		)
	}

	// --- ZenMux Prefix Detection ---
	isZenMux := false
	if strings.HasPrefix(model, "zenmux/") {
		isZenMux = true
		h.logger.Debug("zenmux model detected",
			zap.String("model", model),
		)
	}

	// --- Jiekou.AI Prefix Detection ---
	// Models prefixed with "jiekou/" are routed to Jiekou.AI, an OpenAI-compatible
	// proxy for Moonshot/Kimi, DeepSeek and other Chinese model providers.
	// The prefix must be stripped from the JSON body before forwarding so the
	// upstream Jiekou API receives the clean model name (e.g. "moonshotai/kimi-k3"
	// instead of "jiekou/moonshotai/kimi-k3"). Temperature clamping is also applied
	// because Moonshot/Kimi enforce [0.0, 1.0] while OpenAI allows up to 2.0.
	isJiekou := false
	if strings.HasPrefix(model, "jiekou/") {
		isJiekou = true
		h.logger.Debug("jiekou.ai model detected",
			zap.String("model", model),
		)
	}

	// --- Request Remapping Engine: Google AI Studio (Gemini) ---
	// Normalize both routing forms to the optimized gemini_studio pipeline so
	// Kilo Code agents and standard chat clients bind to the same credential pool:
	//
	//   Phase A — Slash-prefixed routes (Kilo Code agent style):
	//     "gemini/gemini-3.5-flash"  → prefix token "gemini" matches the legacy
	//     namespace; the prefix is stripped later (see Gemini Payload Preparation)
	//     and the clean model is forwarded to Google AI Studio.
	//
	//   Phase B — Standalone routes (standard chat clients):
	//     "gemini-3.5-flash"        → bound by model-name family. No prefix to
	//     strip; the credential's Provider="gemini" field (resolved from the pool)
	//     activates the transpilation pipeline in forwardRequest.
	//
	// Both forms resolve to the ProviderGeminiStudio routing label. The actual
	// transpilation gate remains cred.Provider == ProviderGemini, which is the
	// identity stored on every AI Studio key — so gemini_studio is the logical
	// routing target and gemini is the credential identity.
	//
	// Other namespace prefixes (nvidia/, ollama/, cloudflare/, sarvam/, puter/,
	// zenmux/, 1min/) are intentionally NOT touched here: each has its own dedicated
	// detection + sanitization block below with prefix-specific invariants
	// (e.g. nvidia keeps its prefix for pool lookup). A generic slash-split would
	// break those handlers, so this interceptor is scoped to the gemini family.
	isGemini := false
	studioProvider := ""
	if strings.HasPrefix(model, credentials.ProviderGeminiLegacy+"/") {
		isGemini = true
		studioProvider = credentials.ProviderGeminiStudio
		h.logger.Debug("gemini studio model detected (slash-prefixed)",
			zap.String("model", model),
			zap.String("studio_provider", studioProvider),
		)
	} else if strings.HasPrefix(model, "gemini-") {
		isGemini = true
		studioProvider = credentials.ProviderGeminiStudio
		h.logger.Debug("gemini studio model detected (standalone)",
			zap.String("model", model),
			zap.String("studio_provider", studioProvider),
		)
	}

	// --- Access log: first thing we know enough to emit a useful Info entry ---
	// This fires for EVERY request and is the primary tool for diagnosing
	// Kilo Code instability: you can see exactly what model/tenant/path was
	// requested and whether it ever reached the retry loop.
	requestID, _ := c.Get("request_id")
	tenantName, _ := c.Get("tenant_name")
	h.logger.Info("proxy request received",
		zap.String("request_id", fmt.Sprintf("%v", requestID)),
		zap.String("tenant", fmt.Sprintf("%v", tenantName)),
		zap.String("model", model),
		zap.String("path", c.Request.URL.Path),
		zap.Bool("stream", isStream),
		zap.Bool("is_nvidia", strings.HasPrefix(model, "nvidia/")),
		zap.Bool("is_ollama", strings.HasPrefix(model, "ollama/")),
		zap.Bool("is_1minai", isOneMinAI),
		zap.Bool("is_cloudflare", isCloudflare),
		zap.Bool("is_sarvam", isSarvam),
		zap.Bool("is_puter", isPuter),
		zap.Bool("is_agentrouter", isAgentRouter),
		zap.Bool("is_zenmux", isZenMux),
		zap.Bool("is_jiekou", isJiekou),
		zap.Bool("is_gemini", isGemini),
		zap.String("studio_provider", studioProvider),
		zap.Int("body_bytes", len(body)),
		zap.String("client_ip", c.ClientIP()),
	)

	// Step 4: Cache lookup for routing pool — ~200ns via Ristretto
	poolVal, found := h.cache.Get(cache.PoolKey(model))
	if !found {
		// Try wildcard/prefix matching for model families
		poolVal, found = h.findPoolByPrefix(model)
		if !found {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "no routing pool configured for model: " + model,
			})
			return
		}
	}
	pool := poolVal.(*credentials.BalancedChannelPool)

	// --- NVIDIA Payload Sanitization ---
	// For NVIDIA-prefixed models two transforms are applied:
	//   1. Strip the "nvidia/" routing prefix from the JSON "model" field so
	//      NVIDIA NIM receives the clean model ID (e.g. "z-ai/glm-5.1").
	//   2. Inject reasoning params ONLY for architectures that support thinking
	//      (Nemotron, DeepSeek-R1, etc.). Standard models like GLM reject these
	//      params with a 500, so we gate them behind supportsNvidiaReasoning.
	if isNvidia {
		// Determine the clean upstream model ID (strip "nvidia/" prefix)
		upstreamModel := strings.TrimPrefix(model, "nvidia/")

		// Fix 2: Conditional reasoning injection — only for supported architectures
		if supportsNvidiaReasoning(upstreamModel) {
			body = injectNvidiaParams(body, body, h.logger)
		} else {
			h.logger.Debug("skipping reasoning injection for standard non-thinking model",
				zap.String("model", upstreamModel),
			)
		}

		// Strip the routing prefix from the raw JSON bytes in-place.
		// stripModelPrefixInPlace modifies the buffer directly, avoiding the
		// full-body heap copy that bytes.Replace would allocate.
		body = stripModelPrefixInPlace(body, model, "nvidia/")
	}

	// --- Ollama Payload Sanitization ---
	// For Ollama-prefixed models the routing prefix must be stripped from the
	// JSON "model" field so the upstream Ollama server receives the clean model
	// name it expects (e.g., "llama3:8b" instead of "ollama/llama3:8b").
	// Uses the same in-place byte-level stripping as NVIDIA — zero heap allocation.
	if isOllama {
		body = stripModelPrefixInPlace(body, model, "ollama/")
	}

	// --- Cloudflare Payload Sanitization ---
	// For Cloudflare-prefixed models the "cloudflare/" routing prefix must be
	// stripped from the JSON "model" field so Cloudflare receives the clean
	// model ID (e.g. "@cf/meta/llama-3.1-8b-instruct" instead of
	// "cloudflare/@cf/meta/llama-3.1-8b-instruct").
	// Uses the same in-place byte-level stripping.
	//
	// Unlike nvidia/ollama/sarvam/puter (which use passthroughPath and never
	// embed the model in the upstream URL), Cloudflare's path transformer
	// builds /ai/run/{model}, so the local model variable MUST be synchronized
	// here — otherwise the stale prefix leaks into the URL and Cloudflare
	// rejects the request with error 7000 ("No route for that URI").
	if isCloudflare {
		body = stripModelPrefixInPlace(body, model, "cloudflare/")
		model = strings.TrimPrefix(model, "cloudflare/")
	}

	// --- Sarvam AI Payload Sanitization ---
	// For Sarvam-prefixed models the "sarvam/" routing prefix must be stripped
	// from the JSON "model" field so the upstream Sarvam API receives the clean
	// model name (e.g. "sarvam-105b"). Uses the same single-pass O(n) byte-level
	// replacement as nvidia/ollama/cloudflare.
	//
	// Note: stripping of OpenAI-only request fields that Sarvam's strict schema
	// rejects (stream_options, logprobs, …) is performed in forwardRequest,
	// gated on cred.Provider == "sarvam", so it covers BOTH the prefixed and
	// the clean-alias routing forms.
	if isSarvam {
		body = stripModelPrefixInPlace(body, model, "sarvam/")
	}

	// --- Puter Proactive Completion Optimization ---
	if isPuter {
		body, model = h.OptimizePayloadForKiloCode(body, model)
	}

	// --- Puter Payload Sanitization ---
	if isPuter {
		body = stripModelPrefixInPlace(body, model, "puter/")
	}

	// --- ZenMux Payload Sanitization ---
	if isZenMux {
		body = stripModelPrefixInPlace(body, model, "zenmux/")
		model = strings.TrimPrefix(model, "zenmux/")
	}

	// --- Jiekou.AI Payload Sanitization ---
	// Strip the "jiekou/" routing prefix from the JSON "model" field so the
	// upstream Jiekou API receives the clean model name it expects.
	// The full parameter sanitization (temperature clamp, unsupported field
	// removal) happens in forwardRequest gated on cred.Provider == "jiekou".
	if isJiekou {
		body = stripModelPrefixInPlace(body, model, "jiekou/")
		model = strings.TrimPrefix(model, "jiekou/")
	}

	// --- Gemini Payload Preparation ---
	// For Gemini-routed models the routing prefix (if present) is stripped from
	// both the model name and the JSON body. The actual body transpilation
	// (OpenAI → Gemini format) is deferred to forwardRequest where we have access
	// to the credential and can handle errors cleanly within the retry loop.
	// stripModelPrefixInPlace is a no-op when the model carries no prefix, so the
	// standalone form ("gemini-3.5-flash") passes through here unchanged.
	if isGemini {
		body = stripModelPrefixInPlace(body, model, credentials.ProviderGeminiLegacy+"/")
		model = strings.TrimPrefix(model, credentials.ProviderGeminiLegacy+"/")
	}

	// --- AgentRouter Payload Sanitization ---
	// For AgentRouter-prefixed models the "agentrouter/" routing prefix must be
	// stripped from the JSON "model" field so AgentRouter receives the clean
	// model ID (e.g. "gpt-4o" instead of "agentrouter/gpt-4o").
	// Uses the same single-pass O(n) byte-level replacement.
	// Note: The deep request sanitization (stripping unsupported OpenAI fields,
	// normalizing developer role, etc.) is performed in forwardRequest, gated on
	// cred.Provider == "agentrouter", so it covers BOTH the prefixed and the
	// clean-alias routing forms.
	if isAgentRouter {
		upstreamModel := strings.TrimPrefix(model, "agentrouter/")
		oldToken := []byte(`"` + model + `"`)
		newToken := []byte(`"` + upstreamModel + `"`)
		body = bytes.Replace(body, oldToken, newToken, 1)
	}

	// Step 5-8: Attempt with automatic failover
	// Note: The FULL body (not scanSlice) is forwarded to the upstream provider.
	pctx := &proxyContext{
		model:          model,
		isStream:       isStream,
		isNvidia:       isNvidia,
		isOneMinAI:     isOneMinAI,
		isCloudflare:   isCloudflare,
		isSarvam:       isSarvam,
		isPuter:        isPuter,
		isAgentRouter:  isAgentRouter,
		isZenMux:       isZenMux,
		isGemini:       isGemini,
		isJiekou:       isJiekou,
		studioProvider: studioProvider,
		requestedModel: requestedModel,
		rawBody:        rawBodyCopy,
		body:           body,
		pool:           pool,
	}

	// Retrieve tenant ID from context (set by auth middleware)
	if tenantID, exists := c.Get("tenant_id"); exists {
		pctx.tenantID = tenantID.(string)
	}

	// Try every credential in the pool before giving up.
	// Cap at 100 to avoid runaway loops on huge pools.
	maxAttempts := int(pool.TotalCount)
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	if maxAttempts > 100 {
		maxAttempts = 100
	}
	h.executeWithRetry(c, pctx, requestStart, maxAttempts)
}

// attemptRecord captures the outcome of a single credential attempt within
// the retry loop. A slice of these is built during pool exhaustion and used
// to construct a detailed diagnostic summary in the final OpenAI error envelope.
type attemptRecord struct {
	provider     string
	statusCode   int
	credID       int
	transient400 bool // 400 was an upstream-side failure, not a payload schema error
}

// parallelCoordinator synchronizes multiple parallel key evaluation goroutines.
// It guarantees that at most one worker claims victory and writes to the client connection,
// while canceling non-winning workers.
type parallelCoordinator struct {
	winnerOnce  sync.Once
	winnerIndex int
	hasWinner   atomic.Bool
	cancelFuncs []context.CancelFunc
	cancelMu    sync.Mutex
}

func (pc *parallelCoordinator) cancelOthers(winnerIdx int) {
	pc.cancelMu.Lock()
	defer pc.cancelMu.Unlock()
	for i, cancel := range pc.cancelFuncs {
		if i != winnerIdx && cancel != nil {
			cancel()
		}
	}
}

func (pc *parallelCoordinator) cancelAll() {
	pc.cancelMu.Lock()
	defer pc.cancelMu.Unlock()
	for _, cancel := range pc.cancelFuncs {
		if cancel != nil {
			cancel()
		}
	}
}

func (pc *parallelCoordinator) claimVictory(idx int) bool {
	won := false
	pc.winnerOnce.Do(func() {
		pc.winnerIndex = idx
		pc.hasWinner.Store(true)
		won = true
	})
	if won {
		pc.cancelOthers(idx)
	}
	return won
}

type workerResult struct {
	workerIndex int
	result      *credentials.AcquireResult
	statusCode  int
	upstreamURL string
	errBody     []byte
	err         error
}

// retryRoundsPerPool is the number of complete passes over a pool's
// credentials that executeWithRetry performs before giving up on the pool and
// switching to the exact same model on another provider. Round 1 is the
// initial pass; rounds 2+ retry every credential of the pool again (cooldowns
// permitting). The cross-provider fallback only fires after ALL rounds fail.
//
// Package-level var (not const) so tests can pin fast settings.
var retryRoundsPerPool = 3

// roundTransitionWaitBudget caps how long executeWithRetry waits, between two
// retry rounds on the same pool, for the soonest failed credential's cooldown
// to expire. Transient 5xx cooldowns are 2-5s so a full fresh-key round is
// always affordable; non-retryable rejections (auth/quota/rate-limit) skip
// rounds entirely via isNonRetryableCredentialStatus.
//
// Package-level var (not const) so tests can pin fast settings.
var roundTransitionWaitBudget = 6 * time.Second

// cooldownAcquireWaitBudget caps how long a FRESH request (round 1, including
// the first round on a fallback pool) waits when every credential of the pool
// is already cooling down from earlier failures. Cooldowns longer than this
// budget mean the keys were hard-rejected (suspended account, exhausted quota,
// rate-limit window) — sleeping and probing them again inside this request
// would only stall the client, so the gateway fails over to the exact-model
// fallback (or a meaningful exhaustion error) immediately. Brief penalties
// (≤ budget) are still waited out; retry rounds 2+ are unaffected because
// re-probing their own transient failures is the point of rounds.
//
// Package-level var (not const) so tests can pin fast settings.
var cooldownAcquireWaitBudget = 600 * time.Millisecond

// isNonRetryableCredentialStatus reports whether an upstream status means the
// CREDENTIAL itself was rejected — auth failure, quota/billing exhaustion,
// rate limit, or an invalid payload. Re-trying the same credential inside the
// same request cannot resolve these, so remaining retry rounds are skipped
// and the gateway switches to the exact same model on another provider
// instead of stalling the client on dead keys.
func isNonRetryableCredentialStatus(status int) bool {
	return status == http.StatusBadRequest ||
		status == http.StatusTooManyRequests ||
		isCredentialAuthError(status)
}

// poolSoonestCooldownRemaining returns how much longer the pool's
// soonest-available credential still needs before its cooldown expires.
// Zero means a credential is already fresh. Retry rounds use this to pace
// themselves: each new round starts once the first failed key has cooled down.
func poolSoonestCooldownRemaining(pool *credentials.BalancedChannelPool) time.Duration {
	if pool == nil {
		return 0
	}
	result := pool.AcquireLeastPenalizedToken()
	if result == nil {
		return 0
	}
	remaining := time.Duration(atomic.LoadInt64(&result.Credential.CooldownUntil) - time.Now().UnixNano())
	if remaining < 0 {
		return 0
	}
	return remaining
}

// parallelBatchSize is the number of provider keys evaluated concurrently in each wave.
const parallelBatchSize = 5

func (h *Handler) emitSuccessTelemetry(
	c *gin.Context,
	pctx *proxyContext,
	result *credentials.AcquireResult,
	statusCode int,
	upstreamURL string,
	body []byte,
	requestStart time.Time,
	triedCount int,
) {
	atomic.StoreUint32(&result.Credential.ConsecutiveFailures, 0)
	result.FromPool.ResetCooldown(result.Index)
	h.logger.Info("proxy request completed",
		zap.String("model", pctx.model),
		zap.String("provider", result.Credential.Provider),
		zap.String("upstream_url", upstreamURL),
		zap.String("tenant_id", pctx.tenantID),
		zap.Bool("stream", pctx.isStream),
		zap.Int("status", statusCode),
		zap.Int("unique_tried", triedCount),
		zap.Duration("elapsed", time.Since(requestStart)),
	)

	if h.pipeline == nil {
		return
	}

	promptText := pctx.promptText
	if promptText == "" {
		promptText = extractPromptText(pctx.body)
	}
	var responseText string
	var completionTokens int
	var promptTokens int

	if pctx.isStream && statusCode == http.StatusOK {
		responseText, _ = jsonparser.GetString(body, "text")
		if tok, err := jsonparser.GetInt(body, "tokens"); err == nil {
			completionTokens = int(tok)
		}
	} else {
		responseText = extractResponseText(body)
		completionTokens = extractTokens(body, "completion")
	}

	promptTokens = extractTokens(pctx.body, "prompt")
	if promptTokens == 0 {
		if pctx.anthropicInputTokens > 0 {
			promptTokens = pctx.anthropicInputTokens
		} else {
			promptTokens = len(promptText) / 4
		}
	}
	if completionTokens == 0 && responseText != "" {
		completionTokens = len(responseText) / 4
	}

	entry := telemetry.AcquireEntry()
	entry.TenantID = pctx.tenantID
	entry.Model = pctx.model
	entry.Provider = result.Credential.Provider
	entry.PromptTokens = promptTokens
	entry.CompletionTokens = completionTokens
	entry.LatencyMs = int(time.Since(requestStart).Milliseconds())
	entry.StatusCode = statusCode
	entry.CreatedAt = time.Now()
	entry.Prompt = promptText
	entry.Response = responseText
	h.pipeline.Emit(entry)
}

func (h *Handler) handleCredentialFailure(
	ctx context.Context,
	pctx *proxyContext,
	result *credentials.AcquireResult,
	statusCode int,
	upstreamURL string,
	errBody []byte,
	err error,
	attempts *[]attemptRecord,
	requestID any,
	triedCount int,
	maxAttempts int,
	requestStart time.Time,
	strippedSamplingParams map[string]bool,
) (recStatus int, isFastFail404 bool, isFastFail400 bool) {
	recStatus = statusCode
	if err != nil {
		recStatus = http.StatusInternalServerError
	}
	*attempts = append(*attempts, attemptRecord{
		provider:     result.Credential.Provider,
		statusCode:   recStatus,
		credID:       result.Credential.ID,
		transient400: recStatus == http.StatusBadRequest && isTransientUpstream400(errBody),
	})

	atomic.AddUint32(&result.Credential.ConsecutiveFailures, 1)

	if h.AlertManager != nil {
		errStr := "request failed"
		if err != nil {
			errStr = err.Error()
		} else if len(errBody) > 0 {
			if len(errBody) > 100 {
				errStr = string(errBody[:100]) + "..."
			} else {
				errStr = string(errBody)
			}
		}
		_ = h.AlertManager.TrackFailover(ctx, result.Credential.ID, pctx.model, errStr)
	}

	cooldownDuration := cooldownForStatus(recStatus)

	isDepletedAccount := false
	if recStatus == http.StatusPaymentRequired {
		isDepletedAccount = true
	} else if len(errBody) > 0 {
		bodyLower := strings.ToLower(string(errBody))
		if strings.Contains(bodyLower, "not_enough_balance") ||
			strings.Contains(bodyLower, "insufficient_balance") ||
			strings.Contains(bodyLower, "insufficient_funds") ||
			strings.Contains(bodyLower, "account suspended") ||
			strings.Contains(bodyLower, "account_suspended") ||
			strings.Contains(bodyLower, "quota_exceeded") ||
			strings.Contains(bodyLower, "credit limit") ||
			strings.Contains(bodyLower, "usage-limited-chat") ||
			strings.Contains(bodyLower, "error_400_from_delegate") ||
			strings.Contains(bodyLower, "not enough credits") ||
			strings.Contains(bodyLower, "insufficient credits") ||
			strings.Contains(bodyLower, "insufficient_quota") ||
			strings.Contains(bodyLower, "insufficient_credits") ||
			strings.Contains(bodyLower, "agent_low_balance") ||
			strings.Contains(bodyLower, "enter.pollinations.ai") ||
			strings.Contains(bodyLower, "upstream_error_in_success_body") {
			isDepletedAccount = true
		}
	}

	isSingleKey := pctx.pool != nil && pctx.pool.TotalCount == 1

	if isCloudflareModelAgreementError(errBody) {
		cooldownDuration = 1 * time.Second
		isDepletedAccount = false
		accountID := strings.TrimPrefix(result.Credential.BaseURL, "cloudflare:")
		_ = autoAcceptCloudflareModelAgreement(ctx, h.client, accountID, result.Credential.APIKey, pctx.model)
	} else if isDepletedAccount {
		if isSingleKey {
			cooldownDuration = 15 * time.Second
		} else {
			cooldownDuration = 30 * time.Minute
		}
		h.logger.Warn("credential account balance exhausted or rate limited — applying cooldown",
			zap.String("model", pctx.model),
			zap.String("provider", result.Credential.Provider),
			zap.Int("credential_id", result.Credential.ID),
			zap.Duration("cooldown", cooldownDuration),
			zap.ByteString("error_body", errBody),
		)
	} else if result.Credential.Provider == "puter" {
		isQuotaError := recStatus == http.StatusPaymentRequired ||
			recStatus == http.StatusForbidden ||
			recStatus == http.StatusTooManyRequests
		if isQuotaError {
			if isSingleKey {
				cooldownDuration = 15 * time.Second
			} else {
				cooldownDuration = 30 * time.Minute
			}
		}
	}

	isPuterQuota400 := false
	if result.Credential.Provider == "puter" && recStatus == http.StatusBadRequest && len(errBody) > 0 {
		bodyStr := string(errBody)
		if strings.Contains(bodyStr, "usage-limited-chat") ||
			strings.Contains(bodyStr, "insufficient_funds") ||
			strings.Contains(bodyStr, "error_400_from_delegate") ||
			strings.Contains(bodyStr, "Permission denied") {
			isPuterQuota400 = true
		}
	}

	if isSingleKey && cooldownDuration > 15*time.Second {
		cooldownDuration = 15 * time.Second
	}

	if isCredentialAuthError(recStatus) || isPuterQuota400 || isDepletedAccount {
		result.FromPool.PenalizeToken(result.Index, cooldownDuration)
		h.broadcaster.PublishPenalize(result.FromPool.ModelPattern, result.Credential.ID, result.Index, time.Now().Add(cooldownDuration))
		h.logger.Warn("credential rejected by upstream (quota/auth/balance limit) — rotating key",
			zap.String("request_id", fmt.Sprintf("%v", requestID)),
			zap.String("model", pctx.model),
			zap.String("provider", result.Credential.Provider),
			zap.String("upstream_url", upstreamURL),
			zap.Int("status", recStatus),
			zap.Int("credential_id", result.Credential.ID),
			zap.Int("keys_tried", triedCount),
			zap.Duration("cooldown", cooldownDuration),
		)
		return recStatus, false, false
	}

	if recStatus == http.StatusNotFound {
		h.logger.Warn("upstream returned 404 not found — aborting retry loop to protect pool keys",
			zap.String("model", pctx.model),
			zap.String("provider", result.Credential.Provider),
			zap.ByteString("error_body", errBody),
		)
		return recStatus, true, false
	}

	if recStatus == http.StatusBadRequest {
		provider400Count := 0
		for _, a := range *attempts {
			if a.provider == result.Credential.Provider && a.statusCode == http.StatusBadRequest && !a.transient400 {
				provider400Count++
			}
		}
		if provider400Count >= 2 && !isTransientUpstream400(errBody) {
			h.logger.Error("payload schema error — aborting rotation to protect remaining pool keys",
				zap.String("model", pctx.model),
				zap.String("provider", result.Credential.Provider),
				zap.Int("status", recStatus),
				zap.Int("provider_400_count", provider400Count),
				zap.ByteString("error_body", errBody),
			)
			return recStatus, false, true
		}
	}

	if isSingleKey {
		cooldownDuration = 300 * time.Millisecond
	}
	result.FromPool.PenalizeToken(result.Index, cooldownDuration)
	h.broadcaster.PublishPenalize(result.FromPool.ModelPattern, result.Credential.ID, result.Index, time.Now().Add(cooldownDuration))
	h.logger.Warn("upstream request failed — rotating to next key",
		zap.String("request_id", fmt.Sprintf("%v", requestID)),
		zap.String("model", pctx.model),
		zap.String("provider", result.Credential.Provider),
		zap.String("upstream_url", upstreamURL),
		zap.String("tenant_id", pctx.tenantID),
		zap.Int("status", recStatus),
		zap.Int("max_attempts", maxAttempts),
		zap.Int("keys_tried", triedCount),
		zap.Duration("cooldown", cooldownDuration),
		zap.Duration("elapsed", time.Since(requestStart)),
		zap.NamedError("transport_err", err),
	)

	return recStatus, false, false
}

// executeWithRetry attempts the proxy request, evaluating provider credentials
// in parallel waves before falling back to another provider hosting the exact same model.
func (h *Handler) executeWithRetry(c *gin.Context, pctx *proxyContext, requestStart time.Time, maxAttempts int) {
	requestID, _ := c.Get("request_id")
	var lastErrBody []byte
	var lastStatus int
	var lastProvider string
	var attempts []attemptRecord
	triedIndices := make(map[int]bool)              // deduplicate — never retry the same index twice per round
	strippedSamplingParams := make(map[string]bool) // self-heal guard — strip each rejected sampling param at most once per request
	triedCount := 0
	triedPools := make(map[string]bool)
	if pctx.pool != nil {
		triedPools[pctx.pool.ModelPattern] = true
	}
	if pctx.requestedModel != "" {
		triedPools[pctx.requestedModel] = true
	}

	// Retry-round + cross-provider fallback bookkeeping.
	round := 1
	roundStartIdx := 0 // index into attempts where the current round began
	poolsAttempted := make([]string, 0, 2)
	roundsByPool := make(map[string]int)
	if pctx.pool != nil {
		poolsAttempted = append(poolsAttempted, pctx.pool.ModelPattern)
	}

	maxSpins := maxAttempts*3 + 1
	spins := 0

retryLoop:
	for triedCount < maxAttempts {
		if c.Request.Context().Err() != nil {
			h.logger.Warn("client context canceled or timed out — aborting retry loop immediately to protect pool credentials",
				zap.String("model", pctx.model),
				zap.String("tenant_id", pctx.tenantID),
				zap.Int("keys_tried", triedCount),
				zap.Error(c.Request.Context().Err()),
			)
			if !c.Writer.Written() {
				if pctx.anthropicFront {
					AnthropicErrorResponse(c, http.StatusGatewayTimeout, "api_error", "Client request timed out or was canceled")
				} else {
					c.AbortWithStatusJSON(http.StatusGatewayTimeout, gin.H{
						"error": gin.H{
							"message": "Client request timed out or was canceled",
							"type":    "timeout_error",
							"code":    "context_canceled",
						},
					})
				}
			}
			return
		}

		spins++
		if spins > maxSpins {
			h.logger.Warn("retry loop safety valve — all remaining tokens penalized by concurrent requests",
				zap.String("model", pctx.model),
				zap.String("tenant_id", pctx.tenantID),
				zap.Int("unique_tried", triedCount),
				zap.Int("max_attempts", maxAttempts),
				zap.Int("spins", spins),
			)
			break retryLoop
		}

		batch := pctx.pool.AcquireBatchActiveTokens(parallelBatchSize, triedIndices)
		if len(batch) == 0 {
			allTried := pctx.pool != nil && len(triedIndices) >= int(pctx.pool.TotalCount)
			if allTried {
				break retryLoop
			}

			soonest := pctx.pool.AcquireLeastPenalizedToken()
			if soonest == nil {
				h.logger.Error("no credentials in pool at all",
					zap.String("model", pctx.model),
					zap.String("tenant_id", pctx.tenantID),
				)
				finalBody := formatOpenAIError(http.StatusServiceUnavailable, nil,
					"no credentials are configured for model: "+pctx.model)
				c.Data(http.StatusServiceUnavailable, "application/json", finalBody)
				return
			}
			now := time.Now().UnixNano()
			cooldownUntil := atomic.LoadInt64(&soonest.Credential.CooldownUntil)
			if cooldownUntil > now {
				sleepFor := time.Duration(cooldownUntil - now)
				if round == 1 && sleepFor > cooldownAcquireWaitBudget {
					h.logger.Warn("all pool credentials cooling down for longer than the acquire wait budget — skipping cooldown wait and failing over",
						zap.String("model", pctx.model),
						zap.String("tenant_id", pctx.tenantID),
						zap.String("provider", soonest.Credential.Provider),
						zap.Duration("soonest_cooldown_remaining", sleepFor),
						zap.Int("unique_tried", triedCount),
					)
					break retryLoop
				}
				const maxSleep = 600 * time.Millisecond
				if sleepFor > maxSleep {
					sleepFor = maxSleep
				}
				h.logger.Warn("all tokens cooling down; sleeping until soonest is ready",
					zap.String("model", pctx.model),
					zap.String("provider", soonest.Credential.Provider),
					zap.Duration("sleep", sleepFor),
					zap.Int("unique_tried", triedCount),
				)
				time.Sleep(sleepFor)
			}
			continue retryLoop
		}

		// Single-key synchronous path (zero goroutine/coordination overhead)
		if len(batch) == 1 {
			result := batch[0]
			triedIndices[result.Index] = true
			triedCount++
			pctx.credential = result
			pctx.coordinator = nil
			pctx.workerIndex = 0
			pctx.requestContext = nil

			if result.Credential.Provider == "gemini" {
				if h.isKeyRateLimitedInRedis(c.Request.Context(), result.Credential) {
					result.FromPool.PenalizeToken(result.Index, 5*time.Second)
					continue retryLoop
				}
				h.incrementRedisRPM(c.Request.Context(), result.Credential)
			}

			h.logger.Info("attempting request with credential",
				zap.String("request_id", fmt.Sprintf("%v", requestID)),
				zap.String("model", pctx.model),
				zap.String("provider", result.Credential.Provider),
				zap.Int("credential_id", result.Credential.ID),
				zap.Int("round", round),
				zap.Int("attempt", triedCount),
				zap.Int("max_attempts", maxAttempts),
			)

			statusCode, upstreamURL, errBody, err := h.forwardRequest(c, pctx)
			if c.Request.Context().Err() != nil {
				h.logger.Warn("request terminated due to client context cancellation — aborting rotation",
					zap.String("model", pctx.model),
					zap.String("provider", result.Credential.Provider),
					zap.Int("credential_id", result.Credential.ID),
					zap.Error(c.Request.Context().Err()),
				)
				if !c.Writer.Written() {
					if pctx.anthropicFront {
						AnthropicErrorResponse(c, http.StatusGatewayTimeout, "api_error", "Client request timed out or was canceled")
					} else {
						c.AbortWithStatusJSON(http.StatusGatewayTimeout, gin.H{
							"error": gin.H{
								"message": "Client request timed out or was canceled",
								"type":    "timeout_error",
								"code":    "context_canceled",
							},
						})
					}
				}
				return
			}

			if err == nil && statusCode >= 200 && statusCode < 300 {
				h.emitSuccessTelemetry(c, pctx, result, statusCode, upstreamURL, errBody, requestStart, triedCount)
				return
			}

			if statusCode == http.StatusBadRequest {
				if param, ok := parseUnsupportedSamplingParam(errBody); ok && !strippedSamplingParams[param] {
					if stripped, changed := stripJSONTopLevelKeys(pctx.body, []string{param}); changed {
						strippedSamplingParams[param] = true
						pctx.body = stripped
						delete(triedIndices, result.Index)
						triedCount--
						h.logger.Warn("upstream rejected unsupported sampling parameter — stripped it and retrying same credential",
							zap.String("model", pctx.model),
							zap.String("provider", result.Credential.Provider),
							zap.Int("credential_id", result.Credential.ID),
							zap.String("param", param),
						)
						continue retryLoop
					}
				}
			}

			recStatus, fastFail404, fastFail400 := h.handleCredentialFailure(
				c.Request.Context(), pctx, result, statusCode, upstreamURL, errBody, err,
				&attempts, requestID, triedCount, maxAttempts, requestStart, strippedSamplingParams,
			)
			lastErrBody = errBody
			lastStatus = recStatus
			lastProvider = result.Credential.Provider

			if fastFail404 || fastFail400 {
				break retryLoop
			}
			if recStatus == http.StatusTooManyRequests && pctx.pool != nil && pctx.pool.TotalCount == 1 {
				break retryLoop
			}
			continue retryLoop
		}

		// Multi-key parallel hedged execution path (waves of up to parallelBatchSize)
		for _, cand := range batch {
			triedIndices[cand.Index] = true
		}
		triedCount += len(batch)

		coord := &parallelCoordinator{
			cancelFuncs: make([]context.CancelFunc, len(batch)),
		}
		resCh := make(chan workerResult, len(batch))

		for i := 0; i < len(batch); i++ {
			cand := batch[i]
			workerCtx, cancel := context.WithCancel(c.Request.Context())
			coord.cancelFuncs[i] = cancel

			if cand.Credential.Provider == "gemini" {
				if h.isKeyRateLimitedInRedis(c.Request.Context(), cand.Credential) {
					cand.FromPool.PenalizeToken(cand.Index, 5*time.Second)
					cancel()
					resCh <- workerResult{
						workerIndex: i,
						result:      cand,
						statusCode:  http.StatusTooManyRequests,
						err:         errors.New("redis rpm limit exceeded"),
					}
					continue
				}
				h.incrementRedisRPM(c.Request.Context(), cand.Credential)
			}

			h.logger.Info("attempting parallel request with credential",
				zap.String("request_id", fmt.Sprintf("%v", requestID)),
				zap.String("model", pctx.model),
				zap.String("provider", cand.Credential.Provider),
				zap.Int("credential_id", cand.Credential.ID),
				zap.Int("batch_size", len(batch)),
				zap.Int("worker_index", i),
				zap.Int("attempt", triedCount),
			)

			go func(workerIdx int, credRes *credentials.AcquireResult, reqCtx context.Context) {
				workerPctx := *pctx
				workerPctx.credential = credRes
				workerPctx.coordinator = coord
				workerPctx.workerIndex = workerIdx
				workerPctx.requestContext = reqCtx

				statusCode, upstreamURL, errBody, err := h.forwardRequest(c, &workerPctx)
				resCh <- workerResult{
					workerIndex: workerIdx,
					result:      credRes,
					statusCode:  statusCode,
					upstreamURL: upstreamURL,
					errBody:     errBody,
					err:         err,
				}
			}(i, cand, workerCtx)
		}

		var winner *workerResult
		var shouldBreakLoop bool
		var batchResults []workerResult

		for collected := 0; collected < len(batch); collected++ {
			res := <-resCh

			isCanceledLoser := (res.err == context.Canceled || errors.Is(res.err, context.Canceled)) &&
				coord.hasWinner.Load() && coord.winnerIndex != res.workerIndex

			if isCanceledLoser {
				continue
			}

			batchResults = append(batchResults, res)
			if res.err == nil && res.statusCode >= 200 && res.statusCode < 300 {
				winner = &res
			}
		}

		coord.cancelAll()

		if winner == nil {
			hasParamSelfHeal := false
			for _, res := range batchResults {
				if res.statusCode == http.StatusBadRequest {
					if param, ok := parseUnsupportedSamplingParam(res.errBody); ok && !strippedSamplingParams[param] {
						if stripped, changed := stripJSONTopLevelKeys(pctx.body, []string{param}); changed {
							strippedSamplingParams[param] = true
							pctx.body = stripped
							for _, cand := range batch {
								delete(triedIndices, cand.Index)
							}
							triedCount -= len(batch)
							h.logger.Warn("upstream rejected unsupported sampling parameter — stripped it and retrying batch",
								zap.String("model", pctx.model),
								zap.String("param", param),
							)
							hasParamSelfHeal = true
							break
						}
					}
				}
			}
			if hasParamSelfHeal {
				continue retryLoop
			}

			for _, res := range batchResults {
				recStatus, fastFail404, fastFail400 := h.handleCredentialFailure(
					c.Request.Context(), pctx, res.result, res.statusCode, res.upstreamURL, res.errBody, res.err,
					&attempts, requestID, triedCount, maxAttempts, requestStart, strippedSamplingParams,
				)
				lastErrBody = res.errBody
				lastStatus = recStatus
				lastProvider = res.result.Credential.Provider
				if fastFail404 || fastFail400 {
					shouldBreakLoop = true
				}
			}
		}

		if c.Request.Context().Err() != nil {
			h.logger.Warn("client context canceled or timed out during parallel batch — aborting",
				zap.String("model", pctx.model),
				zap.Error(c.Request.Context().Err()),
			)
			if !c.Writer.Written() {
				if pctx.anthropicFront {
					AnthropicErrorResponse(c, http.StatusGatewayTimeout, "api_error", "Client request timed out or was canceled")
				} else {
					c.AbortWithStatusJSON(http.StatusGatewayTimeout, gin.H{
						"error": gin.H{
							"message": "Client request timed out or was canceled",
							"type":    "timeout_error",
							"code":    "context_canceled",
						},
					})
				}
			}
			return
		}

		if winner != nil {
			h.emitSuccessTelemetry(c, pctx, winner.result, winner.statusCode, winner.upstreamURL, winner.errBody, requestStart, triedCount)
			return
		}

		if shouldBreakLoop {
			break retryLoop
		}
	}

	// ── Retry-Round Policy (never switch providers on the first failure) ──────
	// The pool is abandoned only after retryRoundsPerPool COMPLETE passes over
	// its credentials. After every credential has failed once, ALL credentials
	// are retried (bounded wait for the soonest key's cooldown) so transient
	// upstream errors — 5xx blips, overloads, timeouts — recover without a
	// provider switch.
	//
	// Rounds are skipped when every failure in the last round was a
	// non-retryable credential rejection (auth/quota/billing/rate-limit/
	// payload): re-trying those keys inside the same request cannot succeed,
	// so the gateway proceeds to the exact-model fallback immediately instead
	// of stalling the client.
	//
	// Fast-fail paths above (404, repeated 400, single-key 429, spin guard,
	// whole-pool long-cooldown acquire) break the loop early; if they fired
	// before every key was tried they skip this block (triedCount <
	// maxAttempts). If they fired on the very last key they still land here,
	// but their statuses are all non-retryable, so the guard below routes them
	// straight to the cross-provider fallback instead of burning retry rounds.
	if (triedCount >= maxAttempts || (pctx.pool != nil && len(triedIndices) >= int(pctx.pool.TotalCount))) && round < retryRoundsPerPool {
		lastRoundRetryable := false
		for _, a := range attempts[roundStartIdx:] {
			if !isNonRetryableCredentialStatus(a.statusCode) {
				lastRoundRetryable = true
				break
			}
		}

		if !lastRoundRetryable {
			h.logger.Warn("every credential rejected with non-retryable errors — skipping remaining retry rounds and switching provider",
				zap.String("model", pctx.model),
				zap.String("tenant_id", pctx.tenantID),
				zap.Int("rounds_completed", round),
				zap.Int("rounds_max", retryRoundsPerPool),
				zap.Int("keys_tried", triedCount),
				zap.Duration("elapsed", time.Since(requestStart)),
			)
		} else {
			// Wait — bounded — for the soonest failed key's cooldown so the
			// next round probes fresh credentials, then re-try the whole pool.
			wait := poolSoonestCooldownRemaining(pctx.pool)
			if wait > roundTransitionWaitBudget {
				wait = roundTransitionWaitBudget
			}
			if wait > 0 {
				select {
				case <-time.After(wait):
				case <-c.Request.Context().Done():
					// Client gone: the loop-top context guard aborts cleanly.
				}
			}
			round++
			roundStartIdx = len(attempts)
			triedCount = 0
			triedIndices = make(map[int]bool)
			spins = 0
			h.logger.Warn("all pool credentials failed this round — retrying every credential before any provider switch",
				zap.String("model", pctx.model),
				zap.String("tenant_id", pctx.tenantID),
				zap.String("pool", pctx.pool.ModelPattern),
				zap.Int("round", round),
				zap.Int("rounds_max", retryRoundsPerPool),
				zap.Int("credentials", maxAttempts),
				zap.Duration("waited_for_cooldown", wait),
			)
			goto retryLoop
		}
	}

	// ── Exact-Model Cross-Provider Fallback ────────────────────────────────────
	// When all credentials in the current pool fail or are exhausted, attempt to
	// fall back to other pools hosting the EXACT same model ID and same version.
	// We strictly prohibit falling back to different, cheaper, or arbitrary models.
	// If no healthy exact-model pool exists, the upstream error is thrown directly.
	if h.syncManager != nil {
		// Record how many complete rounds the pool being abandoned consumed.
		if pctx.pool != nil {
			roundsByPool[pctx.pool.ModelPattern] = round
		}
		allPools := h.syncManager.GetAllPools()
		targetModelRef := pctx.requestedModel
		if targetModelRef == "" && pctx.pool != nil {
			targetModelRef = pctx.pool.ModelPattern
		}

		fallbacks := FindExactModelFallbacks(targetModelRef, triedPools, allPools)
		if len(fallbacks) == 0 && pctx.pool != nil && pctx.pool.ModelPattern != targetModelRef {
			fallbacks = FindExactModelFallbacks(pctx.pool.ModelPattern, triedPools, allPools)
		}

		// Prioritize configured DB fallback pool if healthy and untried
		if pctx.pool != nil && pctx.pool.FallbackPool != nil {
			fb := pctx.pool.FallbackPool
			if !triedPools[fb.ModelPattern] && fb.HealthyCount() > 0 {
				alreadyPresent := false
				for _, cand := range fallbacks {
					if cand.ModelPattern == fb.ModelPattern {
						alreadyPresent = true
						break
					}
				}
				if alreadyPresent {
					var reordered []*credentials.BalancedChannelPool
					reordered = append(reordered, fb)
					for _, cand := range fallbacks {
						if cand.ModelPattern != fb.ModelPattern {
							reordered = append(reordered, cand)
						}
					}
					fallbacks = reordered
				} else {
					fallbacks = append([]*credentials.BalancedChannelPool{fb}, fallbacks...)
				}
			}
		}

		if len(fallbacks) > 0 {
			fallbackPool := fallbacks[0]
			triedPools[fallbackPool.ModelPattern] = true
			poolsAttempted = append(poolsAttempted, fallbackPool.ModelPattern)

			h.logger.Warn("pool exhausted across all retry rounds — switching to exact-model cross-provider fallback",
				zap.String("original_model", pctx.requestedModel),
				zap.String("failed_pool", pctx.pool.ModelPattern),
				zap.Int("failed_pool_rounds", round),
				zap.String("fallback_pool", fallbackPool.ModelPattern),
				zap.Int("fallback_healthy_keys", fallbackPool.HealthyCount()),
			)

			h.switchProxyContextToFallback(pctx, fallbackPool)

			// Reset retry counters and give the fallback pool its own retry rounds.
			triedCount = 0
			triedIndices = make(map[int]bool)
			round = 1
			roundStartIdx = len(attempts)
			maxAttempts = int(fallbackPool.TotalCount)
			if maxAttempts < 1 {
				maxAttempts = 1
			}
			if maxAttempts > 100 {
				maxAttempts = 100
			}
			maxSpins = maxAttempts*3 + 1
			spins = 0

			goto retryLoop
		}
	}

	// ── All credentials exhausted ─────────────────────────────────────────────
	// Every credential of every pool hosting the exact same model has failed
	// across all retry rounds. Record the final pool's rounds and return a
	// meaningful canonical error describing every pool, round, and attempt.
	if pctx.pool != nil {
		roundsByPool[pctx.pool.ModelPattern] = round
	}
	// Prefer the client-requested model name so the error describes what the
	// user actually asked for, not the last pool's routing pattern.
	summaryModel := pctx.requestedModel
	if summaryModel == "" {
		summaryModel = pctx.model
	}

	h.logger.Error("all pool credentials exhausted",
		zap.String("model", pctx.model),
		zap.String("tenant_id", pctx.tenantID),
		zap.Int("keys_tried", triedCount),
		zap.Int("rounds_on_last_pool", round),
		zap.Int("rounds_max_per_pool", retryRoundsPerPool),
		zap.Strings("pools_attempted", poolsAttempted),
		zap.Int("last_status", lastStatus),
		zap.String("last_provider", lastProvider),
		zap.Duration("total_elapsed", time.Since(requestStart)),
	)

	// Build the diagnostic summary once — telemetry and the client response
	// must describe the same failure.
	summary := buildAttemptSummary(summaryModel, attempts, roundsByPool, poolsAttempted)
	respStatus := http.StatusBadGateway
	if lastStatus >= 400 && lastStatus < 600 {
		respStatus = lastStatus
	}
	if len(attempts) == 0 {
		// No credential was ever attempted this request: every pool credential
		// was already cooling down from earlier failures (long-cooldown
		// fast-fail) or was penalized by concurrent requests. The model is not
		// broken, just saturated — report temporary unavailability with the
		// soonest retry time so clients can back off intelligently.
		respStatus = http.StatusServiceUnavailable
		if pctx.pool != nil {
			if wait := poolSoonestCooldownRemaining(pctx.pool); wait > 0 {
				summary = fmt.Sprintf(
					"model %q is temporarily unavailable — all %d pool credential(s) are cooling down after earlier failures (soonest retry in ~%s); no upstream request was sent",
					summaryModel, pctx.pool.TotalCount, wait.Round(time.Second))
			}
		}
	}

	// Emit exhaustion telemetry (zero-alloc pool pattern)
	if h.pipeline != nil {
		promptText := extractPromptText(pctx.body)
		promptTokens := extractTokens(pctx.body, "prompt")
		if promptTokens == 0 {
			promptTokens = len(promptText) / 4
		}
		entry := telemetry.AcquireEntry()
		entry.TenantID = pctx.tenantID
		entry.Model = pctx.model
		entry.Provider = lastProvider
		entry.PromptTokens = promptTokens
		entry.LatencyMs = int(time.Since(requestStart).Milliseconds())
		entry.StatusCode = respStatus
		entry.ErrorMessage = summary
		entry.CreatedAt = time.Now()
		entry.Prompt = promptText
		h.pipeline.Emit(entry)
	}

	// Never dump raw upstream bytes — always return a canonical error envelope.
	// Anthropic front-door requests receive the Anthropic-shaped envelope
	// ({"type":"error","error":{...}}); OpenAI requests keep the flat envelope.
	var finalBody []byte
	if pctx.anthropicFront {
		finalBody = formatAnthropicGatewayError(respStatus, lastErrBody, summary)
	} else {
		finalBody = formatOpenAIError(respStatus, lastErrBody, summary)
	}
	c.Data(respStatus, "application/json", finalBody)
}

// switchProxyContextToFallback updates the proxy context to route subsequent retries
// to an exact-model fallback pool on a different provider.
func (h *Handler) switchProxyContextToFallback(pctx *proxyContext, newPool *credentials.BalancedChannelPool) {
	newPattern := newPool.ModelPattern
	newUpstreamModel := FormatUpstreamModelForPattern(newPattern)

	// Rewrite model in request body
	if len(pctx.rawBody) > 0 {
		origModelInBody, err := jsonparser.GetString(pctx.rawBody, "model")
		if err == nil && origModelInBody != "" {
			oldToken := []byte(`"` + origModelInBody + `"`)
			newToken := []byte(`"` + newUpstreamModel + `"`)
			pctx.body = bytes.Replace(pctx.rawBody, oldToken, newToken, 1)
		} else {
			pctx.body = make([]byte, len(pctx.rawBody))
			copy(pctx.body, pctx.rawBody)
		}
	} else if len(pctx.body) > 0 {
		currModelInBody, err := jsonparser.GetString(pctx.body, "model")
		if err == nil && currModelInBody != "" {
			oldToken := []byte(`"` + currModelInBody + `"`)
			newToken := []byte(`"` + newUpstreamModel + `"`)
			pctx.body = bytes.Replace(pctx.body, oldToken, newToken, 1)
		}
	}

	pctx.model = newPattern
	pctx.pool = newPool

	// Update provider flags based on the target provider
	pctx.isNvidia = strings.HasPrefix(newPattern, "nvidia/")
	pctx.isOneMinAI = strings.HasPrefix(newPattern, "1min/")
	pctx.isCloudflare = strings.HasPrefix(newPattern, "cloudflare/")
	pctx.isSarvam = strings.HasPrefix(newPattern, "sarvam/")
	pctx.isPuter = strings.HasPrefix(newPattern, "puter/")
	pctx.isAgentRouter = strings.HasPrefix(newPattern, "agentrouter/")
	pctx.isZenMux = strings.HasPrefix(newPattern, "zenmux/")
	pctx.isJiekou = strings.HasPrefix(newPattern, "jiekou/")
	pctx.isGemini = strings.HasPrefix(newPattern, credentials.ProviderGeminiLegacy+"/")

	// Special provider adjustments:
	if pctx.isCloudflare {
		// Cloudflare path transformer embeds the model in /ai/run/{model}
		pctx.model = strings.TrimPrefix(newPattern, "cloudflare/")
	}
	if pctx.isNvidia && supportsNvidiaReasoning(newUpstreamModel) {
		pctx.body = injectNvidiaParams(pctx.body, pctx.body, h.logger)
	}
}

// forwardRequest sends the request to the upstream provider and returns the
// result without writing anything to c.Writer for retryable error statuses.
//
// Return values:
//   - statusCode: the HTTP status from upstream (or 0 on transport error)
//   - upstreamURL: the exact URL called (for logging)
//   - errBody: the upstream error body read into memory for retryable statuses;
//     nil for success paths (those are streamed/written directly to c.Writer)
//   - err: non-nil only for transport-level failures (DNS, TLS, timeout)
//
// Flaw A fix: For retryable statuses (500, 503, 429…) the upstream body is
// captured in errBody and returned WITHOUT touching c.Writer. executeWithRetry
// can then loop to the next credential. Only once all retries are exhausted
// does the caller flush errBody to the client — ensuring HTTP headers are
// written exactly once on a clean wire.
func (h *Handler) forwardRequest(c *gin.Context, pctx *proxyContext) (statusCode int, upstreamURL string, errBody []byte, err error) {
	defer func() {
		if r := recover(); r != nil {
			h.logger.Error("recovered from forward request panic",
				zap.Any("panic", r),
				zap.String("model", pctx.model),
				zap.String("provider", pctx.credential.Credential.Provider),
				zap.String("upstream_url", upstreamURL),
				zap.ByteString("stack", debug.Stack()),
			)
			statusCode = http.StatusInternalServerError
			err = fmt.Errorf("internal panic: %v", r)
		}
	}()

	cred := pctx.credential.Credential
	modelName := pctx.model
	bodyBytes := pctx.body

	// ── Anthropic front door: per-credential request preparation ──────────
	// The Anthropic-compatible front API (/v1/messages) may route to native
	// Anthropic-protocol credentials (zero-copy passthrough) or to any other
	// provider (canonical OpenAI payload produced by TranslateAnthropicToOpenAI).
	// The decision is per-credential because a pool can mix providers.
	anthropicNative := false
	if pctx.anthropicFront {
		anthropicNative = isNativeAnthropicProvider(cred.Provider)
		if anthropicNative {
			prepared, pErr := PrepareNativeAnthropicPayload(pctx.anthropicRawBody, stripAnthropicRoutingModel(modelName, ""))
			if pErr != nil {
				return http.StatusBadRequest, upstreamURL, formatAnthropicGatewayError(http.StatusBadRequest,
					nil, "failed to prepare native Anthropic payload: "+pErr.Error()), nil
			}
			bodyBytes = prepared
		} else if pctx.anthropicTranslated == nil {
			// The pool was all-native so translation was deferred; the first
			// non-native credential (usually a cross-provider fallback) needs
			// the canonical OpenAI payload now. Translate from the current
			// body so any fallback model rewrite is preserved.
			translated, tErr := TranslateAnthropicToOpenAI(pctx.body, FormatUpstreamModelForPattern(modelName), anthropicFallbackMaxTokens)
			if tErr != nil {
				return http.StatusBadRequest, upstreamURL, formatAnthropicGatewayError(http.StatusBadRequest,
					nil, "failed to translate Anthropic request: "+tErr.Error()), nil
			}
			if pctx.isNvidia && supportsNvidiaReasoning(modelName) {
				translated = injectNvidiaParams(translated, translated, h.logger)
			}
			pctx.anthropicTranslated = translated
			pctx.body = translated
			bodyBytes = translated
		}
	}

	// If the credential has an optional prefix, we must strip it from both the model ID
	// passed to the rewriter and the JSON payload sent upstream.
	if cred.Prefix != "" {
		prefixSlash := cred.Prefix + "/"
		if strings.HasPrefix(modelName, prefixSlash) {
			modelName = strings.TrimPrefix(modelName, prefixSlash)

			// Replace model name inside JSON body bytes.
			// Same O(n) replacement logic as nvidia/ollama.
			// (Native Anthropic payloads are rebuilt below with the stripped name.)
			if !anthropicNative {
				oldToken := []byte(`"` + pctx.model + `"`)
				newToken := []byte(`"` + modelName + `"`)
				bodyBytes = bytes.Replace(bodyBytes, oldToken, newToken, 1)
			}
		}
	}

	// --- Claude ≥ Opus 4.5 sampling-param sanitization ---
	// This Claude generation hard-rejects OpenAI sampling parameters with a
	// 400 "Validation error at temperature: `temperature` is not supported on
	// this model. Remove it from your request." (Cloudflare Workers AI,
	// code 7003). Strip temperature/top_p/top_k up-front for these models on
	// every provider — aggregators already drop them, strict Anthropic-schema
	// endpoints reject them. See sampling_params.go for the full rationale.
	// Native Anthropic passthrough is exempt: temperature/top_p/top_k are
	// legal Anthropic sampling controls that the client sent deliberately.
	if !anthropicNative && modelDisallowsSamplingParams(modelName) {
		bodyBytes = stripSamplingParams(bodyBytes)
	}

	// --- 1min.ai Request Body Translation ---
	// The 1min.ai API uses a completely different request format
	// (type/model/promptObject). The translator converts the OpenAI-compatible
	// body into the 1min.ai Feature API format. For multipart requests (audio
	// transcription), the audio file is uploaded to the 1min.ai Asset API first.
	var contentTypeOverride string
	if cred.Provider == "1minai" {
		translated, ctOverride, trErr := translateOneMinAIRequest(
			pctx.model, c.Request.URL.Path, bodyBytes,
			c.Request.Header.Get("Content-Type"), cred.APIKey, h.client,
		)
		if trErr != nil {
			h.logger.Error("1min.ai request translation failed",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			return http.StatusInternalServerError, upstreamURL, nil, trErr
		}
		bodyBytes = translated
		contentTypeOverride = ctOverride
	}

	// --- Cloudflare Workers AI Request Body Translation ---
	// Native @cf/* text-to-image models served via /ai/run/{model} reject unknown
	// OpenAI fields (model, n, size, response_format) with "Invalid input" (code 8002).
	// We translate image-generation requests down to bare {"prompt":"..."} for native models.
	//
	// Third-party AI Gateway image models (openai/gpt-image-2, recraft/*, krea/*, etc.)
	// accept the full OpenAI payload natively — pass those through unchanged.
	//
	// Chat completions are served via Cloudflare's OpenAI-compatible /ai/v1 endpoint
	// and pass through unchanged regardless of model type.
	if cred.Provider == "cloudflare" && isCloudflareImageRequest(c.Request.URL.Path) {
		if isCloudflareNativeImageModel(modelName) {
			// Native @cf/* model: translate to bare {prompt:...} format
			translated, ctOverride, trErr := translateCloudflareImageRequest(bodyBytes)
			if trErr != nil {
				h.logger.Error("cloudflare native image request translation failed",
					zap.String("model", pctx.model),
					zap.Error(trErr),
				)
				return http.StatusInternalServerError, upstreamURL, nil, trErr
			}
			bodyBytes = translated
			contentTypeOverride = ctOverride
		}
		// Third-party AI Gateway image models: pass OpenAI payload through unchanged
	}

	// --- Cloudflare Workers AI VLM Image Inlining, Vision Translation & Sanitization ---
	if cred.Provider == "cloudflare" && isCloudflareNativeVisionModel(modelName) {
		if inlined, inErr := ConvertImageURLsToBase64(c.Request.Context(), bodyBytes); inErr == nil && len(inlined) > 0 {
			bodyBytes = inlined
		}
		translated, ctOverride, trErr := translateCloudflareVisionRequest(bodyBytes)
		if trErr != nil {
			h.logger.Error("cloudflare native vision request translation failed",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			return http.StatusBadRequest, upstreamURL, nil, trErr
		}
		bodyBytes = translated
		if ctOverride != "" {
			contentTypeOverride = ctOverride
		}
	} else if cred.Provider == "cloudflare" && !isCloudflareImageRequest(c.Request.URL.Path) {
		if inlined, inErr := ConvertImageURLsToBase64(c.Request.Context(), bodyBytes); inErr == nil && len(inlined) > 0 {
			bodyBytes = inlined
		}
		bodyBytes = sanitizeCloudflareRequest(bodyBytes)
	}

	// --- Sarvam AI Request Sanitization ---
	// Sarvam's chat-completions schema is a strict subset of OpenAI's. Unknown
	// top-level fields (stream_options, logprobs, service_tier, …) are rejected
	// with HTTP 422, which the gateway would otherwise mis-attribute to a broken
	// key (20–30 min cooldown per credential). Strip them here so any
	// OpenAI-compatible frontend works with Sarvam unchanged. This runs for BOTH
	// routing forms (prefixed "sarvam/…" and clean alias) since it is gated on
	// the resolved credential's provider, not the model prefix. The common
	// clean-request path is allocation-free (fast presence probe).
	if cred.Provider == "sarvam" {
		bodyBytes = sanitizeSarvamRequest(bodyBytes)
	}

	if cred.Provider == "puter" {
		bodyBytes = sanitizePuterRequest(bodyBytes)
	}

	// --- AgentRouter Request Transpilation ---
	// AgentRouter requires requests in Anthropic's /v1/messages?beta=true format
	// with mandatory Claude Code CLI wire-image headers.
	if cred.Provider == "agentrouter" {
		agentBody, trErr := TranslateOpenAIToAgentRouter(bodyBytes)
		if trErr != nil {
			h.logger.Error("agentrouter request transpilation failed",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			return http.StatusBadRequest, upstreamURL, nil, trErr
		}
		// Debug: log key details of the translated Anthropic request
		debugMsgs, _, _, _ := jsonparser.Get(agentBody, "messages")
		debugMsgCount := 0
		if len(debugMsgs) > 0 {
			json.Unmarshal(debugMsgs, &[]interface{}{}) // count only
			var tmp []interface{}
			if json.Unmarshal(debugMsgs, &tmp) == nil {
				debugMsgCount = len(tmp)
			}
		}
		hasTools, _, _, _ := jsonparser.Get(agentBody, "tools")
		h.logger.Info("agentrouter translated request",
			zap.String("model", pctx.model),
			zap.Int("body_bytes", len(agentBody)),
			zap.Int("msg_count", debugMsgCount),
			zap.Bool("has_tools", len(hasTools) > 0),
		)
		bodyBytes = agentBody
	}

	// --- Jiekou.AI Request Sanitization ---
	// Jiekou proxies Moonshot/Kimi and other Chinese models behind an OpenAI-
	// compatible API. Two issues cause HTTP 400 "invalid request" errors:
	//
	//  1. If any "jiekou/" prefix survived into bodyBytes (e.g., the credential
	//     was resolved from a clean alias pool without a jiekou/ prefix token in
	//     pctx.model), strip it now.
	//
	//  2. Moonshot/Kimi enforce temperature ∈ [0.0, 1.0]. OpenAI allows up to 2.0.
	//     Clients (Cline, Cursor, VSCode) commonly send temperature: 1.2 or 1.5.
	//
	//  3. Several OpenAI-only fields (logit_bias, user, logprobs, stream_options…)
	//     are passed through by Jiekou to Moonshot, which then rejects them.
	if cred.Provider == "jiekou" {
		bodyBytes = sanitizeJiekouRequest(bodyBytes)
	}

	// --- NVIDIA Request Sanitization ---
	// NVIDIA NIM's API schema strictly validates top-level request fields and
	// rejects unknown OpenAI-specific keys (stream_options, logprobs, service_tier,
	// etc.) with HTTP 400 "Unsupported parameter(s)". Without this sanitizer the
	// gateway's 400 fast-fail block would abort after the first attempt and return
	// a misleading 502 gateway_exhaustion_error to the client.
	//
	// Note: reasoning_budget and chat_template_kwargs injection is handled
	// separately in Handle() via injectNvidiaParams(), gated on
	// supportsNvidiaReasoning(). The sanitizer here provides defence-in-depth
	// for all other OpenAI-specific fields that neither reasoning nor standard
	// NVIDIA models accept.
	if cred.Provider == "nvidia" {
		bodyBytes = sanitizeNvidiaRequest(bodyBytes)
	}

	// --- Gemini AI Studio: SDK-native path ---
	// Replaces the raw HTTP proxy path for Gemini. The official google.golang.org/genai
	// SDK handles auth, URL construction, serialization, chunked stream framing, and
	// thought_signature bypass for Gemini 3+ models. executeGeminiWithSDK writes
	// the response directly to c.Writer and returns a telemetry blob as errBody.
	// The early return naturally bypasses URL construction, h.client.Do, and the old
	// Gemini response translation below — those blocks remain as dead code for rollback.
	if cred.Provider == "gemini" && (cred.BaseURL == "" || strings.HasPrefix(cred.BaseURL, "https://generativelanguage.googleapis.com")) {
		sdkStatus, sdkBody, sdkErr := h.executeGeminiWithSDK(c, pctx, cred.APIKey, bodyBytes)
		if sdkErr != nil {
			h.logger.Error("gemini sdk executor error",
				zap.String("model", pctx.model),
				zap.Error(sdkErr),
			)
			return http.StatusInternalServerError, "sdk://gemini", nil, nil
		}
		return sdkStatus, "sdk://gemini", sdkBody, nil
	}

	// --- Gemini AI Studio Request Body Transpilation (LEGACY — unreachable via SDK path) ---
	// Google AI Studio uses a fundamentally different request format from OpenAI.
	// The transpiler converts the entire body — messages, tools, generation config,
	// thinking budget, and safety settings — into Gemini's native generateContent
	// format. This runs for BOTH the prefixed ("gemini/...") and clean ("gemini-2.5-pro")
	// routing forms since it is gated on the resolved credential's provider.
	if cred.Provider == "gemini_legacy_disabled" {
		geminiBody, trErr := transpileOpenAIToGemini(bodyBytes)
		if trErr != nil {
			h.logger.Error("gemini request transpilation failed",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			return http.StatusBadRequest, upstreamURL, nil, trErr
		}
		bodyBytes = geminiBody
	}

	upstreamURL = h.rewriter.RewriteURL(cred.Provider, cred.BaseURL, c.Request.URL.Path, modelName)

	// --- Gemini URL Adjustments ---
	// Two fixes applied after URL construction:
	//
	//  1. Non-stream URL switch: geminiPath always produces streamGenerateContent?alt=sse
	//     as a safe default. For non-streaming client requests we switch to the plain
	//     generateContent endpoint (lower latency, simpler response parsing).
	//
	//  2. API key injection: Google AI Studio requires ?key=<apiKey> as a query
	//     parameter. No Authorization header is used (see rewriter.go).
	if cred.Provider == "gemini" {
		if !pctx.isStream {
			// Switch from streamGenerateContent?alt=sse to generateContent
			upstreamURL = strings.Replace(upstreamURL, ":streamGenerateContent?alt=sse", ":generateContent", 1)
		}
		// Inject API key as query parameter
		if strings.Contains(upstreamURL, "?") {
			upstreamURL += "&key=" + cred.APIKey
		} else {
			upstreamURL += "?key=" + cred.APIKey
		}
	}

	// --- 1min.ai Streaming ---
	// 1min.ai enables streaming via the ?isStreaming=true query parameter
	if cred.Provider == "1minai" && pctx.isStream {
		upstreamURL += "?isStreaming=true"
	}

	reqCtx := c.Request.Context()
	if pctx.requestContext != nil {
		reqCtx = pctx.requestContext
	}

	upstreamReq, err := http.NewRequestWithContext(
		reqCtx,
		c.Request.Method,
		upstreamURL,
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return http.StatusInternalServerError, upstreamURL, nil, err
	}

	h.rewriter.RewriteHeaders(upstreamReq, cred.Provider, cred.APIKey, c.Request.Header)

	// ── Anthropic prompt caching (Phase 4) ─────────────────────────────────
	// Native Anthropic upstreams reject cache_control breakpoints unless the
	// request advertises the prompt-caching beta. Inject the flag when the
	// body uses caching and the client did not send an anthropic-beta header.
	// AgentRouter keeps its fixed Claude Code fingerprint (set above); every
	// other provider is untouched.
	if anthropicNative {
		if beta := AnthropicPromptCachingBeta(bodyBytes, upstreamReq.Header.Get("anthropic-beta")); beta != "" {
			upstreamReq.Header.Set("anthropic-beta", beta)
		}
	}

	// --- Puter Edge Firewall De-fingerprinting ---
	if cred.Provider == "puter" {
		randomIP := generateRandomIP()
		upstreamReq.Header.Set("X-Forwarded-For", randomIP)
		upstreamReq.Header.Set("X-Real-IP", randomIP)
		upstreamReq.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
	}

	// Override Content-Type for 1min.ai (e.g., multipart STT → JSON)
	if contentTypeOverride != "" {
		upstreamReq.Header.Set("Content-Type", contentTypeOverride)
	}

	resp, err := h.client.Do(upstreamReq)

	// --- AgentRouter Mirror Domain Failover (ps.air-outer.com <-> agentrouter.org) ---
	if (err != nil || (resp != nil && resp.StatusCode >= 500)) && (cred.Provider == "agentrouter" || pctx.isAgentRouter) {
		var fallbackDomain string
		if strings.Contains(upstreamURL, "ps.air-outer.com") {
			fallbackDomain = "agentrouter.org"
		} else if strings.Contains(upstreamURL, "agentrouter.org") {
			fallbackDomain = "ps.air-outer.com"
		}

		if fallbackDomain != "" {
			fallbackURL := strings.Replace(upstreamURL, "ps.air-outer.com", fallbackDomain, 1)
			if fallbackURL == upstreamURL {
				fallbackURL = strings.Replace(upstreamURL, "agentrouter.org", fallbackDomain, 1)
			}

			h.logger.Warn("agentrouter primary domain failed — attempting mirror failover",
				zap.String("primary_url", upstreamURL),
				zap.String("fallback_url", fallbackURL),
				zap.Error(err),
			)

			fallbackReq, fbErr := http.NewRequestWithContext(
				reqCtx,
				c.Request.Method,
				fallbackURL,
				bytes.NewReader(bodyBytes),
			)
			if fbErr == nil {
				h.rewriter.RewriteHeaders(fallbackReq, cred.Provider, cred.APIKey, c.Request.Header)
				fbResp, fbErrDo := h.client.Do(fallbackReq)
				if fbErrDo == nil && fbResp.StatusCode < 500 {
					if resp != nil && resp.Body != nil {
						resp.Body.Close()
					}
					resp = fbResp
					err = nil
					upstreamURL = fallbackURL
				} else if fbResp != nil {
					fbResp.Body.Close()
				}
			}
		}
	}

	if err != nil {
		// Map transport-level failures to numeric HTTP status codes so the
		// total-exhaustion retry loop can penalize and rotate to the next key.
		// forwardRequest never returns a non-nil Go error for transport issues;
		// only unrecovered panics (caught by defer/recover above) do.
		mapped := http.StatusBadGateway // connection reset, DNS, TLS handshake
		if isContextTimeoutError(err) {
			mapped = http.StatusGatewayTimeout
		}
		h.logger.Error("upstream transport error — mapped to status for retry",
			zap.String("model", pctx.model),
			zap.String("provider", cred.Provider),
			zap.String("upstream_url", upstreamURL),
			zap.Int("mapped_status", mapped),
			zap.Error(err),
		)
		return mapped, upstreamURL, []byte(err.Error()), nil
	}
	defer resp.Body.Close()

	// ── Total exhaustion policy: capture ALL non-2xx without touching c.Writer ──
	// forwardRequest never writes error responses to the wire. executeWithRetry
	// decides whether to rotate to another key or flush the normalised error.
	// This invariant prevents double WriteHeader corruption on retried connections
	// and enables any error code (400, 404, 422, 444, 599, …) to trigger rotation.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		const maxErrBodyBytes = 4096
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBodyBytes))

		// Auto-Agree Handshake for Cloudflare Meta / Restricted Models
		if cred.Provider == "cloudflare" && isCloudflareModelAgreementError(body) {
			h.logger.Info("cloudflare model agreement required — performing automatic 'agree' handshake",
				zap.String("model", pctx.model),
				zap.Int("credential_id", cred.ID),
			)
			accountID := strings.TrimPrefix(cred.BaseURL, "cloudflare:")
			_ = autoAcceptCloudflareModelAgreement(c.Request.Context(), h.client, accountID, cred.APIKey, modelName)

			// Retry the original request with the agreed credential
			retryReq, reqErr := http.NewRequestWithContext(
				reqCtx,
				c.Request.Method,
				upstreamURL,
				bytes.NewReader(bodyBytes),
			)
			if reqErr == nil {
				h.rewriter.RewriteHeaders(retryReq, cred.Provider, cred.APIKey, c.Request.Header)
				if contentTypeOverride != "" {
					retryReq.Header.Set("Content-Type", contentTypeOverride)
				}
				retryResp, retryDoErr := h.client.Do(retryReq)
				if retryDoErr == nil && retryResp.StatusCode >= 200 && retryResp.StatusCode < 300 {
					if resp != nil && resp.Body != nil {
						resp.Body.Close()
					}
					resp = retryResp
					err = nil
					goto handleSuccess
				} else if retryResp != nil {
					defer retryResp.Body.Close()
					body, _ = io.ReadAll(io.LimitReader(retryResp.Body, maxErrBodyBytes))
				}
			}
		}

		h.logger.Error("upstream returned error (buffered for retry evaluation)",
			zap.String("model", pctx.model),
			zap.String("provider", cred.Provider),
			zap.String("upstream_url", upstreamURL),
			zap.Int("status", resp.StatusCode),
			zap.ByteString("upstream_error_body", body),
		)
		return resp.StatusCode, upstreamURL, body, nil
	}

handleSuccess:
	// ── Error-in-success-body sniffing ────────────────────────────────────
	// Some upstreams (Pollinations being the reported case) return HTTP 200
	// with the actual error message — credit/balance exhaustion, marketing
	// top-up links — embedded in the response body in place of the model's
	// answer. Detect this BEFORE anything is written to the client and
	// convert it into a synthetic 402 so executeWithRetry penalizes the
	// credential, rotates to a healthy key, and can fall back to another
	// provider hosting the same model. See response_error_sniffer.go.
	{
		ct := resp.Header.Get("Content-Type")
		if isSniffableContentType(ct) {
			if pctx.isStream && resp.StatusCode == http.StatusOK {
				res := sniffStreamForError(resp.Body)
				if res.stalled {
					// Upstream accepted the request but produced no bytes
					// within the idle window — a stalled stream, not an error
					// body. Map to 504 (retryable, short cooldown) so the
					// rotation loop tries the next credential instead of
					// hanging until the client's intermediary kills the
					// connection (~100-110s idle timeouts in the wild).
					h.logger.Warn("upstream stalled before first stream byte — rotating to next credential",
						zap.String("model", pctx.model),
						zap.String("provider", cred.Provider),
						zap.Int("credential_id", cred.ID),
						zap.String("upstream_url", upstreamURL),
						zap.Duration("idle_timeout", streamFirstByteIdleTimeout),
					)
					return http.StatusGatewayTimeout, upstreamURL,
						[]byte(`{"error":{"message":"upstream stalled before sending the first byte of the stream","type":"server_error","code":"upstream_stalled"}}`), nil
				}
				if res.verdict != nil {
					h.logger.Warn("upstream 200 OK body is actually an error — converting to retryable failure",
						zap.String("model", pctx.model),
						zap.String("provider", cred.Provider),
						zap.Int("credential_id", cred.ID),
						zap.String("reason", res.verdict.Reason),
						zap.Bool("stream", true),
						zap.String("detail", truncateSnippet(res.verdict.Message, 300)),
					)
					return http.StatusPaymentRequired, upstreamURL, res.verdict.SyntheticBody(), nil
				}
				// Clean: replay the sniffed bytes byte-exactly. The
				// function-level `defer resp.Body.Close()` captured the
				// original body at defer time, so it is still closed exactly
				// once; ProxyStream closing the NopCloser wrapper is a no-op.
				resp.Body = res.body
			} else if !pctx.isStream && resp.StatusCode >= 200 && resp.StatusCode < 300 {
				prefix, truncated := readPrefix(resp.Body, nonStreamSniffLimit)
				if len(prefix) > 0 && (ct != "" || isLikelyTextual(prefix)) {
					if v := detectSuccessBodyError(prefix, ct, !truncated); v != nil {
						h.logger.Warn("upstream 200 OK body is actually an error — converting to retryable failure",
							zap.String("model", pctx.model),
							zap.String("provider", cred.Provider),
							zap.Int("credential_id", cred.ID),
							zap.String("reason", v.Reason),
							zap.Bool("stream", false),
							zap.String("detail", truncateSnippet(v.Message, 300)),
						)
						return http.StatusPaymentRequired, upstreamURL, v.SyntheticBody(), nil
					}
				}
				resp.Body = io.NopCloser(&prefixReplayReader{prefix: prefix, rest: resp.Body})
			}
		}
	}

	if pctx.coordinator != nil {
		if !pctx.coordinator.claimVictory(pctx.workerIndex) {
			return resp.StatusCode, upstreamURL, nil, context.Canceled
		}
	}

	// --- Success stream path ---
	if pctx.isStream && resp.StatusCode == http.StatusOK {
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		if anthropicNative {
			// Zero-copy passthrough: the upstream already speaks the Anthropic
			// SSE protocol the client expects — relay bytes verbatim. Phase 3
			// will add the reverse (OpenAI → Anthropic) SSE encoder for the
			// non-native providers that reach the branch below.
			_, completionTokens := h.stream.ProxyStreamRaw(c, resp)
			type rawStreamResult struct {
				Text   string `json:"text"`
				Tokens int    `json:"tokens"`
			}
			resJSON, _ := json.Marshal(rawStreamResult{Text: "", Tokens: completionTokens})
			return resp.StatusCode, upstreamURL, resJSON, nil
		}
		// Anthropic front door via a non-native provider (Phase 3+4): the
		// client expects Anthropic Messages SSE, so the relay re-encodes the
		// OpenAI-format chunks produced by the transmux layer.
		proxyStreamOpts := []ProxyStreamOption(nil)
		if pctx.anthropicFront {
			msgID := "msg_gate_" + strconv.FormatInt(time.Now().UnixNano(), 36)
			proxyStreamOpts = []ProxyStreamOption{WithAnthropicStream(msgID, pctx.requestedModel, pctx.anthropicInputTokens)}
		}
		responseText, completionTokens := h.stream.ProxyStream(c, resp, cred.Provider, pctx.requestedModel, proxyStreamOpts...)

		// Pack completion tokens and responseText into a temporary json to pass back to the retry worker
		type streamResult struct {
			Text   string `json:"text"`
			Tokens int    `json:"tokens"`
		}
		resJSON, _ := json.Marshal(streamResult{Text: responseText, Tokens: completionTokens})
		return resp.StatusCode, upstreamURL, resJSON, nil
	}

	// All non-2xx responses are captured by the universal error block above.
	// Only 2xx responses reach this point — safe to write to c.Writer.

	// --- AgentRouter native response translation (non-streaming) ---
	// Skipped for Anthropic front-door native passthrough: the response is
	// already in the exact Anthropic shape the client requested.
	if cred.Provider == "agentrouter" && !anthropicNative {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}
		if translated, ok := translateAgentRouterResponseToOpenAI(respBody, pctx.requestedModel); ok {
			respBody = translated
		}
		respBody = h.normalizeNonStreamingReasoning(respBody)
		respBody = h.sanitizeNonStreamingToolCalls(respBody)
		respBody = h.rewriteResponseModel(respBody, pctx.requestedModel)
		if pctx.anthropicFront {
			anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(respBody, pctx.requestedModel, pctx.anthropicInputTokens)
			if aErr == nil && len(anthBytes) > 0 {
				respBody = anthBytes
			}
		}
		for key, vals := range resp.Header {
			if strings.EqualFold(key, "Content-Length") {
				continue
			}
			for _, val := range vals {
				c.Writer.Header().Add(key, val)
			}
		}
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Writer.Header().Set("Content-Length", strconv.Itoa(len(respBody)))
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(respBody) //nolint:errcheck
		return resp.StatusCode, upstreamURL, respBody, nil
	}

	// --- Ollama native response translation (non-streaming) ---
	// Ollama Cloud returns a native JSON body for non-stream requests that must
	// be translated to OpenAI chat completion format before sending to the client.
	// Only Ollama requires buffering — all other providers stream directly below.
	if cred.Provider == "ollama" {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}
		if translated, ok := translateOllamaResponse(respBody); ok {
			respBody = translated
			respBody = h.normalizeNonStreamingReasoning(respBody)
			respBody = h.sanitizeNonStreamingToolCalls(respBody)
			respBody = h.rewriteResponseModel(respBody, pctx.requestedModel)
			if pctx.anthropicFront {
				anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(respBody, pctx.requestedModel, pctx.anthropicInputTokens)
				if aErr == nil && len(anthBytes) > 0 {
					respBody = anthBytes
				}
			}
			for key, vals := range resp.Header {
				for _, val := range vals {
					c.Writer.Header().Add(key, val)
				}
			}
			c.Writer.Header().Set("Content-Type", "application/json")
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(resp.StatusCode)
			c.Writer.Write(respBody) //nolint:errcheck
			return resp.StatusCode, upstreamURL, respBody, nil
		}
		// Ollama response didn't match native format — fall through to stream as-is
		if pctx.anthropicFront {
			anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(respBody, pctx.requestedModel, pctx.anthropicInputTokens)
			if aErr == nil && len(anthBytes) > 0 {
				respBody = anthBytes
			}
		}
		for key, vals := range resp.Header {
			for _, val := range vals {
				c.Writer.Header().Add(key, val)
			}
		}
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(respBody) //nolint:errcheck
		return resp.StatusCode, upstreamURL, respBody, nil
	}

	// --- 1min.ai Response Translation (non-streaming) ---
	// The 1min.ai API returns responses in an aiRecord envelope that must be
	// translated back to OpenAI-compatible format (chat completions, images, etc.)
	if cred.Provider == "1minai" {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}

		// Gap 1 Fix: 1min.ai can return HTTP 200 with an internal failure inside
		// the aiRecord envelope (e.g. insufficient credits, model unavailable).
		// Detect this and route through the retry loop so the credential is
		// penalized and failover can occur — same as a real HTTP error.
		if status, _ := jsonparser.GetString(respBody, "aiRecord", "status"); status == "FAILED" {
			errMsg, _ := jsonparser.GetString(respBody, "aiRecord", "aiRecordDetail", "resultObject", "[0]")
			if errMsg == "" {
				errMsg = "1min.ai internal failure"
			}
			h.logger.Warn("1min.ai returned internal failure (HTTP 200, status=FAILED)",
				zap.String("model", pctx.model),
				zap.String("error", errMsg),
				zap.ByteString("response", respBody),
			)
			// Return 502 so executeWithRetry penalizes the key and rotates
			return http.StatusBadGateway, upstreamURL, respBody, nil
		}

		translated, contentType, trErr := translateOneMinAIResponse(pctx.model, respBody)
		if trErr == nil && translated != nil {
			translated = h.normalizeNonStreamingReasoning(translated)
			translated = h.sanitizeNonStreamingToolCalls(translated)
			translated = h.rewriteResponseModel(translated, pctx.requestedModel)
			if pctx.anthropicFront {
				anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(translated, pctx.requestedModel, pctx.anthropicInputTokens)
				if aErr == nil && len(anthBytes) > 0 {
					translated = anthBytes
					contentType = "application/json"
				}
			}
			c.Writer.Header().Set("Content-Type", contentType)
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(resp.StatusCode)
			c.Writer.Write(translated) //nolint:errcheck
			return resp.StatusCode, upstreamURL, translated, nil
		}
		// Translation failed — write original body as fallback
		h.logger.Warn("1min.ai response translation failed, passing through original",
			zap.String("model", pctx.model),
			zap.Error(trErr),
		)
		for key, vals := range resp.Header {
			for _, val := range vals {
				c.Writer.Header().Add(key, val)
			}
		}
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(respBody) //nolint:errcheck
		return resp.StatusCode, upstreamURL, respBody, nil
	}

	// --- Gemini AI Studio Non-Streaming Response Translation ---
	// Google AI Studio's generateContent endpoint returns a completely different
	// response envelope from OpenAI. For non-streaming requests we translate the
	// entire response into an OpenAI chat completion JSON so clients are unaware
	// of the backend format.
	//
	// Note: Streaming requests (pctx.isStream) are handled by the transmux layer
	// in internal/transmux/gemini.go via the ProxyStream path — they never reach
	// this block.
	if cred.Provider == "gemini" && !pctx.isStream {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}

		// Gemini sometimes returns HTTP 200 with an error payload embedded in the body.
		// Detect and normalize these before attempting response translation.
		if errMsg, _, _, detectErr := jsonparser.Get(respBody, "error", "message"); detectErr == nil && len(errMsg) > 0 {
			normalized := normalizeGeminiError(resp.StatusCode, respBody)
			h.logger.Warn("gemini returned error in 200 body",
				zap.String("model", pctx.model),
				zap.ByteString("error_msg", errMsg),
			)
			c.Writer.Header().Set("Content-Type", "application/json")
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(normalized.HTTPStatus)
			c.Writer.Write(normalized.Body) //nolint:errcheck
			return normalized.HTTPStatus, upstreamURL, normalized.Body, nil
		}

		// Translate Gemini generateContent → OpenAI chat completion format
		translated, trErr := translateGeminiResponse(respBody, pctx.requestedModel)
		if trErr != nil {
			h.logger.Warn("gemini non-stream response translation failed, passing raw body",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			// Fall through with raw body — better than a 502 to the client
			c.Writer.Header().Set("Content-Type", "application/json")
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(resp.StatusCode)
			c.Writer.Write(respBody) //nolint:errcheck
			return resp.StatusCode, upstreamURL, respBody, nil
		}

		// Apply thinking-tag normalization and model rewriting
		translated = h.normalizeNonStreamingReasoning(translated)
		translated = h.sanitizeNonStreamingToolCalls(translated)
		translated = h.rewriteResponseModel(translated, pctx.requestedModel)
		if pctx.anthropicFront {
			anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(translated, pctx.requestedModel, pctx.anthropicInputTokens)
			if aErr == nil && len(anthBytes) > 0 {
				translated = anthBytes
			}
		}
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(translated) //nolint:errcheck
		return resp.StatusCode, upstreamURL, translated, nil
	}

	// --- Cloudflare Workers AI / AI Gateway Image Response Translation ---

	// Two response formats can arrive depending on the model type:
	//
	//  Native @cf/* models (/ai/run/{model}):
	//    Return a Cloudflare-specific envelope: {"success":true,"result":{"image":"<b64>"}}
	//    → Must be translated to OpenAI format: {"created":0,"data":[{"b64_json":"..."}]}
	//
	//  Third-party AI Gateway models (openai/*, recraft/*, krea/*, etc.):
	//    Already return OpenAI-native format directly: {"created":...,"data":[...]}
	//    → Pass through to the client unchanged.
	if cred.Provider == "cloudflare" && isCloudflareImageRequest(c.Request.URL.Path) {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}

		// Detect native Cloudflare envelope vs. OpenAI-native passthrough.
		// isCloudflareNativeImageResponse checks for a top-level "success" boolean
		// which is the canonical marker of the Cloudflare {success, result} envelope.
		if isCloudflareNativeImageResponse(respBody) {
			// Native @cf/* envelope: check for success=false failures first
			if success, err := jsonparser.GetBoolean(respBody, "success"); err == nil && !success {
				h.logger.Warn("cloudflare returned internal failure (HTTP 200, success=false)",
					zap.String("model", pctx.model),
					zap.ByteString("response", respBody),
				)
				return http.StatusBadGateway, upstreamURL, respBody, nil
			}

			// Translate native Cloudflare envelope → OpenAI format
			translated, contentType, trErr := translateCloudflareImageResponse(respBody)
			if trErr == nil && translated != nil {
				c.Writer.Header().Set("Content-Type", contentType)
				c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
				c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
				c.Writer.WriteHeader(resp.StatusCode)
				c.Writer.Write(translated) //nolint:errcheck
				return resp.StatusCode, upstreamURL, translated, nil
			}
			// Translation failed — write original body as fallback
			h.logger.Warn("cloudflare native image response translation failed, passing through original",
				zap.String("model", pctx.model),
				zap.Error(trErr),
			)
			for key, vals := range resp.Header {
				for _, val := range vals {
					c.Writer.Header().Add(key, val)
				}
			}
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(resp.StatusCode)
			c.Writer.Write(respBody) //nolint:errcheck
			return resp.StatusCode, upstreamURL, respBody, nil
		}

		// Third-party AI Gateway image model — response is already OpenAI-native format.
		// Pass through directly without any translation.
		h.logger.Debug("cloudflare third-party image model: passing through OpenAI-native response",
			zap.String("model", pctx.model),
		)
		for key, vals := range resp.Header {
			for _, val := range vals {
				c.Writer.Header().Add(key, val)
			}
		}
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(respBody) //nolint:errcheck
		return resp.StatusCode, upstreamURL, respBody, nil
	}

	// --- Cloudflare Workers AI Native Vision Response Translation ---
	if cred.Provider == "cloudflare" && isCloudflareNativeVisionModel(pctx.model) {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return http.StatusInternalServerError, upstreamURL, nil, readErr
		}

		translated, contentType, trErr := translateCloudflareVisionResponse(respBody, pctx.model)
		if trErr == nil && translated != nil {
			c.Writer.Header().Set("Content-Type", contentType)
			c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
			c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
			c.Writer.WriteHeader(resp.StatusCode)
			c.Writer.Write(translated) //nolint:errcheck
			return resp.StatusCode, upstreamURL, translated, nil
		}
		h.logger.Warn("cloudflare native vision response translation failed, passing through original",
			zap.String("model", pctx.model),
			zap.Error(trErr),
		)
		for key, vals := range resp.Header {
			for _, val := range vals {
				c.Writer.Header().Add(key, val)
			}
		}
		c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(resp.StatusCode)
		c.Writer.Write(respBody) //nolint:errcheck
		return resp.StatusCode, upstreamURL, respBody, nil
	}

	// Normal success path — capture text responses for logging history.
	// Normal success path — capture text responses for logging history.
	// For JSON (application/json) content types, we read into memory to allow telemetry indexing.
	// For Image generation endpoints that return raw binary (image/png, image/jpeg), translate to OpenAI b64_json format.
	// For other binary formats (audio stream, file downloads), stream directly via io.Copy to avoid memory bloat.
	isJSON := strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json")
	var respBody []byte

	if strings.Contains(c.Request.URL.Path, "/images/") && !isJSON {
		respBody, _ = io.ReadAll(resp.Body)
		if isBinaryImage(respBody) {
			b64 := base64.StdEncoding.EncodeToString(respBody)
			respJSON := map[string]interface{}{
				"created": time.Now().Unix(),
				"data": []map[string]interface{}{
					{"b64_json": b64},
				},
			}
			respBody, _ = json.Marshal(respJSON)
			isJSON = true
			resp.Header.Set("Content-Type", "application/json")
		}
	} else if isJSON {
		respBody, _ = io.ReadAll(resp.Body)
		// Normalize inline <think> tags into structured reasoning_content so
		// front-ends render a thinking panel instead of raw reasoning text.
		respBody = h.normalizeNonStreamingReasoning(respBody)
		respBody = h.sanitizeNonStreamingToolCalls(respBody)
		respBody = h.rewriteResponseModel(respBody, pctx.requestedModel)
	}

	// ── Anthropic front door: non-streaming response translation (Phase 3+4) ──
	// Everything below writes the OpenAI chat-completion body to the client.
	// Anthropic front-door requests served by non-native providers must
	// receive an Anthropic Messages envelope instead, so re-shape the
	// normalized OpenAI body now (thinking blocks, text, tool_use, usage with
	// prompt-cache fields, Anthropic stop_reason vocabulary).
	if pctx.anthropicFront && !anthropicNative && isJSON && len(respBody) > 0 {
		anthBytes, _, _, _, aErr := TranslateOpenAIToAnthropicResponse(respBody, pctx.requestedModel, pctx.anthropicInputTokens)
		if aErr == nil && len(anthBytes) > 0 {
			respBody = anthBytes
		}
	}

	for key, vals := range resp.Header {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, val := range vals {
			c.Writer.Header().Add(key, val)
		}
	}
	if isJSON {
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Writer.Header().Set("Content-Length", strconv.Itoa(len(respBody)))
	}
	c.Writer.Header().Set("X-Gateway-Provider", cred.Provider)
	c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
	c.Writer.WriteHeader(resp.StatusCode)

	if isJSON {
		c.Writer.Write(respBody)
		return resp.StatusCode, upstreamURL, respBody, nil
	} else {
		io.Copy(c.Writer, resp.Body) //nolint:errcheck
		return resp.StatusCode, upstreamURL, nil, nil
	}
}

// normalizeNonStreamingReasoning extracts inline <think>…</think> blocks from a
// non-streaming chat completion's message content and surfaces them as a
// structured "reasoning_content" field. This lets OpenAI-compatible front-ends
// (OpenWebUI, LibreChat, Cline, …) render a dedicated thinking panel instead of
// dumping the raw reasoning as chat text.
//
// The function is a fast no-op when:
//   - the body is not a chat-completion shape (no choices[0].message.content)
//   - a structured reasoning_content field is already present (e.g. DeepSeek)
//   - no <think> tag is present in the content
//
// Only the "content" and "reasoning_content" keys are patched via jsonparser.Set,
// preserving every other field on the message object (tool_calls, name, refusal,
// annotations, …) without a full unmarshal/marshal round-trip.
func (h *Handler) normalizeNonStreamingReasoning(body []byte) []byte {
	// GetString (not Get) returns a properly unescaped Go string so json.Marshal
	// re-encodes it correctly — avoiding the double-escaping (\n → \\n) that raw
	// jsonparser.Get bytes would cause. Same caveat documented in ollama.go.
	content, err := jsonparser.GetString(body, "choices", "[0]", "message", "content")
	if err != nil || len(content) == 0 {
		return body
	}

	// Skip if structured reasoning_content is already present — the upstream
	// already separates thinking for us.
	if reasoning, rErr := jsonparser.GetString(body, "choices", "[0]", "message", "reasoning_content"); rErr == nil && reasoning != "" {
		return body
	}

	// Extract every <think>…</think> block, accumulating reasoning text and the
	// remaining clean content. Handles multiple blocks and an unclosed trailing
	// <think> gracefully.
	var reasoningParts []string
	var cleanBuilder strings.Builder
	remaining := content
	for {
		startIdx := strings.Index(remaining, "<think>")
		if startIdx == -1 {
			cleanBuilder.WriteString(remaining)
			break
		}
		cleanBuilder.WriteString(remaining[:startIdx])
		afterTag := remaining[startIdx+len("<think>"):]
		endIdx := strings.Index(afterTag, "</think>")
		if endIdx == -1 {
			if trimmed := strings.TrimSpace(afterTag); trimmed != "" {
				reasoningParts = append(reasoningParts, trimmed)
			}
			break
		}
		if trimmed := strings.TrimSpace(afterTag[:endIdx]); trimmed != "" {
			reasoningParts = append(reasoningParts, trimmed)
		}
		remaining = afterTag[endIdx+len("</think>"):]
	}

	if len(reasoningParts) == 0 {
		return body
	}

	cleanContent := strings.TrimSpace(cleanBuilder.String())
	reasoningBlock := strings.Join(reasoningParts, "\n")

	cleanJSON, err := json.Marshal(cleanContent)
	if err != nil {
		return body
	}
	updated, err := jsonparser.Set(body, cleanJSON, "choices", "[0]", "message", "content")
	if err != nil {
		return body
	}
	reasoningJSON, err := json.Marshal(reasoningBlock)
	if err != nil {
		return body
	}
	updated, err = jsonparser.Set(updated, reasoningJSON, "choices", "[0]", "message", "reasoning_content")
	if err != nil {
		return body
	}
	return updated
}

// sanitizeNonStreamingToolCalls ensures that tool_calls in non-streaming responses
// contain valid non-null strings for id, type, function.name, and function.arguments.
func (h *Handler) sanitizeNonStreamingToolCalls(body []byte) []byte {
	if len(body) == 0 || !bytes.Contains(body, []byte(`"tool_calls"`)) {
		return body
	}
	if !bytes.Contains(body, []byte(`:null`)) && !bytes.Contains(body, []byte(`: null`)) {
		return body
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return body
	}

	choices, ok := payload["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return body
	}

	modified := false
	for _, choice := range choices {
		chMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}
		msg, ok := chMap["message"].(map[string]interface{})
		if !ok {
			continue
		}
		tcs, ok := msg["tool_calls"].([]interface{})
		if !ok {
			continue
		}
		for i, tc := range tcs {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}
			if tcMap["id"] == nil || tcMap["id"] == "" {
				tcMap["id"] = fmt.Sprintf("call_%x_%d", time.Now().UnixNano(), i)
				modified = true
			}
			if tcMap["type"] == nil || tcMap["type"] == "" {
				tcMap["type"] = "function"
				modified = true
			}
			if fn, ok := tcMap["function"].(map[string]interface{}); ok {
				var toolName string
				if fn["name"] == nil {
					fn["name"] = ""
					modified = true
				} else if nameStr, isStr := fn["name"].(string); isStr {
					toolName = nameStr
				}
				if fn["arguments"] == nil {
					fn["arguments"] = "{}"
					modified = true
				} else if argsStr, isStr := fn["arguments"].(string); isStr {
					repaired := RepairToolArguments(toolName, argsStr)
					if repaired != argsStr {
						fn["arguments"] = repaired
						modified = true
					}
				}
			}
		}
	}

	if !modified {
		return body
	}
	cleaned, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return cleaned
}

// translateOllamaResponse converts a native Ollama non-streaming response body
// into an OpenAI-compatible chat completion JSON.
//
// Ollama /api/chat (non-stream) returns:
//
//	{"model":"llama4","message":{"role":"assistant","content":"Hello!"},"done":true,...}
//
// Ollama /api/generate (non-stream) returns:
//
//	{"model":"llama4","response":"Hello!","done":true,...}
//
// Both are translated to the OpenAI /v1/chat/completions shape.
// Returns (translated, true) on success; (nil, false) if not a known Ollama shape.
func translateOllamaResponse(data []byte) ([]byte, bool) {
	var content string

	// Try /api/chat shape first.
	// Use GetString (not Get) to avoid double-escaping: Get returns raw JSON bytes
	// where \n is still 2 bytes (`\` + `n`). GetString returns a Go string with
	// actual characters (real newline, >, etc.) which json.Marshal then encodes correctly.
	if msgContent, err := jsonparser.GetString(data, "message", "content"); err == nil {
		content = msgContent
	} else if response, err := jsonparser.GetString(data, "response"); err == nil {
		// /api/generate shape
		content = response
	} else {
		// Not a recognised Ollama native response — let it pass through unchanged
		return nil, false
	}

	model, _ := jsonparser.GetString(data, "model")
	promptTokens, _ := jsonparser.GetInt(data, "prompt_eval_count")
	completionTokens, _ := jsonparser.GetInt(data, "eval_count")

	// Build an OpenAI-compatible chat completion response
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type choice struct {
		Index        int     `json:"index"`
		Message      message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	}
	type usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	}
	type completion struct {
		ID      string   `json:"id"`
		Object  string   `json:"object"`
		Model   string   `json:"model"`
		Choices []choice `json:"choices"`
		Usage   usage    `json:"usage"`
	}

	result := completion{
		ID:     "chatcmpl-gate",
		Object: "chat.completion",
		Model:  model,
		Choices: []choice{
			{
				Index:        0,
				Message:      message{Role: "assistant", Content: content},
				FinishReason: "stop",
			},
		},
		Usage: usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	out, err := json.Marshal(result)
	if err != nil {
		return nil, false
	}
	return out, true
}

// findPoolByPrefix searches for a pool matching a model name prefix.
// E.g., "gpt-4o-2024-05-13" matches pool "gpt-4o".
//
// Named-provider blocks handle known slash-prefixed namespaces:
//
//	nvidia/, ollama/, 1min/, cloudflare/, sarvam/, puter/, zenmux/, gemini/
//
// A generic catch-all at the end applies the same progressive-prefix loop
// to any other slash-containing model (e.g. "jiekou/baidu/ernie-4.5-300b-a47b-paddle",
// or custom providers registered via the credential prefix field), so new
// provider namespaces work automatically without requiring a new hard-coded block.
func (h *Handler) findPoolByPrefix(model string) (interface{}, bool) {
	// Handle NVIDIA slash-separated model names (e.g., "nvidia/nvidia/nemotron-3-super-120b-a12b")
	if strings.HasPrefix(model, "nvidia/") {
		// Try progressively shorter slash-separated prefixes
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "nvidia"
		if val, found := h.cache.Get(cache.PoolKey("nvidia")); found {
			return val, true
		}
	}

	// Handle Ollama slash-separated model names (e.g., "ollama/llama3:8b")
	if strings.HasPrefix(model, "ollama/") {
		// Try progressively shorter slash-separated prefixes
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "ollama"
		if val, found := h.cache.Get(cache.PoolKey("ollama")); found {
			return val, true
		}
	}

	// Handle 1min.ai slash-separated model names (e.g., "1min/gpt-4o", "1min/flux-schnell")
	if strings.HasPrefix(model, "1min/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "1min"
		if val, found := h.cache.Get(cache.PoolKey("1min")); found {
			return val, true
		}
	}

	// Handle Cloudflare slash-separated model names
	// e.g. "cloudflare/@cf/meta/llama-3.1-8b-instruct"
	if strings.HasPrefix(model, "cloudflare/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "cloudflare"
		if val, found := h.cache.Get(cache.PoolKey("cloudflare")); found {
			return val, true
		}
	}

	// Handle Sarvam AI slash-separated model names (e.g. "sarvam/sarvam-105b")
	if strings.HasPrefix(model, "sarvam/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "sarvam"
		if val, found := h.cache.Get(cache.PoolKey("sarvam")); found {
			return val, true
		}
	}

	// Handle Puter slash-separated model names (e.g. "puter/gpt-4o-mini")
	if strings.HasPrefix(model, "puter/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "puter"
		if val, found := h.cache.Get(cache.PoolKey("puter")); found {
			return val, true
		}
	}

	// Handle AgentRouter slash-separated model names (e.g. "agentrouter/gpt-4o")
	if strings.HasPrefix(model, "agentrouter/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "agentrouter"
		if val, found := h.cache.Get(cache.PoolKey("agentrouter")); found {
			return val, true
		}
	}

	// Handle ZenMux slash-separated model names (e.g. "zenmux/anthropic/claude-3-5-sonnet")
	if strings.HasPrefix(model, "zenmux/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// Also try just the provider namespace key "zenmux"
		if val, found := h.cache.Get(cache.PoolKey("zenmux")); found {
			return val, true
		}
	}
	// Handle Gemini slash-separated model names (e.g. "gemini/gemini-3.5-flash")
	if strings.HasPrefix(model, "gemini/") {
		// 1. Try to match the exact pattern pool first
		if val, found := h.cache.Get(cache.PoolKey(model)); found {
			return val, true
		}
		// 2. If not found, normalize the model name and try matching that pool
		cleanModel := strings.TrimPrefix(model, "gemini/")
		mappedClean := normalizeGeminiModel(cleanModel)
		mappedPool := "gemini/" + mappedClean
		if val, found := h.cache.Get(cache.PoolKey(mappedPool)); found {
			h.logger.Debug("mapping unknown gemini model to fallback pool",
				zap.String("requested", model),
				zap.String("mapped", mappedPool),
			)
			return val, true
		}
		// 3. Progressive path matching fallback
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				return val, true
			}
		}
		// 4. Fallback to generic gemini key namespace
		if val, found := h.cache.Get(cache.PoolKey("gemini")); found {
			return val, true
		}
	}

	// Generic catch-all for any slash-prefixed model not matched by the
	// named-provider blocks above (e.g. "jiekou/baidu/ernie-4.5-300b-a47b-paddle",
	// "mycompany/gpt-4o", or any other custom prefix registered via the
	// credential prefix field).
	//
	// Strategy (mirrors the per-provider blocks):
	//  1. Try the exact full model string first (already tried above via
	//     h.cache.Get(cache.PoolKey(model)) in Handle; repeated here for clarity).
	//  2. Walk progressively shorter slash-joined prefixes longest → shortest.
	//  3. Try just the first slash token alone (the bare namespace key).
	if strings.Contains(model, "/") {
		slashParts := strings.Split(model, "/")
		for i := len(slashParts) - 1; i >= 1; i-- {
			prefix := strings.Join(slashParts[:i], "/")
			if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
				h.logger.Debug("generic slash-prefix pool match",
					zap.String("model", model),
					zap.String("matched_prefix", prefix),
				)
				return val, true
			}
		}
	}

	// Try progressively shorter dash-separated prefixes
	parts := strings.Split(model, "-")
	for i := len(parts) - 1; i >= 1; i-- {
		prefix := strings.Join(parts[:i], "-")
		if val, found := h.cache.Get(cache.PoolKey(prefix)); found {
			return val, true
		}
	}
	// Try provider-specific patterns (e.g., "gemini-" prefix)
	if strings.HasPrefix(model, "gemini-") {
		mappedClean := normalizeGeminiModel(model)
		if val, found := h.cache.Get(cache.PoolKey(mappedClean)); found {
			h.logger.Debug("mapping unknown clean gemini model to fallback pool",
				zap.String("requested", model),
				zap.String("mapped", mappedClean),
			)
			return val, true
		}
		// Fall back to the generic gemini studio pool namespace so a standalone
		// "gemini-<unknown>" request still binds to an AI Studio credential pool
		// rather than failing with "no routing pool configured". Mirrors the
		// slash-prefixed branch above so both routing forms share one fallback.
		if val, found := h.cache.Get(cache.PoolKey(credentials.ProviderGemini)); found {
			h.logger.Debug("mapping unknown clean gemini model to studio namespace pool",
				zap.String("requested", model),
				zap.String("namespace", credentials.ProviderGemini),
			)
			return val, true
		}
	}
	for _, prefix := range []string{"gpt-", "claude-", "gemini-", "deepseek-"} {
		if strings.HasPrefix(model, prefix) {
			baseModel := strings.TrimSuffix(prefix, "-")
			if val, found := h.cache.Get(cache.PoolKey(baseModel)); found {
				return val, true
			}
		}
	}
	return nil, false
}

// isCredentialAuthError returns true for status codes that indicate the
// specific API key is rejected by the provider, or the model is not accessible
// on this account's plan. All of these warrant an immediate key rotation with
// a long cooldown — the key is broken for this model, not the request itself.
//
//   - 400 Bad Request:          malformed auth header, or model alias not found on account
//   - 401 Unauthorized:         invalid or expired API key
//   - 402 Payment Required:     billing lapsed / quota exhausted
//   - 403 Forbidden:            key tier insufficient for this model
//   - 404 Not Found:            model not accessible under this key's plan
//   - 422 Unprocessable Entity: key accepted but provider rejects this request shape
func isCredentialAuthError(status int) bool {
	switch status {
	case http.StatusUnauthorized, // 401
		http.StatusPaymentRequired,     // 402
		http.StatusForbidden,           // 403
		http.StatusNotFound,            // 404
		http.StatusUnprocessableEntity: // 422
		return true
	}
	return false
}

// cooldownForStatus returns the appropriate cooldown duration for a given
// upstream HTTP status code. Covers standard codes, auth anomalies, and
// non-standard codes from load balancers and custom proxies (444, 520, 524, 599…).
// Uses short transient cooldowns so credentials and models are never permanently
// or semi-permanently deactivated on failure.
func cooldownForStatus(status int) time.Duration {
	switch {
	case status == http.StatusUnauthorized ||
		status == http.StatusPaymentRequired ||
		status == http.StatusForbidden ||
		status == http.StatusNotFound:
		// Quota depletion, suspended account, or invalid credentials
		return 30 * time.Minute

	case status == http.StatusUnprocessableEntity:
		return 10 * time.Minute

	case status == http.StatusTooManyRequests:
		return 15 * time.Second // Rate limited — short wait for window reset

	case status == http.StatusInternalServerError || status == http.StatusBadGateway:
		return 3 * time.Second // Server error — transient cooldown

	case status == http.StatusServiceUnavailable:
		return 5 * time.Second // Overloaded — transient cooldown

	case status == http.StatusGatewayTimeout:
		return 2 * time.Second // Timeout — short cooldown, try others first

	default:
		// Non-standard codes (nginx 444, Cloudflare 520/524, custom 599, etc.)
		return 5 * time.Second
	}
}

// ModelDetail represents model information in dual OpenAI and Anthropic format.
// Both schemas are additive and non-conflicting:
// OpenAI fields: id, object, created, owned_by, capabilities
// Anthropic fields: id, type, display_name, created_at
type ModelDetail struct {
	ID           string          `json:"id"`
	Object       string          `json:"object"`
	Type         string          `json:"type"`
	DisplayName  string          `json:"display_name"`
	Created      int64           `json:"created"`
	CreatedAt    string          `json:"created_at"`
	OwnedBy      string          `json:"owned_by"`
	Capabilities map[string]bool `json:"capabilities,omitempty"`
}

// ModelListResponse represents the dual-compatible list models response.
// OpenAI fields: object, data
// Anthropic fields: data, has_more, first_id, last_id
type ModelListResponse struct {
	Object  string        `json:"object"`
	Data    []ModelDetail `json:"data"`
	HasMore bool          `json:"has_more"`
	FirstID string        `json:"first_id"`
	LastID  string        `json:"last_id"`
}

func newModelDetail(pattern string, caps map[string]bool, now int64) ModelDetail {
	return ModelDetail{
		ID:           pattern,
		Object:       "model",
		Type:         "model",
		DisplayName:  pattern,
		Created:      now,
		CreatedAt:    time.Unix(now, 0).UTC().Format(time.RFC3339),
		OwnedBy:      "clever-ai-gate",
		Capabilities: caps,
	}
}

func newModelListResponse(data []ModelDetail) ModelListResponse {
	if data == nil {
		data = []ModelDetail{}
	}
	var firstID, lastID string
	if len(data) > 0 {
		firstID = data[0].ID
		lastID = data[len(data)-1].ID
	}
	return ModelListResponse{
		Object:  "list",
		Data:    data,
		HasMore: false,
		FirstID: firstID,
		LastID:  lastID,
	}
}

// ListModels returns a list of configured active model pools with their
// detected capabilities in dual OpenAI and Anthropic format.
//
// Read order:
//  1. If refresh requested (?refresh=true or Cache-Control: no-cache): bypass cache and query DB
//  2. Redis L2 (< 1ms, shared across all gateway replicas)
//  3. Ristretto L1 (sub-microsecond, local in-process)
//  4. PostgreSQL DB fallback (if caches missed, empty, or refresh requested)
func (h *Handler) ListModels(c *gin.Context) {
	var data []ModelDetail
	now := time.Now().Unix()

	forceRefresh := c.Query("refresh") == "true" || c.GetHeader("Cache-Control") == "no-cache"

	if !forceRefresh {
		// --- Tier 1: Redis L2 cache (serves all gateway replicas with single-digit ms) ---
		if h.redisCacheMgr != nil {
			var redisModels []cache.ActiveModelEntry
			found, _ := h.redisCacheMgr.GetJSON(c.Request.Context(), cache.KeyActiveModels, &redisModels)
			if found && len(redisModels) > 0 {
				data = make([]ModelDetail, len(redisModels))
				for i, m := range redisModels {
					data[i] = newModelDetail(m.Pattern, m.Capabilities, now)
				}
				c.JSON(http.StatusOK, newModelListResponse(data))
				return
			}
		}

		// --- Tier 2: Ristretto L1 cache (sub-microsecond, local in-process) ---
		val, found := h.cache.Get("system:active_models")
		if found {
			// New enriched format: []credentials.ActiveModel
			if models, ok := val.([]credentials.ActiveModel); ok && len(models) > 0 {
				data = make([]ModelDetail, len(models))
				for i, m := range models {
					data[i] = newModelDetail(m.Pattern, m.Capabilities, now)
				}
				// Back-fill Redis L2 asynchronously from Ristretto L1 data.
				// This heals a Redis cache miss without blocking the response.
				if h.redisCacheMgr != nil && len(models) > 0 {
					go func(m []credentials.ActiveModel) {
						entries := make([]cache.ActiveModelEntry, len(m))
						for i, am := range m {
							entries[i] = cache.ActiveModelEntry{
								Pattern:      am.Pattern,
								Capabilities: am.Capabilities,
							}
						}
						h.redisCacheMgr.SetJSON(context.Background(), cache.KeyActiveModels, entries, cache.DefaultCacheTTL)
					}(models)
				}
				c.JSON(http.StatusOK, newModelListResponse(data))
				return
			}
		}
	}

	// --- Tier 3: PostgreSQL DB fallback (when cache missed, empty, or refresh requested) ---
	if h.db != nil {
		rows, err := h.db.Query(c.Request.Context(),
			`SELECT model_pattern, capabilities FROM model_pools WHERE is_active = true ORDER BY model_pattern ASC`)
		if err == nil {
			defer rows.Close()
			var activeModels []credentials.ActiveModel
			for rows.Next() {
				var pattern string
				var capsJSON []byte
				if scanErr := rows.Scan(&pattern, &capsJSON); scanErr == nil {
					am := credentials.ActiveModel{Pattern: pattern}
					if len(capsJSON) > 0 && string(capsJSON) != "{}" {
						_ = json.Unmarshal(capsJSON, &am.Capabilities)
					}
					activeModels = append(activeModels, am)
				}
			}

			if len(activeModels) > 0 {
				data = make([]ModelDetail, len(activeModels))
				for i, m := range activeModels {
					data[i] = newModelDetail(m.Pattern, m.Capabilities, now)
				}

				// Populate both Ristretto L1 and Redis L2
				h.cache.Set("system:active_models", activeModels, 1000)
				if h.redisCacheMgr != nil {
					redisModels := make([]cache.ActiveModelEntry, len(activeModels))
					for i, am := range activeModels {
						redisModels[i] = cache.ActiveModelEntry{
							Pattern:      am.Pattern,
							Capabilities: am.Capabilities,
						}
					}
					h.redisCacheMgr.SetJSON(context.Background(), cache.KeyActiveModels, redisModels, cache.DefaultCacheTTL)
				}
			}
		}
	}

	if data == nil {
		data = []ModelDetail{}
	}

	c.JSON(http.StatusOK, newModelListResponse(data))
}

// GetModel retrieves a single model by ID, supporting slash-prefixed model names
// (e.g. "anthropic/claude-3-7-sonnet" or "deepseek-ai/DeepSeek-R1").
// Returns a dual-compatible model representation with both OpenAI and Anthropic fields.
func (h *Handler) GetModel(c *gin.Context) {
	rawModel := c.Param("model")
	modelID := strings.TrimPrefix(rawModel, "/")
	if modelID == "" {
		h.ListModels(c)
		return
	}

	now := time.Now().Unix()
	var foundModel *ModelDetail

	// 1. Check SyncManager if available
	if h.syncManager != nil {
		if pool := h.syncManager.GetPool(modelID); pool != nil {
			m := newModelDetail(pool.ModelPattern, nil, now)
			foundModel = &m
		}
	}

	// 2. Check in-process cache (system:active_models)
	if h.cache != nil {
		if val, found := h.cache.Get("system:active_models"); found {
			if models, ok := val.([]credentials.ActiveModel); ok {
				for _, m := range models {
					if m.Pattern == modelID {
						if foundModel != nil {
							foundModel.Capabilities = m.Capabilities
						} else {
							detail := newModelDetail(m.Pattern, m.Capabilities, now)
							foundModel = &detail
						}
						break
					}
				}
			}
		}
	}

	// 3. Check Redis L2 cache if available
	if foundModel == nil && h.redisCacheMgr != nil {
		var redisModels []cache.ActiveModelEntry
		found, _ := h.redisCacheMgr.GetJSON(c.Request.Context(), cache.KeyActiveModels, &redisModels)
		if found {
			for _, m := range redisModels {
				if m.Pattern == modelID {
					detail := newModelDetail(m.Pattern, m.Capabilities, now)
					foundModel = &detail
					break
				}
			}
		}
	}

	// 4. DB Fallback if available
	if foundModel == nil && h.db != nil {
		var pattern string
		var capsJSON []byte
		err := h.db.QueryRow(c.Request.Context(),
			`SELECT model_pattern, capabilities FROM model_pools WHERE model_pattern = $1 AND is_active = true`,
			modelID,
		).Scan(&pattern, &capsJSON)
		if err == nil {
			var caps map[string]bool
			if len(capsJSON) > 0 && string(capsJSON) != "{}" {
				_ = json.Unmarshal(capsJSON, &caps)
			}
			detail := newModelDetail(pattern, caps, now)
			foundModel = &detail
		}
	}

	if foundModel != nil {
		c.JSON(http.StatusOK, foundModel)
		return
	}

	// Not found: dual-protocol error response
	isAnthropic := c.GetHeader("anthropic-version") != "" || (c.GetHeader("x-api-key") != "" && !strings.HasPrefix(c.GetHeader("Authorization"), "Bearer "))
	if isAnthropic {
		AnthropicErrorResponse(c, http.StatusNotFound, "not_found_error", fmt.Sprintf("model not found: %s", modelID))
		return
	}

	c.JSON(http.StatusNotFound, gin.H{
		"error": gin.H{
			"message": fmt.Sprintf("The model '%s' does not exist", modelID),
			"type":    "invalid_request_error",
			"param":   "model",
			"code":    "model_not_found",
		},
	})
}

// --- In-Place Model Prefix Stripping ---

// stripModelPrefixInPlace removes a routing prefix from the JSON "model" field
// value directly within the existing byte slice, avoiding the full-body copy
// that bytes.Replace would allocate.
//
// For example, given body containing `"model":"nvidia/glm-5.1"` and
// prefixToStrip="nvidia/", the result is `"model":"glm-5.1"` with the tail
// of the body shifted backward to close the gap.
//
// The caller must ensure that:
//   - body has enough capacity (always true when shrinking)
//   - the model string (fullModel) is an independent copy, NOT an unsafe
//     view into body — the in-place modification would corrupt such a view.
//
// If fullModel does not actually carry prefixToStrip, the body is returned
// unchanged. This guard makes the function safe to call for routing forms that
// have no prefix to strip (e.g. a standalone "gemini-3.5-flash" routed through
// the isGemini path), where the unguarded slice expression below would otherwise
// compute a bogus upstream model and silently corrupt the request body.
func stripModelPrefixInPlace(body []byte, fullModel, prefixToStrip string) []byte {
	if !strings.HasPrefix(fullModel, prefixToStrip) {
		return body
	}
	oldToken := []byte(`"` + fullModel + `"`)
	upstreamModel := fullModel[len(prefixToStrip):]
	newToken := []byte(`"` + upstreamModel + `"`)

	idx := bytes.Index(body, oldToken)
	if idx == -1 {
		return body
	}

	diff := len(oldToken) - len(newToken)

	// Shift the tail backward to close the gap left by the shorter replacement.
	// Go's copy() handles overlapping dst/src correctly when dst < src.
	copy(body[idx+len(newToken):], body[idx+len(oldToken):])
	copy(body[idx:], newToken)

	return body[:len(body)-diff]
}

// --- NVIDIA Reasoning Parameter Injection ---

// injectNvidiaParams injects NVIDIA-specific reasoning parameters into the request body.
//
// Parameters injected:
//   - reasoning_budget: Set to match max_tokens (default 4096 if absent)
//   - chat_template_kwargs: {"enable_thinking": true}
//
// This enables NVIDIA models to return structured reasoning content
// that the NvidiaTransmuxer can normalize into reasoning_content deltas.
//
// The injection uses byte-level manipulation to avoid JSON unmarshal/marshal overhead.
// The caller receives a new byte slice — the original body is not modified.
func injectNvidiaParams(body, scanSlice []byte, logger *zap.Logger) []byte {
	// Check if reasoning_budget is already present (avoid double injection)
	if bytes.Contains(scanSlice, []byte(`"reasoning_budget"`)) {
		return body
	}

	// Extract max_tokens for reasoning_budget (default 4096)
	reasoningBudget := 4096
	if maxTokens, err := jsonparser.GetInt(scanSlice, "max_tokens"); err == nil && maxTokens > 0 {
		reasoningBudget = int(maxTokens)
	}

	// Build the injection payload
	injection := []byte(`,"reasoning_budget":` + strconv.Itoa(reasoningBudget) + `,"chat_template_kwargs":{"enable_thinking":true}`)

	// Find the last '}' in the body (the closing brace of the root JSON object)
	lastBrace := bytes.LastIndexByte(body, '}')
	if lastBrace < 0 {
		logger.Warn("nvidia param injection skipped: no closing brace found in body")
		return body
	}

	// Build the new body: everything before '}' + injection + '}'
	newBody := make([]byte, 0, len(body)+len(injection))
	newBody = append(newBody, body[:lastBrace]...)
	newBody = append(newBody, injection...)
	newBody = append(newBody, body[lastBrace:]...)

	logger.Debug("nvidia reasoning params injected",
		zap.Int("reasoning_budget", reasoningBudget),
		zap.Int("original_size", len(body)),
		zap.Int("new_size", len(newBody)),
	)

	return newBody
}

// supportsNvidiaReasoning reports whether the upstream model (nvidia/ prefix already
// stripped) supports NVIDIA's reasoning extensions (reasoning_budget,
// chat_template_kwargs.enable_thinking). Standard models like GLM, Llama-base,
// and Mistral reject these params with a 500, so they must be excluded.
//
// IMPORTANT: Only the final slash-separated segment (the model name itself) is
// matched against reasoning keywords. Matching the full string would cause false
// positives when the *organisation/namespace* portion contains a keyword — for
// example "thinkingmachines/inkling" would incorrectly trigger on "think" inside
// "thinkingmachines", causing reasoning_budget injection that NVIDIA rejects.
func supportsNvidiaReasoning(upstreamModel string) bool {
	// Extract only the model name (last "/"-separated segment).
	// e.g. "thinkingmachines/inkling"  → "inkling"   (org name excluded)
	//      "deepseek/deepseek-r1"      → "deepseek-r1" (still matches -r1)
	//      "nvidia/nemotron-3-super"   → "nemotron-3-super" (still matches nemotron)
	modelName := upstreamModel
	if lastSlash := strings.LastIndex(upstreamModel, "/"); lastSlash >= 0 {
		modelName = upstreamModel[lastSlash+1:]
	}
	lower := strings.ToLower(modelName)
	return strings.Contains(lower, "nemotron") ||
		strings.Contains(lower, "-r1") ||
		strings.Contains(lower, "reasoning") ||
		strings.Contains(lower, "think")
}

// extractModelFromMultipart extracts the "model" field value from a
// multipart/form-data body using a fast byte-level scan.
//
// In multipart encoding, the model field appears as:
//
//	Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n
//
// We search for the pattern `name="model"` followed by the double CRLF
// separator, then read until the next \r\n or boundary marker.
// This avoids parsing the full multipart structure (no heap allocations
// from mime/multipart) and handles Whisper, DALL-E, and file upload
// requests in sub-microsecond time.
func extractModelFromMultipart(body []byte) string {
	// Look for the field marker
	marker := []byte(`name="model"`)
	idx := bytes.Index(body, marker)
	if idx < 0 {
		return ""
	}

	// Skip past the marker to find the value.
	// After name="model" there may be a CRLF, then optional headers, then
	// a blank CRLF line, then the actual value.
	start := idx + len(marker)
	rest := body[start:]

	// Find the double CRLF (\r\n\r\n) that separates headers from the value
	sep := bytes.Index(rest, []byte("\r\n\r\n"))
	if sep < 0 {
		// Try LF-only variant (some clients use \n\n)
		sep = bytes.Index(rest, []byte("\n\n"))
		if sep < 0 {
			return ""
		}
		rest = rest[sep+2:]
	} else {
		rest = rest[sep+4:]
	}

	// The value ends at the next \r\n (before the boundary)
	end := bytes.Index(rest, []byte("\r\n"))
	if end < 0 {
		end = bytes.Index(rest, []byte("\n"))
		if end < 0 {
			return ""
		}
	}

	value := bytes.TrimSpace(rest[:end])
	return string(value)
}

// --- Telemetry Log Extraction Helpers ---

func extractPromptText(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	// Try "prompt" field first
	if prompt, err := jsonparser.GetString(body, "prompt"); err == nil {
		return prompt
	}
	// Extract the last user message content
	var lastMessage string
	_, _ = jsonparser.ArrayEach(body, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		if content, err := jsonparser.GetString(value, "content"); err == nil {
			lastMessage = content
		}
	}, "messages")
	return lastMessage
}

func extractResponseText(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var sb strings.Builder
	if content, err := jsonparser.GetString(body, "choices", "[0]", "message", "content"); err == nil && content != "" {
		sb.WriteString(content)
	}
	if reasoning, err := jsonparser.GetString(body, "choices", "[0]", "message", "reasoning_content"); err == nil && reasoning != "" {
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString(reasoning)
	}
	if sb.Len() > 0 {
		return sb.String()
	}
	if content, err := jsonparser.GetString(body, "choices", "[0]", "text"); err == nil {
		return content
	}
	// Anthropic response format: content: [{"type":"text","text":"..."}, {"type":"thinking","thinking":"..."}]
	if cVal, cType, _, err := jsonparser.Get(body, "content"); err == nil && cType == jsonparser.Array {
		jsonparser.ArrayEach(cVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if t, _ := jsonparser.GetString(block, "type"); t == "text" {
				if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
					if sb.Len() > 0 {
						sb.WriteByte('\n')
					}
					sb.WriteString(txt)
				}
			} else if t == "thinking" {
				if th, _ := jsonparser.GetString(block, "thinking"); th != "" {
					if sb.Len() > 0 {
						sb.WriteByte('\n')
					}
					sb.WriteString(th)
				}
			}
		})
	}
	return sb.String()
}

func extractTokens(body []byte, promptOrCompletion string) int {
	if val, err := jsonparser.GetInt(body, "usage", promptOrCompletion+"_tokens"); err == nil {
		return int(val)
	}
	// Anthropic response format
	if promptOrCompletion == "prompt" {
		if val, err := jsonparser.GetInt(body, "usage", "input_tokens"); err == nil {
			return int(val)
		}
	} else if promptOrCompletion == "completion" {
		if val, err := jsonparser.GetInt(body, "usage", "output_tokens"); err == nil {
			return int(val)
		}
	}
	return 0
}

// isContextTimeoutError reports whether a transport error is a context timeout
// or deadline exceeded, enabling more precise HTTP status mapping (504 vs 502).
func isContextTimeoutError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "context deadline exceeded") ||
		strings.Contains(msg, "context canceled") ||
		strings.Contains(msg, "timeout")
}

// buildAttemptSummary renders the diagnostic message embedded in the final
// canonical OpenAI error envelope when every credential — across every retry
// round and every exact-model fallback pool — has been exhausted. It reports:
//   - the total number of upstream credential attempts,
//   - every pool that was tried and how many complete retry rounds it consumed,
//   - every failing credential, with repeat failures across retry rounds
//     aggregated into compact "cred#id(provider)→status ×N" groups.
func buildAttemptSummary(model string, attempts []attemptRecord, roundsByPool map[string]int, poolsAttempted []string) string {
	if len(attempts) == 0 {
		return fmt.Sprintf("all upstream credentials for model %q were exhausted with no successful response", model)
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "all %d credential attempt(s) exhausted for model %q", len(attempts), model)

	if len(poolsAttempted) > 0 {
		sb.WriteString(" across pool(s):")
		for i, p := range poolsAttempted {
			if i > 0 {
				sb.WriteString(",")
			}
			if rounds, ok := roundsByPool[p]; ok && rounds > 0 {
				fmt.Fprintf(&sb, " %s (%d round(s))", p, rounds)
			} else {
				fmt.Fprintf(&sb, " %s", p)
			}
		}
	}

	// Aggregate repeat failures (retry rounds re-hit the same keys) into ×N
	// groups, preserving first-seen order.
	type attemptKey struct {
		credID   int
		provider string
		status   int
	}
	order := make([]attemptKey, 0, len(attempts))
	counts := make(map[attemptKey]int, len(attempts))
	for _, a := range attempts {
		k := attemptKey{credID: a.credID, provider: a.provider, status: a.statusCode}
		if counts[k] == 0 {
			order = append(order, k)
		}
		counts[k]++
	}

	const maxListedGroups = 12
	sb.WriteString(". Attempts: [")
	for i, k := range order {
		if i == maxListedGroups {
			fmt.Fprintf(&sb, ", + %d more attempt group(s)", len(order)-maxListedGroups)
			break
		}
		if i > 0 {
			sb.WriteString(", ")
		}
		if counts[k] > 1 {
			fmt.Fprintf(&sb, "cred#%d(%s)→%d ×%d", k.credID, k.provider, k.status, counts[k])
		} else {
			fmt.Fprintf(&sb, "cred#%d(%s)→%d", k.credID, k.provider, k.status)
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// formatOpenAIError returns a canonical OpenAI error envelope for any failure
// context. It is the single exit point for all error responses sent to clients
// — never raw upstream bytes, never unformatted text or HTML.
//
// Detection priority:
//  1. rawBody is already a valid OpenAI error envelope AND summary is empty
//     → return as-is (preserves provider transparency). When a summary IS
//     provided (total exhaustion), the gateway context must reach the client,
//     so the envelope falls through and is re-wrapped with the summary plus
//     the extracted upstream message.
//  2. rawBody is JSON with a "message", "error.message", or "detail" field
//     → extract and re-wrap into the canonical schema
//  3. HTML, plain-text, or empty body → sanitize and embed in the message field
func formatOpenAIError(statusCode int, rawBody []byte, summary string) []byte {
	type innerError struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    string  `json:"code"`
	}
	type envelope struct {
		Error innerError `json:"error"`
	}

	// 1. Already a valid OpenAI error envelope AND no gateway summary provided?
	//    Return unchanged. With a summary (total exhaustion), fall through so
	//    the client sees the gateway context plus the upstream message.
	if summary == "" && len(rawBody) > 0 {
		if errVal, dataType, _, parseErr := jsonparser.Get(rawBody, "error"); parseErr == nil &&
			dataType == jsonparser.Object && len(errVal) > 0 {
			if _, _, _, msgErr := jsonparser.Get(errVal, "message"); msgErr == nil {
				return rawBody // perfect OpenAI shape — preserve provider details
			}
		}
	}

	// 2. Try to extract any upstream message text
	var upstreamMsg string
	if len(rawBody) > 0 {
		if msg, err := jsonparser.GetString(rawBody, "message"); err == nil && msg != "" {
			upstreamMsg = msg
		}
		if upstreamMsg == "" {
			if msg, err := jsonparser.GetString(rawBody, "error", "message"); err == nil && msg != "" {
				upstreamMsg = msg
			}
		}
		if upstreamMsg == "" {
			// FastAPI / Python upstream pattern
			if msg, err := jsonparser.GetString(rawBody, "detail"); err == nil && msg != "" {
				upstreamMsg = msg
			}
		}
	}

	// 3. Build the canonical message
	message := summary
	if upstreamMsg != "" {
		message = summary + ". Last upstream message: " + upstreamMsg
	} else if len(rawBody) > 0 {
		raw := string(rawBody)
		if strings.Contains(raw, "<html") || strings.Contains(raw, "<!DOCTYPE") {
			raw = "[HTML response from upstream load balancer or CDN]"
		} else if len(raw) > 512 {
			raw = raw[:512] + "…"
		}
		if isPrintable(raw) {
			message = summary + ". Last upstream body: " + raw
		}
	}

	out, _ := json.Marshal(envelope{
		Error: innerError{
			Message: message,
			Type:    "gateway_exhaustion_error",
			Param:   nil,
			Code:    "all_providers_failed",
		},
	})
	return out
}

// isPrintable returns true if s consists only of printable ASCII/UTF-8 text.
// Used to filter out binary garbage from upstream responses before embedding
// them in human-readable error messages.
func isPrintable(s string) bool {
	for _, r := range s {
		if r < 0x20 && r != '\n' && r != '\r' && r != '\t' {
			return false
		}
	}
	return true
}

// OptimizePayloadForKiloCode mutates the request model for small completion tasks to save credits.
func (h *Handler) OptimizePayloadForKiloCode(bodyBytes []byte, model string) ([]byte, string) {
	maxTokens, err := jsonparser.GetInt(bodyBytes, "max_tokens")
	if err == nil && maxTokens > 0 && maxTokens <= 64 {
		// Inline completion detected! Downgrade to cheap puter/gpt-4o-mini model
		if model != "puter/gpt-4o-mini" && strings.HasPrefix(model, "puter/") {
			oldToken := []byte(`"` + model + `"`)
			newToken := []byte(`"puter/gpt-4o-mini"`)
			bodyBytes = bytes.Replace(bodyBytes, oldToken, newToken, 1)
			return bodyBytes, "puter/gpt-4o-mini"
		}
	}
	return bodyBytes, model
}

// generateRandomIP produces a random public IPv4 address string to de-fingerprint outbound edge calls.
func generateRandomIP() string {
	return fmt.Sprintf("%d.%d.%d.%d",
		rand.Intn(223-1)+1, // Avoid 0.x and multicast/reserved range
		rand.Intn(256),
		rand.Intn(256),
		rand.Intn(256),
	)
}

// rewriteResponseModel modifies the "model" field in a JSON response body to match the user's originally requested model name.
func (h *Handler) rewriteResponseModel(respBody []byte, requestedModel string) []byte {
	if len(respBody) == 0 || requestedModel == "" {
		return respBody
	}
	if _, err := jsonparser.GetString(respBody, "model"); err == nil {
		if updated, err := jsonparser.Set(respBody, []byte(`"`+requestedModel+`"`), "model"); err == nil {
			return updated
		}
	}
	return respBody
}

// isEmbeddingModel reports whether a model identifier is an embedding model.
func isEmbeddingModel(model string) bool {
	lower := strings.ToLower(model)
	return strings.Contains(lower, "embed") ||
		strings.Contains(lower, "nemoretriever") ||
		strings.Contains(lower, "bge-") ||
		strings.Contains(lower, "e5-") ||
		strings.Contains(lower, "gte-") ||
		strings.Contains(lower, "sentence-")
}
