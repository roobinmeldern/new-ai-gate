package router

import (
	"io/fs"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/skadraneshghn/clever-ai-gate/api/admin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/config"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/health"
	"github.com/skadraneshghn/clever-ai-gate/internal/jobs"
	"github.com/skadraneshghn/clever-ai-gate/internal/jobs/ui"
	"github.com/skadraneshghn/clever-ai-gate/internal/middleware"
	"github.com/skadraneshghn/clever-ai-gate/internal/playground"
	"github.com/skadraneshghn/clever-ai-gate/internal/proxy"
	"github.com/skadraneshghn/clever-ai-gate/internal/redisclient"
	"github.com/skadraneshghn/clever-ai-gate/internal/telemetry"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"go.uber.org/zap"
)

// Dependencies holds all the shared dependencies for route handlers.
type Dependencies struct {
	Config        *config.Config
	DB            *pgxpool.Pool
	Cache         *cache.Store
	TenantCache   *cache.RedisTenantCache    // nil when Redis not configured
	Redis         *redisclient.Client        // nil when Redis not configured
	RedisCacheMgr *cache.RedisCacheManager   // nil when Redis not configured; drives L2 cache
	Vault         *credentials.Vault
	Logger        *zap.Logger
	Health        *health.Handler
	Proxy         *proxy.Handler
	LogHub                *telemetry.LogHub // non-blocking log broadcaster for the admin log viewer
	Scheduler             *jobs.Scheduler  // nil when not initialized
	HealthCheckBroadcaster chan jobs.HealthCheckSSEEvent // real-time SSE broadcaster for model health monitor
	SyncManager           *credentials.SyncManager // nil when not initialized
}

// NewEngine creates and configures the Gin engine with all routes.
//
// Route architecture:
//   - Health routes: no middleware (must always respond)
//   - Proxy routes: auth + rate limiting only (zero overhead)
//   - Admin routes: auth + recovery + CORS (full middleware stack)
//   - Swagger: no auth (documentation is public)
func NewEngine(deps *Dependencies) *gin.Engine {
	// Configure Gin for production
	gin.SetMode(deps.Config.GinMode)

	// Create engine WITHOUT default middleware
	// Default middleware adds logging and recovery which we don't want on the hot-path
	engine := gin.New()

	if deps.Proxy != nil && deps.SyncManager != nil {
		deps.Proxy.SetSyncManager(deps.SyncManager)
	}

	// --- Health routes (no middleware) ---
	engine.GET("/health", deps.Health.Liveness)
	engine.GET("/ready", deps.Health.Readiness)

	// --- Swagger UI (no auth) ---
	engine.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// --- Developer Playground (protected by HTTP Basic Auth) ---
	subFS, err := fs.Sub(playground.DistFS, "dist")
	if err != nil {
		panic("failed to read embedded playground dist: " + err.Error())
	}

	// Read index.html at startup to serve it directly
	// This avoids http.FileServer's built-in redirect logic for "index.html" (which redirects to ./ and causes 404/redirection loops)
	indexContent, err := fs.ReadFile(subFS, "index.html")
	if err != nil {
		panic("failed to read embedded playground index.html: " + err.Error())
	}

	// All playground routes are protected by HTTP Basic Auth.
	// This secures both the SPA HTML and the compiled JS/CSS bundles.
	playgroundGroup := engine.Group("", middleware.PlaygroundBasicAuth(deps.Config.PlaygroundUser, deps.Config.PlaygroundPass))
	{
		// Serve Svelte client static assets from the subFS using http.FileServer
		fileServer := http.StripPrefix("/playground", http.FileServer(http.FS(subFS)))

		// Serve Svelte index.html for requests to /playground
		playgroundGroup.GET("/playground", func(c *gin.Context) {
			c.Data(200, "text/html; charset=utf-8", indexContent)
		})

		// Serve Svelte index.html or dynamic static assets for /playground/*any
		// SPA routing strategy: only serve real files if the path has a file extension
		// (e.g. .js, .css, .png, .json). All other paths are client-side routes → index.html
		playgroundGroup.GET("/playground/*any", func(c *gin.Context) {
			path := c.Param("any")
			if path == "" || path == "/" {
				c.Data(200, "text/html; charset=utf-8", indexContent)
				return
			}

			// If the path contains a dot after the last slash, it's a real static asset
			// (e.g. /playground/_app/immutable/chunks/foo.js, /playground/favicon.png)
			// Otherwise it's a SPA route like /playground/logs → serve index.html
			lastSegment := path[strings.LastIndex(path, "/")+1:]
			if strings.Contains(lastSegment, ".") {
				// Real static asset — let the file server handle it
				fileServer.ServeHTTP(c.Writer, c.Request)
				return
			}

			// SPA client-side route — always serve index.html
			c.Data(200, "text/html; charset=utf-8", indexContent)
		})

		// Config endpoint to return admin key and default tenant key to Basic Auth logged-in user
		playgroundGroup.GET("/api/v1/playground/config", func(c *gin.Context) {
			var tenantKey string
			err := deps.DB.QueryRow(c.Request.Context(), `
				SELECT api_key FROM tenants WHERE is_active = true ORDER BY created_at LIMIT 1
			`).Scan(&tenantKey)
			if err != nil {
				// No active tenant exists, let's create a default one
				tenantKey = "cag_default_tenant_key_salman_136517"
				_, err = deps.DB.Exec(c.Request.Context(), `
					INSERT INTO tenants (name, api_key, token_balance, rate_limit_rpm)
					VALUES ($1, $2, $3, $4)
					ON CONFLICT (api_key) DO NOTHING
				`, "Default Tenant", tenantKey, 1000000000, 120)
				if err != nil {
					c.JSON(500, gin.H{"error": "failed to seed default tenant", "details": err.Error()})
					return
				}
			}

			c.JSON(200, gin.H{
				"admin_key":          deps.Config.AdminAPIKey,
				"tenant_key":         tenantKey,
				"anthropic_endpoint": "/v1/messages",
				"openai_endpoint":    "/v1/chat/completions",
			})
		})
	}

	// --- Proxy routes (minimal middleware for maximum throughput) ---
	proxyGroup := engine.Group("")
	{
		// CORS + OPTIONS preflight handling for browser-based clients.
		// Browser Anthropic SDKs targeting /v1/messages send x-api-key /
		// anthropic-version / anthropic-beta headers, which trigger CORS
		// preflights that must be answered before auth runs (they carry no
		// API key). Must be registered before the auth middleware.
		proxyGroup.Use(middleware.CORS())

		// Auth: prefer two-layer Ristretto+Redis cache when available; accepts Admin key for playground
		if deps.TenantCache != nil {
			proxyGroup.Use(middleware.ProxyAuthWithRedis(deps.TenantCache, deps.Config.AdminAPIKey))
		} else {
			proxyGroup.Use(middleware.ProxyAuth(deps.Cache, deps.Config.AdminAPIKey))
		}

		// Rate limiting: prefer Redis Lua sliding-window when available
		if deps.Redis != nil && deps.Redis.Unwrap() != nil {
			redisRL := middleware.NewRedisRateLimiter(deps.Redis.Unwrap(), deps.Config.DefaultRateLimitRPM)
			proxyGroup.Use(redisRL.Middleware())
		} else {
			rateLimiter := middleware.NewRateLimiter(deps.Config.DefaultRateLimitRPM)
			proxyGroup.Use(rateLimiter.Middleware())
		}

		// Single catch-all for all OpenAI-compatible endpoints:
		// /v1/chat/completions, /v1/embeddings, /v1/images/generations,
		// /v1/audio/transcriptions, /v1/audio/speech, etc.
		// All routes use the same proxy handler — model-based routing
		// is determined by the "model" field in the JSON body, not the URL path.
		proxyGroup.POST("/v1/*proxyPath", deps.Proxy.Handle)
		proxyGroup.POST("/messages", deps.Proxy.Handle)
		proxyGroup.POST("/messages/*proxyPath", deps.Proxy.Handle)
		proxyGroup.GET("/v1/models", deps.Proxy.ListModels)
		proxyGroup.GET("/v1/models/*model", deps.Proxy.GetModel)
		proxyGroup.GET("/models", deps.Proxy.ListModels)
		proxyGroup.GET("/models/*model", deps.Proxy.GetModel)

		// CORS preflight route: browsers send OPTIONS before any authenticated
		// request. The CORS middleware registered above intercepts OPTIONS and
		// answers 204 + the allow headers before auth/rate-limit run — this
		// route only exists so Gin does not answer preflights with 404
		// (Gin routes by method).
		proxyGroup.OPTIONS("/v1/*proxyPath", func(c *gin.Context) {
			c.Status(http.StatusNoContent)
		})
		proxyGroup.OPTIONS("/messages", func(c *gin.Context) {
			c.Status(http.StatusNoContent)
		})
		proxyGroup.OPTIONS("/messages/*proxyPath", func(c *gin.Context) {
			c.Status(http.StatusNoContent)
		})

		// Playground Chats CRUD (Tenant-isolated, requires tenant API Key)
		chatHandler := admin.NewChatHandler(deps.DB)
		proxyGroup.GET("/api/v1/playground/tenant", chatHandler.GetTenantInfo)
		proxyGroup.GET("/api/v1/playground/chats", chatHandler.List)
		proxyGroup.GET("/api/v1/playground/chats/:id", chatHandler.Get)
		proxyGroup.POST("/api/v1/playground/chats", chatHandler.Create)
		proxyGroup.PUT("/api/v1/playground/chats/:id", chatHandler.Update)
		proxyGroup.DELETE("/api/v1/playground/chats/:id", chatHandler.Delete)
	}

	// --- Admin API routes (full middleware stack) ---
	adminGroup := engine.Group("/api/v1/admin")
	{
		adminGroup.Use(middleware.CORS())
		adminGroup.Use(middleware.Recovery(deps.Logger))
		adminGroup.Use(middleware.RequestID())
		adminGroup.Use(middleware.AdminAuth(deps.Config.AdminAPIKey))

		// Tenant management
		tenantHandler := admin.NewTenantHandler(deps.DB)
		adminGroup.GET("/tenants", tenantHandler.List)
		adminGroup.POST("/tenants", tenantHandler.Create)
		adminGroup.GET("/tenants/:id", tenantHandler.Get)
		adminGroup.PUT("/tenants/:id", tenantHandler.Update)
		adminGroup.DELETE("/tenants/:id", tenantHandler.Delete)

		// Model pool management
		poolHandler := admin.NewPoolHandler(deps.DB, deps.Vault, deps.Scheduler, deps.RedisCacheMgr, deps.SyncManager)
		adminGroup.GET("/pools", poolHandler.List)
		adminGroup.POST("/pools", poolHandler.Create)
		// NOTE: static routes must be registered before Gin's parameterised :id patterns.
		adminGroup.POST("/pools/bulk-test", poolHandler.BulkTest)
		adminGroup.POST("/pools/bulk-delete", poolHandler.BulkDelete)
		adminGroup.POST("/pools/bulk-activate", poolHandler.BulkActivate)
		adminGroup.POST("/pools/purge-unhealthy", poolHandler.PurgeUnhealthyPools)
		adminGroup.GET("/pools/:id", poolHandler.Get)
		adminGroup.PUT("/pools/:id", poolHandler.Update)
		adminGroup.DELETE("/pools/:id", poolHandler.Delete)
		adminGroup.GET("/pools/:id/logs", poolHandler.GetLogs)
		adminGroup.POST("/pools/:id/credentials/:cred_id/test", poolHandler.TestCredential)

		// Credential management
		credHandler := admin.NewCredentialHandler(deps.DB, deps.Vault, deps.Scheduler, deps.RedisCacheMgr, deps.SyncManager)
		adminGroup.GET("/credentials", credHandler.List)
		adminGroup.POST("/credentials", credHandler.Create)
		adminGroup.GET("/credentials/:id", credHandler.Get)
		adminGroup.PUT("/credentials/:id", credHandler.Update)
		adminGroup.DELETE("/credentials/:id", credHandler.Delete)
		adminGroup.POST("/credentials/bulk-delete", credHandler.BulkDelete)

		// Provider auto-discovery (registers all models for a given provider key)
		adminGroup.POST("/providers/nvidia", credHandler.RegisterNvidiaProvider)
		adminGroup.POST("/providers/ollama", credHandler.RegisterOllamaProvider)
		adminGroup.POST("/providers/custom", credHandler.RegisterCustomProvider)
		adminGroup.POST("/providers/custom/batch", credHandler.RegisterCustomProvider)
		adminGroup.POST("/providers/openrouter", credHandler.RegisterOpenRouterProvider)
		adminGroup.POST("/providers/1minai", credHandler.RegisterOneMinAIProvider)
		adminGroup.POST("/providers/cloudflare", credHandler.RegisterCloudflareProvider)
		adminGroup.POST("/providers/sarvam", credHandler.RegisterSarvamProvider)
		adminGroup.POST("/providers/puter", credHandler.RegisterPuterProvider)
		adminGroup.POST("/providers/agentrouter", credHandler.RegisterAgentRouterProvider)
		adminGroup.POST("/providers/zenmux", credHandler.RegisterZenMuxProvider)
		adminGroup.POST("/providers/gemini", credHandler.RegisterGeminiProvider)
		adminGroup.POST("/providers/refresh", credHandler.RefreshAllProviders)
		// Re-discovery: async scan of all provider endpoints for new models
		adminGroup.POST("/providers/rediscover", credHandler.TriggerReDiscovery)
		adminGroup.GET("/providers/rediscover/status", credHandler.GetReDiscoveryStatus)

		// Live log streaming and daily log file download
		if deps.LogHub != nil {
			logCtrl := admin.NewLogAdminController(deps.LogHub)
			adminGroup.GET("/logs/stream", logCtrl.StreamLiveCoreLogs)
			adminGroup.GET("/logs/download", logCtrl.DownloadTodayLogFile)
		}

		// Dashboard statistics & metrics
		metricsCtrl := admin.NewMetricsController(deps.DB)
		adminGroup.GET("/metrics", metricsCtrl.GetDashboardStats)

		// Exhaustive Model Health Check & Real-time SSE Monitor
		hcHandler := admin.NewHealthCheckHandler(deps.DB, deps.Vault, deps.Logger, deps.HealthCheckBroadcaster)
		adminGroup.POST("/health-check/trigger", hcHandler.TriggerHealthCheck)
		adminGroup.GET("/health-check/stream", hcHandler.StreamHealthCheckSSE)
		adminGroup.GET("/health-check/sessions", hcHandler.GetSessions)
		adminGroup.GET("/health-check/sessions/:id", hcHandler.GetSessionDetails)

		// Job scheduling system
		if deps.Scheduler != nil {
			jobHandler := admin.NewJobHandler(deps.DB, deps.Scheduler, deps.Logger)

			// Job CRUD & lifecycle
			adminGroup.GET("/jobs", jobHandler.ListJobs)
			adminGroup.POST("/jobs", jobHandler.CreateJob)
			adminGroup.GET("/jobs/runs", jobHandler.GetAllRuns)
			adminGroup.GET("/jobs/:id", jobHandler.GetJob)
			adminGroup.PUT("/jobs/:id", jobHandler.UpdateJob)
			adminGroup.DELETE("/jobs/:id", jobHandler.DeleteJob)
			adminGroup.POST("/jobs/:id/trigger", jobHandler.TriggerJob)
			adminGroup.POST("/jobs/:id/pause", jobHandler.PauseJob)
			adminGroup.POST("/jobs/:id/resume", jobHandler.ResumeJob)
			adminGroup.GET("/jobs/:id/runs", jobHandler.GetJobRuns)
			adminGroup.DELETE("/jobs/runs/:run_id", jobHandler.DeleteRun)

			// Scheduler settings & operations
			adminGroup.GET("/scheduler/settings", jobHandler.GetSchedulerSettings)
			adminGroup.PUT("/scheduler/settings", jobHandler.UpdateSchedulerSettings)
			adminGroup.GET("/scheduler/stats", jobHandler.GetSchedulerStats)
			adminGroup.GET("/scheduler/types", jobHandler.GetRegisteredTypes)
			adminGroup.POST("/scheduler/restart", jobHandler.RestartScheduler)
		}
	}

	// --- Job Scheduler Admin UI (protected by HTTP Basic Auth) ---
	if deps.Scheduler != nil {
		jobsSubFS, err := fs.Sub(ui.DistFS, "dist")
		if err != nil {
			deps.Logger.Warn("failed to read embedded jobs UI", zap.Error(err))
		} else {
			jobsIndexContent, err := fs.ReadFile(jobsSubFS, "index.html")
			if err != nil {
				deps.Logger.Warn("failed to read jobs UI index.html", zap.Error(err))
			} else {
				jobsGroup := engine.Group("", middleware.PlaygroundBasicAuth(deps.Config.PlaygroundUser, deps.Config.PlaygroundPass))
				jobsFileServer := http.StripPrefix("/jobs", http.FileServer(http.FS(jobsSubFS)))

				jobsGroup.GET("/jobs", func(c *gin.Context) {
					c.Data(200, "text/html; charset=utf-8", jobsIndexContent)
				})
				jobsGroup.GET("/jobs/*any", func(c *gin.Context) {
					path := c.Param("any")
					if path == "" || path == "/" {
						c.Data(200, "text/html; charset=utf-8", jobsIndexContent)
						return
					}
					lastSegment := path[strings.LastIndex(path, "/")+1:]
					if strings.Contains(lastSegment, ".") {
						jobsFileServer.ServeHTTP(c.Writer, c.Request)
						return
					}
					c.Data(200, "text/html; charset=utf-8", jobsIndexContent)
				})
			}
		}
	}

	return engine
}
