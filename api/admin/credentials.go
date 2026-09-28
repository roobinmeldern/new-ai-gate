package admin

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/skadraneshghn/clever-ai-gate/api/dto"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/database"
	"github.com/skadraneshghn/clever-ai-gate/internal/jobs"
)

// CredentialHandler provides CRUD operations for provider credentials.
type CredentialHandler struct {
	db            *pgxpool.Pool
	vault         *credentials.Vault
	scheduler     *jobs.Scheduler
	redisCacheMgr *cache.RedisCacheManager // nil-safe; invalidates on every mutation
	syncManager   *credentials.SyncManager // nil-safe; triggers immediate synchronous reload
}

// NewCredentialHandler creates a new credential handler.
func NewCredentialHandler(db *pgxpool.Pool, vault *credentials.Vault, scheduler *jobs.Scheduler, redisCacheMgr *cache.RedisCacheManager, syncManager ...*credentials.SyncManager) *CredentialHandler {
	var sm *credentials.SyncManager
	if len(syncManager) > 0 {
		sm = syncManager[0]
	}
	return &CredentialHandler{db: db, vault: vault, scheduler: scheduler, redisCacheMgr: redisCacheMgr, syncManager: sm}
}

// SetSyncManager attaches the sync manager for immediate cache reloading.
func (h *CredentialHandler) SetSyncManager(sm *credentials.SyncManager) {
	h.syncManager = sm
}

func (h *CredentialHandler) syncCaches(ctx context.Context) {
	if h.syncManager != nil {
		_ = h.syncManager.ReloadSync(ctx)
	} else if h.redisCacheMgr != nil {
		h.redisCacheMgr.InvalidateAndPublish(ctx)
	}
}

// List returns provider credentials with masked keys.
//
// When the `limit` query parameter is supplied the endpoint responds with a
// paginated envelope ({data, total, limit, offset}) supporting server-side
// pagination, filtering (`provider`, `search`) and virtualized rendering.
// When `limit` is omitted the endpoint preserves the legacy behaviour of
// returning the full flat array of credentials (backward compatibility).
//
// @Summary      List credentials
// @Description  Returns provider credentials across all pools with masked API keys. Supports optional pagination via limit/offset and filtering via provider/search.
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Param        limit     query  int     false  "Page size (enables paginated envelope when present)"  example(100)
// @Param        offset    query  int     false  "Page offset"                                          example(0)
// @Param        provider  query  string  false  "Exact provider filter"                                example(openai)
// @Param        search    query  string  false  "Case-insensitive search across provider/base_url/model_pattern"  example(openai)
// @Success      200  {array}   dto.CredentialResponse
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials [get]
func (h *CredentialHandler) List(c *gin.Context) {
	limitStr := c.Query("limit")

	// Legacy path: no pagination requested -> return the full flat list.
	if limitStr == "" {
		creds, err := database.ListAllCredentials(c.Request.Context(), h.db)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to list credentials", Details: err.Error()})
			return
		}

		resp := make([]dto.CredentialResponse, len(creds))
		for i, cr := range creds {
			resp[i] = dto.CredentialResponse{
				ID:           cr.ID,
				PoolID:       cr.PoolID,
				Provider:     cr.Provider,
				BaseURL:      cr.BaseURL,
				Weight:       cr.Weight,
				IsHealthy:    cr.IsHealthy,
				LastError:    cr.LastError,
				KeyMask:      dto.MaskAPIKey(cr.EncryptedKey),
				ModelPattern: cr.ModelPattern,
				Prefix:       cr.Prefix,
				CreatedAt:    cr.CreatedAt,
			}
		}
		c.JSON(http.StatusOK, resp)
		return
	}

	// Paginated path.
	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit <= 0 {
		limit = 100
	}
	offset := 0
	if offStr := c.Query("offset"); offStr != "" {
		if v, err := strconv.Atoi(offStr); err == nil && v >= 0 {
			offset = v
		}
	}
	search := c.Query("search")
	provider := c.Query("provider")

	creds, total, err := database.ListCredentialsPaginated(c.Request.Context(), h.db, limit, offset, search, provider)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to list credentials", Details: err.Error()})
		return
	}

	resp := make([]dto.CredentialResponse, len(creds))
	for i, cr := range creds {
		resp[i] = dto.CredentialResponse{
			ID:           cr.ID,
			PoolID:       cr.PoolID,
			Provider:     cr.Provider,
			BaseURL:      cr.BaseURL,
			Weight:       cr.Weight,
			IsHealthy:    cr.IsHealthy,
			LastError:    cr.LastError,
			KeyMask:      dto.MaskAPIKey(cr.EncryptedKey),
			ModelPattern: cr.ModelPattern,
			Prefix:       cr.Prefix,
			CreatedAt:    cr.CreatedAt,
		}
	}

	c.JSON(http.StatusOK, dto.PaginatedCredentialsResponse{
		Data:   resp,
		Total:  total,
		Limit:  limit,
		Offset: offset,
	})
}

// Create adds a new provider credential to a pool.
// @Summary      Add credential
// @Description  Adds a new provider API key to a model routing pool
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.CreateCredentialRequest  true  "Credential details"
// @Success      201   {object}  dto.CredentialResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials [post]
func (h *CredentialHandler) Create(c *gin.Context) {
	var req dto.CreateCredentialRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	// Validate pool exists
	pool, err := database.GetModelPool(c.Request.Context(), h.db, req.PoolID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to validate pool"})
		return
	}
	if pool == nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "pool not found", Details: "the specified pool_id does not exist"})
		return
	}

	// Encrypt API key before storage
	encryptedKey, err := h.vault.Encrypt(req.APIKey)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to encrypt API key"})
		return
	}

	id, err := database.CreateCredential(c.Request.Context(), h.db, req.PoolID, req.Provider, encryptedKey, req.BaseURL, req.Weight, req.Prefix)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to create credential", Details: err.Error()})
		return
	}

	// Synchronously reload caches so all cluster nodes pick up the new credential.
	h.syncCaches(c.Request.Context())

	c.JSON(http.StatusCreated, dto.CredentialResponse{
		ID:        id,
		PoolID:    req.PoolID,
		Provider:  req.Provider,
		BaseURL:   req.BaseURL,
		Weight:    req.Weight,
		IsHealthy: true,
		KeyMask:   dto.MaskAPIKey(req.APIKey),
		Prefix:    req.Prefix,
	})
}

// Get returns a single credential with masked key.
// @Summary      Get credential
// @Description  Returns a credential by ID with masked API key
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Param        id   path      int  true  "Credential ID"
// @Success      200  {object}  dto.CredentialResponse
// @Failure      404  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials/{id} [get]
func (h *CredentialHandler) Get(c *gin.Context) {
	id, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid credential ID"})
		return
	}

	cred, err := database.GetCredential(c.Request.Context(), h.db, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to get credential"})
		return
	}
	if cred == nil {
		c.JSON(http.StatusNotFound, dto.ErrorResponse{Error: "credential not found"})
		return
	}

	c.JSON(http.StatusOK, dto.CredentialResponse{
		ID:        cred.ID,
		PoolID:    cred.PoolID,
		Provider:  cred.Provider,
		BaseURL:   cred.BaseURL,
		Weight:    cred.Weight,
		IsHealthy: cred.IsHealthy,
		LastError: cred.LastError,
		KeyMask:   dto.MaskAPIKey(cred.EncryptedKey),
		Prefix:    cred.Prefix,
		CreatedAt: cred.CreatedAt,
	})
}

// Update updates a credential's configuration.
// @Summary      Update credential
// @Description  Updates a credential's provider, key, URL, weight, and health
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        id    path      int                         true  "Credential ID"
// @Param        body  body      dto.UpdateCredentialRequest  true  "Updated credential details"
// @Success      200   {object}  dto.SuccessResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials/{id} [put]
func (h *CredentialHandler) Update(c *gin.Context) {
	id, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid credential ID"})
		return
	}

	var req dto.UpdateCredentialRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	// If API key is provided, encrypt it
	encryptedKey := ""
	if req.APIKey != "" {
		encrypted, err := h.vault.Encrypt(req.APIKey)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to encrypt API key"})
			return
		}
		encryptedKey = encrypted
	} else {
		// Keep existing key
		existing, err := database.GetCredential(c.Request.Context(), h.db, id)
		if err != nil || existing == nil {
			c.JSON(http.StatusNotFound, dto.ErrorResponse{Error: "credential not found"})
			return
		}
		encryptedKey = existing.EncryptedKey
	}

	err = database.UpdateCredential(c.Request.Context(), h.db, id, req.Provider, encryptedKey, req.BaseURL, req.Weight, req.IsHealthy, req.Prefix)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to update credential", Details: err.Error()})
		return
	}

	// Synchronously reload caches
	h.syncCaches(c.Request.Context())

	c.JSON(http.StatusOK, dto.SuccessResponse{Message: "credential updated successfully"})
}

// Delete removes a credential.
// @Summary      Delete credential
// @Description  Permanently removes a provider credential
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Param        id   path      int  true  "Credential ID"
// @Success      200  {object}  dto.SuccessResponse
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials/{id} [delete]
func (h *CredentialHandler) Delete(c *gin.Context) {
	id, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid credential ID"})
		return
	}

	err = database.DeleteCredential(c.Request.Context(), h.db, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to delete credential", Details: err.Error()})
		return
	}

	// Synchronously reload caches so all cluster nodes stop routing to deleted credential.
	h.syncCaches(c.Request.Context())

	c.JSON(http.StatusOK, dto.SuccessResponse{Message: "credential deleted successfully"})
}

// RegisterNvidiaProvider auto-discovers all models available under an NVIDIA API key,
// creates model pools for each, and binds the credential to all of them in one transaction.
// @Summary      Auto-discover NVIDIA Models
// @Description  Submits an NVIDIA key, hits their /models endpoint, and registers all active models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverProviderRequest  true  "NVIDIA provider details"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/nvidia [post]
func (h *CredentialHandler) RegisterNvidiaProvider(c *gin.Context) {
	var req dto.DiscoverProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.Provider != "nvidia" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid provider, must be 'nvidia'"})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for NVIDIA provider"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterNvidiaModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.BaseURL,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "NVIDIA auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       "Successfully synchronized all NVIDIA models",
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Invalidate and synchronously reload cache so models are immediately visible
	h.syncCaches(c.Request.Context())
}

// RegisterOllamaProvider auto-discovers all models available on an Ollama instance,
// creates model pools for each, and binds the credential to all of them in one transaction.
// Multiple Ollama accounts can be registered — identical models are grouped into the same
// pool for automatic load balancing and failover.
// @Summary      Auto-discover Ollama Models
// @Description  Connects to an Ollama instance, hits the /v1/models endpoint, and registers all available models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverProviderRequest  true  "Ollama provider details"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/ollama [post]
func (h *CredentialHandler) RegisterOllamaProvider(c *gin.Context) {
	var req dto.DiscoverProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.Provider != "ollama" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid provider, must be 'ollama'"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	// API key is optional for Ollama — local instances typically require no authentication.
	// When provided, it is sent as a Bearer token (useful for Ollama behind an auth proxy).
	count, models, err := credentials.DiscoverAndRegisterOllamaModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.BaseURL,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "Ollama auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       "Successfully synchronized all Ollama models",
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered Ollama models.
	h.syncCaches(c.Request.Context())
}

// RegisterOpenRouterProvider auto-discovers all FREE models available on OpenRouter,
// creates model pools for each, and binds the credential to all of them in one transaction.
// Free-tier models are identified by the `:free` suffix in their ID (OpenRouter's canonical
// free-tier designation) and/or zero prompt/completion pricing.
//
// @Summary      Auto-discover OpenRouter Free Models
// @Description  Connects to OpenRouter, fetches the model catalog, and registers only free-tier models
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverProviderRequest  true  "OpenRouter provider details (api_key required, base_url ignored)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/openrouter [post]
func (h *CredentialHandler) RegisterOpenRouterProvider(c *gin.Context) {
	var req dto.DiscoverProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for OpenRouter auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterOpenRouterModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "OpenRouter free-model discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d free OpenRouter models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered OpenRouter models.
	h.syncCaches(c.Request.Context())
}

// RegisterCustomProvider handles POST /api/v1/admin/providers/custom
// It connects to any OpenAI-compatible provider, validates the API key,
// discovers all available models, and registers them for load balancing.
//
// @Summary     Register a generic OpenAI-compatible provider
// @Description Auto-discovers models from any OpenAI-compatible endpoint.
// @Tags        providers
// @Accept      json
// @Produce     json
// @Param       body body     dto.DiscoverProviderRequest true "Provider details"
// @Success     200  {object} dto.DiscoverProviderResponse
// @Failure     400  {object} dto.ErrorResponse
// @Failure     500  {object} dto.ErrorResponse
// @Router      /api/v1/admin/providers/custom [post]
func (h *CredentialHandler) RegisterCustomProvider(c *gin.Context) {
	var req dto.DiscoverProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	var keys []string
	for _, k := range req.APIKeys {
		k = strings.TrimSpace(k)
		if k != "" {
			keys = append(keys, k)
		}
	}

	if len(keys) == 0 && req.APIKey != "" {
		rawKeys := strings.FieldsFunc(req.APIKey, func(r rune) bool {
			return r == '\n' || r == '\r' || r == ','
		})
		for _, rk := range rawKeys {
			rk = strings.TrimSpace(rk)
			if rk != "" {
				keys = append(keys, rk)
			}
		}
	}

	if len(keys) == 0 {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key or api_keys is required for OpenAI-compatible provider discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	// Use the label field as the provider name, falling back to "custom"
	providerLabel := req.Label
	if providerLabel == "" {
		providerLabel = "custom"
	}

	prefix := req.Prefix
	if prefix == "" && req.Label != "" {
		prefix = providerLabel
	}

	// Single key workflow
	if len(keys) == 1 && len(req.APIKeys) == 0 {
		count, models, err := credentials.DiscoverAndRegisterCustomModels(
			c.Request.Context(),
			h.db,
			h.vault,
			keys[0],
			req.BaseURL,
			providerLabel,
			req.Weight,
			prefix,
		)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "OpenAI-compatible provider discovery failed", Details: err.Error()})
			return
		}

		c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
			Message:       fmt.Sprintf("Successfully synchronized %d models from %s", count, providerLabel),
			ModelsCount:   count,
			DiscoveredIDs: models,
		})

		// Synchronously reload caches so all cluster nodes instantly see new custom provider models.
		h.syncCaches(c.Request.Context())
		return
	}

	// Batch keys workflow
	totalKeys, successCount, failedCount, totalModels, results, allDiscovered, err := credentials.DiscoverAndRegisterCustomModelsBatch(
		c.Request.Context(),
		h.db,
		h.vault,
		keys,
		req.BaseURL,
		providerLabel,
		req.Weight,
		prefix,
	)
	if err != nil && successCount == 0 {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "OpenAI-compatible batch provider discovery failed", Details: err.Error()})
		return
	}

	batchResults := make([]dto.BatchKeyResult, len(results))
	for i, r := range results {
		batchResults[i] = dto.BatchKeyResult{
			Index:         r.Index,
			KeyMasked:     r.KeyMasked,
			Success:       r.Success,
			ModelsCount:   r.ModelsCount,
			DiscoveredIDs: r.DiscoveredIDs,
			Error:         r.Error,
		}
	}

	msg := fmt.Sprintf("Successfully synchronized %d unique models across %d of %d keys from %s", totalModels, successCount, totalKeys, providerLabel)
	if failedCount > 0 {
		msg = fmt.Sprintf("Batch discovery completed: %d of %d keys succeeded, %d failed (%d unique models synchronized)", successCount, totalKeys, failedCount, totalModels)
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       msg,
		ModelsCount:   totalModels,
		DiscoveredIDs: allDiscovered,
		TotalKeys:     totalKeys,
		SuccessCount:  successCount,
		FailedCount:   failedCount,
		Results:       batchResults,
	})

	if successCount > 0 {
		h.syncCaches(c.Request.Context())
	}
}

// RegisterOneMinAIProvider auto-discovers all models available on 1min.ai
// across all five modalities (chat, code, image, audio, video), creates model
// pools for each, and binds the credential to all of them in one transaction
// per key.
//
// Bulk registration is supported: pass an array of keys via api_keys (or a
// newline/comma-separated list in api_key). Keys are processed one by one — a
// failing key is reported in the results without aborting the batch, and every
// valid key is inserted into the credentials table bound to the full 1min.ai
// manifest catalog for round-robin load balancing. 1min.ai does not expose a
// /v1/models endpoint, so the catalog comes from a static manifest and each
// key is validated via a lightweight chat request before registration. The
// base URL is hardcoded to https://api.1min.ai.
//
// @Summary      Auto-discover 1min.ai Models
// @Description  Submits one or more 1min.ai keys and registers all models across all modalities automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverOneMinAIRequest  true  "1min.ai provider details (api_key or api_keys required, base_url ignored)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/1minai [post]
func (h *CredentialHandler) RegisterOneMinAIProvider(c *gin.Context) {
	var req dto.DiscoverOneMinAIRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	// Collect keys: prefer the explicit api_keys array, then fall back to
	// newline/comma-separated keys in api_key.
	var keys []string
	for _, k := range req.APIKeys {
		k = strings.TrimSpace(k)
		if k != "" {
			keys = append(keys, k)
		}
	}

	if len(keys) == 0 && req.APIKey != "" {
		rawKeys := strings.FieldsFunc(req.APIKey, func(r rune) bool {
			return r == '\n' || r == '\r' || r == ','
		})
		for _, rk := range rawKeys {
			rk = strings.TrimSpace(rk)
			if rk != "" {
				keys = append(keys, rk)
			}
		}
	}

	if len(keys) == 0 {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key or api_keys is required for 1min.ai auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	// Single key workflow (legacy response shape).
	if len(keys) == 1 && len(req.APIKeys) == 0 {
		count, models, err := credentials.DiscoverAndRegisterOneMinAIModels(
			c.Request.Context(),
			h.db,
			h.vault,
			keys[0],
			req.Weight,
		)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "1min.ai auto-discovery failed", Details: err.Error()})
			return
		}

		c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
			Message:       fmt.Sprintf("Successfully synchronized %d 1min.ai models across all modalities", count),
			ModelsCount:   count,
			DiscoveredIDs: models,
		})

		// Synchronously reload caches so all cluster nodes instantly see newly registered 1min.ai models.
		h.syncCaches(c.Request.Context())
		return
	}

	// Batch keys workflow: each key is processed one by one; failures are
	// reported per key without aborting the remaining keys.
	totalKeys, successCount, failedCount, totalModels, results, allDiscovered, err := credentials.DiscoverAndRegisterOneMinAIModelsBatch(
		c.Request.Context(),
		h.db,
		h.vault,
		keys,
		req.Weight,
	)
	if err != nil && successCount == 0 {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "1min.ai batch auto-discovery failed", Details: err.Error()})
		return
	}

	batchResults := make([]dto.BatchKeyResult, len(results))
	for i, r := range results {
		batchResults[i] = dto.BatchKeyResult{
			Index:         r.Index,
			KeyMasked:     r.KeyMasked,
			Success:       r.Success,
			ModelsCount:   r.ModelsCount,
			DiscoveredIDs: r.DiscoveredIDs,
			Error:         r.Error,
		}
	}

	msg := fmt.Sprintf("Successfully synchronized %d unique 1min.ai models across %d of %d keys", totalModels, successCount, totalKeys)
	if failedCount > 0 {
		msg = fmt.Sprintf("Batch discovery completed: %d of %d keys succeeded, %d failed (%d unique models synchronized)", successCount, totalKeys, failedCount, totalModels)
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       msg,
		ModelsCount:   totalModels,
		DiscoveredIDs: allDiscovered,
		TotalKeys:     totalKeys,
		SuccessCount:  successCount,
		FailedCount:   failedCount,
		Results:       batchResults,
	})

	if successCount > 0 {
		h.syncCaches(c.Request.Context())
	}
}

// RegisterCloudflareProvider auto-discovers all Cloudflare Workers AI models
// available under the given Account ID, creates model pools for each one,
// and binds the API token credential to all of them in a single atomic transaction.
//
// Credential storage convention (zero schema migration):
//   - The API Token is encrypted and stored in credentials.encrypted_key.
//   - The Account ID is stored in credentials.base_url as "cloudflare:<accountID>".
//     The rewriter parses it at request time with strings.TrimPrefix.
//
// Each model is registered under two pool patterns for maximum client compatibility:
//   - "cloudflare/@cf/meta/llama-3.1-8b-instruct" (explicit prefix form)
//   - "@cf/meta/llama-3.1-8b-instruct" (clean form)
//
// @Summary      Auto-discover Cloudflare Workers AI Models
// @Description  Connects to Cloudflare, fetches all Workers AI models for the account, and registers them
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverCloudflareRequest  true  "Cloudflare account_id and api_token"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/cloudflare [post]
func (h *CredentialHandler) RegisterCloudflareProvider(c *gin.Context) {
	var req dto.DiscoverCloudflareRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.AccountID == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "account_id is required for Cloudflare Workers AI"})
		return
	}

	if req.APIToken == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_token is required for Cloudflare Workers AI"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterCloudflareModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.AccountID,
		req.APIToken,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{
			Error:   "Cloudflare Workers AI auto-discovery failed",
			Details: err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d Cloudflare Workers AI models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered Cloudflare Workers AI models.
	h.syncCaches(c.Request.Context())
}

// RegisterSarvamProvider auto-discovers all Sarvam AI chat models available under
// an API subscription key, creates model pools for each, and binds the credential
// to all of them in one transaction.
//
// Sarvam AI is natively OpenAI-compatible (POST /v1/chat/completions, SSE
// streaming, reasoning_content) but does not expose a /v1/models endpoint — the
// chat model enum is fixed (sarvam-30b, sarvam-105b), so discovery uses a static
// manifest. Adding the provider requires only the API key; the base URL is
// hardcoded to https://api.sarvam.ai.
//
// @Summary      Auto-discover Sarvam AI Models
// @Description  Submits a Sarvam AI key, validates it, and registers all OpenAI-compatible chat models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverSarvamRequest  true  "Sarvam AI provider details (api_key required)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/sarvam [post]
func (h *CredentialHandler) RegisterSarvamProvider(c *gin.Context) {
	var req dto.DiscoverSarvamRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for Sarvam AI auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterSarvamModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "Sarvam AI auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d Sarvam AI models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered Sarvam AI models.
	h.syncCaches(c.Request.Context())
}

// RegisterGeminiProvider auto-discovers all models available under a Google AI Studio
// API key, creates model pools for each, and binds the credential to all of them in
// one transaction.
//
// Google AI Studio exposes a live /v1beta/models endpoint that returns the full
// model catalog. Each model is registered under two pool patterns:
//   - "gemini/<modelID>"  — explicit provider routing prefix
//   - "<modelID>"          — clean name for direct use in Cline, Continue, etc.
//
// The discovery engine filters models to only those supporting generateContent
// (chat) or embedContent (embeddings). The base URL is hardcoded to
// https://generativelanguage.googleapis.com.
//
// @Summary      Auto-discover Google AI Studio (Gemini) Models
// @Description  Submits a Google AI Studio API key, hits the /v1beta/models endpoint, and registers all available models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverGeminiRequest  true  "Google AI Studio provider details (api_key required)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/gemini [post]
func (h *CredentialHandler) RegisterGeminiProvider(c *gin.Context) {
	var req dto.DiscoverGeminiRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for Google AI Studio auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterGeminiModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "Google AI Studio auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d Google AI Studio (Gemini) models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered Gemini models.
	h.syncCaches(c.Request.Context())
}

// RegisterPuterProvider auto-discovers all Puter.com AI models available under
// one or more API tokens, creates model pools for each, and binds every
// credential to all of them in one transaction per key.
//
// Bulk registration is supported: pass an array of tokens via api_keys (or a
// newline/comma-separated list in api_key). Keys are processed one by one — a
// failing key is reported in the results without aborting the batch, and every
// valid key is inserted into the credentials table bound to the full Puter
// model catalog for round-robin load balancing.

//
// @Summary      Auto-discover Puter.com Models
// @Description  Submits one or more Puter tokens, fetches the model details, and registers all models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverPuterRequest  true  "Puter provider details (api_key or api_keys required)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/puter [post]
func (h *CredentialHandler) RegisterPuterProvider(c *gin.Context) {
	var req dto.DiscoverPuterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	// Collect keys: prefer the explicit api_keys array, then fall back to
	// newline/comma-separated tokens in api_key.
	var keys []string
	for _, k := range req.APIKeys {
		k = strings.TrimSpace(k)
		if k != "" {
			keys = append(keys, k)
		}
	}

	if len(keys) == 0 && req.APIKey != "" {
		rawKeys := strings.FieldsFunc(req.APIKey, func(r rune) bool {
			return r == '\n' || r == '\r' || r == ','
		})
		for _, rk := range rawKeys {
			rk = strings.TrimSpace(rk)
			if rk != "" {
				keys = append(keys, rk)
			}
		}
	}

	if len(keys) == 0 {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key or api_keys is required for Puter.com auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	// Single key workflow (legacy response shape).
	if len(keys) == 1 && len(req.APIKeys) == 0 {
		count, models, err := credentials.DiscoverAndRegisterPuterModels(
			c.Request.Context(),
			h.db,
			h.vault,
			keys[0],
			req.Weight,
		)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "Puter.com auto-discovery failed", Details: err.Error()})
			return
		}

		c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
			Message:       fmt.Sprintf("Successfully synchronized %d Puter.com models", count),
			ModelsCount:   count,
			DiscoveredIDs: models,
		})

		// Synchronously reload caches so all cluster nodes instantly see newly registered Puter models.
		h.syncCaches(c.Request.Context())
		return
	}

	// Batch keys workflow: each key is processed one by one; failures are
	// reported per key without aborting the remaining keys.
	totalKeys, successCount, failedCount, totalModels, results, allDiscovered, err := credentials.DiscoverAndRegisterPuterModelsBatch(
		c.Request.Context(),
		h.db,
		h.vault,
		keys,
		req.Weight,
	)
	if err != nil && successCount == 0 {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "Puter.com batch auto-discovery failed", Details: err.Error()})
		return
	}

	batchResults := make([]dto.BatchKeyResult, len(results))
	for i, r := range results {
		batchResults[i] = dto.BatchKeyResult{
			Index:         r.Index,
			KeyMasked:     r.KeyMasked,
			Success:       r.Success,
			ModelsCount:   r.ModelsCount,
			DiscoveredIDs: r.DiscoveredIDs,
			Error:         r.Error,
		}
	}

	msg := fmt.Sprintf("Successfully synchronized %d unique Puter.com models across %d of %d keys", totalModels, successCount, totalKeys)
	if failedCount > 0 {
		msg = fmt.Sprintf("Batch discovery completed: %d of %d keys succeeded, %d failed (%d unique models synchronized)", successCount, totalKeys, failedCount, totalModels)
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       msg,
		ModelsCount:   totalModels,
		DiscoveredIDs: allDiscovered,
		TotalKeys:     totalKeys,
		SuccessCount:  successCount,
		FailedCount:   failedCount,
		Results:       batchResults,
	})

	if successCount > 0 {
		h.syncCaches(c.Request.Context())
	}
}

// RegisterAgentRouterProvider auto-discovers all models available on AgentRouter.org
// under an API key, creates model pools for each, and binds the credential to all
// of them in one transaction. Multiple keys can be added for round-robin rotation
// with automatic 429/401 failover.
//
// AgentRouter.org is an AI API aggregator that exposes an OpenAI-compatible endpoint.
// The gateway applies request sanitization (OmniRoute Issue #1921 fix) to strip
// unsupported fields that AgentRouter's diverse upstream backends reject.
//
// @Summary      Auto-discover AgentRouter Models
// @Description  Submits an AgentRouter key, fetches the model catalog, and registers all models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverAgentRouterRequest  true  "AgentRouter provider details (api_key required)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/agentrouter [post]
func (h *CredentialHandler) RegisterAgentRouterProvider(c *gin.Context) {
	var req dto.DiscoverAgentRouterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for AgentRouter auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterAgentRouterModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "AgentRouter auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d AgentRouter models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Invalidate Redis cache so all cluster nodes instantly see new AgentRouter models.
	h.syncCaches(c.Request.Context())
}

// RegisterZenMuxProvider auto-discovers all ZenMux AI models available under
// an API key, creates model pools for each, and binds the credential
// to all of them in one transaction.
//
// @Summary      Auto-discover ZenMux Models
// @Description  Submits a ZenMux API key, fetches the model details, and registers all models automatically
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        body  body      dto.DiscoverZenMuxRequest  true  "ZenMux provider details (api_key required)"
// @Success      200   {object}  dto.DiscoverProviderResponse
// @Failure      400   {object}  dto.ErrorResponse
// @Failure      500   {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/zenmux [post]
func (h *CredentialHandler) RegisterZenMuxProvider(c *gin.Context) {
	var req dto.DiscoverZenMuxRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	if req.APIKey == "" {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "api_key is required for ZenMux auto-discovery"})
		return
	}

	if req.Weight <= 0 {
		req.Weight = 1
	}

	count, models, err := credentials.DiscoverAndRegisterZenMuxModels(
		c.Request.Context(),
		h.db,
		h.vault,
		req.APIKey,
		req.Weight,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "ZenMux auto-discovery failed", Details: err.Error()})
		return
	}

	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       fmt.Sprintf("Successfully synchronized %d ZenMux models", count),
		ModelsCount:   count,
		DiscoveredIDs: models,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly registered ZenMux models.
	h.syncCaches(c.Request.Context())
}



// RefreshAllProviders re-runs auto-discovery for every distinct provider key
// already registered in the database. This is the "sync all" action: it reads
// every unique (provider, encrypted_key, base_url, prefix) combination from the
// credentials table, decrypts the key, and calls the appropriate discovery
// function. New clean-alias pools (e.g. "dall-e-3" alongside "1min/dall-e-3")
// are provisioned automatically — no manual re-submission of keys required.
//
// Providers supported for re-discovery:
//   - nvidia    → DiscoverAndRegisterNvidiaModels
//   - ollama    → DiscoverAndRegisterOllamaModels
//   - openrouter → DiscoverAndRegisterOpenRouterModels
//   - 1minai    → DiscoverAndRegisterOneMinAIModels
//   - cloudflare → DiscoverAndRegisterCloudflareModels
//   - sarvam    → DiscoverAndRegisterSarvamModels
//   - puter     → DiscoverAndRegisterPuterModels
//   - agentrouter → DiscoverAndRegisterAgentRouterModels
//   - all others → DiscoverAndRegisterCustomModels (any OpenAI-compatible)
//
// Results are aggregated across all provider accounts and returned as a single
// summary. Per-provider errors are collected but do not abort the overall refresh.
//
// @Summary      Refresh all providers
// @Description  Re-runs auto-discovery for every provider key already in the database, provisioning any missing alias pools
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Success      200  {object}  dto.DiscoverProviderResponse
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/refresh [post]
func (h *CredentialHandler) RefreshAllProviders(c *gin.Context) {
	ctx := c.Request.Context()

	// 1. Query distinct provider accounts from the DB.
	//    We group by (provider, encrypted_key, base_url, prefix) so that each
	//    unique key+endpoint combination is re-discovered exactly once, even if
	//    that key is bound to hundreds of model pools.
	type providerAccount struct {
		Provider     string
		EncryptedKey string
		BaseURL      string
		Prefix       string
		Weight       int
	}

	rows, err := h.db.Query(ctx, `
		SELECT DISTINCT ON (provider, encrypted_key, base_url, COALESCE(prefix,''))
		       provider, encrypted_key, base_url, COALESCE(prefix,'') AS prefix,
		       MAX(weight) OVER (PARTITION BY provider, encrypted_key, base_url, COALESCE(prefix,'')) AS weight
		FROM credentials
		ORDER BY provider, encrypted_key, base_url, COALESCE(prefix,'')
	`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{
			Error:   "failed to query provider accounts",
			Details: err.Error(),
		})
		return
	}
	defer rows.Close()

	var accounts []providerAccount
	for rows.Next() {
		var acc providerAccount
		if err := rows.Scan(&acc.Provider, &acc.EncryptedKey, &acc.BaseURL, &acc.Prefix, &acc.Weight); err != nil {
			continue
		}
		accounts = append(accounts, acc)
	}
	rows.Close()

	if len(accounts) == 0 {
		c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
			Message:       "No provider accounts found — nothing to refresh",
			ModelsCount:   0,
			DiscoveredIDs: []string{},
		})
		return
	}

	// 2. Re-run discovery for each unique account.
	var allDiscovered []string
	var providerErrors []string

	for _, acc := range accounts {
		// Decrypt the stored key before passing it to the discovery function.
		apiKey, decErr := h.vault.Decrypt(acc.EncryptedKey)
		if decErr != nil {
			providerErrors = append(providerErrors,
				fmt.Sprintf("%s(%s): decrypt error: %v", acc.Provider, acc.BaseURL, decErr))
			continue
		}

		weight := acc.Weight
		if weight <= 0 {
			weight = 1
		}

		var count int
		var discovered []string
		var discErr error

		switch acc.Provider {
		case "nvidia":
			count, discovered, discErr = credentials.DiscoverAndRegisterNvidiaModels(
				ctx, h.db, h.vault, apiKey, acc.BaseURL, weight)

		case "ollama":
			count, discovered, discErr = credentials.DiscoverAndRegisterOllamaModels(
				ctx, h.db, h.vault, apiKey, acc.BaseURL, weight)

		case "openrouter":
			count, discovered, discErr = credentials.DiscoverAndRegisterOpenRouterModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "1minai":
			count, discovered, discErr = credentials.DiscoverAndRegisterOneMinAIModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "cloudflare":
			// Recover the account ID from the stored base_url convention.
			// base_url is stored as "cloudflare:<accountID>" by RegisterCloudflareProvider.
			accountID := strings.TrimPrefix(acc.BaseURL, "cloudflare:")
			count, discovered, discErr = credentials.DiscoverAndRegisterCloudflareModels(
				ctx, h.db, h.vault, accountID, apiKey, weight)

		case "sarvam":
			count, discovered, discErr = credentials.DiscoverAndRegisterSarvamModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "puter":
			count, discovered, discErr = credentials.DiscoverAndRegisterPuterModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "agentrouter":
			count, discovered, discErr = credentials.DiscoverAndRegisterAgentRouterModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "zenmux":
			count, discovered, discErr = credentials.DiscoverAndRegisterZenMuxModels(
				ctx, h.db, h.vault, apiKey, weight)

		case "gemini":
			count, discovered, discErr = credentials.DiscoverAndRegisterGeminiModels(
				ctx, h.db, h.vault, apiKey, weight)

		default:
			// Any OpenAI-compatible provider (openai, anthropic, deepseek, custom, …)
			count, discovered, discErr = credentials.DiscoverAndRegisterCustomModels(
				ctx, h.db, h.vault, apiKey, acc.BaseURL, acc.Provider, weight, acc.Prefix)
		}

		if discErr != nil {
			providerErrors = append(providerErrors,
				fmt.Sprintf("%s(%s): %v", acc.Provider, acc.BaseURL, discErr))
			continue
		}
		_ = count
		allDiscovered = append(allDiscovered, discovered...)
	}

	// 3. Build the response summary.
	msg := fmt.Sprintf("Refreshed %d provider account(s) — %d model pools synchronized",
		len(accounts)-len(providerErrors), len(allDiscovered))
	if len(providerErrors) > 0 {
		msg += fmt.Sprintf(" (%d account(s) failed: see details)", len(providerErrors))
	}

	// Return partial-success (200) even when some accounts failed so the client
	// can surface the successful ones without treating the whole refresh as fatal.
	c.JSON(http.StatusOK, dto.DiscoverProviderResponse{
		Message:       msg,
		ModelsCount:   len(allDiscovered),
		DiscoveredIDs: allDiscovered,
	})

	// Synchronously reload caches so all cluster nodes instantly see newly refreshed models.
	h.syncCaches(ctx)
}

// TriggerReDiscovery launches an asynchronous re-discovery job that scans all
// registered provider endpoints for newly available models and auto-registers
// them into the pool system. The job runs in the background via the scheduler
// framework and produces a structured JSON report stored in job_runs.output.
//
// @Summary      Trigger model re-discovery
// @Description  Launches an async scan of all provider endpoints to discover and register new models
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Success      202  {object}  map[string]string
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /api/v1/admin/providers/rediscover [post]
func (h *CredentialHandler) TriggerReDiscovery(c *gin.Context) {
	if h.scheduler == nil {
		c.JSON(http.StatusServiceUnavailable, dto.ErrorResponse{
			Error:   "Job scheduler is not available",
			Details: "The scheduler is required for async re-discovery but was not initialized",
		})
		return
	}

	ctx := c.Request.Context()

	// Look up the re-discovery job by its job_type. If it doesn't exist yet,
	// auto-create it as a manual-trigger-only singleton job with a proper UUID.
	var jobID string
	err := h.db.QueryRow(ctx,
		`SELECT id FROM jobs WHERE job_type = 'provider_rediscovery' LIMIT 1`,
	).Scan(&jobID)

	if err != nil {
		// Job definition doesn't exist yet — create it with a generated UUID.
		err = h.db.QueryRow(ctx, `
			INSERT INTO jobs (name, description, job_type, schedule_type, payload, timezone, max_retries, timeout_seconds, is_enabled, is_singleton)
			VALUES ('Provider Re-Discovery', 'Scans all registered provider endpoints for new AI models and auto-registers them', 'provider_rediscovery', 'manual', '{}'::jsonb, 'UTC', 0, 600, true, true)
			RETURNING id
		`).Scan(&jobID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, dto.ErrorResponse{
				Error:   "Failed to create re-discovery job definition",
				Details: err.Error(),
			})
			return
		}
	}

	// Check if a run is already in progress (singleton guard)
	var runningCount int
	_ = h.db.QueryRow(ctx, `
		SELECT COUNT(*) FROM job_runs
		WHERE job_id = $1 AND status IN ('pending', 'running')
	`, jobID).Scan(&runningCount)
	if runningCount > 0 {
		c.JSON(http.StatusConflict, dto.ErrorResponse{
			Error:   "Re-discovery is already running",
			Details: "Please wait for the current scan to complete before starting another",
		})
		return
	}

	runID, err := h.scheduler.TriggerNow(ctx, jobID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{
			Error:   "Failed to dispatch re-discovery job",
			Details: err.Error(),
		})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{
		"message": "Re-discovery scan started — scanning all provider endpoints for new models",
		"run_id":  runID,
	})
}

// GetReDiscoveryStatus returns the status and report from the most recent
// re-discovery job run. The frontend polls this endpoint to display progress
// and the expandable results banner.
//
// @Summary      Get re-discovery status
// @Description  Returns the status and report from the latest re-discovery run
// @Tags         Credentials
// @Produce      json
// @Security     BearerAuth
// @Success      200  {object}  map[string]string
// @Router       /api/v1/admin/providers/rediscover/status [get]
func (h *CredentialHandler) GetReDiscoveryStatus(c *gin.Context) {
	var runID string
	var status string
	var result *string
	var errorMsg *string
	var durationMs *int64
	var startedAt *string
	var finishedAt *string

	// Join through the jobs table to find runs by job_type (avoids hardcoded UUID)
	err := h.db.QueryRow(c.Request.Context(), `
		SELECT jr.id, jr.status, jr.output, jr.error_message, jr.duration_ms,
		       jr.started_at::text, jr.finished_at::text
		FROM job_runs jr
		JOIN jobs j ON j.id = jr.job_id
		WHERE j.job_type = 'provider_rediscovery'
		ORDER BY jr.created_at DESC LIMIT 1
	`).Scan(&runID, &status, &result, &errorMsg, &durationMs, &startedAt, &finishedAt)

	if err != nil {
		// No runs found — system is idle
		c.JSON(http.StatusOK, gin.H{"status": "IDLE"})
		return
	}

	resp := gin.H{
		"run_id": runID,
		"status": strings.ToUpper(status),
	}
	if result != nil {
		resp["report"] = *result
	}
	if errorMsg != nil {
		resp["error"] = *errorMsg
	}
	if durationMs != nil {
		resp["duration_ms"] = *durationMs
	}
	if startedAt != nil {
		resp["started_at"] = *startedAt
	}
	if finishedAt != nil {
		resp["finished_at"] = *finishedAt
	}

	c.JSON(http.StatusOK, resp)
}

// BulkDelete deletes multiple credentials at once.
// @Summary      Bulk delete credentials
// @Description  Permanently removes multiple provider credentials by their IDs
// @Tags         Credentials
// @Accept       json
// @Produce      json
// @Security     BearerAuth
// @Param        request   body      dto.BulkDeleteRequest  true  "IDs of credentials to delete"
// @Success      200       {object}  dto.SuccessResponse
// @Failure      400       {object}  dto.ErrorResponse
// @Failure      500       {object}  dto.ErrorResponse
// @Router       /api/v1/admin/credentials/bulk-delete [post]
func (h *CredentialHandler) BulkDelete(c *gin.Context) {
	var req dto.BulkDeleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, dto.ErrorResponse{Error: "invalid request body", Details: err.Error()})
		return
	}

	err := database.DeleteCredentialsBulk(c.Request.Context(), h.db, req.IDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, dto.ErrorResponse{Error: "failed to delete credentials in bulk", Details: err.Error()})
		return
	}

	// Synchronously reload caches so all cluster nodes stop routing to deleted credentials.
	h.syncCaches(c.Request.Context())

	c.JSON(http.StatusOK, dto.SuccessResponse{Message: fmt.Sprintf("%d credentials deleted successfully", len(req.IDs))})
}
