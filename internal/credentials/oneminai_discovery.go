package credentials

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DiscoverAndRegisterOneMinAIModels validates the 1min.ai API key, then iterates
// over the static model manifest to auto-provision model_pools and bind the
// credential to every model across all five modalities (chat, code, image,
// audio, video) in a single atomic transaction.
//
// Unlike OpenAI-compatible providers, 1min.ai does not expose a /v1/models
// endpoint. Instead, the full catalog is maintained as a hardcoded manifest
// (oneminai_manifest.go). Key validity is confirmed via a lightweight chat
// request before registration begins.
//
// Returns the count of registered models, their patterns, and any error.
func DiscoverAndRegisterOneMinAIModels(ctx context.Context, db *pgxpool.Pool, vault *Vault, apiKey string, weight int) (int, []string, error) {
	if apiKey == "" {
		return 0, nil, fmt.Errorf("1min.ai api key is required for model discovery")
	}

	if weight <= 0 {
		weight = 1
	}

	// 1. Validate the API key with a minimal chat request
	if err := validateOneMinAIKey(ctx, apiKey); err != nil {
		return 0, nil, err
	}

	// 2. Encrypt the API key once before the database loop
	encryptedKey, err := vault.Encrypt(apiKey)
	if err != nil {
		return 0, nil, fmt.Errorf("vault encryption failed: %w", err)
	}

	// 2.5 Find which pools already have this apiKey bound to avoid duplicates.
	alreadyBound := make(map[int]bool)
	rows, err := db.Query(ctx, `SELECT pool_id, encrypted_key FROM credentials WHERE provider = $1`, "1minai")
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var poolID int
			var encKey string
			if err := rows.Scan(&poolID, &encKey); err == nil {
				decrypted, decErr := vault.Decrypt(encKey)
				if decErr == nil && decrypted == apiKey {
					alreadyBound[poolID] = true
				}
			}
		}
	}

	// 3. Open a transaction to atomically write all pools and credentials
	tx, err := db.Begin(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck — intentional deferred cleanup

	var discoveredModels []string

	manifest := OneMinAIManifest()

	for _, entry := range manifest {
		// Each manifest entry is registered under two pool patterns:
		//
		//   1. The prefixed gateway pattern  (e.g. "1min/dall-e-3") — used by
		//      clients that explicitly select the 1min.ai provider.
		//   2. The clean upstream model name (e.g. "dall-e-3") — required for
		//      client tools (Cline, LobeChat, Open WebUI …) that hardcode a
		//      whitelist of known model names and reject any prefixed variant.
		//
		// Both pools bind the same encrypted 1min.ai credential, so the
		// load-balancer ring treats them independently (and if a native OpenAI
		// key is also configured for "dall-e-3", it joins the same pool and
		// the gateway naturally load-balances across both providers).
		patterns := []string{entry.Pattern}
		if entry.Model != entry.Pattern {
			patterns = append(patterns, entry.Model)
		}

		for _, modelPattern := range patterns {
			// 4. Classify capabilities from the model identifier, then apply
			//    modality-specific overrides so the gateway knows exactly what
			//    each model supports.
			caps := ClassifyModel(modelPattern)
			applyOneMinAIOverrides(&caps, entry.Modality)

			capsJSON, err := json.Marshal(caps.ToMap())
			if err != nil {
				capsJSON = []byte("{}")
			}

			// 5. Upsert the model_pool — updates capabilities on re-discovery
			var poolID int
			err = tx.QueryRow(ctx,
				`INSERT INTO model_pools (model_pattern, strategy, capabilities)
				 VALUES ($1, 'round-robin', $2)
				 ON CONFLICT (model_pattern) DO UPDATE
				 SET capabilities = EXCLUDED.capabilities
				 RETURNING id`,
				modelPattern, capsJSON,
			).Scan(&poolID)
			if err != nil {
				return 0, nil, fmt.Errorf("failed to upsert model pool for %s: %w", modelPattern, err)
			}

			// 6. Bind the 1min.ai credential to this pool (idempotent)
			if !alreadyBound[poolID] {
				_, err = tx.Exec(ctx,
					`INSERT INTO credentials (pool_id, provider, encrypted_key, base_url, weight, is_healthy)
					 VALUES ($1, '1minai', $2, $3, $4, true)`,
					poolID, encryptedKey, OneMinAIBaseURL, weight,
				)
				if err != nil {
					return 0, nil, fmt.Errorf("failed to bind credential for pool %s: %w", modelPattern, err)
				}
				alreadyBound[poolID] = true
			}

			discoveredModels = append(discoveredModels, modelPattern)
		}
	}

	// 7. Notify the SyncManager to instantly hot-reload the routing cache
	if _, err = tx.Exec(ctx, "NOTIFY config_change, 'model_pools:reload'"); err != nil {
		return 0, nil, fmt.Errorf("failed to broadcast config change notification: %w", err)
	}

	return len(discoveredModels), discoveredModels, tx.Commit(ctx)
}

// DiscoverAndRegisterOneMinAIModelsBatch processes multiple 1min.ai API keys one
// by one against the full 1min.ai static model catalog. Keys are deduplicated
// and trimmed while preserving order. A failing key (invalid key, network
// error, …) is recorded in the results and does NOT abort the batch — the
// remaining keys still get discovered and bound, exactly like the custom and
// Puter batch flows. Each valid key is inserted atomically into the
// credentials table and bound to every manifest model pool for round-robin
// load balancing.
func DiscoverAndRegisterOneMinAIModelsBatch(
	ctx context.Context,
	db *pgxpool.Pool,
	vault *Vault,
	apiKeys []string,
	weight int,
) (totalKeys, successCount, failedCount, totalModels int, results []BatchKeyDiscoveryResult, allDiscovered []string, err error) {
	// Deduplicate and trim keys while preserving order.
	seenKeys := make(map[string]bool)
	var cleanKeys []string
	for _, raw := range apiKeys {
		k := strings.TrimSpace(raw)
		if k != "" && !seenKeys[k] {
			seenKeys[k] = true
			cleanKeys = append(cleanKeys, k)
		}
	}

	if len(cleanKeys) == 0 {
		return 0, 0, 0, 0, nil, nil, fmt.Errorf("no valid 1min.ai API keys provided")
	}

	totalKeys = len(cleanKeys)
	results = make([]BatchKeyDiscoveryResult, 0, totalKeys)
	discoveredMap := make(map[string]bool)

	for i, k := range cleanKeys {
		res := BatchKeyDiscoveryResult{
			Index:     i + 1,
			KeyMasked: MaskAPIKey(k),
		}

		count, models, dErr := DiscoverAndRegisterOneMinAIModels(ctx, db, vault, k, weight)
		if dErr != nil {
			res.Success = false
			res.Error = dErr.Error()
			failedCount++
		} else {
			res.Success = true
			res.ModelsCount = count
			res.DiscoveredIDs = models
			successCount++
			for _, m := range models {
				if !discoveredMap[m] {
					discoveredMap[m] = true
					allDiscovered = append(allDiscovered, m)
				}
			}
		}
		results = append(results, res)
	}

	totalModels = len(allDiscovered)
	return totalKeys, successCount, failedCount, totalModels, results, allDiscovered, nil
}

// validateOneMinAIKey sends a minimal chat request to 1min.ai to confirm the
// API key is valid. This consumes a tiny amount of credits but is the most
// reliable validation method since 1min.ai has no dedicated key-check endpoint.
func validateOneMinAIKey(ctx context.Context, apiKey string) error {
	client := &http.Client{Timeout: 15 * time.Second}

	body := `{"type":"UNIFY_CHAT_WITH_AI","model":"gpt-4o-mini","promptObject":{"prompt":"hi"}}`

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		OneMinAIBaseURL+"/api/chat-with-ai", strings.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to build validation request: %w", err)
	}

	req.Header.Set("API-KEY", apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("1min.ai connection failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("1min.ai rejected the api key (%d) — verify your key at app.1min.ai", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("1min.ai returned unexpected status during validation: %d", resp.StatusCode)
	}

	return nil
}

// applyOneMinAIOverrides sets capability flags based on the modality declared
// in the manifest entry. This ensures correct capability metadata even when the
// heuristic classifier might miss a model (e.g. "ideogram" for image generation).
func applyOneMinAIOverrides(caps *ModelCapabilities, modality string) {
	switch modality {
	case "image":
		caps.ImageGeneration = true
	case "audio_tts", "audio_stt":
		caps.Audio = true
	case "video":
		caps.Video = true
	case "code":
		caps.Code = true
	}
}
