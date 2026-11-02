package database

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// --- Tenant Queries ---

// TenantRow represents a tenant record from the database.
type TenantRow struct {
	ID           string
	Name         string
	APIKey       string
	TokenBalance int64
	IsActive     bool
	RateLimitRPM int
	CreatedAt    string
	UpdatedAt    string
}

// GetTenantByAPIKey fetches a tenant by their virtual API key.
func GetTenantByAPIKey(ctx context.Context, pool *pgxpool.Pool, apiKey string) (*TenantRow, error) {
	row := pool.QueryRow(ctx, `
		SELECT id, name, api_key, token_balance, is_active, rate_limit_rpm, 
		       created_at::text, updated_at::text
		FROM tenants WHERE api_key = $1
	`, apiKey)

	t := &TenantRow{}
	err := row.Scan(&t.ID, &t.Name, &t.APIKey, &t.TokenBalance, &t.IsActive, &t.RateLimitRPM, &t.CreatedAt, &t.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	return t, nil
}

// ListTenants returns all tenants.
func ListTenants(ctx context.Context, pool *pgxpool.Pool) ([]*TenantRow, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, name, api_key, token_balance, is_active, rate_limit_rpm,
		       created_at::text, updated_at::text
		FROM tenants ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to list tenants: %w", err)
	}
	defer rows.Close()

	var tenants []*TenantRow
	for rows.Next() {
		t := &TenantRow{}
		if err := rows.Scan(&t.ID, &t.Name, &t.APIKey, &t.TokenBalance, &t.IsActive, &t.RateLimitRPM, &t.CreatedAt, &t.UpdatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan tenant: %w", err)
		}
		tenants = append(tenants, t)
	}
	return tenants, nil
}

// CreateTenant inserts a new tenant and returns the generated ID.
func CreateTenant(ctx context.Context, pool *pgxpool.Pool, name, apiKey string, tokenBalance int64, rateLimitRPM int) (string, error) {
	var id string
	err := pool.QueryRow(ctx, `
		INSERT INTO tenants (name, api_key, token_balance, rate_limit_rpm)
		VALUES ($1, $2, $3, $4)
		RETURNING id
	`, name, apiKey, tokenBalance, rateLimitRPM).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("failed to create tenant: %w", err)
	}
	return id, nil
}

// UpdateTenant updates a tenant's mutable fields.
func UpdateTenant(ctx context.Context, pool *pgxpool.Pool, id, name string, tokenBalance int64, isActive bool, rateLimitRPM int) error {
	_, err := pool.Exec(ctx, `
		UPDATE tenants SET name = $2, token_balance = $3, is_active = $4, 
		       rate_limit_rpm = $5, updated_at = NOW()
		WHERE id = $1
	`, id, name, tokenBalance, isActive, rateLimitRPM)
	if err != nil {
		return fmt.Errorf("failed to update tenant: %w", err)
	}
	return nil
}

// DeleteTenant removes a tenant by ID.
func DeleteTenant(ctx context.Context, pool *pgxpool.Pool, id string) error {
	_, err := pool.Exec(ctx, `DELETE FROM tenants WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("failed to delete tenant: %w", err)
	}
	return nil
}

// --- Model Pool Queries ---

// ModelPoolRow represents a model routing pool record.
type ModelPoolRow struct {
	ID             int
	ModelPattern   string
	Strategy       string
	FallbackPoolID *int
	Capabilities   []byte // Raw JSONB bytes — decoded by callers
	CreatedAt      string
}

// ListModelPools returns all model routing pools.
func ListModelPools(ctx context.Context, pool *pgxpool.Pool) ([]*ModelPoolRow, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, model_pattern, strategy, fallback_pool_id, capabilities, created_at::text
		FROM model_pools ORDER BY model_pattern
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to list pools: %w", err)
	}
	defer rows.Close()

	var pools []*ModelPoolRow
	for rows.Next() {
		p := &ModelPoolRow{}
		if err := rows.Scan(&p.ID, &p.ModelPattern, &p.Strategy, &p.FallbackPoolID, &p.Capabilities, &p.CreatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan pool: %w", err)
		}
		pools = append(pools, p)
	}
	return pools, nil
}

// ModelPoolRowWithCount extends ModelPoolRow with the number of credentials
// attached to the pool and the number of healthy ones, used by the paginated
// admin pools list.
type ModelPoolRowWithCount struct {
	ModelPoolRow
	CredentialCount int
	HealthyCount    int
}

// PoolFilter holds the filtering and sorting parameters for paginated model pools.
type PoolFilter struct {
	Search         string
	Strategy       string
	Capabilities   []string
	HasFallback    *bool
	HasCredentials *bool
	HealthStatus   string
	SortBy         string
	SortOrder      string
}

// ListModelPoolsPaginated returns a single page of model routing pools
// together with each pool's credential count and the total number of rows
// matching the filters (ignoring limit/offset).
// This powers the paginated / virtualized admin pools list.
func ListModelPoolsPaginated(ctx context.Context, pool *pgxpool.Pool, limit, offset int, filter PoolFilter) ([]*ModelPoolRowWithCount, int, error) {
	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	var conditions []string
	var havingConditions []string
	var args []interface{}
	argCount := 1

	if s := strings.TrimSpace(filter.Search); s != "" {
		conditions = append(conditions, fmt.Sprintf("(mp.model_pattern ILIKE $%d OR mp.strategy ILIKE $%d)", argCount, argCount))
		args = append(args, "%"+s+"%")
		argCount++
	}

	if filter.Strategy != "" {
		conditions = append(conditions, fmt.Sprintf("mp.strategy = $%d", argCount))
		args = append(args, filter.Strategy)
		argCount++
	}

	if filter.HasFallback != nil {
		if *filter.HasFallback {
			conditions = append(conditions, "mp.fallback_pool_id IS NOT NULL")
		} else {
			conditions = append(conditions, "mp.fallback_pool_id IS NULL")
		}
	}

	if len(filter.Capabilities) > 0 {
		capsMap := make(map[string]bool)
		for _, capVal := range filter.Capabilities {
			if capVal != "" {
				capsMap[capVal] = true
			}
		}
		if len(capsMap) > 0 {
			capsBytes, _ := json.Marshal(capsMap)
			conditions = append(conditions, fmt.Sprintf("mp.capabilities @> $%d::jsonb", argCount))
			args = append(args, capsBytes)
			argCount++
		}
	}

	if filter.HasCredentials != nil {
		if *filter.HasCredentials {
			havingConditions = append(havingConditions, "COUNT(c.id) > 0")
		} else {
			havingConditions = append(havingConditions, "COUNT(c.id) = 0")
		}
	}

	if filter.HealthStatus != "" {
		switch filter.HealthStatus {
		case "healthy":
			havingConditions = append(havingConditions, "COUNT(c.id) > 0 AND SUM(CASE WHEN NOT c.is_healthy THEN 1 ELSE 0 END) = 0")
		case "unhealthy":
			havingConditions = append(havingConditions, "SUM(CASE WHEN NOT c.is_healthy THEN 1 ELSE 0 END) > 0")
		case "empty":
			havingConditions = append(havingConditions, "COUNT(c.id) = 0")
		}
	}

	sortBy := "mp.model_pattern"
	if filter.SortBy != "" {
		switch filter.SortBy {
		case "id":
			sortBy = "mp.id"
		case "created_at":
			sortBy = "mp.created_at"
		case "credential_count":
			sortBy = "credential_count"
		case "strategy":
			sortBy = "mp.strategy"
		case "health_percent":
			sortBy = "CASE WHEN COUNT(c.id) = 0 THEN -1 ELSE ROUND(100.0 * SUM(CASE WHEN c.is_healthy THEN 1 ELSE 0 END) / COUNT(c.id)) END"
		}
	}

	sortOrder := "ASC"
	if strings.ToLower(filter.SortOrder) == "desc" {
		sortOrder = "DESC"
	}

	query := `
		SELECT mp.id, mp.model_pattern, mp.strategy, mp.fallback_pool_id, mp.capabilities, mp.created_at::text,
		       COUNT(c.id) AS credential_count,
		       COALESCE(SUM(CASE WHEN c.is_healthy THEN 1 ELSE 0 END), 0) AS healthy_count,
		       COUNT(*) OVER() AS total_count
		FROM model_pools mp
		LEFT JOIN credentials c ON c.pool_id = mp.id
	`

	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	query += " GROUP BY mp.id, mp.model_pattern, mp.strategy, mp.fallback_pool_id, mp.capabilities, mp.created_at"

	if len(havingConditions) > 0 {
		query += " HAVING " + strings.Join(havingConditions, " AND ")
	}

	query += fmt.Sprintf(" ORDER BY %s %s LIMIT $%d OFFSET $%d", sortBy, sortOrder, argCount, argCount+1)
	args = append(args, limit, offset)

	rows, err := pool.Query(ctx, query, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list pools (paginated): %w", err)
	}
	defer rows.Close()

	var pools []*ModelPoolRowWithCount
	total := 0
	for rows.Next() {
		p := &ModelPoolRowWithCount{}
		if err := rows.Scan(&p.ID, &p.ModelPattern, &p.Strategy, &p.FallbackPoolID, &p.Capabilities, &p.CreatedAt, &p.CredentialCount, &p.HealthyCount, &total); err != nil {
			return nil, 0, fmt.Errorf("failed to scan pool: %w", err)
		}
		pools = append(pools, p)
	}
	// total stays 0 when there are no rows, which is the correct count.
	return pools, total, nil
}

// GetModelPool returns a single model pool by ID.
func GetModelPool(ctx context.Context, dbPool *pgxpool.Pool, id int) (*ModelPoolRow, error) {
	row := dbPool.QueryRow(ctx, `
		SELECT id, model_pattern, strategy, fallback_pool_id, capabilities, created_at::text
		FROM model_pools WHERE id = $1
	`, id)

	p := &ModelPoolRow{}
	err := row.Scan(&p.ID, &p.ModelPattern, &p.Strategy, &p.FallbackPoolID, &p.Capabilities, &p.CreatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get pool: %w", err)
	}
	return p, nil
}

// CreateModelPool inserts a new model routing pool.
func CreateModelPool(ctx context.Context, pool *pgxpool.Pool, modelPattern, strategy string, fallbackPoolID *int) (int, error) {
	var id int
	err := pool.QueryRow(ctx, `
		INSERT INTO model_pools (model_pattern, strategy, fallback_pool_id)
		VALUES ($1, $2, $3)
		RETURNING id
	`, modelPattern, strategy, fallbackPoolID).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to create pool: %w", err)
	}
	return id, nil
}

// UpdateModelPool updates a model pool's configuration.
func UpdateModelPool(ctx context.Context, pool *pgxpool.Pool, id int, modelPattern, strategy string, fallbackPoolID *int) error {
	_, err := pool.Exec(ctx, `
		UPDATE model_pools SET model_pattern = $2, strategy = $3, fallback_pool_id = $4
		WHERE id = $1
	`, id, modelPattern, strategy, fallbackPoolID)
	if err != nil {
		return fmt.Errorf("failed to update pool: %w", err)
	}
	return nil
}

// DeleteModelPool removes a model pool and its credentials (CASCADE).
func DeleteModelPool(ctx context.Context, pool *pgxpool.Pool, id int) error {
	_, err := pool.Exec(ctx, `DELETE FROM model_pools WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("failed to delete pool: %w", err)
	}
	return nil
}

// DeleteModelPoolsBulk removes multiple model pools by their IDs (CASCADE).
func DeleteModelPoolsBulk(ctx context.Context, pool *pgxpool.Pool, ids []int) error {
	if len(ids) == 0 {
		return nil
	}
	_, err := pool.Exec(ctx, `DELETE FROM model_pools WHERE id = ANY($1)`, ids)
	if err != nil {
		return fmt.Errorf("failed to delete pools in bulk: %w", err)
	}
	return nil
}

// DeleteZeroHealthPools deletes all model pools where every credential is
// unhealthy (is_healthy = FALSE) or where the pool has no credentials at all.
// Credentials are cascade-deleted by the foreign key constraint.
// Returns the number of pools deleted.
func DeleteZeroHealthPools(ctx context.Context, pool *pgxpool.Pool) (int64, error) {
	result, err := pool.Exec(ctx, `
		DELETE FROM model_pools
		WHERE id IN (
			SELECT mp.id
			FROM model_pools mp
			LEFT JOIN credentials c ON c.pool_id = mp.id
			GROUP BY mp.id
			HAVING COUNT(c.id) = 0
			    OR SUM(CASE WHEN c.is_healthy THEN 1 ELSE 0 END) = 0
		)
	`)
	if err != nil {
		return 0, fmt.Errorf("failed to purge zero-health pools: %w", err)
	}
	return result.RowsAffected(), nil
}

// --- Credential Queries ---

// CredentialRow represents a provider credential record.
type CredentialRow struct {
	ID           int
	PoolID       int
	Provider     string
	EncryptedKey string
	BaseURL      string
	Weight       int
	IsHealthy    bool
	LastError    *string
	CreatedAt    string
	Prefix       string
}

// ListCredentialsByPool returns all credentials for a given pool.
func ListCredentialsByPool(ctx context.Context, pool *pgxpool.Pool, poolID int) ([]*CredentialRow, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, pool_id, provider, encrypted_key, base_url, weight, 
		       is_healthy, last_error, created_at::text, prefix
		FROM credentials WHERE pool_id = $1 ORDER BY id
	`, poolID)
	if err != nil {
		return nil, fmt.Errorf("failed to list credentials: %w", err)
	}
	defer rows.Close()

	var creds []*CredentialRow
	for rows.Next() {
		c := &CredentialRow{}
		if err := rows.Scan(&c.ID, &c.PoolID, &c.Provider, &c.EncryptedKey, &c.BaseURL, &c.Weight, &c.IsHealthy, &c.LastError, &c.CreatedAt, &c.Prefix); err != nil {
			return nil, fmt.Errorf("failed to scan credential: %w", err)
		}
		creds = append(creds, c)
	}
	return creds, nil
}

// CredentialWithPool extends CredentialRow with the model pattern from the
// associated pool, used by the management dashboard list view.
type CredentialWithPool struct {
	CredentialRow
	ModelPattern string
	Capabilities []byte // JSONB raw bytes from model_pools
}

// ListAllCredentials returns all credentials across all pools, joined with
// the model_pools table to include the model_pattern string and capabilities.
func ListAllCredentials(ctx context.Context, pool *pgxpool.Pool) ([]*CredentialWithPool, error) {
	rows, err := pool.Query(ctx, `
		SELECT c.id, c.pool_id, c.provider, c.encrypted_key, c.base_url, c.weight,
		       c.is_healthy, c.last_error, c.created_at::text,
		       COALESCE(mp.model_pattern, '') AS model_pattern, c.prefix,
		       COALESCE(mp.capabilities, '{}'::jsonb) AS capabilities
		FROM credentials c
		LEFT JOIN model_pools mp ON c.pool_id = mp.id
		ORDER BY c.id
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to list all credentials: %w", err)
	}
	defer rows.Close()

	var creds []*CredentialWithPool
	for rows.Next() {
		c := &CredentialWithPool{}
		if err := rows.Scan(
			&c.ID, &c.PoolID, &c.Provider, &c.EncryptedKey, &c.BaseURL, &c.Weight,
			&c.IsHealthy, &c.LastError, &c.CreatedAt, &c.ModelPattern, &c.Prefix,
			&c.Capabilities,
		); err != nil {
			return nil, fmt.Errorf("failed to scan credential: %w", err)
		}
		creds = append(creds, c)
	}
	return creds, nil
}

// ListCredentialsPaginated returns a single page of credentials joined with
// their pool model pattern, plus the total number of rows matching the same
// filters (ignoring limit/offset). This powers the paginated / virtualized
// admin credentials list.
//
//   - limit  <= 0 defaults to 100
//   - search (non-empty after trim) performs a case-insensitive ILIKE match
//     across provider, base_url and model_pattern
//   - provider (non-empty after trim) performs an exact provider filter
//
// The total count is obtained via a COUNT(*) OVER() window function so it is
// fetched in the same query round-trip.
func ListCredentialsPaginated(ctx context.Context, pool *pgxpool.Pool, limit, offset int, search, provider string) ([]*CredentialWithPool, int, error) {
	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	var searchVal, providerVal *string
	if s := strings.TrimSpace(search); s != "" {
		v := "%" + s + "%"
		searchVal = &v
	}
	if p := strings.TrimSpace(provider); p != "" {
		providerVal = &p
	}

	rows, err := pool.Query(ctx, `
		SELECT c.id, c.pool_id, c.provider, c.encrypted_key, c.base_url, c.weight,
		       c.is_healthy, c.last_error, c.created_at::text,
		       COALESCE(mp.model_pattern, '') AS model_pattern, c.prefix,
		       COUNT(*) OVER() AS total_count
		FROM credentials c
		LEFT JOIN model_pools mp ON c.pool_id = mp.id
		WHERE ($1::text IS NULL OR c.provider = $1)
		  AND ($2::text IS NULL
		       OR c.provider ILIKE $2
		       OR c.base_url ILIKE $2
		       OR COALESCE(mp.model_pattern, '') ILIKE $2)
		ORDER BY c.id
		LIMIT $3 OFFSET $4
	`, providerVal, searchVal, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list credentials (paginated): %w", err)
	}
	defer rows.Close()

	var creds []*CredentialWithPool
	total := 0
	for rows.Next() {
		c := &CredentialWithPool{}
		if err := rows.Scan(
			&c.ID, &c.PoolID, &c.Provider, &c.EncryptedKey, &c.BaseURL, &c.Weight,
			&c.IsHealthy, &c.LastError, &c.CreatedAt, &c.ModelPattern, &c.Prefix,
			&total,
		); err != nil {
			return nil, 0, fmt.Errorf("failed to scan credential: %w", err)
		}
		creds = append(creds, c)
	}
	// total stays 0 when there are no rows, which is the correct count.
	return creds, total, nil
}

// GetCredential returns a single credential by ID.
func GetCredential(ctx context.Context, pool *pgxpool.Pool, id int) (*CredentialRow, error) {
	row := pool.QueryRow(ctx, `
		SELECT id, pool_id, provider, encrypted_key, base_url, weight,
		       is_healthy, last_error, created_at::text, prefix
		FROM credentials WHERE id = $1
	`, id)

	c := &CredentialRow{}
	err := row.Scan(&c.ID, &c.PoolID, &c.Provider, &c.EncryptedKey, &c.BaseURL, &c.Weight, &c.IsHealthy, &c.LastError, &c.CreatedAt, &c.Prefix)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get credential: %w", err)
	}
	return c, nil
}

// CreateCredential inserts a new provider credential.
func CreateCredential(ctx context.Context, pool *pgxpool.Pool, poolID int, provider, encryptedKey, baseURL string, weight int, prefix string) (int, error) {
	var id int
	err := pool.QueryRow(ctx, `
		INSERT INTO credentials (pool_id, provider, encrypted_key, base_url, weight, prefix)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id
	`, poolID, provider, encryptedKey, baseURL, weight, prefix).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to create credential: %w", err)
	}
	return id, nil
}

// UpdateCredential updates a credential's mutable fields.
func UpdateCredential(ctx context.Context, pool *pgxpool.Pool, id int, provider, encryptedKey, baseURL string, weight int, isHealthy bool, prefix string) error {
	_, err := pool.Exec(ctx, `
		UPDATE credentials SET provider = $2, encrypted_key = $3, base_url = $4, 
		       weight = $5, is_healthy = $6, prefix = $7
		WHERE id = $1
	`, id, provider, encryptedKey, baseURL, weight, isHealthy, prefix)
	if err != nil {
		return fmt.Errorf("failed to update credential: %w", err)
	}
	return nil
}

// DeleteCredential removes a credential by ID.
func DeleteCredential(ctx context.Context, pool *pgxpool.Pool, id int) error {
	_, err := pool.Exec(ctx, `DELETE FROM credentials WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("failed to delete credential: %w", err)
	}
	return nil
}

// DeleteCredentialsBulk removes multiple credentials by their IDs.
func DeleteCredentialsBulk(ctx context.Context, pool *pgxpool.Pool, ids []int) error {
	if len(ids) == 0 {
		return nil
	}
	_, err := pool.Exec(ctx, `DELETE FROM credentials WHERE id = ANY($1)`, ids)
	if err != nil {
		return fmt.Errorf("failed to delete credentials in bulk: %w", err)
	}
	return nil
}

// --- Bulk Loading (for cache warm-up) ---

// LoadAllPoolsWithCredentials loads all model pools and their credentials
// for initial cache population at startup in two fast queries instead of N+1 queries.
func LoadAllPoolsWithCredentials(ctx context.Context, pool *pgxpool.Pool) ([]*ModelPoolRow, map[int][]*CredentialRow, error) {
	pools, err := ListModelPools(ctx, pool)
	if err != nil {
		return nil, nil, err
	}

	rows, err := pool.Query(ctx, `
		SELECT id, pool_id, provider, encrypted_key, base_url, weight, 
		       is_healthy, last_error, created_at::text, prefix
		FROM credentials ORDER BY pool_id, id
	`)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query all credentials: %w", err)
	}
	defer rows.Close()

	credsByPool := make(map[int][]*CredentialRow, len(pools))
	for rows.Next() {
		c := &CredentialRow{}
		if err := rows.Scan(&c.ID, &c.PoolID, &c.Provider, &c.EncryptedKey, &c.BaseURL, &c.Weight, &c.IsHealthy, &c.LastError, &c.CreatedAt, &c.Prefix); err != nil {
			return nil, nil, fmt.Errorf("failed to scan credential: %w", err)
		}
		credsByPool[c.PoolID] = append(credsByPool[c.PoolID], c)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("error reading credentials: %w", err)
	}

	return pools, credsByPool, nil
}

// --- Playground Conversation Queries ---

// ConversationRow represents a conversation history session stored in the database.
type ConversationRow struct {
	ID        string
	TenantID  string
	Title     string
	Messages  []byte // Raw JSONB bytes
	CreatedAt string
	UpdatedAt string
}

// ListConversations returns all conversations for a tenant sorted by last updated.
func ListConversations(ctx context.Context, pool *pgxpool.Pool, tenantID string) ([]*ConversationRow, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, tenant_id, title, messages, created_at::text, updated_at::text
		FROM conversations 
		WHERE tenant_id = $1
		ORDER BY updated_at DESC
	`, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to list conversations: %w", err)
	}
	defer rows.Close()

	var list []*ConversationRow
	for rows.Next() {
		c := &ConversationRow{}
		if err := rows.Scan(&c.ID, &c.TenantID, &c.Title, &c.Messages, &c.CreatedAt, &c.UpdatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan conversation: %w", err)
		}
		list = append(list, c)
	}
	return list, nil
}

// GetConversation fetches a single conversation by UUID and validates tenant access.
func GetConversation(ctx context.Context, pool *pgxpool.Pool, id string, tenantID string) (*ConversationRow, error) {
	row := pool.QueryRow(ctx, `
		SELECT id, tenant_id, title, messages, created_at::text, updated_at::text
		FROM conversations WHERE id = $1 AND tenant_id = $2
	`, id, tenantID)

	c := &ConversationRow{}
	err := row.Scan(&c.ID, &c.TenantID, &c.Title, &c.Messages, &c.CreatedAt, &c.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get conversation: %w", err)
	}
	return c, nil
}

// CreateConversation inserts a new conversation history session for a tenant.
func CreateConversation(ctx context.Context, pool *pgxpool.Pool, tenantID string, title string, messages []byte) (string, error) {
	var id string
	err := pool.QueryRow(ctx, `
		INSERT INTO conversations (tenant_id, title, messages)
		VALUES ($1, $2, $3)
		RETURNING id
	`, tenantID, title, messages).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("failed to create conversation: %w", err)
	}
	return id, nil
}

// UpdateConversation updates the title and/or messages array of a conversation under tenant validation.
func UpdateConversation(ctx context.Context, pool *pgxpool.Pool, id string, tenantID string, title string, messages []byte) error {
	_, err := pool.Exec(ctx, `
		UPDATE conversations SET title = $3, messages = $4, updated_at = NOW()
		WHERE id = $1 AND tenant_id = $2
	`, id, tenantID, title, messages)
	if err != nil {
		return fmt.Errorf("failed to update conversation: %w", err)
	}
	return nil
}

// DeleteConversation deletes a conversation session from the database under tenant validation.
func DeleteConversation(ctx context.Context, pool *pgxpool.Pool, id string, tenantID string) error {
	_, err := pool.Exec(ctx, `DELETE FROM conversations WHERE id = $1 AND tenant_id = $2`, id, tenantID)
	if err != nil {
		return fmt.Errorf("failed to delete conversation: %w", err)
	}
	return nil
}

// --- Credential Health State Updates ---

// UpdateCredentialHealthState updates the health status and last error message of a credential.
func UpdateCredentialHealthState(ctx context.Context, pool *pgxpool.Pool, id int, isHealthy bool, lastError *string) error {
	_, err := pool.Exec(ctx, `
		UPDATE credentials SET is_healthy = $2, last_error = $3
		WHERE id = $1
	`, id, isHealthy, lastError)
	if err != nil {
		return fmt.Errorf("failed to update credential health: %w", err)
	}
	return nil
}

// --- Request Telemetry Log Retrieval & Semantic Search ---

// LogWithVectorRow represents a merged telemetry log entry with prompt/response text and similarity.
type LogWithVectorRow struct {
	ID               int64   `json:"id"`
	TenantID         *string `json:"tenant_id,omitempty"`
	TenantName       *string `json:"tenant_name,omitempty"`
	Model            string  `json:"model"`
	Provider         string  `json:"provider"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	LatencyMs        int     `json:"latency_ms"`
	StatusCode       int     `json:"status_code"`
	ErrorMessage     *string `json:"error_message,omitempty"`
	CreatedAt        string  `json:"created_at"`
	PromptText       *string `json:"prompt_text,omitempty"`
	ResponseText     *string `json:"response_text,omitempty"`
	Similarity       float64 `json:"similarity,omitempty"`
}

// ListLogsForPool queries and filters telemetry logs for a pool pattern, supporting semantic search fallback.
func ListLogsForPool(
	ctx context.Context,
	pool *pgxpool.Pool,
	modelPattern string,
	limit, offset int,
	tenantFilter, statusFilter, searchKeyword string,
	semanticQuery string,
	embeddingVector []float32,
) ([]*LogWithVectorRow, error) {
	// Base query variables
	var tenantIDVal *string
	if tenantFilter != "" {
		tenantIDVal = &tenantFilter
	}
	var statusVal *string
	if statusFilter != "" {
		statusVal = &statusFilter
	}
	var searchVal *string
	if searchKeyword != "" {
		wrapped := "%" + searchKeyword + "%"
		searchVal = &wrapped
	}

	// Determine matching logic for model pattern:
	// Handle wildcard models like "claude-*" using LIKE 'claude-%'
	var modelPatternQuery string
	var modelArgs []interface{}
	if strings.Contains(modelPattern, "*") {
		prefix := strings.ReplaceAll(modelPattern, "*", "%")
		modelPatternQuery = "l.model LIKE $1"
		modelArgs = append(modelArgs, prefix)
	} else {
		modelPatternQuery = "l.model = $1"
		modelArgs = append(modelArgs, modelPattern)
	}

	// 1. Semantic / vector search path (if pgvector is available and query embedding provided)
	if HasPgVector && len(embeddingVector) > 0 && semanticQuery != "" {
		query := fmt.Sprintf(`
			SELECT l.id, l.tenant_id, t.name AS tenant_name, l.model, l.provider,
			       l.prompt_tokens, l.completion_tokens, l.latency_ms, l.status_code,
			       l.error_message, l.created_at::text,
			       v.prompt_text, v.response_text,
			       (1.0 - (v.prompt_embedding <=> $2::vector)) AS similarity
			FROM request_logs l
			LEFT JOIN request_vector_logs v ON l.id = v.log_id
			LEFT JOIN tenants t ON l.tenant_id = t.id
			WHERE %s
			  AND ($3::text IS NULL OR t.api_key = $3 OR t.id::text = $3)
			  AND ($4::text IS NULL OR 
			       ($4 = 'success' AND l.status_code >= 200 AND l.status_code < 400) OR 
			       ($4 = 'error' AND l.status_code >= 400))
			  AND ($5::text IS NULL OR v.prompt_text ILIKE $5 OR v.response_text ILIKE $5 OR l.error_message ILIKE $5)
			  AND v.prompt_embedding IS NOT NULL
			ORDER BY v.prompt_embedding <=> $2::vector ASC
			LIMIT $6 OFFSET $7
		`, modelPatternQuery)

		// Convert []float32 to vector format for postgres pgvector (e.g. "[0.12,0.34,...]")
		var sb strings.Builder
		sb.WriteString("[")
		for i, val := range embeddingVector {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(fmt.Sprintf("%f", val))
		}
		sb.WriteString("]")
		vectorStr := sb.String()

		rows, err := pool.Query(ctx, query, modelArgs[0], vectorStr, tenantIDVal, statusVal, searchVal, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("failed semantic search: %w", err)
		}
		defer rows.Close()

		var results []*LogWithVectorRow
		for rows.Next() {
			r := &LogWithVectorRow{}
			err := rows.Scan(
				&r.ID, &r.TenantID, &r.TenantName, &r.Model, &r.Provider,
				&r.PromptTokens, &r.CompletionTokens, &r.LatencyMs, &r.StatusCode,
				&r.ErrorMessage, &r.CreatedAt,
				&r.PromptText, &r.ResponseText, &r.Similarity,
			)
			if err != nil {
				return nil, err
			}
			results = append(results, r)
		}
		return results, nil
	}

	// 2. Standard / fallback search path
	query := fmt.Sprintf(`
		SELECT l.id, l.tenant_id, t.name AS tenant_name, l.model, l.provider,
		       l.prompt_tokens, l.completion_tokens, l.latency_ms, l.status_code,
		       l.error_message, l.created_at::text,
		       v.prompt_text, v.response_text,
		       0.0 AS similarity
		FROM request_logs l
		LEFT JOIN request_vector_logs v ON l.id = v.log_id
		LEFT JOIN tenants t ON l.tenant_id = t.id
		WHERE %s
		  AND ($2::text IS NULL OR t.api_key = $2 OR t.id::text = $2)
		  AND ($3::text IS NULL OR 
		       ($3 = 'success' AND l.status_code >= 200 AND l.status_code < 400) OR 
		       ($3 = 'error' AND l.status_code >= 400))
		  AND ($4::text IS NULL OR v.prompt_text ILIKE $4 OR v.response_text ILIKE $4 OR l.error_message ILIKE $4)
		ORDER BY l.created_at DESC
		LIMIT $5 OFFSET $6
	`, modelPatternQuery)

	rows, err := pool.Query(ctx, query, modelArgs[0], tenantIDVal, statusVal, searchVal, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed standard log query: %w", err)
	}
	defer rows.Close()

	var results []*LogWithVectorRow
	for rows.Next() {
		r := &LogWithVectorRow{}
		err := rows.Scan(
			&r.ID, &r.TenantID, &r.TenantName, &r.Model, &r.Provider,
			&r.PromptTokens, &r.CompletionTokens, &r.LatencyMs, &r.StatusCode,
			&r.ErrorMessage, &r.CreatedAt,
			&r.PromptText, &r.ResponseText, &r.Similarity,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

// GenerateEmbedding calculates a deterministic 1536-dimensional L2-normalized float32
// vector derived from text for cosine similarity matching.
func GenerateEmbedding(text string) []float32 {
	vec := make([]float32, 1536)
	if text == "" {
		return vec
	}

	runes := []rune(text)
	for i := 0; i < 1536; i++ {
		h := uint32(i)
		for _, r := range runes {
			h = h*31 + uint32(r)
		}
		// Normalize to float range [-1, 1]
		vec[i] = float32(int32(h)) / 2147483648.0
	}

	// L2 normalization
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	if sum > 0 {
		stdDev := math.Sqrt(sum)
		for i := range vec {
			vec[i] = float32(float64(vec[i]) / stdDev)
		}
	}

	return vec
}

// ActivateCredentialsBulk updates the status of all credentials associated with the specified model pools.
// It sets is_healthy to true and resets last_error to NULL.
// Returns the count of affected credentials.
func ActivateCredentialsBulk(ctx context.Context, pool *pgxpool.Pool, poolIDs []int) (int64, error) {
	if len(poolIDs) == 0 {
		return 0, nil
	}
	result, err := pool.Exec(ctx, `
		UPDATE credentials 
		SET is_healthy = TRUE, last_error = NULL 
		WHERE pool_id = ANY($1)
	`, poolIDs)
	if err != nil {
		return 0, fmt.Errorf("failed to bulk activate credentials: %w", err)
	}
	return result.RowsAffected(), nil
}



