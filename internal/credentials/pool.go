package credentials

import (
	"sync/atomic"
	"time"
)

// RuntimeCredential represents a single upstream provider API key with
// its connection details and lock-free health state.
type RuntimeCredential struct {
	ID                  int
	Provider            string // "openai", "anthropic", "gemini", "deepseek"
	APIKey              string // Decrypted API key (in-memory only)
	BaseURL             string // Provider base URL
	Weight              int    // Distribution weight
	CooldownUntil       int64  // Unix nanosecond; managed via atomic ops
	Prefix              string // Optional routing prefix to strip before forwarding
	ConsecutiveFailures uint32 // Atomic counter for sequential request errors
}

// IsAvailable checks if this credential is past its cooldown period.
// Uses atomic load — no lock contention.
func (rc *RuntimeCredential) IsAvailable() bool {
	return atomic.LoadInt64(&rc.CooldownUntil) <= time.Now().UnixNano()
}

// BalancedChannelPool distributes requests across multiple provider credentials
// using a lock-free atomic round-robin algorithm with nanosecond cooldown barriers.
//
// Design rationale:
//   - No sync.Mutex: atomic.AddUint64 for cursor progression is contention-free
//   - No sync.RWMutex: atomic.LoadInt64 for cooldown checks avoids reader locks
//   - Weighted distribution: credentials with weight > 1 appear multiple times in the ring
//   - Cooldown isolation: a single 429/500 response penalizes only the affected credential
type BalancedChannelPool struct {
	ModelPattern string
	Strategy     string // "round-robin", "weighted-round-robin", "random"
	Credentials  []*RuntimeCredential
	Cursor       uint64
	TotalCount   uint64
	FallbackPool *BalancedChannelPool // Optional fallback for cascading failover
}

// NewBalancedPool creates a pool from a slice of credentials.
// Credentials with weight > 1 are duplicated in the ring for proportional distribution.
func NewBalancedPool(model, strategy string, creds []*RuntimeCredential, fallback *BalancedChannelPool) *BalancedChannelPool {
	// Build weighted ring: a credential with weight=3 appears 3 times
	var ring []*RuntimeCredential
	for _, c := range creds {
		w := c.Weight
		if w < 1 {
			w = 1
		}
		for i := 0; i < w; i++ {
			ring = append(ring, c)
		}
	}

	return &BalancedChannelPool{
		ModelPattern: model,
		Strategy:     strategy,
		Credentials:  ring,
		TotalCount:   uint64(len(ring)),
		FallbackPool: fallback,
	}
}

// AcquireResult holds the result of a credential acquisition.
type AcquireResult struct {
	Credential *RuntimeCredential
	Index      int
	FromPool   *BalancedChannelPool // Which pool provided this credential
}

// AcquireActiveToken selects the next available credential using lock-free
// atomic round-robin with cooldown bypass.
//
// Algorithm:
//  1. Atomically increment cursor to get next index
//  2. Check cooldown via atomic load (zero contention)
//  3. If cooled down, skip to next candidate
//  4. After full ring scan, cascade to fallback pool if available
//  5. Returns nil only when ALL credentials across ALL pools are cooled down
func (p *BalancedChannelPool) AcquireActiveToken() *AcquireResult {
	if p == nil || p.TotalCount == 0 {
		return nil
	}

	now := time.Now().UnixNano()
	limit := p.TotalCount

	baseIdx := atomic.AddUint64(&p.Cursor, 1)

	for i := uint64(0); i < limit; i++ {
		idx := (baseIdx + i) % limit
		cand := p.Credentials[idx]

		// Atomic cooldown check — no mutex, no read lock
		if atomic.LoadInt64(&cand.CooldownUntil) <= now {
			return &AcquireResult{
				Credential: cand,
				Index:      int(idx),
				FromPool:   p,
			}
		}
	}

	// All credentials in this pool are cooled down — cascade to fallback
	if p.FallbackPool != nil {
		return p.FallbackPool.AcquireActiveToken()
	}

	return nil // All pools exhausted
}

// AcquireBatchActiveTokens acquires up to maxCount distinct active credentials
// from the pool that are past cooldown and not in excludedIndices.
// It only inspects the current pool (does NOT cascade to FallbackPool) so that the
// caller can evaluate all keys of the current provider before switching.
func (p *BalancedChannelPool) AcquireBatchActiveTokens(maxCount int, excludedIndices map[int]bool) []*AcquireResult {
	if p == nil || p.TotalCount == 0 || maxCount <= 0 {
		return nil
	}

	now := time.Now().UnixNano()
	limit := p.TotalCount
	baseIdx := atomic.AddUint64(&p.Cursor, uint64(maxCount))

	var results []*AcquireResult
	seenIndices := make(map[int]bool)
	seenCredIDs := make(map[int]bool)

	for i := uint64(0); i < limit && len(results) < maxCount; i++ {
		idx := int((baseIdx + i) % limit)
		if excludedIndices != nil && excludedIndices[idx] {
			continue
		}
		if seenIndices[idx] {
			continue
		}
		cand := p.Credentials[idx]
		if cand == nil || seenCredIDs[cand.ID] {
			continue
		}
		if atomic.LoadInt64(&cand.CooldownUntil) <= now {
			seenIndices[idx] = true
			seenCredIDs[cand.ID] = true
			results = append(results, &AcquireResult{
				Credential: cand,
				Index:      idx,
				FromPool:   p,
			})
		}
	}

	return results
}

// PenalizeToken marks a credential as temporarily unavailable.
// Called on 429 (rate limit), 500 (server error), or 503 (overloaded) responses.
// Uses atomic store — no locks, no contention with concurrent readers.
func (p *BalancedChannelPool) PenalizeToken(index int, duration time.Duration) {
	if index < 0 || index >= int(p.TotalCount) {
		return
	}
	cooldownTime := time.Now().Add(duration).UnixNano()
	atomic.StoreInt64(&p.Credentials[index].CooldownUntil, cooldownTime)
}

// PenalizeByProvider marks all credentials in the pool matching the given provider
// as temporarily unavailable. Used by the circuit breaker to fast-fail an entire provider
// when consecutive credentials suffer quota depletion, billing suspension, or auth errors.
func (p *BalancedChannelPool) PenalizeByProvider(provider string, duration time.Duration) {
	if p == nil {
		return
	}
	cooldownTime := time.Now().Add(duration).UnixNano()
	for _, cred := range p.Credentials {
		if cred.Provider == provider {
			atomic.StoreInt64(&cred.CooldownUntil, cooldownTime)
		}
	}
}

// PenalizeByCredID marks a credential by ID as unavailable until the given
// Unix nanosecond timestamp. Used by the cluster broadcaster to apply events
// received from peer nodes without needing an index.
func (p *BalancedChannelPool) PenalizeByCredID(credID int, untilNs int64) {
	if p == nil {
		return
	}
	for _, cred := range p.Credentials {
		if cred.ID == credID {
			atomic.StoreInt64(&cred.CooldownUntil, untilNs)
			return
		}
	}
}

// ResetByCredID clears the cooldown for a credential identified by ID.
// Used by the cluster broadcaster to apply reset events from peer nodes.
func (p *BalancedChannelPool) ResetByCredID(credID int) {
	if p == nil {
		return
	}
	for _, cred := range p.Credentials {
		if cred.ID == credID {
			atomic.StoreInt64(&cred.CooldownUntil, 0)
			return
		}
	}
}

// ResetCooldown clears the cooldown on a specific credential.
func (p *BalancedChannelPool) ResetCooldown(index int) {
	if index >= 0 && index < int(p.TotalCount) {
		atomic.StoreInt64(&p.Credentials[index].CooldownUntil, 0)
	}
}

// HealthyCount returns the number of credentials currently past their cooldown.
func (p *BalancedChannelPool) HealthyCount() int {
	if p == nil {
		return 0
	}
	now := time.Now().UnixNano()
	count := 0
	seen := make(map[int]bool) // deduplicate weighted entries
	for _, c := range p.Credentials {
		if !seen[c.ID] && atomic.LoadInt64(&c.CooldownUntil) <= now {
			count++
			seen[c.ID] = true
		}
	}
	return count
}

// AcquireLeastPenalizedToken is a safety valve for single-key or small-pool
// environments. It picks the credential with the lowest CooldownUntil timestamp
// — i.e., the one that will become available soonest — even if it is still
// technically on cooldown. The caller is responsible for sleeping until the
// token is ready (if desired) before using the result.
//
// This prevents a single transient NVIDIA 500/503 from locking the gateway
// out for a full 10–30 seconds when only one key is configured.
func (p *BalancedChannelPool) AcquireLeastPenalizedToken() *AcquireResult {
	if p == nil || p.TotalCount == 0 {
		return nil
	}

	var bestCred *RuntimeCredential
	var bestIdx int
	bestCooldown := int64(1<<63 - 1) // max int64

	for idx, cand := range p.Credentials {
		cd := atomic.LoadInt64(&cand.CooldownUntil)
		if cd < bestCooldown {
			bestCooldown = cd
			bestCred = cand
			bestIdx = idx
		}
	}

	if bestCred != nil {
		return &AcquireResult{
			Credential: bestCred,
			Index:      bestIdx,
			FromPool:   p,
		}
	}

	if p.FallbackPool != nil {
		return p.FallbackPool.AcquireLeastPenalizedToken()
	}

	return nil
}

