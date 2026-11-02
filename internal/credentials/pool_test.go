package credentials

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAcquireActiveToken_RoundRobin(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 3, APIKey: "key3", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	// Acquire 6 tokens — should cycle through all 3 twice
	seen := make(map[int]int)
	for i := 0; i < 6; i++ {
		result := pool.AcquireActiveToken()
		if result == nil {
			t.Fatalf("expected non-nil result at iteration %d", i)
		}
		seen[result.Credential.ID]++
	}

	for id, count := range seen {
		if count != 2 {
			t.Errorf("credential %d was used %d times, expected 2", id, count)
		}
	}
}

func TestAcquireActiveToken_Cooldown(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	// Penalize first credential
	pool.PenalizeToken(0, 10*time.Second)

	// Next 3 acquisitions should all return credential 2
	for i := 0; i < 3; i++ {
		result := pool.AcquireActiveToken()
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.Credential.ID != 2 {
			t.Errorf("expected credential 2, got %d", result.Credential.ID)
		}
	}
}

func TestAcquireActiveToken_AllCooledDown(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	// Penalize all credentials
	pool.PenalizeToken(0, 10*time.Second)
	pool.PenalizeToken(1, 10*time.Second)

	result := pool.AcquireActiveToken()
	if result != nil {
		t.Error("expected nil result when all credentials are cooled down")
	}
}

func TestAcquireActiveToken_FallbackPool(t *testing.T) {
	primaryCreds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}
	fallbackCreds := []*RuntimeCredential{
		{ID: 2, APIKey: "key2", Provider: "anthropic", BaseURL: "https://api.anthropic.com", Weight: 1},
	}

	fallbackPool := NewBalancedPool("claude", "round-robin", fallbackCreds, nil)
	primaryPool := NewBalancedPool("gpt-4o", "round-robin", primaryCreds, fallbackPool)

	// Penalize primary credential
	primaryPool.PenalizeToken(0, 10*time.Second)

	// Should cascade to fallback
	result := primaryPool.AcquireActiveToken()
	if result == nil {
		t.Fatal("expected result from fallback pool")
	}
	if result.Credential.ID != 2 {
		t.Errorf("expected credential 2 from fallback, got %d", result.Credential.ID)
	}
	if result.Credential.Provider != "anthropic" {
		t.Errorf("expected anthropic provider from fallback, got %s", result.Credential.Provider)
	}
}

func TestAcquireActiveToken_WeightedDistribution(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 3},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "weighted-round-robin", creds, nil)

	// Pool ring should have 4 entries (3 + 1)
	if pool.TotalCount != 4 {
		t.Errorf("expected 4 ring entries, got %d", pool.TotalCount)
	}
}

func TestAcquireActiveToken_ConcurrentAccess(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 3, APIKey: "key3", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	var wg sync.WaitGroup
	var nilCount int64
	iterations := 10000

	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result := pool.AcquireActiveToken()
			if result == nil {
				atomic.AddInt64(&nilCount, 1)
			}
		}()
	}

	wg.Wait()
	if nilCount > 0 {
		t.Errorf("got %d nil results in concurrent test, expected 0", nilCount)
	}
}

func BenchmarkAcquireActiveToken(b *testing.B) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 3, APIKey: "key3", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pool.AcquireActiveToken()
	}
}

func BenchmarkAcquireActiveToken_Parallel(b *testing.B) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 2, APIKey: "key2", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
		{ID: 3, APIKey: "key3", Provider: "openai", BaseURL: "https://api.openai.com", Weight: 1},
	}

	pool := NewBalancedPool("gpt-4o", "round-robin", creds, nil)

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			pool.AcquireActiveToken()
		}
	})
}

func TestAcquireBatchActiveTokens(t *testing.T) {
	creds := []*RuntimeCredential{
		{ID: 1, APIKey: "key1", Provider: "cleanapis", BaseURL: "https://cleanapis.com", Weight: 2},
		{ID: 2, APIKey: "key2", Provider: "cleanapis", BaseURL: "https://cleanapis.com", Weight: 1},
		{ID: 3, APIKey: "key3", Provider: "cleanapis", BaseURL: "https://cleanapis.com", Weight: 1},
		{ID: 4, APIKey: "key4", Provider: "cleanapis", BaseURL: "https://cleanapis.com", Weight: 1},
	}
	fallbackCreds := []*RuntimeCredential{
		{ID: 99, APIKey: "fb-key", Provider: "fallback", BaseURL: "https://fallback.com", Weight: 1},
	}
	fbPool := NewBalancedPool("model", "round-robin", fallbackCreds, nil)
	pool := NewBalancedPool("model", "round-robin", creds, fbPool)

	// Ring length is 2 + 1 + 1 + 1 = 5. Cred ID 1 is duplicated in the ring.
	// Acquire batch of 5: should return distinct physical credentials (IDs: 1, 2, 3, 4 - count 4)
	batch := pool.AcquireBatchActiveTokens(5, nil)
	if len(batch) != 4 {
		t.Fatalf("expected 4 distinct active credentials in batch, got %d", len(batch))
	}
	seenIDs := make(map[int]bool)
	for _, res := range batch {
		if seenIDs[res.Credential.ID] {
			t.Errorf("duplicate physical credential ID %d in batch", res.Credential.ID)
		}
		seenIDs[res.Credential.ID] = true
	}

	// Now penalize ID 2 and exclude index of ID 3
	pool.PenalizeToken(batch[1].Index, 10*time.Second) // penalize one
	excluded := map[int]bool{batch[2].Index: true}

	batch2 := pool.AcquireBatchActiveTokens(5, excluded)
	for _, res := range batch2 {
		if res.Credential.ID == 2 {
			t.Errorf("cooled-down credential 2 was returned in batch")
		}
		if res.Index == batch[2].Index {
			t.Errorf("excluded index %d was returned in batch", res.Index)
		}
	}

	// Ensure it does NOT cascade to fallback pool
	for _, cand := range creds {
		cand.CooldownUntil = time.Now().Add(10 * time.Minute).UnixNano()
	}
	batchEmpty := pool.AcquireBatchActiveTokens(5, nil)
	if len(batchEmpty) != 0 {
		t.Errorf("expected empty batch when all current pool tokens are cooled down, got %d (must not cascade to fallback)", len(batchEmpty))
	}
}

