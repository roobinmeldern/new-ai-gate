package credentials

import (
	"context"
	"testing"
)

func TestDiscoverAndRegisterOneMinAIModelsBatch_NoValidKeys(t *testing.T) {
	// Blank/whitespace-only input must be rejected before any DB, vault or
	// network access (same guard as the custom and Puter batch flows).
	totalKeys, successCount, failedCount, totalModels, results, allDiscovered, err := DiscoverAndRegisterOneMinAIModelsBatch(
		context.Background(), nil, nil, []string{"", "   ", ""}, 1,
	)
	if err == nil {
		t.Fatalf("expected error for empty key list, got nil")
	}
	if totalKeys != 0 || successCount != 0 || failedCount != 0 || totalModels != 0 {
		t.Errorf("expected all counters to be zero, got total=%d success=%d failed=%d models=%d", totalKeys, successCount, failedCount, totalModels)
	}
	if results != nil || allDiscovered != nil {
		t.Errorf("expected nil results and discovered list, got %+v / %+v", results, allDiscovered)
	}
}

func TestDiscoverAndRegisterOneMinAIModelsBatch_DedupesAndTrimsKeys(t *testing.T) {
	// Deduplication/trimming happens before any network call: with only
	// duplicate/whitespace variants of one key, the first per-key discovery
	// call would attempt an outbound request and fail against nil deps — but
	// the batch itself must still report exactly one processed key.
	totalKeys, _, failedCount, _, results, _, err := DiscoverAndRegisterOneMinAIModelsBatch(
		context.Background(), nil, nil, []string{"  key-a  ", "key-a", ""}, 1,
	)
	if err != nil {
		t.Fatalf("batch coordination itself must not fail, got %v", err)
	}
	if totalKeys != 1 {
		t.Errorf("expected 1 unique key, got %d", totalKeys)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result entry, got %d", len(results))
	}
	if results[0].Index != 1 || results[0].KeyMasked != MaskAPIKey("key-a") {
		t.Errorf("unexpected result metadata: %+v", results[0])
	}
	if failedCount != 1 {
		t.Errorf("expected the single discovery attempt (nil deps) to be recorded as failed, got failed=%d", failedCount)
	}
	if results[0].Success {
		t.Errorf("expected failure for the discovery attempt with nil deps")
	}
	if results[0].Error == "" {
		t.Errorf("expected a non-empty error message on the failed key result")
	}
}
