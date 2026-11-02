package proxy

import (
	"regexp"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
)

// KnownProviderPrefixes lists standard routing namespaces used in the gateway.
var KnownProviderPrefixes = []string{
	"routerai/",
	"true-sota/",
	"tokenrouter/",
	"cleanapis/",
	"nvidia/",
	"openrouter/",
	"ollama/",
	"gemini/",
	"1min/",
	"cloudflare/",
	"sarvam/",
	"puter/",
	"agentrouter/",
	"zenmux/",
	"jiekou/",
	"novita/",
	"together/",
	"groq/",
	"cerebras/",
	"sambanova/",
	"siliconflow/",
	"fireworks/",
	"mistral/",
	"perplexity/",
	"ofox/",
	"aplikasinala/",
}

// KnownPublisherPrefixes lists common model vendor/organization namespaces.
var KnownPublisherPrefixes = []string{
	"@cf/meta/",
	"@cf/mistral/",
	"@cf/qwen/",
	"@cf/google/",
	"@cf/deepseek-ai/",
	"@cf/baai/",
	"@cf/defog/",
	"meta-llama/",
	"meta/",
	"deepseek-ai/",
	"deepseek/",
	"openai/",
	"anthropic/",
	"google/",
	"mistralai/",
	"mistral/",
	"qwen/",
	"01-ai/",
	"baichuan-inc/",
	"z-ai/",
	"thudm/",
	"bigcode/",
	"databricks/",
	"microsoft/",
	"cohere/",
}

var versionDotRegex = regexp.MustCompile(`(\d+)\.(\d+)`)

// StripProviderPrefix removes any known or custom leading provider routing namespace.
func StripProviderPrefix(pattern string) (cleanModel, provider string) {
	for _, p := range KnownProviderPrefixes {
		if strings.HasPrefix(pattern, p) {
			return pattern[len(p):], strings.TrimSuffix(p, "/")
		}
	}
	// Also handle any arbitrary custom single-segment prefix e.g. "myprovider/foo"
	if idx := strings.IndexByte(pattern, '/'); idx != -1 {
		prefix := pattern[:idx]
		isPublisher := false
		for _, pub := range KnownPublisherPrefixes {
			if strings.TrimSuffix(pub, "/") == prefix {
				isPublisher = true
				break
			}
		}
		if !isPublisher {
			return pattern[idx+1:], prefix
		}
	}
	return pattern, ""
}

// FormatUpstreamModelForPattern returns the clean model ID expected in the upstream
// request body when routing to the given pool pattern.
func FormatUpstreamModelForPattern(pattern string) string {
	clean, _ := StripProviderPrefix(pattern)
	return clean
}

// ExtractCoreModelAndVersion extracts the canonical model ID and version from a model
// pattern by stripping routing prefixes, vendor namespaces, and normalizing version separators.
//
// Examples:
//
//	"nvidia/meta/llama-3.3-70b-instruct"          -> "llama-3-3-70b-instruct"
//	"openrouter/meta-llama/llama-3.3-70b-instruct" -> "llama-3-3-70b-instruct"
//	"cloudflare/@cf/meta/llama-3.3-70b-instruct"   -> "llama-3-3-70b-instruct"
//	"ollama/llama3.3:70b-instruct"                 -> "llama-3-3-70b-instruct"
//	"puter/gpt-4o"                                 -> "gpt-4o"
//	"openrouter/openai/gpt-4o"                     -> "gpt-4o"
//	"deepseek/deepseek-r1:free"                    -> "deepseek-r1"
//	"nvidia/deepseek-ai/deepseek-r1"               -> "deepseek-r1"
//	"anthropic/claude-3-5-sonnet-20241022"         -> "claude-3-5-sonnet-20241022"
//	"claude-3.5-sonnet"                            -> "claude-3-5-sonnet"
func ExtractCoreModelAndVersion(pattern string) string {
	raw := strings.ToLower(strings.TrimSpace(pattern))
	if raw == "" {
		return ""
	}

	// 1. Strip common suffixes like :free, :latest, :beta, :preview
	raw = strings.TrimSuffix(raw, ":free")
	raw = strings.TrimSuffix(raw, ":latest")
	raw = strings.TrimSuffix(raw, ":beta")
	raw = strings.TrimSuffix(raw, ":preview")

	// 2. Strip leading routing provider prefix (e.g. nvidia/, openrouter/, puter/)
	for _, p := range KnownProviderPrefixes {
		if strings.HasPrefix(raw, p) {
			raw = strings.TrimPrefix(raw, p)
			break
		}
	}

	// 3. Strip publisher / organization prefixes (e.g. meta-llama/, deepseek-ai/, @cf/meta/)
	for _, pub := range KnownPublisherPrefixes {
		if strings.HasPrefix(raw, pub) {
			raw = strings.TrimPrefix(raw, pub)
			break
		}
	}

	// 4. If any slash remains (e.g. custom provider "myprov/meta/llama-3.3-70b"),
	// take the final segment if it appears to be the model name
	if idx := strings.LastIndexByte(raw, '/'); idx != -1 {
		raw = raw[idx+1:]
	}

	// 5. Normalize version dots to hyphens: e.g. "3.5" -> "3-5", "3.3" -> "3-3"
	// This ensures "claude-3.5-sonnet" and "claude-3-5-sonnet" match identically.
	raw = versionDotRegex.ReplaceAllString(raw, "${1}-${2}")

	// 6. Normalize colon separators (common in Ollama tags e.g. "llama3-3:70b" -> "llama3-3-70b")
	raw = strings.ReplaceAll(raw, ":", "-")

	// 7. Normalize "llama3-3" with "llama-3-3"
	if strings.HasPrefix(raw, "llama3-") {
		raw = "llama-3-" + strings.TrimPrefix(raw, "llama3-")
	} else if strings.HasPrefix(raw, "llama2-") {
		raw = "llama-2-" + strings.TrimPrefix(raw, "llama2-")
	}

	return strings.TrimSpace(raw)
}

// IsSameModelAndVersion returns true if patternA and patternB refer to the exact same
// model architecture, parameter size, and version.
func IsSameModelAndVersion(patternA, patternB string) bool {
	coreA := ExtractCoreModelAndVersion(patternA)
	coreB := ExtractCoreModelAndVersion(patternB)
	return coreA != "" && coreA == coreB
}

// ProviderReliabilityScore returns a baseline reputation score for an upstream provider.
// Higher scores are prioritized for cross-provider fallbacks.
func ProviderReliabilityScore(provider string) int {
	switch strings.ToLower(provider) {
	case "anthropic", "openai":
		return 120
	case "deepseek", "gemini", "groq", "cerebras":
		return 110
	case "together", "fireworks", "sambanova":
		return 105
	case "routerai":
		return 100
	case "true-sota":
		return 95
	case "tokenrouter":
		return 90
	case "zenmux":
		return 85
	case "cleanapis":
		return 80
	case "novita", "siliconflow", "cloudflare", "nvidia":
		return 75
	case "openrouter":
		return 70
	case "pollinations":
		return -30
	case "puter", "jiekou", "ofox", "aplikasinala":
		return -50
	default:
		return 50
	}
}

// CalculatePoolScore evaluates a candidate pool for fallback suitability based on
// provider reliability, healthy key availability, and failure history.
func CalculatePoolScore(pool *credentials.BalancedChannelPool) int {
	if pool == nil {
		return -1000
	}

	// 1. Identify provider
	provider := ""
	if len(pool.Credentials) > 0 && pool.Credentials[0] != nil {
		provider = pool.Credentials[0].Provider
	}
	if provider == "" {
		_, provider = StripProviderPrefix(pool.ModelPattern)
	}

	score := ProviderReliabilityScore(provider)

	// 2. Health availability ratio bonus (up to +20 points)
	total := int(pool.TotalCount)
	if total > 0 {
		healthy := pool.HealthyCount()
		ratio := float64(healthy) / float64(total)
		score += int(ratio * 20)
	}

	// 3. Subtract penalty for accumulated consecutive failures
	for _, cred := range pool.Credentials {
		if cred != nil {
			failures := atomic.LoadUint32(&cred.ConsecutiveFailures)
			score -= int(failures) * 10
		}
	}

	return score
}

// FindExactModelFallbacks searches all loaded routing pools for other pools that host
// the exact same model and version as currentModel, excluding already tried pools and
// pools with zero healthy credentials. Candidates are prioritized and sorted deterministically
// by provider reputation, healthy key ratio, and failure history.
func FindExactModelFallbacks(currentModel string, triedPools map[string]bool, allPools map[string]*credentials.BalancedChannelPool) []*credentials.BalancedChannelPool {
	targetCore := ExtractCoreModelAndVersion(currentModel)
	if targetCore == "" || len(allPools) == 0 {
		return nil
	}

	var candidates []*credentials.BalancedChannelPool
	seen := make(map[string]bool)

	for candPattern, candPool := range allPools {
		if candPool == nil {
			continue
		}
		// Never retry the same pattern or an already tried pool
		if candPattern == currentModel || candPool.ModelPattern == currentModel ||
			triedPools[candPattern] || triedPools[candPool.ModelPattern] ||
			seen[candPool.ModelPattern] {
			continue
		}

		// Must be the EXACT same model ID and version
		candCore := ExtractCoreModelAndVersion(candPattern)
		if candCore != targetCore {
			continue
		}

		// Must have at least one healthy credential past cooldown
		if candPool.HealthyCount() == 0 {
			continue
		}

		seen[candPool.ModelPattern] = true
		candidates = append(candidates, candPool)
	}

	// Sort candidates by score descending, then by ModelPattern ascending for determinism
	sort.SliceStable(candidates, func(i, j int) bool {
		scoreI := CalculatePoolScore(candidates[i])
		scoreJ := CalculatePoolScore(candidates[j])
		if scoreI != scoreJ {
			return scoreI > scoreJ
		}
		return candidates[i].ModelPattern < candidates[j].ModelPattern
	})

	return candidates
}
