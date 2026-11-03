package proxy

import (
	"net/http"

	"go.uber.org/zap"
)

// EdgeProber represents the edge prober service.
// This feature is disabled in favor of native DNS resolution and standard OS networking behavior.
// All methods are safe no-ops to preserve interface compatibility without running background
// DNS probing goroutines or emitting edge prober log entries to the console.
type EdgeProber struct{}

// NewEdgeProber creates an edge prober (disabled; native behavior active).
func NewEdgeProber(hostCSV string, logger *zap.Logger) *EdgeProber {
	return &EdgeProber{}
}

// Start is a no-op since edge probing is disabled.
func (ep *EdgeProber) Start() {}

// StartPreWarming is a no-op since edge probing and pre-warming are disabled.
func (ep *EdgeProber) StartPreWarming(client *http.Client) {}

// Stop is a no-op since edge probing is disabled.
func (ep *EdgeProber) Stop() {}

// GetFastestIP always returns an empty string to allow native DNS resolution to handle all requests.
func (ep *EdgeProber) GetFastestIP(host string) string {
	return ""
}
