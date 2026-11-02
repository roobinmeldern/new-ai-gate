package proxy

import (
	"context"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const (
	defaultProbeInterval = 30 * time.Second
	defaultProbeTimeout  = 150 * time.Millisecond
	defaultWarmInterval  = 15 * time.Second
)

// EdgeProber continuously tracks low-latency Anycast CDN edge IPs for
// targeted upstream providers. It runs two background loops:
//
//  1. DNS probing: resolves each configured host, TCP-probes every
//     resolved IPv4 address, and stores the fastest IP per host.
//  2. Connection pre-warming: periodically sends lightweight HEAD
//     requests through the main transport to keep HTTP/2 idle
//     connections warm in the connection pool.
//
// The hot-path lookup (GetFastestIP) uses an RWMutex read lock —
// sub-nanosecond contention-free.
type EdgeProber struct {
	mu        sync.RWMutex
	fastestIP map[string]string
	hosts     []string
	probeInt  time.Duration
	probeTO   time.Duration
	warmInt   time.Duration
	logger    *zap.Logger
	cancel    context.CancelFunc
	wg        sync.WaitGroup
}

// NewEdgeProber creates an edge prober for the given comma-separated host list.
// Hosts should be bare hostnames without scheme or port (e.g. "api.cloudflare.com").
func NewEdgeProber(hostCSV string, logger *zap.Logger) *EdgeProber {
	hosts := parseHostList(hostCSV)
	return &EdgeProber{
		fastestIP: make(map[string]string),
		hosts:     hosts,
		probeInt:  defaultProbeInterval,
		probeTO:   defaultProbeTimeout,
		warmInt:   defaultWarmInterval,
		logger:    logger,
	}
}

// Start launches the DNS probing background loop.
func (ep *EdgeProber) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	ep.cancel = cancel
	ep.wg.Add(1)
	go ep.probeLoop(ctx)
}

// StartPreWarming launches the connection pre-warming loop using the main
// HTTP client. This keeps idle HTTP/2 connections alive in the transport's
// connection pool so hot-path requests never pay TCP+TLS setup cost.
func (ep *EdgeProber) StartPreWarming(client *http.Client) {
	ctx, cancel := context.WithCancel(context.Background())
	// Chain with the probe loop's cancel so Stop() kills both
	prevCancel := ep.cancel
	ep.cancel = func() {
		prevCancel()
		cancel()
	}
	ep.wg.Add(1)
	go ep.warmLoop(ctx, client)
}

// Stop gracefully shuts down all background goroutines.
func (ep *EdgeProber) Stop() {
	if ep.cancel != nil {
		ep.cancel()
	}
	ep.wg.Wait()
}

// GetFastestIP returns the historically lowest-latency IP for a host,
// or an empty string if no probe data is available yet.
func (ep *EdgeProber) GetFastestIP(host string) string {
	ep.mu.RLock()
	ip := ep.fastestIP[host]
	ep.mu.RUnlock()
	return ip
}

// --- Internal: probing ---

func (ep *EdgeProber) probeLoop(ctx context.Context) {
	defer ep.wg.Done()

	ep.probeAll()

	ticker := time.NewTicker(ep.probeInt)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ep.probeAll()
		case <-ctx.Done():
			return
		}
	}
}

func (ep *EdgeProber) probeAll() {
	for _, host := range ep.hosts {
		ep.probeHost(host)
	}
}

func (ep *EdgeProber) probeHost(host string) {
	ips, err := net.LookupIP(host)
	if err != nil || len(ips) == 0 {
		return
	}

	var bestIP string
	var bestRTT time.Duration = 999 * time.Millisecond

	for _, ip := range ips {
		ipv4 := ip.To4()
		if ipv4 == nil {
			continue
		}
		ipStr := ipv4.String()
		rtt, err := probeIPLatency(ipStr, "443", ep.probeTO)
		if err == nil && rtt < bestRTT {
			bestRTT = rtt
			bestIP = ipStr
		}
	}

	if bestIP == "" {
		return
	}

	ep.mu.Lock()
	old := ep.fastestIP[host]
	ep.fastestIP[host] = bestIP
	ep.mu.Unlock()

	if old != bestIP {
		ep.logger.Info("edge prober: selected fastest IP",
			zap.String("host", host),
			zap.String("ip", bestIP),
			zap.Duration("rtt", bestRTT),
			zap.String("previous_ip", old),
		)
	}
}

func probeIPLatency(ip, port string, timeout time.Duration) (time.Duration, error) {
	start := time.Now()
	dialer := &net.Dialer{Timeout: timeout}
	conn, err := dialer.Dial("tcp", net.JoinHostPort(ip, port))
	if err != nil {
		return 0, err
	}
	_ = conn.Close()
	return time.Since(start), nil
}

// --- Internal: pre-warming ---

func (ep *EdgeProber) warmLoop(ctx context.Context, client *http.Client) {
	defer ep.wg.Done()

	ticker := time.NewTicker(ep.warmInt)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ep.warmAll(client)
		case <-ctx.Done():
			return
		}
	}
}

func (ep *EdgeProber) warmAll(client *http.Client) {
	for _, host := range ep.hosts {
		go ep.warmHost(client, host)
	}
}

func (ep *EdgeProber) warmHost(client *http.Client, host string) {
	url := "https://" + host + "/"
	req, err := http.NewRequest(http.MethodHead, url, nil)
	if err != nil {
		return
	}
	req.Header.Set("User-Agent", "CleverAIGate/EdgeWarmer")
	resp, err := client.Do(req)
	if err != nil {
		ep.logger.Debug("edge warmer: connection failed",
			zap.String("host", host),
			zap.Error(err),
		)
		return
	}
	resp.Body.Close()
}

// --- Helpers ---

func parseHostList(csv string) []string {
	if csv == "" {
		return nil
	}
	var hosts []string
	for _, h := range strings.Split(csv, ",") {
		h = strings.TrimSpace(h)
		if h != "" {
			hosts = append(hosts, h)
		}
	}
	return hosts
}
