//go:build !windows

package proxy

import "syscall"

const (
	// Linux kernel constant for TCP_QUICKACK. Not defined in Go's syscall
	// package on all Unix variants, so we use the raw value.
	// On systems that don't support it (e.g. macOS), SetsockoptInt returns
	// an error that the caller silently ignores.
	tcpQuickAck        = 12
	tcpFastOpenConnect = 30
)

// setTCPNoDelay disables Nagle's algorithm on the given socket file descriptor.
// On Unix systems, syscall.SetsockoptInt expects the fd as an int.
func setTCPNoDelay(fd uintptr) error {
	return syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_NODELAY, 1)
}

// setTCPQuickAck enables immediate ACK sending instead of the kernel's
// delayed-ACK heuristic. This prevents streaming acknowledgment stalling
// where the kernel holds ACKs for 40-200ms, causing the sender (upstream
// provider) to wait before sending the next batch of SSE tokens.
//
// Linux-only (kernel 2.4+). Silently ignored on macOS/BSD.
func setTCPQuickAck(fd uintptr) error {
	return syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, tcpQuickAck, 1)
}

// setTCPFastOpenConnect enables TCP Fast Open in client mode (Linux 4.11+).
// When an idle connection is dropped from the pool and a new one is
// established, the initial SYN packet carries the HTTP request payload,
// saving one full RTT on reconnect.
//
// Linux-only. Silently ignored on macOS/BSD.
func setTCPFastOpenConnect(fd uintptr) error {
	return syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, tcpFastOpenConnect, 1)
}
