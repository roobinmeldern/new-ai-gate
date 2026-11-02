//go:build windows

package proxy

import "syscall"

const (
	// Defined here because syscall.TCP_NODELAY is not available on Windows.
	ipprotoTCP = 6
	tcpNoDelay = 1
)

// setTCPNoDelay disables Nagle's algorithm on the given socket file descriptor.
// On Windows, syscall.SetsockoptInt expects the fd as a syscall.Handle (uintptr).
func setTCPNoDelay(fd uintptr) error {
	return syscall.SetsockoptInt(syscall.Handle(fd), ipprotoTCP, tcpNoDelay, 1)
}

// setTCPQuickAck is a Linux-only optimization. On Windows, the TCP stack
// does not support TCP_QUICKACK, so this is a no-op.
func setTCPQuickAck(fd uintptr) error {
	return nil
}

// setTCPFastOpenConnect is a Linux-only optimization (Linux 4.11+).
// On Windows, TCP Fast Open is managed differently and this is a no-op.
func setTCPFastOpenConnect(fd uintptr) error {
	return nil
}
