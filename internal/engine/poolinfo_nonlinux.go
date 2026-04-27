//go:build !linux

package engine

func modelPoolSizeForInfo() int {
	return 1
}
