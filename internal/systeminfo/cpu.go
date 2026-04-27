package systeminfo

import (
	"os"
	"runtime"
	"strings"
)

func CPUFeatureFlags() map[string]bool {
	flags := map[string]bool{
		"avx":      false,
		"avx2":     false,
		"avx512f":  false,
		"avx512bw": false,
	}
	if runtime.GOOS != "linux" {
		return flags
	}
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return flags
	}
	text := strings.ToLower(string(data))
	flags["avx"] = strings.Contains(text, " avx ")
	flags["avx2"] = strings.Contains(text, " avx2 ")
	flags["avx512f"] = strings.Contains(text, " avx512f ")
	flags["avx512bw"] = strings.Contains(text, " avx512bw ")
	return flags
}
