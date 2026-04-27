// PlateX: High Performance Chinese License Plate Recognition Server
//
// Usage:
//
//	lpr-server -config configs/config.yaml
//	lpr-server -port 8080
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/vesaa/platex/internal/api"
	"github.com/vesaa/platex/internal/config"
	"github.com/vesaa/platex/internal/engine"
	"github.com/vesaa/platex/internal/modeldl"
	"github.com/vesaa/platex/internal/systeminfo"
)

var (
	version   = "dev"
	buildTime = "unknown"
	gitCommit = "unknown"
)

func main() {
	// Command line flags
	configPath := flag.String("config", "", "Path to config file (YAML)")
	port := flag.Int("port", 0, "Override server port")
	host := flag.String("host", "", "Override server host")
	workers := flag.Int("workers", 0, "Override worker count")
	logLevel := flag.String("log-level", "", "Log level: debug, info, warn, error")
	downloadModels := flag.Bool("download", false, "Download required ONNX models and exit")
	flag.Parse()

	// Setup logging
	setupLogging(*logLevel)

	if *downloadModels {
		slog.Info("Starting model download process...")
		// Use the internal/modeldl package to download models to the "models" directory
		if err := modeldl.DownloadModels("models"); err != nil {
			slog.Error("Failed to download models", "error", err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	slog.Info("Starting PlateX Server",
		"version", version,
		"build_time", buildTime,
		"git_commit", gitCommit,
		"pid", os.Getpid(),
	)

	// Load configuration
	cfg, err := loadConfig(*configPath)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}
	// Apply ENV overrides (Docker-friendly). CLI flags still have higher priority.
	applyEnvOverrides(cfg)

	// Apply CLI overrides
	if *port > 0 {
		cfg.Server.Port = *port
	}
	if *host != "" {
		cfg.Server.Host = *host
	}
	if *workers > 0 {
		cfg.Engine.Workers = *workers
	}

	// Initialize engine
	eng, err := engine.New(&cfg.Engine)
	if err != nil {
		slog.Error("Failed to initialize engine", "error", err)
		os.Exit(1)
	}
	defer eng.Close()

	// Create API server
	srv := api.NewServer(eng, version)

	// Create HTTP server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	httpServer := &http.Server{
		Addr:         addr,
		Handler:      srv.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		rc := eng.GetRuntimeConfig()
		features := systeminfo.CPUFeatureFlags()
		slog.Info("HTTP server listening", "address", addr)
		fmt.Printf("\n  🚗 JSAI-LPR Server v%s\n", version)
		fmt.Printf("  ├── Listening on: http://%s\n", addr)
		fmt.Printf("  ├── API endpoint: http://%s/api/v1/recognize\n", addr)
		fmt.Printf("  ├── Health check: http://%s/api/v1/health\n", addr)
		fmt.Printf("  ├── Workers: %v\n", rc["workers"])
		fmt.Printf("  ├── Model Pool Size: %v\n", rc["model_pool_size"])
		fmt.Printf("  ├── URL Fetch Concurrency: %v\n", rc["url_max_fetch_concurrency"])
		fmt.Printf("  ├── ONNX Threads/Session: %v\n", rc["onnx_threads_per_session"])
		fmt.Printf("  ├── CPU: cores=%d gomaxprocs=%d\n", runtime.NumCPU(), runtime.GOMAXPROCS(0))
		fmt.Printf("  └── SIMD: AVX=%t AVX2=%t AVX512F=%t\n\n",
			features["avx"],
			features["avx2"],
			features["avx512f"],
		)

		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("HTTP server error", "error", err)
			os.Exit(1)
		}
	}()

	// Wait for shutdown signal
	sig := <-done
	slog.Info("Received shutdown signal", "signal", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		slog.Error("Server shutdown error", "error", err)
	}

	slog.Info("Server stopped")
}

// loadConfig loads configuration from file or returns defaults.
func loadConfig(path string) (*config.Config, error) {
	if path != "" {
		return config.Load(path)
	}

	// Try default paths
	for _, p := range []string{"configs/config.yaml", "config.yaml"} {
		if _, err := os.Stat(p); err == nil {
			slog.Info("Using config file", "path", p)
			return config.Load(p)
		}
	}

	slog.Info("No config file found, using defaults")
	return config.DefaultConfig(), nil
}

func applyEnvOverrides(cfg *config.Config) {
	if cfg == nil {
		return
	}
	if v, ok := getEnvInt("PLATEX_WORKERS"); ok && v > 0 {
		cfg.Engine.Workers = v
		slog.Info("Applied env override", "key", "PLATEX_WORKERS", "value", v)
	}
	if v, ok := getEnvInt("PLATEX_ONNX_THREADS_PER_SESSION"); ok && v > 0 {
		cfg.Engine.ONNX.ThreadsPerSession = v
		slog.Info("Applied env override", "key", "PLATEX_ONNX_THREADS_PER_SESSION", "value", v)
	}
	if v, ok := getEnvInt("PLATEX_URL_MAX_FETCH_CONCURRENCY"); ok && v > 0 {
		cfg.Engine.URL.MaxFetchConcurrency = v
		slog.Info("Applied env override", "key", "PLATEX_URL_MAX_FETCH_CONCURRENCY", "value", v)
	}
	if v, ok := getEnvInt("PLATEX_SUBMIT_TIMEOUT_MS"); ok && v > 0 {
		cfg.Engine.SubmitTimeoutMs = v
		slog.Info("Applied env override", "key", "PLATEX_SUBMIT_TIMEOUT_MS", "value", v)
	}
	if v, ok := getEnvFloat32("PLATEX_FULL_EARLY_STOP_CONF"); ok && v > 0 {
		cfg.Engine.Rec.FullEarlyStopConf = v
		slog.Info("Applied env override", "key", "PLATEX_FULL_EARLY_STOP_CONF", "value", v)
	}
}

func getEnvInt(key string) (int, bool) {
	raw, ok := os.LookupEnv(key)
	if !ok {
		return 0, false
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		slog.Warn("Ignore invalid integer env", "key", key, "value", raw, "error", err)
		return 0, false
	}
	return v, true
}

func getEnvFloat32(key string) (float32, bool) {
	raw, ok := os.LookupEnv(key)
	if !ok {
		return 0, false
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		slog.Warn("Ignore invalid float env", "key", key, "value", raw, "error", err)
		return 0, false
	}
	return float32(v), true
}

// setupLogging configures the slog default logger.
func setupLogging(level string) {
	var logLevel slog.Level
	switch level {
	case "debug":
		logLevel = slog.LevelDebug
	case "warn":
		logLevel = slog.LevelWarn
	case "error":
		logLevel = slog.LevelError
	default:
		logLevel = slog.LevelInfo
	}

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: logLevel,
	})
	slog.SetDefault(slog.New(handler))
}
