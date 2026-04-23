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
	"syscall"
	"time"

	"github.com/vesaa/platex/internal/api"
	"github.com/vesaa/platex/internal/config"
	"github.com/vesaa/platex/internal/engine"
	"github.com/vesaa/platex/internal/modeldl"
)

const version = "0.1.0"

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
		"pid", os.Getpid(),
	)

	// Load configuration
	cfg, err := loadConfig(*configPath)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

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
		slog.Info("HTTP server listening", "address", addr)
		fmt.Printf("\n  🚗 JSAI-LPR Server v%s\n", version)
		fmt.Printf("  ├── Listening on: http://%s\n", addr)
		fmt.Printf("  ├── API endpoint: http://%s/api/v1/recognize\n", addr)
		fmt.Printf("  ├── Health check: http://%s/api/v1/health\n", addr)
		fmt.Printf("  └── Workers: %d\n\n", cfg.Engine.Workers)

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
