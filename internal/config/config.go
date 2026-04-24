// Package config handles application configuration.
package config

import (
	"fmt"
	"os"
	"runtime"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration structure.
type Config struct {
	Server ServerConfig `yaml:"server"`
	Engine EngineConfig `yaml:"engine"`
	Log    LogConfig    `yaml:"logging"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	ReadTimeout    string `yaml:"read_timeout"`
	WriteTimeout   string `yaml:"write_timeout"`
	MaxRequestBody string `yaml:"max_request_body"`
}

// EngineConfig holds inference engine settings.
type EngineConfig struct {
	Mode    string       `yaml:"mode"`    // "crop" or "full"
	Workers int          `yaml:"workers"` // 0 = auto
	Models  ModelsConfig `yaml:"models"`
	ONNX    ONNXConfig   `yaml:"onnx"`
	Rec     RecConfig    `yaml:"recognition"`
	URL     URLConfig    `yaml:"url"`
}

// ModelsConfig holds model file paths.
type ModelsConfig struct {
	Recognizer string `yaml:"recognizer"`
	Detector   string `yaml:"detector"`
	Color      string `yaml:"color"`
}

// ONNXConfig holds ONNX Runtime settings.
type ONNXConfig struct {
	ThreadsPerSession int `yaml:"threads_per_session"`
	OptimizationLevel int `yaml:"optimization_level"`
}

// RecConfig holds recognition parameters.
type RecConfig struct {
	MinConfidence float32 `yaml:"min_confidence"`
	MaxPlates     int     `yaml:"max_plates"`
}

// URLConfig holds settings for URL image input fetching.
type URLConfig struct {
	Enabled             bool   `yaml:"enabled"`
	FetchTimeoutMs      int    `yaml:"fetch_timeout_ms"`
	MaxImageBytes       int64  `yaml:"max_image_bytes"`
	MaxFetchConcurrency int    `yaml:"max_fetch_concurrency"`
	BlockPrivateIP      bool   `yaml:"block_private_ip"`
	AllowedSchemes      []string `yaml:"allowed_schemes"`
}

// LogConfig holds logging settings.
type LogConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
	Output string `yaml:"output"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() *Config {
	workers := runtime.NumCPU() * 3 / 4
	if workers < 1 {
		workers = 1
	}
	return &Config{
		Server: ServerConfig{
			Host:           "0.0.0.0",
			Port:           8080,
			ReadTimeout:    "30s",
			WriteTimeout:   "30s",
			MaxRequestBody: "100MB",
		},
		Engine: EngineConfig{
			Mode:    "crop",
			Workers: workers,
			Models: ModelsConfig{
				Recognizer: "models/plate_rec.onnx",
				Detector:   "models/plate_detect.onnx",
				Color:      "models/plate_color.onnx",
			},
			ONNX: ONNXConfig{
				ThreadsPerSession: 1,
				OptimizationLevel: 3,
			},
			Rec: RecConfig{
				MinConfidence: 0.6,
				MaxPlates:     10,
			},
			URL: URLConfig{
				Enabled:             true,
				FetchTimeoutMs:      1200,
				MaxImageBytes:       5 * 1024 * 1024,
				MaxFetchConcurrency: 16,
				BlockPrivateIP:      true,
				AllowedSchemes:      []string{"http", "https"},
			},
		},
		Log: LogConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}
}

// Load reads configuration from a YAML file.
// Missing fields fall back to default values.
func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Auto-calculate workers if set to 0
	if cfg.Engine.Workers <= 0 {
		cfg.Engine.Workers = runtime.NumCPU() * 3 / 4
		if cfg.Engine.Workers < 1 {
			cfg.Engine.Workers = 1
		}
	}

	return cfg, nil
}
