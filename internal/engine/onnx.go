package engine

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

// Model wraps an ONNX Runtime inference session.
type Model struct {
	name     string
	filePath string
}

// loadModel loads an ONNX model file and validates it exists.
// The actual ONNX Runtime session creation happens in platform-specific files.
func loadModel(modelPath string, threads int, optLevel int) (*Model, error) {
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", absPath)
	}

	name := filepath.Base(modelPath)
	slog.Info("Loading ONNX model", "name", name, "path", absPath)

	return &Model{
		name:     name,
		filePath: absPath,
	}, nil
}

// Close releases the model resources.
func (m *Model) Close() {
	slog.Info("Model released", "name", m.name)
}

// initONNXRuntime initializes the ONNX Runtime shared library.
func initONNXRuntime(libraryPath string) error {
	slog.Info("ONNX Runtime initialization",
		"note", "Will use ONNX Runtime when available on Linux",
	)
	return nil
}

// destroyONNXRuntime cleans up ONNX Runtime resources.
func destroyONNXRuntime() error {
	return nil
}
