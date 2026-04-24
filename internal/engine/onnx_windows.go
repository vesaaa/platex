//go:build !linux

package engine

import (
	"fmt"
	"log/slog"
	"path/filepath"
)

// Model wraps an ONNX Runtime inference session.
type Model struct {
	name        string
	filePath    string
	outputShape []int64
}

// loadModel loads an ONNX model file (Mock for Windows)
func loadModel(modelPath string, threads int, optLevel int) (*Model, error) {
	name := filepath.Base(modelPath)
	slog.Info("Loading ONNX model (Mock for Windows)", "name", name, "path", modelPath)

	return &Model{
		name:     name,
		filePath: modelPath,
	}, nil
}

// RunInference executes the ONNX model (Mock for Windows).
func (m *Model) RunInference(inputData []float32) ([]float32, error) {
	slog.Warn("Running mock inference on Windows - please deploy to Linux for real ONNX acceleration")
	return nil, fmt.Errorf("real model inference is only supported on Linux")
}

// GetOutputShape returns the dimensions of the model's output tensor (Mock).
func (m *Model) GetOutputShape() []int64 {
	return []int64{1, 20, 78}
}

// Close releases the model resources.
func (m *Model) Close() {
	slog.Info("Model released (Mock)", "name", m.name)
}

// initONNXRuntime initializes the ONNX Runtime shared library.
func initONNXRuntime(libraryPath string) error {
	slog.Info("ONNX Runtime initialization",
		"note", "Using mock ONNX Runtime on Windows. Real ONNX will be used on Linux.",
	)
	return nil
}

// destroyONNXRuntime cleans up ONNX Runtime resources.
func destroyONNXRuntime() error {
	return nil
}
