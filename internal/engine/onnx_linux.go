//go:build linux

package engine

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// Model wraps an ONNX Runtime inference session.
type Model struct {
	session      *ort.AdvancedSession
	name         string
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

// initONNXRuntime initializes the ONNX Runtime shared library.
func initONNXRuntime(libraryPath string) error {
	if libraryPath != "" {
		ort.SetSharedLibraryPath(libraryPath)
	} else {
		// Look for libonnxruntime.so in the current directory as a fallback
		if _, err := os.Stat("./libonnxruntime.so.1.18.1"); err == nil {
			ort.SetSharedLibraryPath("./libonnxruntime.so.1.18.1")
		} else if _, err := os.Stat("./libonnxruntime.so"); err == nil {
			ort.SetSharedLibraryPath("./libonnxruntime.so")
		}
	}
	return ort.InitializeEnvironment()
}

// destroyONNXRuntime cleans up ONNX Runtime resources.
func destroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

// loadModel loads an ONNX model file and creates a real session on Linux.
func loadModel(modelPath string, threads int, optLevel int) (*Model, error) {
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", absPath)
	}

	name := filepath.Base(modelPath)
	slog.Info("Loading ONNX model (Linux Native)", "name", name, "path", absPath)

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	if err := opts.SetIntraOpNumThreads(threads); err != nil {
		return nil, fmt.Errorf("set threads: %w", err)
	}

	modelBytes, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("read model: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(modelBytes)
	if err != nil {
		return nil, fmt.Errorf("get model info: %w", err)
	}

	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, fmt.Errorf("invalid model: no inputs or outputs")
	}

	inShape := inputs[0].Dimensions
	if inShape[0] == -1 {
		inShape[0] = 1
	}
	slog.Info("Model input info", "name", inputs[0].Name, "shape", inShape)

	inTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(inShape...))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outShape := outputs[0].Dimensions
	if outShape[0] == -1 {
		outShape[0] = 1
	}
	slog.Info("Model output info", "name", outputs[0].Name, "shape", outShape)
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(outShape...))
	if err != nil {
		inTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(absPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		[]ort.ArbitraryTensor{inTensor},
		[]ort.ArbitraryTensor{outTensor},
		opts,
	)
	if err != nil {
		inTensor.Destroy()
		outTensor.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Model{
		session:      session,
		name:         name,
		inputTensor:  inTensor,
		outputTensor: outTensor,
	}, nil
}

// RunInference executes the ONNX model by copying data in and out.
func (m *Model) RunInference(inputData []float32) ([]float32, error) {
	if m.session == nil {
		return nil, fmt.Errorf("session is nil")
	}

	inData := m.inputTensor.GetData()
	if len(inputData) > len(inData) {
		return nil, fmt.Errorf("input data too large for tensor: %d > %d", len(inputData), len(inData))
	}

	// Copy input data into the tensor
	copy(inData, inputData)

	// Run execution
	if err := m.session.Run(); err != nil {
		return nil, fmt.Errorf("session run: %w", err)
	}

	// Copy output data out of the tensor
	outData := m.outputTensor.GetData()
	result := make([]float32, len(outData))
	copy(result, outData)

	return result, nil
}

// GetOutputShape returns the dimensions of the model's output tensor.
func (m *Model) GetOutputShape() []int64 {
	if m.outputTensor == nil {
		return nil
	}
	return m.outputTensor.GetShape().GetDimensions()
}

// Close releases the model resources.
func (m *Model) Close() {
	if m.session != nil {
		m.session.Destroy()
	}
	if m.inputTensor != nil {
		m.inputTensor.Destroy()
	}
	if m.outputTensor != nil {
		m.outputTensor.Destroy()
	}
	slog.Info("Model released", "name", m.name)
}
