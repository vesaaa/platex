//go:build linux

package engine

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

type modelRunner struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

// Model wraps an ONNX Runtime inference session.
type Model struct {
	name        string
	outputShape []int64
	runners     []*modelRunner
	pool        chan *modelRunner
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

func resolvePoolSize() int {
	// Optional runtime override for throughput tuning.
	if raw := os.Getenv("PLATEX_MODEL_POOL_SIZE"); raw != "" {
		trimmed := strings.TrimSpace(raw)
		if v, err := strconv.Atoi(trimmed); err == nil && v > 0 {
			if v > 16 {
				slog.Warn("PLATEX_MODEL_POOL_SIZE is capped",
					"raw", raw,
					"effective", 16,
				)
				return 16
			}
			slog.Info("Using env model pool size",
				"raw", raw,
				"effective", v,
			)
			return v
		}
		slog.Warn("Invalid PLATEX_MODEL_POOL_SIZE, fallback to auto",
			"raw", raw,
		)
	}
	cpu := runtime.NumCPU()
	auto := cpu / 4
	if auto < 2 {
		auto = 2
	}
	if auto > 6 {
		auto = 6
	}
	slog.Info("Using auto model pool size",
		"cpu", cpu,
		"effective", auto,
	)
	return auto
}

func createRunner(absPath string, opts *ort.SessionOptions, inputName, outputName string, inShape, outShape []int64) (*modelRunner, error) {
	inTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(inShape...))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(outShape...))
	if err != nil {
		inTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	session, err := ort.NewAdvancedSession(absPath,
		[]string{inputName},
		[]string{outputName},
		[]ort.ArbitraryTensor{inTensor},
		[]ort.ArbitraryTensor{outTensor},
		opts,
	)
	if err != nil {
		inTensor.Destroy()
		outTensor.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &modelRunner{
		session:      session,
		inputTensor:  inTensor,
		outputTensor: outTensor,
	}, nil
}

// loadModel loads an ONNX model file and creates real sessions on Linux.
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

	outShape := outputs[0].Dimensions
	if outShape[0] == -1 {
		outShape[0] = 1
	}
	slog.Info("Model output info", "name", outputs[0].Name, "shape", outShape)

	poolSize := resolvePoolSize()
	runners := make([]*modelRunner, 0, poolSize)
	pool := make(chan *modelRunner, poolSize)
	for i := 0; i < poolSize; i++ {
		runner, err := createRunner(absPath, opts, inputs[0].Name, outputs[0].Name, inShape, outShape)
		if err != nil {
			for _, r := range runners {
				r.session.Destroy()
				r.inputTensor.Destroy()
				r.outputTensor.Destroy()
			}
			return nil, err
		}
		runners = append(runners, runner)
		pool <- runner
	}
	slog.Info("Model session pool initialized", "name", name, "pool_size", poolSize)

	return &Model{
		name:        name,
		outputShape: outShape,
		runners:     runners,
		pool:        pool,
	}, nil
}

// RunInference executes the ONNX model by copying data in and out.
func (m *Model) RunInference(inputData []float32) ([]float32, error) {
	if len(m.runners) == 0 || m.pool == nil {
		return nil, fmt.Errorf("session pool is empty")
	}
	runner := <-m.pool
	defer func() { m.pool <- runner }()

	inData := runner.inputTensor.GetData()
	if len(inputData) > len(inData) {
		return nil, fmt.Errorf("input data too large for tensor: %d > %d", len(inputData), len(inData))
	}

	// Copy input data into the tensor
	copy(inData, inputData)

	// Run execution
	if err := runner.session.Run(); err != nil {
		return nil, fmt.Errorf("session run: %w", err)
	}

	// Copy output data out of the tensor
	outData := runner.outputTensor.GetData()
	result := make([]float32, len(outData))
	copy(result, outData)

	return result, nil
}

// GetOutputShape returns the dimensions of the model's output tensor.
func (m *Model) GetOutputShape() []int64 {
	return m.outputShape
}

// Close releases the model resources.
func (m *Model) Close() {
	for _, runner := range m.runners {
		if runner.session != nil {
			runner.session.Destroy()
		}
		if runner.inputTensor != nil {
			runner.inputTensor.Destroy()
		}
		if runner.outputTensor != nil {
			runner.outputTensor.Destroy()
		}
	}
	slog.Info("Model released", "name", m.name)
}
