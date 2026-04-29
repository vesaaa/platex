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
	session       *ort.AdvancedSession
	inputTensor   *ort.Tensor[float32]
	outputTensor  *ort.Tensor[float32]
	extraOutputs  []*ort.Tensor[float32]
	extraOutShape [][]int64
}

// Model wraps an ONNX Runtime inference session.
type Model struct {
	name             string
	outputShape      []int64
	runners          []*modelRunner
	pool             chan *modelRunner
	extraOutputShape [][]int64
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
	return createRunnerMulti(absPath, opts, inputName, []string{outputName}, inShape, [][]int64{outShape})
}

func createRunnerMulti(absPath string, opts *ort.SessionOptions, inputName string, outputNames []string, inShape []int64, outShapes [][]int64) (*modelRunner, error) {
	if len(outputNames) == 0 {
		return nil, fmt.Errorf("no outputs requested")
	}
	if len(outputNames) != len(outShapes) {
		return nil, fmt.Errorf("output name/shape count mismatch: %d vs %d", len(outputNames), len(outShapes))
	}
	inTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(inShape...))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	outTensors := make([]*ort.Tensor[float32], 0, len(outputNames))
	for i, sh := range outShapes {
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(sh...))
		if err != nil {
			inTensor.Destroy()
			for _, prev := range outTensors {
				prev.Destroy()
			}
			return nil, fmt.Errorf("create output tensor %d: %w", i, err)
		}
		outTensors = append(outTensors, t)
	}
	arbInputs := []ort.ArbitraryTensor{inTensor}
	arbOutputs := make([]ort.ArbitraryTensor, 0, len(outTensors))
	for _, t := range outTensors {
		arbOutputs = append(arbOutputs, t)
	}
	session, err := ort.NewAdvancedSession(absPath,
		[]string{inputName},
		outputNames,
		arbInputs,
		arbOutputs,
		opts,
	)
	if err != nil {
		inTensor.Destroy()
		for _, t := range outTensors {
			t.Destroy()
		}
		return nil, fmt.Errorf("create session: %w", err)
	}
	runner := &modelRunner{
		session:     session,
		inputTensor: inTensor,
	}
	if len(outTensors) >= 1 {
		runner.outputTensor = outTensors[0]
	}
	if len(outTensors) > 1 {
		runner.extraOutputs = outTensors[1:]
		runner.extraOutShape = outShapes[1:]
	}
	return runner, nil
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

// loadModelDual loads a model with TWO outputs (e.g. plate-rec-color which emits
// CTC logits and color logits in a single forward). It is otherwise identical to
// loadModel and reuses the same session pool semantics.
func loadModelDual(modelPath string, threads int, optLevel int) (*Model, error) {
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", absPath)
	}

	name := filepath.Base(modelPath)
	slog.Info("Loading dual-output ONNX model (Linux Native)", "name", name, "path", absPath)

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
	if len(inputs) == 0 || len(outputs) < 2 {
		return nil, fmt.Errorf("dual model expects >=2 outputs, got %d", len(outputs))
	}

	inShape := inputs[0].Dimensions
	if inShape[0] == -1 {
		inShape[0] = 1
	}
	slog.Info("Model input info", "name", inputs[0].Name, "shape", inShape)

	outNames := make([]string, 0, len(outputs))
	outShapes := make([][]int64, 0, len(outputs))
	for _, o := range outputs {
		sh := o.Dimensions
		if len(sh) > 0 && sh[0] == -1 {
			sh[0] = 1
		}
		slog.Info("Model output info", "name", o.Name, "shape", sh)
		outNames = append(outNames, o.Name)
		outShapes = append(outShapes, sh)
	}

	poolSize := resolvePoolSize()
	runners := make([]*modelRunner, 0, poolSize)
	pool := make(chan *modelRunner, poolSize)
	for i := 0; i < poolSize; i++ {
		runner, err := createRunnerMulti(absPath, opts, inputs[0].Name, outNames, inShape, outShapes)
		if err != nil {
			for _, r := range runners {
				r.session.Destroy()
				r.inputTensor.Destroy()
				if r.outputTensor != nil {
					r.outputTensor.Destroy()
				}
				for _, ex := range r.extraOutputs {
					ex.Destroy()
				}
			}
			return nil, err
		}
		runners = append(runners, runner)
		pool <- runner
	}
	slog.Info("Dual model session pool initialized", "name", name, "pool_size", poolSize)

	return &Model{
		name:             name,
		outputShape:      outShapes[0],
		runners:          runners,
		pool:             pool,
		extraOutputShape: outShapes[1:],
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

	copy(inData, inputData)

	if err := runner.session.Run(); err != nil {
		return nil, fmt.Errorf("session run: %w", err)
	}

	outData := runner.outputTensor.GetData()
	result := make([]float32, len(outData))
	copy(result, outData)

	return result, nil
}

// RunInferenceDual executes a dual-output session and returns both output tensors
// as fresh slices. The caller owns the returned slices.
func (m *Model) RunInferenceDual(inputData []float32) ([]float32, []float32, error) {
	if len(m.runners) == 0 || m.pool == nil {
		return nil, nil, fmt.Errorf("session pool is empty")
	}
	runner := <-m.pool
	defer func() { m.pool <- runner }()

	if len(runner.extraOutputs) < 1 {
		return nil, nil, fmt.Errorf("model is not dual-output")
	}

	inData := runner.inputTensor.GetData()
	if len(inputData) > len(inData) {
		return nil, nil, fmt.Errorf("input data too large for tensor: %d > %d", len(inputData), len(inData))
	}
	copy(inData, inputData)

	if err := runner.session.Run(); err != nil {
		return nil, nil, fmt.Errorf("session run: %w", err)
	}

	a := runner.outputTensor.GetData()
	b := runner.extraOutputs[0].GetData()
	out0 := append([]float32(nil), a...)
	out1 := append([]float32(nil), b...)
	return out0, out1, nil
}

// GetOutputShape returns the dimensions of the model's output tensor.
func (m *Model) GetOutputShape() []int64 {
	return m.outputShape
}

// GetExtraOutputShape returns the shape of the i-th additional output tensor
// when the model was loaded via loadModelDual, or nil otherwise.
func (m *Model) GetExtraOutputShape(i int) []int64 {
	if i < 0 || i >= len(m.extraOutputShape) {
		return nil
	}
	return m.extraOutputShape[i]
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
		for _, t := range runner.extraOutputs {
			t.Destroy()
		}
	}
	slog.Info("Model released", "name", m.name)
}
