package engine

import (
	"fmt"
	"image"
	"math"
	"strings"

	"github.com/vesaa/platex/internal/types"
)

// Recognizer handles license plate character recognition using CRNN model.
type Recognizer struct {
	model       *Model
	inputWidth  int
	inputHeight int
}

// NewRecognizer creates a new plate character recognizer.
func NewRecognizer(modelPath string, threads, optLevel int) (*Recognizer, error) {
	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load recognizer model: %w", err)
	}

	return &Recognizer{
		model:       model,
		inputWidth:  168, // CRNN standard input width
		inputHeight: 48,  // CRNN standard input height
	}, nil
}

// Recognize performs character recognition on a cropped plate image.
// Returns the plate number string, per-character confidences, and overall confidence.
func (r *Recognizer) Recognize(img image.Image) (string, []float32, float32, error) {
	// Preprocess: resize to model input size and convert to tensor
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	tensor := imageToTensorCHW(img, r.inputWidth, r.inputHeight, mean, std)

	// Run inference
	output, err := r.runInference(tensor)
	if err != nil {
		return "", nil, 0, fmt.Errorf("inference: %w", err)
	}

	// CTC decode
	plateNumber, charConfs, avgConf := ctcDecode(output, r.getOutputTimeSteps(), len(plateChars))

	return plateNumber, charConfs, avgConf, nil
}

// runInference executes the ONNX model.
func (r *Recognizer) runInference(input []float32) ([]float32, error) {
	return r.model.RunInference(input)
}

// getOutputTimeSteps returns the number of time steps in CRNN output.
func (r *Recognizer) getOutputTimeSteps() int {
	// CRNN output width depends on input width: typically inputWidth/4
	return r.inputWidth / 4 // 168/4 = 42 time steps
}

// ctcDecode performs CTC (Connectionist Temporal Classification) greedy decoding.
// Input: raw output probabilities [timeSteps * numClasses]
// Returns: decoded string, per-character confidences, average confidence.
func ctcDecode(output []float32, timeSteps, numClasses int) (string, []float32, float32) {
	if len(output) == 0 {
		return "", nil, 0
	}

	var chars []string
	var confs []float32
	prevIdx := 0 // CTC blank index

	for t := 0; t < timeSteps; t++ {
		// Find the class with max probability at this time step
		maxIdx := 0
		maxVal := float32(-math.MaxFloat32)

		for c := 0; c < numClasses; c++ {
			idx := t*numClasses + c
			if idx < len(output) && output[idx] > maxVal {
				maxVal = output[idx]
				maxIdx = c
			}
		}

		// Apply softmax-like confidence (the raw value after softmax)
		conf := softmaxMax(output, t*numClasses, numClasses)

		// CTC rules: skip blanks (index 0) and repeated characters
		if maxIdx != 0 && maxIdx != prevIdx {
			if maxIdx < len(plateChars) {
				chars = append(chars, plateChars[maxIdx])
				confs = append(confs, conf)
			}
		}
		prevIdx = maxIdx
	}

	plateNumber := strings.Join(chars, "")

	// Calculate average confidence
	var avgConf float32
	if len(confs) > 0 {
		var sum float32
		for _, c := range confs {
			sum += c
		}
		avgConf = sum / float32(len(confs))
	}

	return plateNumber, confs, avgConf
}

// softmaxMax computes the softmax probability of the maximum element.
func softmaxMax(data []float32, offset, size int) float32 {
	maxVal := float32(-math.MaxFloat32)
	for i := 0; i < size; i++ {
		idx := offset + i
		if idx < len(data) && data[idx] > maxVal {
			maxVal = data[idx]
		}
	}

	var sumExp float64
	for i := 0; i < size; i++ {
		idx := offset + i
		if idx < len(data) {
			sumExp += math.Exp(float64(data[idx] - maxVal))
		}
	}

	if sumExp == 0 {
		return 0
	}
	return float32(1.0 / sumExp)
}

// classifyPlateType determines the plate type from the recognized characters.
func classifyPlateType(plateNumber string) types.PlateType {
	runes := []rune(plateNumber)
	switch {
	case len(runes) == 8:
		return types.PlateTypeNewEnergy
	case len(runes) == 7:
		return types.PlateTypeStandard7
	default:
		return types.PlateTypeUnknown
	}
}

// Close releases recognizer resources.
func (r *Recognizer) Close() {
	if r.model != nil {
		r.model.Close()
	}
}
