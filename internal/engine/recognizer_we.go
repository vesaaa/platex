package engine

import (
	"fmt"
	"image"
	"log/slog"
	"math"

	"github.com/vesaa/platex/internal/types"
)

// RecognizerWE wraps the we0091234 plate_rec_color ONNX model. Compared to the
// HyperLPR3 v3 SVTR-LCNet model, it:
//   - takes BGR input (no channel flip), normalized with mean=0.588, std=0.193
//   - emits two outputs in a single forward: CTC plate logits + 5-class color
//   - uses a 78-token charset that includes 民/航/危/险/品 and lacks I/O
//
// Used as a complementary recognizer in ensemble mode to fix tilted-plate /
// length-drift cases that the primary recognizer mispredicts.
type RecognizerWE struct {
	model       *Model
	inputWidth  int
	inputHeight int
	timeSteps   int
	numClasses  int
	numColors   int
}

// WEResult is the structured output from the dual-head we recognizer.
type WEResult struct {
	PlateNumber string
	Confidence  float32
	CharConfs   []float32
	Color       int
	ColorConf   float32
}

// NewRecognizerWE loads the we0091234 plate_rec_color ONNX model. Returns nil
// without error when the file is missing - callers can treat that as "ensemble
// disabled" and stay on the primary recognizer alone.
func NewRecognizerWE(modelPath string, threads, optLevel int) (*RecognizerWE, error) {
	if modelPath == "" {
		return nil, nil
	}
	model, err := loadModelDual(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load we recognizer: %w", err)
	}
	r := &RecognizerWE{
		model:       model,
		inputWidth:  168,
		inputHeight: 48,
		timeSteps:   21,
		numClasses:  78,
		numColors:   5,
	}
	shape := model.GetOutputShape()
	if len(shape) >= 3 {
		r.timeSteps = int(shape[len(shape)-2])
		r.numClasses = int(shape[len(shape)-1])
	}
	if extra := model.GetExtraOutputShape(0); len(extra) >= 1 {
		r.numColors = int(extra[len(extra)-1])
	}
	slog.Info("WE recognizer loaded",
		"timesteps", r.timeSteps,
		"classes", r.numClasses,
		"colors", r.numColors,
	)
	return r, nil
}

// Recognize runs the we0091234 model on a single plate image and returns plate
// text plus color. The image is expected to be a roughly axis-aligned plate
// crop (pre-warped if necessary).
func (r *RecognizerWE) Recognize(img image.Image) (*WEResult, error) {
	if r == nil || r.model == nil {
		return nil, fmt.Errorf("we recognizer not initialized")
	}
	tensor := r.preprocess(img)
	plateOut, colorOut, err := r.model.RunInferenceDual(tensor)
	if err != nil {
		return nil, fmt.Errorf("we inference: %w", err)
	}
	plate, confs, avg := decodeWEPlate(plateOut, r.timeSteps, r.numClasses)
	colorIdx, colorConf := decodeWEColor(colorOut, r.numColors)
	return &WEResult{
		PlateNumber: plate,
		Confidence:  avg,
		CharConfs:   confs,
		Color:       colorIdx,
		ColorConf:   colorConf,
	}, nil
}

// preprocess converts a Go image into the [1, 3, 48, 168] BGR tensor expected
// by the we model: cv2-style BGR pixel ordering, (x/255 - 0.588) / 0.193.
func (r *RecognizerWE) preprocess(img image.Image) []float32 {
	resized := resizeImage(img, r.inputWidth, r.inputHeight)
	tensor := make([]float32, 3*r.inputHeight*r.inputWidth)
	channelSize := r.inputHeight * r.inputWidth
	const mean = 0.588
	const std = 0.193
	for y := 0; y < r.inputHeight; y++ {
		for x := 0; x < r.inputWidth; x++ {
			rr, gg, bb, _ := resized.At(x, y).RGBA()
			idx := y*r.inputWidth + x
			// BGR order, normalized.
			tensor[0*channelSize+idx] = (float32(bb>>8)/255.0 - mean) / std // B
			tensor[1*channelSize+idx] = (float32(gg>>8)/255.0 - mean) / std // G
			tensor[2*channelSize+idx] = (float32(rr>>8)/255.0 - mean) / std // R
		}
	}
	return tensor
}

// Close releases the underlying ONNX session pool.
func (r *RecognizerWE) Close() {
	if r != nil && r.model != nil {
		r.model.Close()
	}
}

// MapColor translates a we color index into our internal PlateColor code.
func MapWEColor(idx int) types.PlateColor {
	if idx < 0 || idx >= len(weColorLabels) {
		return types.ColorOther
	}
	return types.PlateColor(weColorLabels[idx])
}

// decodeWEPlate runs greedy CTC decode on the [T, C] plate logits tensor.
func decodeWEPlate(output []float32, timeSteps, numClasses int) (string, []float32, float32) {
	if len(output) == 0 || timeSteps <= 0 || numClasses <= 0 {
		return "", nil, 0
	}
	chars := make([]string, 0, timeSteps)
	confs := make([]float32, 0, timeSteps)
	prev := -1
	for t := 0; t < timeSteps; t++ {
		maxIdx := 0
		maxVal := float32(-math.MaxFloat32)
		for c := 0; c < numClasses; c++ {
			idx := t*numClasses + c
			if idx >= len(output) {
				break
			}
			if output[idx] > maxVal {
				maxVal = output[idx]
				maxIdx = c
			}
		}
		if maxIdx != 0 && maxIdx != prev && maxIdx < len(weChars) {
			chars = append(chars, weChars[maxIdx])
			confs = append(confs, maxVal)
		}
		prev = maxIdx
	}
	plate := ""
	for _, ch := range chars {
		plate += ch
	}
	avg := float32(0)
	if len(confs) > 0 {
		var sum float32
		for _, c := range confs {
			sum += c
		}
		avg = sum / float32(len(confs))
	}
	return plate, confs, avg
}

// decodeWEColor runs argmax + softmax-style normalization on the color head.
func decodeWEColor(output []float32, numColors int) (int, float32) {
	if len(output) == 0 || numColors <= 0 {
		return 0, 0
	}
	maxIdx := 0
	maxVal := output[0]
	for i := 1; i < numColors && i < len(output); i++ {
		if output[i] > maxVal {
			maxVal = output[i]
			maxIdx = i
		}
	}
	var sumExp float64
	for i := 0; i < numColors && i < len(output); i++ {
		sumExp += math.Exp(float64(output[i] - maxVal))
	}
	conf := float32(0)
	if sumExp > 0 {
		conf = float32(1.0 / sumExp)
	}
	return maxIdx, conf
}
