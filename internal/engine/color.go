package engine

import (
	"fmt"
	"image"
	"log/slog"
	"math"
)

// ColorClassifier handles license plate color classification.
type ColorClassifier struct {
	model       *Model
	inputWidth  int
	inputHeight int
	useModel    bool // false = use heuristic fallback
}

// NewColorClassifier creates a new color classifier.
// If modelPath is empty or model loading fails, falls back to heuristic-based classification.
func NewColorClassifier(modelPath string, threads, optLevel int) *ColorClassifier {
	if modelPath == "" {
		slog.Info("Color model path not set, using heuristic fallback")
		return &ColorClassifier{useModel: false}
	}

	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		slog.Warn("Failed to load color model, using heuristic fallback", "error", err)
		return &ColorClassifier{useModel: false}
	}

	return &ColorClassifier{
		model:       model,
		inputWidth:  96,
		inputHeight: 24,
		useModel:    true,
	}
}

// Classify determines the color of a license plate image.
// Returns color code (0-5) and confidence.
func (c *ColorClassifier) Classify(img image.Image) (int, float32) {
	if !c.useModel {
		return getDominantColor(img), 0.5 // Heuristic with low confidence
	}

	// Preprocess
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	tensor := imageToTensorCHW(img, c.inputWidth, c.inputHeight, mean, std)

	// Run inference
	output, err := c.runInference(tensor)
	if err != nil {
		slog.Warn("Color inference failed, using heuristic", "error", err)
		return getDominantColor(img), 0.5
	}

	// Find max class
	return argmaxWithConf(output)
}

// runInference executes the color classification model.
func (c *ColorClassifier) runInference(input []float32) ([]float32, error) {
	return c.model.RunInference(input)
}

// argmaxWithConf returns the index of the maximum value and its softmax confidence.
func argmaxWithConf(data []float32) (int, float32) {
	if len(data) == 0 {
		return 5, 0 // Other
	}

	maxIdx := 0
	maxVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
			maxIdx = i
		}
	}

	// Softmax for confidence
	var sumExp float64
	for _, v := range data {
		sumExp += math.Exp(float64(v - maxVal))
	}
	conf := float32(1.0 / sumExp)

	if maxIdx < len(plateColorLabels) {
		return plateColorLabels[maxIdx], conf
	}
	return 5, conf
}

// Close releases classifier resources.
func (c *ColorClassifier) Close() {
	if c.model != nil {
		c.model.Close()
	}
}
