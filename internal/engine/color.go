package engine

import (
	"image"
	"log/slog"
	"math"

	"github.com/vesaa/platex/internal/types"
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

	// HyperLPR3 v3 color model expects a square 96x96 RGB tensor and emits 3
	// class probabilities (blue/green/yellow). The previous defaults (96x24,
	// 6 labels) were a leftover from an older variant and silently disabled
	// the model in practice.
	return &ColorClassifier{
		model:       model,
		inputWidth:  96,
		inputHeight: 96,
		useModel:    true,
	}
}

// Classify determines the color of a license plate image.
// Returns color code (0-5) and confidence.
func (c *ColorClassifier) Classify(img image.Image) (int, float32) {
	if !c.useModel {
		color, conf := getDominantColorWithConf(img)
		slog.Debug("Color classified by heuristic fallback", "color", color, "conf", conf)
		return color, conf
	}

	// Preprocess
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	tensor := imageToTensorCHW(img, c.inputWidth, c.inputHeight, mean, std, false)

	// Run inference
	output, err := c.runInference(tensor)
	if err != nil {
		slog.Warn("Color inference failed, using heuristic", "error", err)
		color, conf := getDominantColorWithConf(img)
		return color, conf
	}

	// Find max class
	return argmaxWithConf(output)
}

func getDominantColorWithConf(img image.Image) (int, float32) {
	color := getDominantColor(img)
	// Heuristic fallback confidence should stay conservative.
	// Green is usually harder under warm/low-light conditions.
	switch color {
	case 5: // green
		return color, 0.62
	case 4: // yellow
		return color, 0.58
	default:
		return color, 0.55
	}
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

// Plate color rules. Mainland plate color is fixed by the plate format
// (GA 36): NEV green, police white, HK/Macau/embassy black, learner/trailer
// yellow; only standard 7-char plates can be blue (small car) or yellow
// (large vehicle). On the 20万 camera crops the 3-class color model called
// ~2% of blue plates yellow (the cameras tint blue towards cyan and white
// text towards yellow) and cannot output white or black at all, so color is
// derived from the recognized number first, and for standard plates from the
// share of saturated yellow pixels in the plate center.
const (
	yellowPlateMinFraction = 0.45
	// Yellow in OpenCV HSV terms: H 12..38 (of 180), S and V >= 90 (of 255).
	yellowHueMinDeg = 24.0
	yellowHueMaxDeg = 76.0
	yellowMinSV     = 90.0
)

// plateColorByFormat returns the plate color implied by the plate number,
// consulting the image only for standard 7-char plates (blue vs yellow).
// ok=false means the format doesn't determine the color.
func plateColorByFormat(plate string, img image.Image) (color int, ok bool) {
	r := []rune(plate)
	if len(r) < 7 || !looksLikeMainlandPlatePrefix(r) {
		return 0, false
	}
	switch last := r[len(r)-1]; {
	case last == '警':
		return int(types.ColorWhite), true
	case last == '港' || last == '澳' || last == '使' || last == '领':
		return int(types.ColorBlack), true
	case last == '学' || last == '挂':
		return int(types.ColorYellow), true
	}
	if isNewEnergyPlateFormat(r) {
		return int(types.ColorGreen), true
	}
	if len(r) == 7 && isStandardPlateTail(r[2:]) {
		if yellowPixelFraction(img) >= yellowPlateMinFraction {
			return int(types.ColorYellow), true
		}
		return int(types.ColorBlue), true
	}
	return 0, false
}

// isStandardPlateTail reports whether the 5 chars after province+city are
// all letters/digits (excludes special suffixes handled above).
func isStandardPlateTail(r []rune) bool {
	for _, c := range r {
		if !(c >= '0' && c <= '9') && !(c >= 'A' && c <= 'Z') {
			return false
		}
	}
	return len(r) == 5
}

// yellowPixelFraction is the share of saturated yellow pixels in the central
// region of the plate (96x32 resample, margins trimmed so car body around a
// tight crop doesn't count).
func yellowPixelFraction(img image.Image) float64 {
	small := resizeImage(img, 96, 32)
	total, yellow := 0, 0
	for y := 5; y < 27; y++ {
		for x := 14; x < 82; x++ {
			c := small.NRGBAAt(x, y)
			h, s, v := rgbToHSV(float64(c.R), float64(c.G), float64(c.B))
			total++
			if h >= yellowHueMinDeg && h <= yellowHueMaxDeg && s >= yellowMinSV && v >= yellowMinSV {
				yellow++
			}
		}
	}
	return float64(yellow) / float64(total)
}
