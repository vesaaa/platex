package engine

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log/slog"
	"math"
	"os"
	"strings"

	"github.com/vesaa/platex/internal/types"
)

// Recognizer handles license plate character recognition using CRNN model.
type Recognizer struct {
	model      *Model
	inputWidth  int
	inputHeight int
	resizeMode  string // "auto", "letterbox", or "stretch"
}

// NewRecognizer creates a new plate character recognizer.
func NewRecognizer(modelPath string, threads, optLevel int) (*Recognizer, error) {
	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load recognizer model: %w", err)
	}

	return &Recognizer{
		model:      model,
		inputWidth:  160, // HyperLPR3 rpv3_mdict model input width
		inputHeight: 48,  // HyperLPR3 rpv3_mdict model input height
		resizeMode:  "auto",
	}, nil
}

// Recognize performs character recognition on a cropped plate image.
// Returns the plate number string, per-character confidences, and overall confidence.
func (r *Recognizer) Recognize(img image.Image) (string, []float32, float32, error) {
	// Determine resize strategy
	useLetterbox := r.shouldUseLetterbox(img)
	_ = useLetterbox // reserved for future use

	// Preprocess: match HyperLPR3's encode_images() exactly
	// 1. Resize height to 48, width proportionally (capped at 160)
	// 2. Normalize: (pixel - 127.5) / 127.5 → range [-1, 1]
	// 3. Channel order: BGR (OpenCV convention)
	// 4. Left-aligned zero-padding to fill 160 width
	tensor := r.preprocessPlate(img)

	// DEBUG: Save preprocessed tensor to image to verify preprocessing
	r.saveDebugImage(tensor, "debug_preprocessed.jpg")

	// Run inference
	output, err := r.runInference(tensor)
	if err != nil {
		return "", nil, 0, fmt.Errorf("inference: %w", err)
	}

	// DEBUG: Log raw output stats
	if len(output) > 0 {
		max := float32(-1e10)
		min := float32(1e10)
		for _, v := range output {
			if v > max {
				max = v
			}
			if v < min {
				min = v
			}
		}
		slog.Debug("Raw model output stats", "len", len(output), "min", min, "max", max, "first_5", output[:minInt(len(output), 5)])
	}

	// CTC decode: model outputs [40, 6625] but only first 77 indices are valid chars
	numClasses := len(output) / r.getOutputTimeSteps() // = 6625
	plateNumber, charConfs, avgConf := ctcDecode(output, r.getOutputTimeSteps(), numClasses)

	slog.Debug("Recognition result", "plate", plateNumber, "conf", avgConf, "steps", r.getOutputTimeSteps(), "classes", numClasses)

	return plateNumber, charConfs, avgConf, nil
}

// runInference executes the ONNX model.
func (r *Recognizer) runInference(input []float32) ([]float32, error) {
	return r.model.RunInference(input)
}

// getOutputTimeSteps returns the number of time steps in CRNN output.
func (r *Recognizer) getOutputTimeSteps() int {
	// CRNN output width depends on input width: typically inputWidth/4
	return r.inputWidth / 4 // 160/4 = 40 time steps
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

		// Apply confidence (the raw value, which is already softmaxed by the model)
		conf := maxVal

		// CTC rules: skip blanks (index 0) and repeated characters
		if maxIdx != 0 && maxIdx != prevIdx {
			if maxIdx < len(plateChars) {
				chars = append(chars, plateChars[maxIdx])
				confs = append(confs, conf)
				slog.Debug("CTC token", "t", t, "idx", maxIdx, "char", plateChars[maxIdx], "conf", conf)
			} else {
				slog.Debug("CTC token ignored (index out of range)", "t", t, "idx", maxIdx, "conf", conf)
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

// preprocessPlate replicates HyperLPR3's encode_images() exactly:
//   - Resize to height=48, width proportional (capped at 160, min 48)
//   - Normalize: (pixel - 127.5) / 127.5 → [-1, 1]
//   - Channel order: BGR
//   - Left-aligned, zero-padded to full 160 width
func (r *Recognizer) preprocessPlate(img image.Image) []float32 {
	bounds := img.Bounds()
	srcW := float64(bounds.Dx())
	srcH := float64(bounds.Dy())

	imgH := r.inputHeight // 48
	imgW := r.inputWidth  // 160

	// Calculate proportional width (same as Python: ratio_imgH)
	ratio := srcW / srcH
	resizedW := int(math.Ceil(float64(imgH) * ratio))
	if resizedW < 48 {
		resizedW = 48
	}
	if resizedW > imgW {
		resizedW = imgW
	}

	// Resize to (resizedW, 48)
	resized := resizeImage(img, resizedW, imgH)

	// Create zero-padded tensor [3, 48, 160] in BGR order
	tensor := make([]float32, 3*imgH*imgW) // all zeros = zero-padding
	channelSize := imgH * imgW

	for y := 0; y < imgH; y++ {
		for x := 0; x < resizedW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := y*imgW + x

			// BGR order, normalize: (pixel - 127.5) / 127.5
			tensor[0*channelSize+idx] = (float32(b>>8) - 127.5) / 127.5 // B
			tensor[1*channelSize+idx] = (float32(g>>8) - 127.5) / 127.5 // G
			tensor[2*channelSize+idx] = (float32(r>>8) - 127.5) / 127.5 // R
		}
	}
	// Remaining columns (resizedW to 160) stay as 0.0 = zero-padding

	return tensor
}

// shouldUseLetterbox decides the resize strategy based on resizeMode and image aspect ratio.
// In "auto" mode: if the input aspect ratio is within 10% of the model's target (3.33:1),
// use stretch (faster, negligible distortion); otherwise use letterbox.
func (r *Recognizer) shouldUseLetterbox(img image.Image) bool {
	switch r.resizeMode {
	case "stretch":
		return false
	case "letterbox":
		return true
	default: // "auto"
		bounds := img.Bounds()
		imgRatio := float64(bounds.Dx()) / float64(bounds.Dy())
		modelRatio := float64(r.inputWidth) / float64(r.inputHeight) // 160/48 = 3.33

		// If aspect ratio within 10% of target, stretch is fine
		diff := math.Abs(imgRatio-modelRatio) / modelRatio
		return diff > 0.10
	}
}

// Close releases recognizer resources.
func (r *Recognizer) Close() {
	if r.model != nil {
		r.model.Close()
	}
}

func (r *Recognizer) saveDebugImage(tensor []float32, filename string) {
	img := image.NewNRGBA(image.Rect(0, 0, r.inputWidth, r.inputHeight))
	channelSize := r.inputWidth * r.inputHeight
	for y := 0; y < r.inputHeight; y++ {
		for x := 0; x < r.inputWidth; x++ {
			idx := y*r.inputWidth + x
			// Tensor is BGR, convert back to RGB for saving
			b := uint8(tensor[0*channelSize+idx]*127.5 + 127.5)
			g := uint8(tensor[1*channelSize+idx]*127.5 + 127.5)
			re := uint8(tensor[2*channelSize+idx]*127.5 + 127.5)
			img.SetNRGBA(x, y, color.NRGBA{R: re, G: g, B: b, A: 255})
		}
	}
	f, err := os.Create(filename)
	if err == nil {
		jpeg.Encode(f, img, nil)
		f.Close()
		slog.Debug("Saved debug image", "path", filename)
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

