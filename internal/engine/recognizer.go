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
	"unicode"

	"github.com/vesaa/platex/internal/types"
)

// Recognizer handles license plate character recognition using CRNN model.
type Recognizer struct {
	model       *Model
	inputWidth  int
	inputHeight int
	resizeMode  string // "auto", "letterbox", or "stretch"
	timeSteps   int    // Detected from model output
	numClasses  int    // Detected from model output
}

// NewRecognizer creates a new plate character recognizer.
func NewRecognizer(modelPath string, threads, optLevel int) (*Recognizer, error) {
	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load recognizer model: %w", err)
	}

	r := &Recognizer{
		model:       model,
		inputWidth:  160,
		inputHeight: 48,
		resizeMode:  "auto",
		timeSteps:   20, // Default fallback
		numClasses:  78, // Default fallback
	}

	// Try to detect actual shape from model
	shape := model.GetOutputShape()
	if len(shape) >= 3 {
		r.timeSteps = int(shape[len(shape)-2])
		r.numClasses = int(shape[len(shape)-1])
		slog.Info("Recognizer adapted to model shape", "timeSteps", r.timeSteps, "numClasses", r.numClasses)
	}

	return r, nil
}

// Recognize performs character recognition on a cropped plate image.
// Returns the plate number string, per-character confidences, and overall confidence.
func (r *Recognizer) Recognize(img image.Image) (string, []float32, float32, error) {
	// Stage 1: fast path for normal images.
	plateNumber, charConfs, avgConf, err := r.recognizeSingleAngle(img)
	if err != nil {
		return "", nil, 0, err
	}
	if !needRecoverySearch(plateNumber, avgConf) {
		slog.Info("Recognition result", "plate", plateNumber, "conf", avgConf, "steps", r.timeSteps, "classes", r.numClasses)
		return plateNumber, charConfs, avgConf, nil
	}

	// Stage 2: expensive candidate search only for abnormal outputs.
	best := recognizeCandidateResult{
		plate: plateNumber,
		confs: charConfs,
		conf:  avgConf,
		score: scorePlateCandidate(plateNumber, avgConf),
	}
	crops := r.candidateCrops(img)
	crops = append(crops, r.candidateCrops(upscaleImage(img, 2))...)
	angles := r.candidateAngles(img)
	for _, cimg := range crops {
		for _, angle := range angles {
			candImg := cimg
			if angle != 0 {
				candImg = rotateImageGrayBG(cimg, angle)
			}
			pn, pc, cf, e := r.recognizeSingleAngle(candImg)
			if e != nil {
				continue
			}
			score := scorePlateCandidate(pn, cf)
			if score > best.score {
				best = recognizeCandidateResult{
					plate: pn,
					confs: pc,
					conf:  cf,
					score: score,
				}
			}
		}
	}

	if best.score <= float32(-1e8) {
		return "", nil, 0, fmt.Errorf("inference: no valid candidate")
	}

	// Stage 3 (recovery only): lightweight ambiguous-char rerank.
	best.plate, best.confs, best.score = rerankAmbiguousPlate(best.plate, best.confs, best.score)

	slog.Info("Recognition result", "plate", best.plate, "conf", best.conf, "steps", r.timeSteps, "classes", r.numClasses)
	return best.plate, best.confs, best.conf, nil
}

func needRecoverySearch(plate string, conf float32) bool {
	r := []rune(strings.TrimSpace(plate))
	if len(r) < 6 {
		return true
	}
	if !looksLikeMainlandPlatePrefix(r) {
		return true
	}
	return conf < 0.70
}

func (r *Recognizer) candidateCrops(img image.Image) []image.Image {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if h == 0 {
		return []image.Image{img}
	}
	ratio := float64(w) / float64(h)
	if ratio >= 2.2 {
		return []image.Image{img}
	}
	// For square-ish inputs, include slight left-shift crops to focus on plate body.
	cands := []image.Image{
		img,
		cropWithOffset(img, 0.72, -0.25, 0),
		cropWithOffset(img, 0.68, -0.25, 0),
		cropWithOffset(img, 0.64, -0.25, 0),
		cropWithOffset(img, 0.60, -0.25, 0),
		cropWithOffset(img, 0.72, -0.15, 0),
		cropWithOffset(img, 0.68, -0.15, 0),
	}
	return cands
}

func cropWithOffset(src image.Image, ratio, xOffset, yOffset float64) image.Image {
	if ratio >= 0.999 || ratio <= 0 {
		return src
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	cw := int(float64(w) * ratio)
	ch := int(float64(h) * ratio)
	if cw < 8 || ch < 8 {
		return src
	}
	maxShiftX := (w - cw) / 2
	maxShiftY := (h - ch) / 2
	shiftX := int(float64(maxShiftX) * xOffset)
	shiftY := int(float64(maxShiftY) * yOffset)
	x0 := b.Min.X + (w-cw)/2 + shiftX
	y0 := b.Min.Y + (h-ch)/2 + shiftY
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x0+cw > b.Max.X {
		x0 = b.Max.X - cw
	}
	if y0+ch > b.Max.Y {
		y0 = b.Max.Y - ch
	}
	dst := image.NewNRGBA(image.Rect(0, 0, cw, ch))
	for y := 0; y < ch; y++ {
		for x := 0; x < cw; x++ {
			dst.Set(x, y, src.At(x0+x, y0+y))
		}
	}
	return dst
}

type recognizeCandidateResult struct {
	plate string
	confs []float32
	conf  float32
	score float32
}

func (r *Recognizer) recognizeSingleAngle(img image.Image) (string, []float32, float32, error) {
	// Determine resize strategy
	useLetterbox := r.shouldUseLetterbox(img)
	_ = useLetterbox // reserved for future use

	tensor := r.preprocessPlate(img)
	output, err := r.runInference(tensor)
	if err != nil {
		return "", nil, 0, fmt.Errorf("inference: %w", err)
	}

	plateNumber, charConfs, avgConf := ctcDecode(output, r.timeSteps, r.numClasses)
	plateNumber, charConfs = normalizePlateNumberWithConfidence(plateNumber, charConfs)
	return plateNumber, charConfs, avgConf, nil
}

func (r *Recognizer) candidateAngles(img image.Image) []float64 {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if h == 0 {
		return []float64{0}
	}
	ratio := float64(w) / float64(h)
	// Square-ish/tilted crops need more angle search.
	if ratio < 2.2 {
		return []float64{
			0,
			-30, -25, -20, -16, -12, -8, -4,
			4, 8, 12, 16, 20, 25, 30,
		}
	}
	if ratio < 3.0 {
		return []float64{0, -18, -12, -8, -4, 4, 8, 12, 18}
	}
	return []float64{0, -12, -8, -4, 4, 8, 12}
}

func scorePlateCandidate(plate string, conf float32) float32 {
	r := []rune(strings.TrimSpace(plate))
	if len(r) == 0 {
		return -100
	}
	score := conf * 100

	// Prefer standard CN plate lengths.
	switch len(r) {
	case 7:
		score += 8
	case 8:
		score += 10
	case 6:
		score -= 12
	default:
		score -= 20
	}
	if looksLikeMainlandPlatePrefix(r) {
		score += 6
	} else {
		score -= 10
	}
	score += mainlandFormatScore(r)
	return score
}

func rerankAmbiguousPlate(plate string, confs []float32, baseScore float32) (string, []float32, float32) {
	r := []rune(strings.TrimSpace(plate))
	if len(r) != 7 || len(confs) != len(r) || !looksLikeMainlandPlatePrefix(r) {
		return plate, confs, baseScore
	}

	bestPlate := plate
	bestConfs := append([]float32(nil), confs...)
	bestScore := baseScore

	// Candidate A: repeated-digit tail often comes from alnum confusion in tilted crops.
	// Example: 粤L02166 -> 粤L021Y6
	if unicode.IsDigit(r[4]) && unicode.IsDigit(r[5]) && unicode.IsDigit(r[6]) && r[5] == r[6] {
		for _, repl := range ambiguousLettersForDigit(r[5]) {
			cand := append([]rune(nil), r...)
			cand[5] = repl
			candConfs := append([]float32(nil), confs...)
			if candConfs[5] < 0.68 {
				candConfs[5] = 0.68
			}
			// Replacement penalty keeps this as a recovery-only fallback.
			candScore := scorePlateCandidate(string(cand), meanConfs(candConfs)) - 1.0
			if candScore > bestScore {
				bestScore = candScore
				bestPlate = string(cand)
				bestConfs = candConfs
			}
		}
	}

	// Candidate B: adjacent transposition in tail is common on tilted/noisy crops.
	// Example: 粤L02Y16 -> 粤L021Y6
	if isASCIILetter(r[4]) && unicode.IsDigit(r[5]) && unicode.IsDigit(r[6]) {
		cand := append([]rune(nil), r...)
		cand[4], cand[5] = cand[5], cand[4]
		candConfs := append([]float32(nil), confs...)
		// keep conservative confidence for swapped chars
		if candConfs[4] < 0.72 {
			candConfs[4] = 0.72
		}
		if candConfs[5] < 0.68 {
			candConfs[5] = 0.68
		}
		candScore := scorePlateCandidate(string(cand), meanConfs(candConfs)) - 0.6
		if candScore > bestScore {
			bestScore = candScore
			bestPlate = string(cand)
			bestConfs = candConfs
		}
	}

	return bestPlate, bestConfs, bestScore
}

func ambiguousLettersForDigit(d rune) []rune {
	switch d {
	case '0':
		return []rune{'D', 'Q'}
	case '1':
		return []rune{'I', 'L'}
	case '5':
		return []rune{'S'}
	case '6':
		return []rune{'Y', 'G'}
	case '8':
		return []rune{'B'}
	default:
		return []rune{'Y'}
	}
}

func meanConfs(confs []float32) float32 {
	if len(confs) == 0 {
		return 0
	}
	var sum float32
	for _, c := range confs {
		sum += c
	}
	return sum / float32(len(confs))
}

func upscaleImage(src image.Image, scale int) image.Image {
	if scale <= 1 {
		return src
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w == 0 || h == 0 {
		return src
	}
	dst := image.NewNRGBA(image.Rect(0, 0, w*scale, h*scale))
	for y := 0; y < h*scale; y++ {
		for x := 0; x < w*scale; x++ {
			sx := b.Min.X + x/scale
			sy := b.Min.Y + y/scale
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst
}

func mainlandFormatScore(r []rune) float32 {
	// Prefer canonical mainland plate patterns:
	// 7-char: [汉][A-Z][A-Z0-9]{5}
	// 8-char(new-energy): [汉][A-Z][A-Z0-9]{6}
	if len(r) != 7 && len(r) != 8 {
		return -8
	}
	if !isChineseRune(r[0]) || !isASCIILetter(r[1]) {
		return -12
	}
	score := float32(0)
	for i := 2; i < len(r); i++ {
		ch := r[i]
		if isASCIILetter(ch) || unicode.IsDigit(ch) {
			score += 1.8
		} else {
			score -= 3
		}
	}
	if len(r) == 8 {
		score += 2 // slight preference for recognized new-energy length
	}
	return score
}

func rotateImageGrayBG(src image.Image, angleDeg float64) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w == 0 || h == 0 {
		return src
	}
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	bg := color.NRGBA{R: 128, G: 128, B: 128, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.SetNRGBA(x, y, bg)
		}
	}

	rad := angleDeg * math.Pi / 180.0
	sinA := math.Sin(rad)
	cosA := math.Cos(rad)
	cx := float64(w-1) / 2.0
	cy := float64(h-1) / 2.0

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy

			srcX := cosA*dx + sinA*dy + cx
			srcY := -sinA*dx + cosA*dy + cy

			ix := int(math.Round(srcX))
			iy := int(math.Round(srcY))
			if ix >= 0 && ix < w && iy >= 0 && iy < h {
				dst.Set(x, y, src.At(ix+b.Min.X, iy+b.Min.Y))
			}
		}
	}
	return dst
}

func normalizePlateNumberWithConfidence(s string, confs []float32) (string, []float32) {
	r := []rune(strings.TrimSpace(s))
	if len(r) < 2 || len(confs) != len(r) {
		return s, confs
	}

	// Soft correction 1:
	// Collapse duplicated province prefix only when the duplicate has low confidence.
	if len(r) >= 3 &&
		r[0] == r[1] &&
		isChineseRune(r[0]) &&
		isASCIILetter(r[2]) &&
		confs[1] < 0.72 {
		r = append([]rune{r[0]}, r[2:]...)
		confs = append([]float32{confs[0]}, confs[2:]...)
	}

	// Soft correction 2:
	// For new-energy-like tail, convert '0' -> 'D' only on low confidence.
	// Example: 粤LD07111 -> 粤LDD7111
	if len(r) >= 8 {
		for i := 2; i <= len(r)-4; i++ {
			if r[i] != '0' || confs[i] >= 0.72 {
				continue
			}
			if i > 0 && isASCIILetter(r[i-1]) && allDigits(r[i+1:]) {
				r[i] = 'D'
				confs[i] = 0.72
				break
			}
		}
	}

	// Soft correction 3:
	// If result length is 6 and it looks like a normal mainland plate prefix,
	// duplicate tail digit once to recover common CTC merge drop at the end.
	// Example: 粤LE7G2 -> 粤LE7G22
	if len(r) == 6 &&
		looksLikeMainlandPlatePrefix(r) &&
		unicode.IsDigit(r[len(r)-1]) &&
		confs[len(confs)-1] >= 0.65 {
		r = append(r, r[len(r)-1])
		confs = append(confs, confs[len(confs)-1]*0.95)
	}

	return string(r), confs
}

func isChineseRune(ch rune) bool {
	return unicode.Is(unicode.Han, ch)
}

func isASCIILetter(ch rune) bool {
	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
}

func allDigits(rs []rune) bool {
	if len(rs) == 0 {
		return false
	}
	for _, ch := range rs {
		if !unicode.IsDigit(ch) {
			return false
		}
	}
	return true
}

func looksLikeMainlandPlatePrefix(r []rune) bool {
	if len(r) < 2 {
		return false
	}
	// Typical structure starts with province Chinese char + Latin letter.
	return isChineseRune(r[0]) && isASCIILetter(r[1])
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
				slog.Info("CTC token", "t", t, "idx", maxIdx, "char", plateChars[maxIdx], "conf", conf)
			} else {
				slog.Info("CTC token ignored (index out of range)", "t", t, "idx", maxIdx, "conf", conf)
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

