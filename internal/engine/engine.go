package engine

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg" // Register JPEG decoder
	_ "image/png"  // Register PNG decoder
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/vesaa/platex/internal/config"
	"github.com/vesaa/platex/internal/types"
)

const (
	plateAspectRatioTarget    = 3.33
	plateAspectRatioTolerance = 0.10
)

// Engine is the main license plate recognition engine.
type Engine struct {
	detector     *Detector
	recognizer   *Recognizer
	recognizerWE *RecognizerWE
	color        *ColorClassifier
	config       *config.EngineConfig
	workerCh     chan *recognizeJob
	urlFetchCh   chan struct{}
	httpClient   *http.Client
	wg           sync.WaitGroup

	// Stats
	totalImages atomic.Int64
	totalPlates atomic.Int64
	totalTimeMs atomic.Int64
}

// recognizeJob represents a unit of work for the worker pool.
type recognizeJob struct {
	img        image.Image
	resizeMode string
	resultCh   chan *types.PlateResult
	errCh      chan error
}

// New creates and initializes the recognition engine.
func New(cfg *config.EngineConfig) (*Engine, error) {
	slog.Info("Initializing LPR engine",
		"mode", cfg.Mode,
		"workers", cfg.Workers,
	)

	// Initialize ONNX Runtime
	if err := initONNXRuntime(""); err != nil {
		slog.Warn("Failed to initialize ONNX Runtime, will use fallback", "error", err)
	}

	// Load detector model (for full mode)
	detector, err := NewDetector(
		cfg.Models.Detector,
		cfg.ONNX.ThreadsPerSession,
		cfg.ONNX.OptimizationLevel,
		cfg.Detection,
	)
	if err != nil {
		slog.Warn("Failed to load detector model", "error", err)
		// Continue without detector - full mode will fail on requests
	}

	// Load recognizer model
	recognizer, err := NewRecognizer(
		cfg.Models.Recognizer,
		cfg.ONNX.ThreadsPerSession,
		cfg.ONNX.OptimizationLevel,
	)
	if err != nil {
		slog.Warn("Failed to load recognizer model", "error", err)
		// Continue without model - will fail on recognition attempts
	}

	// Load color classifier (falls back to heuristic if model unavailable)
	colorCls := NewColorClassifier(
		cfg.Models.Color,
		cfg.ONNX.ThreadsPerSession,
		cfg.ONNX.OptimizationLevel,
	)

	// Optional: load the we0091234 plate_rec_color model for ensemble fallback.
	// Missing file is non-fatal; the primary recognizer keeps working alone.
	weRec, err := NewRecognizerWE(
		cfg.Models.RecognizerWE,
		cfg.ONNX.ThreadsPerSession,
		cfg.ONNX.OptimizationLevel,
	)
	if err != nil {
		slog.Warn("Failed to load WE recognizer; ensemble disabled", "error", err)
		weRec = nil
	}

	e := &Engine{
		detector:     detector,
		recognizer:   recognizer,
		recognizerWE: weRec,
		color:        colorCls,
		config:       cfg,
		workerCh:   make(chan *recognizeJob, cfg.Workers*2),
		urlFetchCh: make(chan struct{}, max(1, cfg.URL.MaxFetchConcurrency)),
		httpClient: &http.Client{
			Timeout: time.Duration(max(100, cfg.URL.FetchTimeoutMs)) * time.Millisecond,
			Transport: &http.Transport{
				MaxIdleConns:        max(32, cfg.URL.MaxIdleConns),
				MaxIdleConnsPerHost: max(8, cfg.URL.MaxIdleConnsPerHost),
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}

	// Start worker pool
	for i := 0; i < cfg.Workers; i++ {
		e.wg.Add(1)
		go e.worker(i)
	}

	slog.Info("LPR engine initialized", "workers", cfg.Workers)
	return e, nil
}

// worker processes recognition jobs from the channel.
func (e *Engine) worker(id int) {
	defer e.wg.Done()
	slog.Debug("Worker started", "id", id)

	for job := range e.workerCh {
		result, err := e.recognizeSingle(job.img, job.resizeMode)
		if err != nil {
			job.errCh <- err
		} else {
			job.resultCh <- result
		}
	}

	slog.Debug("Worker stopped", "id", id)
}

// recognizeSingle performs recognition on a single image.
func (e *Engine) recognizeSingle(img image.Image, resizeMode string) (*types.PlateResult, error) {
	if e.recognizer == nil {
		return nil, fmt.Errorf("recognizer model not loaded")
	}

	// Step 1: Primary recognition (HyperLPR3 v3 SVTR-LCNet).
	plateNumber, _, confidence, err := e.recognizer.RecognizeWithResizeMode(img, resizeMode)
	if err != nil {
		return nil, fmt.Errorf("recognition: %w", err)
	}

	// Step 1b (optional): when an ensemble recognizer is available and the
	// primary result looks suspicious, run we0091234 as a cross-check and pick
	// the better candidate.  Suspicious means low confidence, structurally
	// inconsistent (length/prefix), or empty.
	weColorCode := -1
	weColorConf := float32(0)
	if e.recognizerWE != nil {
		if shouldRunEnsemble(plateNumber, confidence) {
			if weRes, weErr := e.recognizerWE.Recognize(img); weErr == nil && weRes != nil {
				if better := pickBetterPlateCandidate(plateNumber, confidence, weRes.PlateNumber, weRes.Confidence); better == "we" {
					slog.Info("Ensemble: WE recognizer wins",
						"primary", plateNumber, "primary_conf", confidence,
						"we", weRes.PlateNumber, "we_conf", weRes.Confidence,
					)
					plateNumber = weRes.PlateNumber
					confidence = weRes.Confidence
					weColorCode = int(MapWEColor(weRes.Color))
					weColorConf = weRes.ColorConf
				}
			}
		}
	}

	if plateNumber == "" || confidence < e.config.Rec.MinConfidence {
		// Not an error - just no plate detected with sufficient confidence
		return nil, nil
	}

	// Step 2: Color classification
	colorCode, colorConf := e.color.Classify(img)
	// Prefer the WE color when its confidence dominates the heuristic-fallback color.
	if weColorCode >= 0 && weColorConf > colorConf {
		colorCode = weColorCode
		colorConf = weColorConf
	}

	// Step 3: Determine plate type
	plateType := classifyPlateType(plateNumber)

	// Lightweight consistency correction:
	// new-energy plates are predominantly green; if color confidence is low,
	// avoid reporting yellow by biasing toward green.
	if plateType == types.PlateTypeNewEnergy &&
		colorCode == int(types.ColorYellow) &&
		colorConf < 0.80 {
		slog.Info("Color corrected for new-energy plate",
			"plate", plateNumber,
			"from", colorCode,
			"to", int(types.ColorGreen),
			"color_conf", colorConf,
		)
		colorCode = int(types.ColorGreen)
	}
	// For standard 7-char civilian plates, low-confidence green predictions are
	// often blue/green boundary errors; bias back to blue conservatively.
	if plateType == types.PlateTypeStandard7 &&
		colorCode == int(types.ColorGreen) &&
		colorConf < 0.90 {
		slog.Info("Color corrected for standard-7 plate",
			"plate", plateNumber,
			"from", colorCode,
			"to", int(types.ColorBlue),
			"color_conf", colorConf,
		)
		colorCode = int(types.ColorBlue)
	}
	// For standard-7, yellow predictions on blue plates are also common under
	// low-light/warm-light conditions. Only correct when model confidence is low
	// and image evidence strongly favors blue over yellow.
	if plateType == types.PlateTypeStandard7 &&
		colorCode == int(types.ColorYellow) &&
		colorConf < 0.90 &&
		hasBlueDominanceOverYellow(img) {
		slog.Info("Color corrected for standard-7 blue/yellow ambiguity",
			"plate", plateNumber,
			"from", colorCode,
			"to", int(types.ColorBlue),
			"color_conf", colorConf,
		)
		colorCode = int(types.ColorBlue)
	}

	colorName := "其他"
	if name, ok := types.ColorNames[types.PlateColor(colorCode)]; ok {
		colorName = name
	}

	return &types.PlateResult{
		PlateNumber: plateNumber,
		Color:       types.PlateColor(colorCode),
		ColorName:   colorName,
		Confidence:  confidence,
		Type:        plateType,
	}, nil
}

// RecognizeBatch processes a batch of image inputs.
func (e *Engine) RecognizeBatch(inputs []types.ImageInput, mode string, opts *types.RecognizeOption) []types.ImageResult {
	start := time.Now()
	results := make([]types.ImageResult, len(inputs))

	minConf := e.config.Rec.MinConfidence
	if opts != nil && opts.MinConfidence > 0 {
		minConf = opts.MinConfidence
	}
	_ = minConf // Will be used when model is integrated

	// Normalize mode
	if mode == "" {
		mode = "auto"
	}

	resizeMode := "auto"
	if opts != nil && opts.ResizeMode != "" {
		resizeMode = opts.ResizeMode
	}

	var wg sync.WaitGroup

	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, inp types.ImageInput) {
			defer wg.Done()

			imgStart := time.Now()
			result := types.ImageResult{ID: inp.ID}

			// Decode image
			img, err := e.decodeInput(inp)
			if err != nil {
				result.Error = fmt.Sprintf("decode error: %v", err)
				result.ElapsedMs = time.Since(imgStart).Milliseconds()
				results[idx] = result
				return
			}

			effectiveMode := mode
			if mode == "auto" {
				if shouldUseCropByAspect(img) {
					effectiveMode = "crop"
				} else {
					effectiveMode = "full"
				}
			}

			if effectiveMode == "full" {
				plates, recErr := e.recognizeFull(img, opts, resizeMode)
				if recErr != nil {
					result.Error = recErr.Error()
				} else {
					result.Plates = plates
					e.totalPlates.Add(int64(len(plates)))
				}
				result.ElapsedMs = time.Since(imgStart).Milliseconds()
				e.totalImages.Add(1)
				e.totalTimeMs.Add(result.ElapsedMs)
				results[idx] = result
				return
			}

			// Submit to worker pool (crop mode)
			job := &recognizeJob{
				img:        img,
				resizeMode: resizeMode,
				resultCh:   make(chan *types.PlateResult, 1),
				errCh:      make(chan error, 1),
			}

			submitTimeout := time.Duration(max(50, e.config.SubmitTimeoutMs)) * time.Millisecond
			timer := time.NewTimer(submitTimeout)
			defer timer.Stop()
			select {
			case e.workerCh <- job:
				// Wait for result
				select {
				case plate := <-job.resultCh:
					if plate != nil {
						result.Plates = []types.PlateResult{*plate}
						e.totalPlates.Add(1)
					} else {
						result.Plates = []types.PlateResult{} // no plate found
					}
				case err := <-job.errCh:
					result.Error = err.Error()
				}
			case <-timer.C:
				result.Error = "worker queue submit timeout, try again later"
			}

			// Crop second-pass retry:
			// If crop mode produced no plate, retry with lightweight image tweaks
			// while staying in crop pipeline (no full-mode fallback).
			if effectiveMode == "crop" && result.Error == "" && len(result.Plates) == 0 {
				if plate := e.retryCropWithTweaks(img, resizeMode); plate != nil {
					result.Plates = []types.PlateResult{*plate}
					e.totalPlates.Add(1)
				}
			}

			result.ElapsedMs = time.Since(imgStart).Milliseconds()
			e.totalImages.Add(1)
			e.totalTimeMs.Add(result.ElapsedMs)
			results[idx] = result
		}(i, input)
	}

	wg.Wait()

	totalMs := time.Since(start).Milliseconds()
	slog.Info("Batch recognition completed",
		"images", len(inputs),
		"total_ms", totalMs,
	)

	return results
}

func (e *Engine) recognizeFull(img image.Image, opts *types.RecognizeOption, resizeMode string) ([]types.PlateResult, error) {
	if e.detector == nil {
		return nil, fmt.Errorf("detector model not loaded")
	}
	tryDirect := shouldTryDirectPlateInFull(img)
	directEarlyStopConf := resolveFullEarlyStopConf(e.config.Rec.FullEarlyStopConf, opts)
	var direct *types.PlateResult
	if tryDirect {
		// Fast short-circuit for plate-like inputs in full mode:
		// if direct whole-image OCR is high quality, skip detector path.
		if plate, recErr := e.recognizeSingle(img, resizeMode); recErr == nil && plate != nil {
			direct = plate
			if isHighQualityFullDirect(*plate, directEarlyStopConf) {
				return []types.PlateResult{*plate}, nil
			}
		}
	}

	dets, err := e.detector.DetectPlates(img)
	if err != nil {
		return nil, fmt.Errorf("detect: %w", err)
	}
	if len(dets) == 0 {
		if direct != nil {
			return []types.PlateResult{*direct}, nil
		}
		return []types.PlateResult{}, nil
	}

	maxPlates := resolveMaxPlatesByMode(e.config.Rec.MaxPlates, e.config.Rec.FullMaxPlates, "full", opts)
	filtered := make([]PlateDetection, 0, len(dets))
	for _, det := range dets {
		if isLikelyPlateBox(img.Bounds(), det.Box) {
			filtered = append(filtered, det)
		}
	}
	if len(filtered) == 0 {
		filtered = dets
	}
	if len(filtered) > maxPlates {
		filtered = filtered[:maxPlates]
	}

	workers := max(1, min(e.config.Workers, len(filtered)))
	sem := make(chan struct{}, workers)
	ordered := make([]*types.PlateResult, len(filtered))
	var wg sync.WaitGroup
	for i, det := range filtered {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, d PlateDetection) {
			defer wg.Done()
			defer func() { <-sem }()
			plate := e.recognizePlateRegion(img, d, resizeMode)
			if plate == nil {
				return
			}
			ordered[idx] = plate
		}(i, det)
	}
	wg.Wait()

	results := make([]types.PlateResult, 0, len(filtered))
	for _, plate := range ordered {
		if plate != nil {
			results = append(results, *plate)
		}
	}

	// For plate-like inputs sent with mode=full (already cropped or near-cropped),
	// include a direct full-image recognition candidate and pick the best one.
	if tryDirect {
		if direct != nil {
			results = append(results, *direct)
		}
		if len(results) > 1 {
			if direct != nil {
				if preferred := preferNewEnergyDOverZero(results, *direct); preferred != nil {
					return []types.PlateResult{*preferred}, nil
				}
			}
			bestIdx := 0
			bestScore := scorePlateCandidate(results[0].PlateNumber, results[0].Confidence)
			for i := 1; i < len(results); i++ {
				s := scorePlateCandidate(results[i].PlateNumber, results[i].Confidence)
				if s > bestScore {
					bestScore = s
					bestIdx = i
				}
			}
			return []types.PlateResult{results[bestIdx]}, nil
		}
	}
	return results, nil
}

func shouldTryDirectPlateInFull(img image.Image) bool {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return false
	}
	ratio := float64(w) / float64(h)
	// Heuristic for pre-cropped/near-cropped plate images:
	// moderate pixel size with plate-like or near-square aspect.
	if ratio < 0.8 || ratio > 5.2 {
		return false
	}
	if w > 1280 || h > 512 {
		return false
	}
	return true
}

func isHighQualityFullDirect(p types.PlateResult, minConf float32) bool {
	if p.Type == types.PlateTypeUnknown {
		return false
	}
	if p.Confidence < minConf {
		return false
	}
	r := []rune(strings.TrimSpace(p.PlateNumber))
	if len(r) != 7 && len(r) != 8 {
		return false
	}
	return looksLikeMainlandPlatePrefix(r)
}

// shouldRunEnsemble decides if the WE cross-check is worth running. The we
// model is most useful for fixing the dominant remaining HyperLPR3 v3 failure
// modes:
//   - 7-char output that should have been an NEV 8-char (truncated tail)
//   - 8-char non-NEV output that should have been 7-char (over-decoded tail)
//   - empty / very short / structurally invalid output
//
// We deliberately keep the trigger narrow so confident clean outputs from the
// primary recognizer never get a chance to be overwritten.
func shouldRunEnsemble(plate string, conf float32) bool {
	r := []rune(strings.TrimSpace(plate))
	if len(r) == 0 {
		return true
	}
	if !looksLikeMainlandPlatePrefix(r) {
		return true
	}
	if len(r) == 7 && conf < 0.97 {
		return true
	}
	if len(r) == 8 {
		if !looksLikeNewEnergyPlate(r) && conf < 0.97 {
			return true
		}
		if conf < 0.85 {
			return true
		}
	}
	if len(r) != 7 && len(r) != 8 {
		return true
	}
	return false
}

// pickBetterPlateCandidate chooses between primary and WE candidates with very
// conservative rules so the WE model never overrides the primary's province
// recognition or middle-char content. The WE candidate may only:
//   - replace an empty/invalid primary
//   - extend a 7-char primary to an 8-char NEV (insertion at index 2 or
//     append at the tail), keeping every other char identical to primary
//   - shorten a 8-char primary to a 7-char by trimming the trailing char,
//     when the primary clearly looks like CTC over-decode
func pickBetterPlateCandidate(primary string, pConf float32, we string, weConf float32) string {
	pr := []rune(strings.TrimSpace(primary))
	wr := []rune(strings.TrimSpace(we))
	if len(wr) == 0 {
		return "primary"
	}
	if !looksLikeMainlandPlatePrefix(wr) {
		return "primary"
	}
	if len(pr) == 0 {
		return "we"
	}
	if !looksLikeMainlandPlatePrefix(pr) {
		// Primary looks broken; accept WE if WE looks healthy.
		if weConf >= 0.80 {
			return "we"
		}
		return "primary"
	}
	// Province must match for WE to win.
	if pr[0] != wr[0] {
		return "primary"
	}

	// 7 -> 8 length recovery (NEV form). We restrict to NEV plates because
	// non-NEV 7->8 is empirically more often a WE false positive than a fix.
	if len(pr) == 7 && len(wr) == 8 && weConf >= 0.86 && looksLikeNewEnergyPlate(wr) {
		if alignmentMatches(pr, wr) >= 6 {
			return "we"
		}
	}
	// 8 -> 7 length trim: WE must equal primary's first 7 chars when primary
	// is not a valid NEV (NEV-shaped should not be trimmed) and primary is
	// mid-confidence at most.
	if len(pr) == 8 && len(wr) == 7 && weConf >= 0.86 && !looksLikeNewEnergyPlate(pr) && pConf < 0.93 {
		if alignmentMatches(pr[:7], wr) >= 6 {
			return "we"
		}
	}
	return "primary"
}

// alignmentMatch checks how many positions match between two equal-or-nearly-
// equal length plates and returns true when at least minMatches positions agree.
// When the lengths differ we consider the longest common prefix + best
// alignment between primary and we (only meaningful for length 7 vs 8 cases).
func alignmentMatch(a, b []rune, minMatches int) int1bool { //nolint:revive
	return alignmentMatches(a, b) >= minMatches
}

type int1bool = bool

func alignmentMatches(a, b []rune) int {
	if len(a) == len(b) {
		matched := 0
		for i := range a {
			if a[i] == b[i] {
				matched++
			}
		}
		return matched
	}
	short, long := a, b
	if len(short) > len(long) {
		short, long = long, short
	}
	best := 0
	for shift := 0; shift+len(short) <= len(long); shift++ {
		matched := 0
		for i := range short {
			if short[i] == long[shift+i] {
				matched++
			}
		}
		if matched > best {
			best = matched
		}
	}
	return best
}

func resolveFullEarlyStopConf(defaultConf float32, opts *types.RecognizeOption) float32 {
	conf := defaultConf
	if conf <= 0 {
		conf = 0.65
	}
	if opts != nil {
		if opts.FullEarlyStopConf > 0 {
			conf = opts.FullEarlyStopConf
		} else if opts.MaxPlates == 1 {
			// Single-plate full-mode workloads prefer precision over aggressive
			// early stop; tighten default short-circuit threshold automatically.
			if conf < 0.90 {
				conf = 0.90
			}
		}
	}
	if conf < 0.50 {
		conf = 0.50
	}
	if conf > 0.99 {
		conf = 0.99
	}
	return conf
}

// If direct candidate and detector candidate only differ on D/0 in NEV marker slot,
// prefer the D variant to reduce common D->0 confusion.
func preferNewEnergyDOverZero(cands []types.PlateResult, direct types.PlateResult) *types.PlateResult {
	directRunes := []rune(strings.TrimSpace(direct.PlateNumber))
	if len(directRunes) != 8 || !looksLikeMainlandPlatePrefix(directRunes) {
		return nil
	}
	if unicode.ToUpper(directRunes[2]) != 'D' {
		return nil
	}
	for i := range cands {
		r := []rune(strings.TrimSpace(cands[i].PlateNumber))
		if len(r) != 8 || !looksLikeMainlandPlatePrefix(r) {
			continue
		}
		if unicode.ToUpper(r[2]) != 'D' {
			continue
		}
		// Compare only D/0 ambiguity at index 3, all others must match.
		if !sameExceptIndex(r, directRunes, 3) {
			continue
		}
		a := unicode.ToUpper(r[3])
		b := unicode.ToUpper(directRunes[3])
		if (a == '0' && b == 'D') || (a == 'D' && b == '0') {
			if b == 'D' {
				return &direct
			}
			return &cands[i]
		}
	}
	return nil
}

func sameExceptIndex(a, b []rune, except int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if i == except {
			continue
		}
		if unicode.ToUpper(a[i]) != unicode.ToUpper(b[i]) {
			return false
		}
	}
	return true
}

func shouldUseCropByAspect(img image.Image) bool {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return false
	}
	ratio := float64(w) / float64(h)
	minRatio := plateAspectRatioTarget * (1.0 - plateAspectRatioTolerance)
	maxRatio := plateAspectRatioTarget * (1.0 + plateAspectRatioTolerance)
	return ratio >= minRatio && ratio <= maxRatio
}

// recognizePlateRegion produces a recognition for a single detected plate.
// When the detector exposed 4 keypoints with a confident score, it warps the
// quadrilateral into a canonical 168x48 plate canvas and prefers the warped
// recognition over the axis-aligned crop. This is the most effective single
// lever to fix tilted/perspective-distorted plates in full mode.
const (
	plateWarpOutW       = 168
	plateWarpOutH       = 48
	plateWarpScoreFloor = float32(0.30)
	plateWarpFastFloor  = float32(0.65)
	plateWarpAcceptDiff = float32(0.02)
)

func (e *Engine) recognizePlateRegion(img image.Image, det PlateDetection, resizeMode string) *types.PlateResult {
	box := det.Box
	useWarp := det.HasKeys && det.Score >= plateWarpScoreFloor && keypointsValid(det.Keypoints, 6.0)

	// Fast path: high-confidence keypoints alone are reliable; skip the crop
	// inference entirely to save a recognizer call. We only second-guess this
	// when the warped result itself looks weak.
	if useWarp && det.Score >= plateWarpFastFloor {
		warped := warpPerspectivePlate(img, det.Keypoints, plateWarpOutW, plateWarpOutH)
		warpPlate, _ := e.recognizeSingle(warped, resizeMode)
		if warpPlate != nil && isAcceptableWarpResult(*warpPlate) {
			return warpPlate
		}
		// Length-boundary recovery: when the primary warp result looks weakly
		// classified (especially 7<->8 char drift, the dominant remaining
		// failure mode), try one extra warp with horizontally expanded
		// keypoints. The extra inference is gated to avoid throughput hits.
		if warpPlate != nil {
			if alt := e.tryExpandedKeypointWarp(img, det, *warpPlate, resizeMode); alt != nil {
				return alt
			}
			return warpPlate
		}
	}

	cropImg := cropImage(img, box[0], box[1], box[2], box[3])
	cropPlate, _ := e.recognizeSingle(cropImg, resizeMode)

	if !useWarp {
		return cropPlate
	}

	warped := warpPerspectivePlate(img, det.Keypoints, plateWarpOutW, plateWarpOutH)
	warpPlate, _ := e.recognizeSingle(warped, resizeMode)
	if warpPlate == nil {
		return cropPlate
	}
	if cropPlate == nil {
		return warpPlate
	}
	// Pick the better candidate by structure-aware score; the warp path also
	// gets a small constant bonus because it tends to be far cleaner for the
	// recognizer once distortion is removed.
	cropScore := scorePlateCandidate(cropPlate.PlateNumber, cropPlate.Confidence)
	warpScore := scorePlateCandidate(warpPlate.PlateNumber, warpPlate.Confidence) + 1.0
	if warpPlate.Confidence > cropPlate.Confidence+plateWarpAcceptDiff || warpScore > cropScore {
		return warpPlate
	}
	return cropPlate
}

// tryExpandedKeypointWarp performs at most one extra recognizer inference with
// horizontally widened keypoints. The expansion direction is chosen by the
// length-mismatch hypothesis on the primary warp result:
//
//   - 7 chars with weak tail confidence -> expected NEV 8: pull right edge out
//   - 8 chars with weak last-char and not NEV-shape -> tail noise: pull right edge in
//
// Anything else returns nil to keep the fast path cheap.
func (e *Engine) tryExpandedKeypointWarp(img image.Image, det PlateDetection, base types.PlateResult, resizeMode string) *types.PlateResult {
	r := []rune(strings.TrimSpace(base.PlateNumber))
	if len(r) != 7 && len(r) != 8 {
		return nil
	}
	if !looksLikeMainlandPlatePrefix(r) {
		return nil
	}
	expand := 0.0
	switch len(r) {
	case 7:
		// Likely truncated NEV (8-char) when ends with digits and last char is unstable.
		// Trigger more eagerly: detector boxes on plate-like crops often clip
		// the rightmost NEV character even at high recognition confidence.
		expand = 0.10
	case 8:
		if looksLikeNewEnergyPlate(r) {
			return nil
		}
		expand = -0.06
	}
	if expand == 0 {
		return nil
	}
	kp := expandKeypointsHorizontally(det.Keypoints, expand)
	if !keypointsValid(kp, 6.0) {
		return nil
	}
	warped := warpPerspectivePlate(img, kp, plateWarpOutW, plateWarpOutH)
	cand, _ := e.recognizeSingle(warped, resizeMode)
	if cand == nil {
		return nil
	}
	baseScore := scorePlateCandidate(base.PlateNumber, base.Confidence)
	candScore := scorePlateCandidate(cand.PlateNumber, cand.Confidence)
	// Accept the alternate warp when its candidate score (which already encodes
	// length/structural priors) clearly outranks the original. The +0.6 cushion
	// keeps the original answer when both look comparable.
	if candScore > baseScore+0.6 && cand.Confidence >= 0.80 {
		return cand
	}
	return nil
}

// expandKeypointsHorizontally returns a new quad whose left/right edges are
// pushed outward (positive ratio) or inward (negative ratio) along the local
// horizontal axis defined by (TR-TL) and (BR-BL).
func expandKeypointsHorizontally(kp PlateKeypoints, ratio float64) PlateKeypoints {
	if ratio == 0 {
		return kp
	}
	tl, tr, br, bl := kp[0], kp[1], kp[2], kp[3]
	topDx := tr[0] - tl[0]
	topDy := tr[1] - tl[1]
	botDx := br[0] - bl[0]
	botDy := br[1] - bl[1]
	out := PlateKeypoints{
		{tl[0] - topDx*ratio, tl[1] - topDy*ratio},
		{tr[0] + topDx*ratio, tr[1] + topDy*ratio},
		{br[0] + botDx*ratio, br[1] + botDy*ratio},
		{bl[0] - botDx*ratio, bl[1] - botDy*ratio},
	}
	return out
}

// isAcceptableWarpResult is the gate for trusting a warp-only recognition. We
// require a structurally valid mainland prefix and a healthy confidence so that
// we only short-circuit when the result is clearly stable.
func isAcceptableWarpResult(p types.PlateResult) bool {
	if p.Type == types.PlateTypeUnknown {
		return false
	}
	if p.Confidence < 0.85 {
		return false
	}
	r := []rune(strings.TrimSpace(p.PlateNumber))
	if len(r) != 7 && len(r) != 8 {
		return false
	}
	return looksLikeMainlandPlatePrefix(r)
}

func (e *Engine) retryCropWithTweaks(img image.Image, resizeMode string) *types.PlateResult {
	// Keep retry path short and generic: each variant is cheap and broadly useful.
	variants := []image.Image{
		unsharpMask(img),
		adaptiveGrayBoost(img),
		enhanceGrayContrast(img),
		trimWhiteFrame(img),
		upscaleImage(img, 2),
	}
	for _, v := range variants {
		plate, err := e.recognizeSingle(v, resizeMode)
		if err != nil || plate == nil {
			continue
		}
		// Guardrail: retry path should only accept structurally reliable outputs.
		// This avoids replacing "no result" with a clearly wrong noisy candidate.
		if plate.Type == types.PlateTypeUnknown || plate.Confidence < max(0.68, e.config.Rec.MinConfidence) {
			continue
		}
		return plate
	}
	return nil
}

// decodeInput converts an ImageInput to a Go image.Image.
func (e *Engine) decodeInput(input types.ImageInput) (image.Image, error) {
	switch input.Type {
	case "base64":
		data, err := base64.StdEncoding.DecodeString(input.Data)
		if err != nil {
			return nil, fmt.Errorf("base64 decode: %w", err)
		}
		return decodeImage(bytes.NewReader(data))

	case "path":
		return loadImage(input.Data)

	case "url":
		return e.fetchImageFromURL(input.Data)

	default:
		return nil, fmt.Errorf("unknown input type: %s", input.Type)
	}
}

func (e *Engine) fetchImageFromURL(rawURL string) (image.Image, error) {
	if !e.config.URL.Enabled {
		return nil, fmt.Errorf("url input is disabled")
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse url: %w", err)
	}
	if !e.isAllowedScheme(parsed.Scheme) {
		return nil, fmt.Errorf("unsupported url scheme: %s", parsed.Scheme)
	}
	if parsed.Hostname() == "" {
		return nil, fmt.Errorf("url host is empty")
	}

	if e.config.URL.BlockPrivateIP {
		if err := ensurePublicHost(parsed.Hostname()); err != nil {
			return nil, err
		}
	}

	e.urlFetchCh <- struct{}{}
	defer func() { <-e.urlFetchCh }()

	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "platex/1.0")

	retries := max(0, e.config.URL.MaxFetchRetries)
	backoff := time.Duration(max(20, e.config.URL.RetryBackoffMs)) * time.Millisecond
	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		img, fetchErr := e.fetchImageFromURLOnce(req)
		if fetchErr == nil {
			return img, nil
		}
		lastErr = fetchErr
		if !shouldRetryStatusError(lastErr) {
			return nil, lastErr
		}
		if attempt < retries {
			time.Sleep(backoff * time.Duration(attempt+1))
		}
	}
	return nil, lastErr
}

func (e *Engine) fetchImageFromURLOnce(req *http.Request) (image.Image, error) {
	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("http status: %d", resp.StatusCode)
	}
	if resp.ContentLength > e.config.URL.MaxImageBytes {
		return nil, fmt.Errorf("image too large: %d > %d", resp.ContentLength, e.config.URL.MaxImageBytes)
	}

	limited := io.LimitReader(resp.Body, e.config.URL.MaxImageBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}
	if int64(len(data)) > e.config.URL.MaxImageBytes {
		return nil, fmt.Errorf("image too large: body exceeds %d bytes", e.config.URL.MaxImageBytes)
	}
	img, err := decodeImage(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return img, nil
}

func (e *Engine) isAllowedScheme(scheme string) bool {
	normalized := strings.ToLower(strings.TrimSpace(scheme))
	for _, allowed := range e.config.URL.AllowedSchemes {
		if normalized == strings.ToLower(strings.TrimSpace(allowed)) {
			return true
		}
	}
	return false
}

func ensurePublicHost(host string) error {
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("resolve host: %w", err)
	}
	if len(ips) == 0 {
		return fmt.Errorf("resolve host: no ip found")
	}
	for _, ip := range ips {
		if isPrivateOrLocalIP(ip) {
			return fmt.Errorf("blocked private/local address: %s", ip.String())
		}
	}
	return nil
}

func isPrivateOrLocalIP(ip net.IP) bool {
	if ip.IsLoopback() || ip.IsLinkLocalMulticast() || ip.IsLinkLocalUnicast() || ip.IsUnspecified() || ip.IsMulticast() {
		return true
	}
	if v4 := ip.To4(); v4 != nil {
		switch {
		case v4[0] == 10:
			return true
		case v4[0] == 172 && v4[1] >= 16 && v4[1] <= 31:
			return true
		case v4[0] == 192 && v4[1] == 168:
			return true
		case v4[0] == 169 && v4[1] == 254:
			return true
		case v4[0] == 127:
			return true
		}
		return false
	}

	// IPv6 private and local ranges
	if len(ip) == net.IPv6len {
		if ip[0]&0xfe == 0xfc { // fc00::/7 unique local
			return true
		}
		if ip[0] == 0xfe && (ip[1]&0xc0) == 0x80 { // fe80::/10 link local
			return true
		}
		if ip.IsLoopback() {
			return true
		}
	}
	return false
}

func resolveMaxPlatesByMode(defaultMax, defaultFullMax int, mode string, opts *types.RecognizeOption) int {
	maxPlates := defaultMax
	if mode == "full" && defaultFullMax > 0 {
		maxPlates = defaultFullMax
	}
	if opts != nil && opts.MaxPlates > 0 {
		maxPlates = opts.MaxPlates
	}
	if maxPlates <= 0 {
		return 10
	}
	return maxPlates
}

func isLikelyPlateBox(bounds image.Rectangle, box [4]int) bool {
	bw := box[2] - box[0]
	bh := box[3] - box[1]
	if bw < 8 || bh < 8 {
		return false
	}
	ratio := float64(bw) / float64(bh)
	if ratio < 1.6 || ratio > 6.5 {
		return false
	}
	imgArea := float64(bounds.Dx() * bounds.Dy())
	boxArea := float64(bw * bh)
	if imgArea <= 0 {
		return false
	}
	areaRatio := boxArea / imgArea
	return areaRatio >= 0.003 && areaRatio <= 0.60
}

func shouldRetryStatusError(err error) bool {
	msg := err.Error()
	if strings.Contains(msg, "http fetch:") {
		return true
	}
	if !strings.HasPrefix(msg, "http status: ") {
		return false
	}
	codeStr := strings.TrimPrefix(msg, "http status: ")
	code, convErr := strconv.Atoi(strings.TrimSpace(codeStr))
	if convErr != nil {
		return false
	}
	if code == http.StatusTooManyRequests {
		return true
	}
	return code >= 500 && code <= 599
}

func hasBlueDominanceOverYellow(img image.Image) bool {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return false
	}

	// Focus center region to reduce border/background interference.
	marginX := w / 8
	marginY := h / 6
	x0, x1 := bounds.Min.X+marginX, bounds.Max.X-marginX
	y0, y1 := bounds.Min.Y+marginY, bounds.Max.Y-marginY
	if x1 <= x0 || y1 <= y0 {
		x0, x1 = bounds.Min.X, bounds.Max.X
		y0, y1 = bounds.Min.Y, bounds.Max.Y
	}

	var blueCount, yellowCount, valid int
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			r16, g16, b16, _ := img.At(x, y).RGBA()
			r, g, b := float64(r16>>8), float64(g16>>8), float64(b16>>8)
			hue, sat, val := rgbToHSV(r, g, b)
			if val < 40 || sat < 35 {
				continue
			}
			valid++
			if hue >= 170 && hue <= 270 {
				blueCount++
				continue
			}
			if hue >= 35 && hue <= 80 {
				yellowCount++
			}
		}
	}
	if valid < 100 {
		return false
	}
	blueRatio := float64(blueCount) / float64(valid)
	yellowRatio := float64(yellowCount) / float64(valid)
	return blueRatio >= 0.18 && blueRatio > yellowRatio*1.6
}

// GetStats returns current engine statistics.
func (e *Engine) GetStats() *types.StatsData {
	total := e.totalImages.Load()
	totalMs := e.totalTimeMs.Load()

	var avgLatency float64
	if total > 0 {
		avgLatency = float64(totalMs) / float64(total)
	}

	return &types.StatsData{
		TotalImages:  total,
		TotalPlates:  e.totalPlates.Load(),
		AvgLatencyMs: avgLatency,
		SuccessRate:  1.0, // TODO: Track failures
	}
}

// GetRuntimeConfig returns key runtime config values for observability.
func (e *Engine) GetRuntimeConfig() map[string]interface{} {
	return map[string]interface{}{
		"mode":                      e.config.Mode,
		"workers":                   e.config.Workers,
		"submit_timeout_ms":         e.config.SubmitTimeoutMs,
		"onnx_threads_per_session":  e.config.ONNX.ThreadsPerSession,
		"url_max_fetch_concurrency": e.config.URL.MaxFetchConcurrency,
		"full_early_stop_conf":      e.config.Rec.FullEarlyStopConf,
		"model_pool_size":           modelPoolSizeForInfo(),
	}
}

// Close shuts down the engine and releases resources.
func (e *Engine) Close() {
	slog.Info("Shutting down LPR engine")
	close(e.workerCh)
	e.wg.Wait()

	if e.recognizer != nil {
		e.recognizer.Close()
	}
	if e.recognizerWE != nil {
		e.recognizerWE.Close()
	}
	if e.detector != nil {
		e.detector.Close()
	}
	if e.color != nil {
		e.color.Close()
	}

	_ = destroyONNXRuntime()
	slog.Info("LPR engine shutdown complete")
}
