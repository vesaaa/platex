package engine

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg" // Register JPEG decoder
	_ "image/png"  // Register PNG decoder
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vesaa/platex/internal/config"
	"github.com/vesaa/platex/internal/types"
)

// Engine is the main license plate recognition engine.
type Engine struct {
	recognizer *Recognizer
	color      *ColorClassifier
	config     *config.EngineConfig
	workerCh   chan *recognizeJob
	wg         sync.WaitGroup

	// Stats
	totalImages atomic.Int64
	totalPlates atomic.Int64
	totalTimeMs atomic.Int64
}

// recognizeJob represents a unit of work for the worker pool.
type recognizeJob struct {
	img      image.Image
	resultCh chan *types.PlateResult
	errCh    chan error
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

	e := &Engine{
		recognizer: recognizer,
		color:      colorCls,
		config:     cfg,
		workerCh:   make(chan *recognizeJob, cfg.Workers*2),
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
		result, err := e.recognizeSingle(job.img)
		if err != nil {
			job.errCh <- err
		} else {
			job.resultCh <- result
		}
	}

	slog.Debug("Worker stopped", "id", id)
}

// recognizeSingle performs recognition on a single image.
func (e *Engine) recognizeSingle(img image.Image) (*types.PlateResult, error) {
	if e.recognizer == nil {
		return nil, fmt.Errorf("recognizer model not loaded")
	}

	// Step 1: Character recognition
	plateNumber, charConfs, confidence, err := e.recognizer.Recognize(img)
	if err != nil {
		return nil, fmt.Errorf("recognition: %w", err)
	}

	if plateNumber == "" || confidence < e.config.Rec.MinConfidence {
		// Not an error - just no plate detected with sufficient confidence
		return nil, nil
	}

	// Step 2: Color classification
	colorCode, _ := e.color.Classify(img)

	// Step 3: Determine plate type
	plateType := classifyPlateType(plateNumber)

	colorName := "其他"
	if name, ok := types.ColorNames[types.PlateColor(colorCode)]; ok {
		colorName = name
	}

	return &types.PlateResult{
		PlateNumber:     plateNumber,
		Color:           types.PlateColor(colorCode),
		ColorName:       colorName,
		Confidence:      confidence,
		CharConfidences: charConfs,
		Type:            plateType,
	}, nil
}

// RecognizeBatch processes a batch of image inputs.
func (e *Engine) RecognizeBatch(inputs []types.ImageInput, opts *types.RecognizeOption) []types.ImageResult {
	start := time.Now()
	results := make([]types.ImageResult, len(inputs))

	minConf := e.config.Rec.MinConfidence
	if opts != nil && opts.MinConfidence > 0 {
		minConf = opts.MinConfidence
	}
	_ = minConf // Will be used when model is integrated

	// Set resize mode: default to auto
	if e.recognizer != nil {
		if opts != nil && opts.ResizeMode != "" {
			e.recognizer.resizeMode = opts.ResizeMode
		} else {
			e.recognizer.resizeMode = "auto"
		}
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

			// Submit to worker pool
			job := &recognizeJob{
				img:      img,
				resultCh: make(chan *types.PlateResult, 1),
				errCh:    make(chan error, 1),
			}

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
			default:
				result.Error = "worker pool full, try again later"
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
		// TODO: Implement HTTP fetch for URL type
		return nil, fmt.Errorf("url type not yet implemented")

	default:
		return nil, fmt.Errorf("unknown input type: %s", input.Type)
	}
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

// Close shuts down the engine and releases resources.
func (e *Engine) Close() {
	slog.Info("Shutting down LPR engine")
	close(e.workerCh)
	e.wg.Wait()

	if e.recognizer != nil {
		e.recognizer.Close()
	}
	if e.color != nil {
		e.color.Close()
	}

	_ = destroyONNXRuntime()
	slog.Info("LPR engine shutdown complete")
}
