package engine

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"sort"

	"github.com/vesaa/platex/internal/config"
)

// Detector handles plate region detection from full images.
type Detector struct {
	model       *Model
	inputWidth  int
	inputHeight int
	confThr     float32
	iouThr      float32
	maxCand     int
	// tileSize > 0 enables tiled detection on images larger than ~1.25x it.
	tileSize int
}

// NewDetector creates a detector for full-image plate localization.
func NewDetector(modelPath string, threads, optLevel int, cfg config.DetectionConfig) (*Detector, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("detector model path is empty")
	}
	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load detector model: %w", err)
	}
	return &Detector{
		model:       model,
		inputWidth:  320,
		inputHeight: 320,
		confThr:     normalizeConfThreshold(cfg.ConfThreshold),
		iouThr:      normalizeIoUThreshold(cfg.IoUThreshold),
		maxCand:     normalizeMaxCandidates(cfg.MaxCandidates),
		tileSize:    cfg.TileSize,
	}, nil
}

// PlateDetection is a single detection result with optional 4-corner keypoints.
type PlateDetection struct {
	Box       [4]int
	Score     float32
	HasKeys   bool
	Keypoints PlateKeypoints
}

// Detect returns detected plate boxes as [x1,y1,x2,y2] on the original image.
// It is kept for backward compatibility with code paths that don't need
// keypoints; new code should prefer DetectPlates.
func (d *Detector) Detect(img image.Image) ([][4]int, error) {
	dets, err := d.DetectPlates(img)
	if err != nil {
		return nil, err
	}
	out := make([][4]int, 0, len(dets))
	for _, det := range dets {
		out = append(out, det.Box)
	}
	return out, nil
}

// DetectPlates returns the full detection result including keypoints when the
// detector head exposes them. Currently parses the y5fu_320x output layout
// `[cx, cy, w, h, obj, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y, cls...]`.
func (d *Detector) DetectPlates(img image.Image) ([]PlateDetection, error) {
	if d.model == nil {
		return nil, fmt.Errorf("detector model not loaded")
	}
	dets, err := d.detectRegion(img)
	if err != nil {
		return nil, err
	}
	tiles := detectorTiles(img.Bounds(), d.tileSize)
	if len(tiles) == 0 {
		return dets, nil
	}
	// The detector sees a fixed 320x320 input, so on large scenes (e.g.
	// 1944x2592 roadside cameras) a plate shrinks below ~20px and is missed.
	// Also detect on overlapping tiles and merge. A tile detection touching an
	// inner tile edge is a cut-off plate; the overlap guarantees the whole
	// plate appears in a neighbouring tile (or in the full-image pass).
	for _, t := range tiles {
		tileDets, err := d.detectRegion(subImage(img, t))
		if err != nil {
			return nil, err
		}
		for _, det := range tileDets {
			if !touchesInnerEdge(det.Box, t, img.Bounds()) {
				dets = append(dets, det)
			}
		}
	}
	return mergeDetections(dets, d.iouThr), nil
}

// detectRegion runs the detector once on img (any bounds; results are in
// img's coordinate space).
func (d *Detector) detectRegion(img image.Image) ([]PlateDetection, error) {

	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("invalid image size")
	}

	// Keep aspect ratio via letterbox, which is standard for YOLO-style detectors.
	letterboxed, scale, padX, padY := letterboxForDetector(img, d.inputWidth, d.inputHeight)
	input := imageToTensorDetector(letterboxed)

	out, err := d.model.RunInference(input)
	if err != nil {
		return nil, err
	}

	rows := decodeDetectorRows(out, d.model.GetOutputShape(), d.confThr, d.iouThr, d.maxCand)
	if len(rows) == 0 {
		return []PlateDetection{}, nil
	}

	// Expand detected boxes by a small margin to avoid clipping plate edge
	// characters (province letter on the left, last digit on the right).
	// The expansion is proportional to box dimensions.
	const boxExpandX = 0.08
	const boxExpandY = 0.06

	restored := make([]PlateDetection, 0, len(rows))
	for _, b := range rows {
		x1 := int(math.Round((float64(b.x1)-padX)/scale)) + srcBounds.Min.X
		y1 := int(math.Round((float64(b.y1)-padY)/scale)) + srcBounds.Min.Y
		x2 := int(math.Round((float64(b.x2)-padX)/scale)) + srcBounds.Min.X
		y2 := int(math.Round((float64(b.y2)-padY)/scale)) + srcBounds.Min.Y

		bw := x2 - x1
		bh := y2 - y1
		dx := int(math.Round(float64(bw) * boxExpandX))
		dy := int(math.Round(float64(bh) * boxExpandY))
		x1 -= dx
		y1 -= dy
		x2 += dx
		y2 += dy

		if x1 < srcBounds.Min.X {
			x1 = srcBounds.Min.X
		}
		if y1 < srcBounds.Min.Y {
			y1 = srcBounds.Min.Y
		}
		if x2 > srcBounds.Max.X {
			x2 = srcBounds.Max.X
		}
		if y2 > srcBounds.Max.Y {
			y2 = srcBounds.Max.Y
		}
		if x2-x1 < 8 || y2-y1 < 8 {
			continue
		}

		det := PlateDetection{
			Box:   [4]int{x1, y1, x2, y2},
			Score: b.score,
		}
		if b.hasKeys {
			var raw [4][2]float64
			for i := 0; i < 4; i++ {
				kx := (float64(b.keys[i][0]) - padX) / scale
				ky := (float64(b.keys[i][1]) - padY) / scale
				kx += float64(srcBounds.Min.X)
				ky += float64(srcBounds.Min.Y)
				raw[i] = [2]float64{kx, ky}
			}
			det.Keypoints = orderPlateKeypoints(raw)
			det.HasKeys = true
		}
		restored = append(restored, det)
	}
	return restored, nil
}

// Close releases detector resources.
func (d *Detector) Close() {
	if d.model != nil {
		d.model.Close()
	}
}

type detBox struct {
	x1, y1  float32
	x2, y2  float32
	score   float32
	hasKeys bool
	keys    [4][2]float32
}

func letterboxForDetector(img image.Image, width, height int) (image.Image, float64, float64, float64) {
	srcBounds := img.Bounds()
	srcW := float64(srcBounds.Dx())
	srcH := float64(srcBounds.Dy())
	targetW := float64(width)
	targetH := float64(height)

	scale := targetW / srcW
	if srcH*scale > targetH {
		scale = targetH / srcH
	}
	newW := int(srcW * scale)
	newH := int(srcH * scale)

	resized := resizeImage(img, newW, newH)
	canvas := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			canvas.Set(x, y, color.NRGBA{R: 114, G: 114, B: 114, A: 255})
		}
	}
	padX := (width - newW) / 2
	padY := (height - newH) / 2
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			canvas.Set(x+padX, y+padY, resized.At(x, y))
		}
	}
	return canvas, scale, float64(padX), float64(padY)
}

func imageToTensorDetector(img image.Image) []float32 {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	tensor := make([]float32, 3*w*h)
	ch := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
			i := y*w + x
			tensor[i] = float32(r>>8) / 255.0
			tensor[ch+i] = float32(g>>8) / 255.0
			tensor[2*ch+i] = float32(bl>>8) / 255.0
		}
	}
	return tensor
}

// decodeDetectorRows parses the detector head into [detBox] entries while
// preserving keypoint information when the head includes them. It supports
// two common layouts:
//
//   - 6-column classic YOLO: [cx, cy, w, h, obj, cls0]
//   - 15-column yolov5face: [cx, cy, w, h, obj, kp1x, kp1y, kp2x, kp2y,
//     kp3x, kp3y, kp4x, kp4y, cls0, cls1]
//
// keypointHead is auto-detected when numCols >= 13.
func decodeDetectorRows(output []float32, shape []int64, confThr, iouThr float32, maxCandidates int) []detBox {
	if len(output) < 6 {
		return nil
	}

	// Expected common layouts: [1, N, C] or [N, C].
	var numRows, numCols int
	if len(shape) >= 3 {
		numRows = int(shape[len(shape)-2])
		numCols = int(shape[len(shape)-1])
	} else if len(shape) == 2 {
		numRows = int(shape[0])
		numCols = int(shape[1])
	}
	if numRows <= 0 || numCols <= 0 || numRows*numCols > len(output) {
		// Fallback: guess a row width commonly used by YOLO heads.
		numCols = 6
		numRows = len(output) / numCols
	}
	if numCols < 6 {
		return nil
	}

	hasKeypoints := numCols >= 13

	boxes := make([]detBox, 0, numRows)
	for i := 0; i < numRows; i++ {
		base := i * numCols
		cx := output[base]
		cy := output[base+1]
		w := output[base+2]
		h := output[base+3]
		obj := output[base+4]
		if obj <= 0 {
			continue
		}

		// Class confidence: skip the keypoint span when present.
		bestCls := float32(0.0)
		clsStart := 5
		if hasKeypoints {
			clsStart = 13
		}
		if numCols > clsStart {
			for c := clsStart; c < numCols; c++ {
				if output[base+c] > bestCls {
					bestCls = output[base+c]
				}
			}
		} else {
			bestCls = 1.0
		}
		score := obj * bestCls
		if score < confThr {
			continue
		}
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		row := detBox{x1: x1, y1: y1, x2: x2, y2: y2, score: score}
		if hasKeypoints {
			row.hasKeys = true
			for k := 0; k < 4; k++ {
				row.keys[k][0] = output[base+5+2*k]
				row.keys[k][1] = output[base+5+2*k+1]
			}
		}
		boxes = append(boxes, row)
	}

	if len(boxes) == 0 {
		return boxes
	}
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].score > boxes[j].score
	})
	boxes = nms(boxes, iouThr)
	if maxCandidates > 0 && len(boxes) > maxCandidates {
		boxes = boxes[:maxCandidates]
	}
	return boxes
}

func nms(boxes []detBox, iouThr float32) []detBox {
	kept := make([]detBox, 0, len(boxes))
	suppressed := make([]bool, len(boxes))
	for i := 0; i < len(boxes); i++ {
		if suppressed[i] {
			continue
		}
		kept = append(kept, boxes[i])
		for j := i + 1; j < len(boxes); j++ {
			if suppressed[j] {
				continue
			}
			if iou(boxes[i], boxes[j]) > iouThr {
				suppressed[j] = true
			}
		}
	}
	return kept
}

func iou(a, b detBox) float32 {
	ix1 := maxf(a.x1, b.x1)
	iy1 := maxf(a.y1, b.y1)
	ix2 := minf(a.x2, b.x2)
	iy2 := minf(a.y2, b.y2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	aa := (a.x2 - a.x1) * (a.y2 - a.y1)
	ab := (b.x2 - b.x1) * (b.y2 - b.y1)
	union := aa + ab - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func normalizeConfThreshold(v float32) float32 {
	if v <= 0 || v >= 1 {
		return 0.30
	}
	return v
}

func normalizeIoUThreshold(v float32) float32 {
	if v <= 0 || v >= 1 {
		return 0.45
	}
	return v
}

func normalizeMaxCandidates(v int) int {
	if v <= 0 {
		return 50
	}
	return v
}

const detectorTileOverlap = 0.3

// detectorTiles splits bounds into an even grid of square-ish tiles of about
// size px overlapping by detectorTileOverlap. Returns nil when the image is
// small enough for a single pass.
func detectorTiles(b image.Rectangle, size int) []image.Rectangle {
	if size <= 0 || max(b.Dx(), b.Dy()) <= size*5/4 {
		return nil
	}
	xs := tileStarts(b.Min.X, b.Dx(), size)
	ys := tileStarts(b.Min.Y, b.Dy(), size)
	tiles := make([]image.Rectangle, 0, len(xs)*len(ys))
	for _, y := range ys {
		for _, x := range xs {
			tiles = append(tiles, image.Rect(x, y, min(x+size, b.Max.X), min(y+size, b.Max.Y)).Intersect(b))
		}
	}
	return tiles
}

func tileStarts(origin, length, size int) []int {
	if length <= size {
		return []int{origin}
	}
	stride := float64(size) * (1 - detectorTileOverlap)
	n := int(math.Ceil(float64(length-size)/stride)) + 1
	step := float64(length-size) / float64(n-1)
	starts := make([]int, n)
	for i := range starts {
		starts[i] = origin + int(math.Round(step*float64(i)))
	}
	return starts
}

// subImage returns the rect part of img keeping img's coordinate space.
func subImage(img image.Image, rect image.Rectangle) image.Image {
	if s, ok := img.(interface {
		SubImage(image.Rectangle) image.Image
	}); ok {
		return s.SubImage(rect)
	}
	dst := image.NewNRGBA(rect)
	draw.Draw(dst, rect, img, rect.Min, draw.Src)
	return dst
}

// touchesInnerEdge reports whether box reaches an edge of tile that is not
// also an edge of the full image.
func touchesInnerEdge(box [4]int, tile, full image.Rectangle) bool {
	const m = 2
	return (tile.Min.X > full.Min.X && box[0] <= tile.Min.X+m) ||
		(tile.Min.Y > full.Min.Y && box[1] <= tile.Min.Y+m) ||
		(tile.Max.X < full.Max.X && box[2] >= tile.Max.X-m) ||
		(tile.Max.Y < full.Max.Y && box[3] >= tile.Max.Y-m)
}

// mergeDetections keeps the highest-scoring detection among overlapping ones
// (IoU above iouThr, or one box mostly inside the other).
func mergeDetections(dets []PlateDetection, iouThr float32) []PlateDetection {
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	kept := make([]PlateDetection, 0, len(dets))
	for _, d := range dets {
		dup := false
		for _, k := range kept {
			if boxIoU(d.Box, k.Box) > iouThr || boxIoMin(d.Box, k.Box) > 0.7 {
				dup = true
				break
			}
		}
		if !dup {
			kept = append(kept, d)
		}
	}
	return kept
}

func boxArea(b [4]int) int { return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) }

func boxInter(a, b [4]int) int {
	return boxArea([4]int{max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])})
}

func boxIoU(a, b [4]int) float32 {
	in := boxInter(a, b)
	u := boxArea(a) + boxArea(b) - in
	if u <= 0 {
		return 0
	}
	return float32(in) / float32(u)
}

func boxIoMin(a, b [4]int) float32 {
	m := min(boxArea(a), boxArea(b))
	if m <= 0 {
		return 0
	}
	return float32(boxInter(a, b)) / float32(m)
}
