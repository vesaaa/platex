package engine

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"
)

// Detector handles plate region detection from full images.
type Detector struct {
	model       *Model
	inputWidth  int
	inputHeight int
}

// NewDetector creates a detector for full-image plate localization.
func NewDetector(modelPath string, threads, optLevel int) (*Detector, error) {
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
	}, nil
}

// Detect returns detected plate boxes as [x1,y1,x2,y2] on the original image.
func (d *Detector) Detect(img image.Image) ([][4]int, error) {
	if d.model == nil {
		return nil, fmt.Errorf("detector model not loaded")
	}

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

	boxes := decodeYOLOLikeOutput(out, d.model.GetOutputShape(), 0.30, 0.45)
	if len(boxes) == 0 {
		return [][4]int{}, nil
	}

	restored := make([][4]int, 0, len(boxes))
	for _, b := range boxes {
		x1 := int(math.Round((float64(b.x1)-padX)/scale)) + srcBounds.Min.X
		y1 := int(math.Round((float64(b.y1)-padY)/scale)) + srcBounds.Min.Y
		x2 := int(math.Round((float64(b.x2)-padX)/scale)) + srcBounds.Min.X
		y2 := int(math.Round((float64(b.y2)-padY)/scale)) + srcBounds.Min.Y

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
		restored = append(restored, [4]int{x1, y1, x2, y2})
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
	x1, y1 float32
	x2, y2 float32
	score  float32
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

func decodeYOLOLikeOutput(output []float32, shape []int64, confThr, iouThr float32) []detBox {
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

		bestCls := float32(1.0)
		if numCols > 5 {
			bestCls = float32(0.0)
			for c := 5; c < numCols; c++ {
				if output[base+c] > bestCls {
					bestCls = output[base+c]
				}
			}
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
		boxes = append(boxes, detBox{x1: x1, y1: y1, x2: x2, y2: y2, score: score})
	}

	if len(boxes) == 0 {
		return boxes
	}
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].score > boxes[j].score
	})
	return nms(boxes, iouThr)
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
