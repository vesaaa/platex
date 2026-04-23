// Package engine implements the license plate recognition inference pipeline.
package engine

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"os"

	"golang.org/x/image/draw"
)

// resizeImage resizes an image to the target dimensions using high-quality interpolation.
func resizeImage(src image.Image, width, height int) *image.NRGBA {
	dst := image.NewNRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)
	return dst
}

// loadImage reads and decodes an image from a file path.
func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()
	return decodeImage(f)
}

// decodeImage decodes an image from a reader.
func decodeImage(r io.Reader) (image.Image, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// imageToTensorCHW converts an image to a float32 tensor in CHW format (channels, height, width).
// The tensor is normalized to [0, 1] by default, or with custom mean/std if provided.
func imageToTensorCHW(img image.Image, width, height int, mean, std [3]float32) []float32 {
	resized := resizeImage(img, width, height)
	tensor := make([]float32, 3*height*width)
	channelSize := height * width

	bounds := resized.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			idx := (y-bounds.Min.Y)*width + (x - bounds.Min.X)

			// Normalize: (pixel/255 - mean) / std
			tensor[0*channelSize+idx] = (float32(r>>8)/255.0 - mean[0]) / std[0] // R
			tensor[1*channelSize+idx] = (float32(g>>8)/255.0 - mean[1]) / std[1] // G
			tensor[2*channelSize+idx] = (float32(b>>8)/255.0 - mean[2]) / std[2] // B
		}
	}
	return tensor
}

// imageToTensorCHWSimple converts an image to a float32 tensor normalized to [0, 1].
func imageToTensorCHWSimple(img image.Image, width, height int) []float32 {
	mean := [3]float32{0.0, 0.0, 0.0}
	std := [3]float32{1.0, 1.0, 1.0}
	return imageToTensorCHW(img, width, height, mean, std)
}

// cropImage extracts a rectangular region from an image.
func cropImage(src image.Image, x1, y1, x2, y2 int) image.Image {
	rect := image.Rect(x1, y1, x2, y2)
	dst := image.NewNRGBA(image.Rect(0, 0, x2-x1, y2-y1))
	draw.Copy(dst, image.Point{}, src, rect, draw.Over, nil)
	return dst
}

// getDominantColor analyzes an image to determine the dominant plate color.
// This is a simple heuristic fallback when the ML color model is not available.
func getDominantColor(img image.Image) int {
	bounds := img.Bounds()
	var totalR, totalG, totalB float64
	var count float64

	// Sample the center region of the plate
	marginX := (bounds.Max.X - bounds.Min.X) / 6
	marginY := (bounds.Max.Y - bounds.Min.Y) / 4

	for y := bounds.Min.Y + marginY; y < bounds.Max.Y-marginY; y++ {
		for x := bounds.Min.X + marginX; x < bounds.Max.X-marginX; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			totalR += float64(r >> 8)
			totalG += float64(g >> 8)
			totalB += float64(b >> 8)
			count++
		}
	}

	if count == 0 {
		return 0 // Other
	}

	avgR := totalR / count
	avgG := totalG / count
	avgB := totalB / count

	// Convert to HSV for better color classification
	h, s, v := rgbToHSV(avgR, avgG, avgB)

	// Classification rules based on HSV
	switch {
	case v < 50:
		return 2 // Black
	case s < 30 && v > 180:
		return 1 // White
	case h >= 100 && h <= 140 && s > 50:
		return 3 // Blue
	case (h >= 40 && h <= 70) && s > 50:
		return 4 // Yellow
	case (h >= 70 && h <= 170) && s > 30:
		return 5 // Green
	default:
		return 0 // Other
	}
}

// rgbToHSV converts RGB values (0-255) to HSV.
func rgbToHSV(r, g, b float64) (h, s, v float64) {
	r /= 255.0
	g /= 255.0
	b /= 255.0

	max := math.Max(r, math.Max(g, b))
	min := math.Min(r, math.Min(g, b))
	delta := max - min

	v = max * 255

	if max == 0 {
		s = 0
	} else {
		s = (delta / max) * 255
	}

	if delta == 0 {
		h = 0
	} else if max == r {
		h = 60 * math.Mod((g-b)/delta, 6)
	} else if max == g {
		h = 60 * ((b-r)/delta + 2)
	} else {
		h = 60 * ((r-g)/delta + 4)
	}

	if h < 0 {
		h += 360
	}

	return h, s, v
}

// Ensure jpeg and png decoders are registered
func init() {
	// Force import of image decoders
	_ = jpeg.Decode
	_ = png.Decode
	_ = color.NRGBA{}
}
