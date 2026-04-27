package engine

import (
	"image"
	"testing"
)

func TestShouldUseCropByAspect(t *testing.T) {
	tests := []struct {
		name string
		w    int
		h    int
		want bool
	}{
		{name: "exact target ratio", w: 333, h: 100, want: true},
		{name: "within tolerance high", w: 360, h: 100, want: true},
		{name: "within tolerance low", w: 300, h: 100, want: true},
		{name: "square image should full", w: 100, h: 100, want: false},
		{name: "very wide should full", w: 500, h: 100, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tt.w, tt.h))
			got := shouldUseCropByAspect(img)
			if got != tt.want {
				t.Fatalf("shouldUseCropByAspect(%dx%d)=%v, want %v", tt.w, tt.h, got, tt.want)
			}
		})
	}
}

func TestIsLikelyPlateBox(t *testing.T) {
	bounds := image.Rect(0, 0, 1920, 1080)
	tests := []struct {
		name string
		box  [4]int
		want bool
	}{
		{name: "normal plate-like box", box: [4]int{400, 500, 800, 620}, want: true},
		{name: "too tall", box: [4]int{400, 400, 520, 760}, want: false},
		{name: "too thin", box: [4]int{400, 500, 1200, 560}, want: false},
		{name: "too tiny area", box: [4]int{10, 10, 30, 20}, want: false},
		{name: "too huge area", box: [4]int{0, 0, 1910, 1000}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLikelyPlateBox(bounds, tt.box)
			if got != tt.want {
				t.Fatalf("isLikelyPlateBox(%v)=%v, want %v", tt.box, got, tt.want)
			}
		})
	}
}

func TestShouldUseCropByAspect_312x90FallsInAutoCropWindow(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 312, 90))
	if !shouldUseCropByAspect(img) {
		t.Fatalf("expected 312x90 to route to crop in auto mode")
	}
}
