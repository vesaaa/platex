package engine

import (
	"image"
	"image/color"
	"testing"
)

func TestHasBlueDominanceOverYellow(t *testing.T) {
	blue := image.NewNRGBA(image.Rect(0, 0, 160, 48))
	for y := 0; y < 48; y++ {
		for x := 0; x < 160; x++ {
			blue.Set(x, y, color.NRGBA{R: 28, G: 72, B: 205, A: 255})
		}
	}
	if !hasBlueDominanceOverYellow(blue) {
		t.Fatalf("expected blue image to satisfy hasBlueDominanceOverYellow")
	}

	yellow := image.NewNRGBA(image.Rect(0, 0, 160, 48))
	for y := 0; y < 48; y++ {
		for x := 0; x < 160; x++ {
			yellow.Set(x, y, color.NRGBA{R: 220, G: 180, B: 30, A: 255})
		}
	}
	if hasBlueDominanceOverYellow(yellow) {
		t.Fatalf("expected yellow image to fail hasBlueDominanceOverYellow")
	}
}
