package engine

import (
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/vesaa/platex/internal/types"
)

func solidPlate(c color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, 200, 60))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: c}, image.Point{}, draw.Src)
	return img
}

func TestPlateColorByFormat(t *testing.T) {
	yellow := solidPlate(color.NRGBA{R: 230, G: 190, B: 30, A: 255})
	// Camera-tinted blue plate: cyan-ish background.
	blue := solidPlate(color.NRGBA{R: 40, G: 150, B: 190, A: 255})
	// Blue plate on a yellow taxi: yellow only around the border.
	taxi := solidPlate(color.NRGBA{R: 230, G: 190, B: 30, A: 255}).(*image.NRGBA)
	draw.Draw(taxi, image.Rect(25, 8, 175, 52), &image.Uniform{C: color.NRGBA{R: 40, G: 90, B: 200, A: 255}}, image.Point{}, draw.Src)

	tests := []struct {
		plate  string
		img    image.Image
		want   types.PlateColor
		wantOK bool
	}{
		{"粤LDD1755", blue, types.ColorGreen, true},
		{"粤B12345D", blue, types.ColorGreen, true},
		{"粤BAX1329", yellow, types.ColorGreen, true},
		{"粤L0352警", blue, types.ColorWhite, true},
		{"粤Z154G港", blue, types.ColorBlack, true},
		{"粤L4989学", blue, types.ColorYellow, true},
		{"粤L12345", blue, types.ColorBlue, true},
		{"粤L12345", yellow, types.ColorYellow, true},
		{"粤L12345", taxi, types.ColorBlue, true},
		{"WJ12345", blue, 0, false},
		{"粤L1234", blue, 0, false},
	}
	for _, tt := range tests {
		got, ok := plateColorByFormat(tt.plate, tt.img)
		if ok != tt.wantOK || (ok && types.PlateColor(got) != tt.want) {
			t.Errorf("plateColorByFormat(%q)=(%d,%v), want (%d,%v)", tt.plate, got, ok, tt.want, tt.wantOK)
		}
	}
}

func TestYellowPixelFraction(t *testing.T) {
	if f := yellowPixelFraction(solidPlate(color.NRGBA{R: 230, G: 190, B: 30, A: 255})); f < 0.99 {
		t.Fatalf("solid yellow fraction=%v, want ~1", f)
	}
	if f := yellowPixelFraction(solidPlate(color.NRGBA{R: 40, G: 150, B: 190, A: 255})); f != 0 {
		t.Fatalf("cyan-blue fraction=%v, want 0", f)
	}
}

func TestExtendedNEVSegments(t *testing.T) {
	for _, p := range []string{"粤BAX1329", "粤ABR8493", "粤BB34996", "粤AEM8195", "粤LGJ1234"} {
		if got := classifyPlateType(p); got != types.PlateTypeNewEnergy {
			t.Errorf("classifyPlateType(%s)=%q, want new_energy", p, got)
		}
	}
	for _, p := range []string{"粤LRA716L", "粤BAX132A", "粤LIA1234"} {
		if got := classifyPlateType(p); got == types.PlateTypeNewEnergy {
			t.Errorf("classifyPlateType(%s)=new_energy, want not", p)
		}
	}
}
