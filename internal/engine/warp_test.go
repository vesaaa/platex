package engine

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestComputeHomographyIdentity(t *testing.T) {
	src := [4][2]float64{{0, 0}, {10, 0}, {10, 5}, {0, 5}}
	dst := src
	H, ok := computeHomography(src, dst)
	if !ok {
		t.Fatalf("expected solvable identity homography")
	}
	// Apply H to (3,2) and expect ~(3,2).
	x, y := 3.0, 2.0
	denom := H[6]*x + H[7]*y + H[8]
	if denom == 0 {
		t.Fatalf("zero denom")
	}
	X := (H[0]*x + H[1]*y + H[2]) / denom
	Y := (H[3]*x + H[4]*y + H[5]) / denom
	if math.Abs(X-3) > 1e-6 || math.Abs(Y-2) > 1e-6 {
		t.Fatalf("identity homography failed, got (%.6f, %.6f)", X, Y)
	}
}

func TestWarpPerspectivePlateRecoversAxisAligned(t *testing.T) {
	// Build a 64x32 source with a unique color per quadrant so we can verify
	// orientation after warping.
	src := image.NewNRGBA(image.Rect(0, 0, 64, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 64; x++ {
			var c color.NRGBA
			switch {
			case x < 32 && y < 16:
				c = color.NRGBA{R: 255, A: 255}
			case x >= 32 && y < 16:
				c = color.NRGBA{G: 255, A: 255}
			case x < 32 && y >= 16:
				c = color.NRGBA{B: 255, A: 255}
			default:
				c = color.NRGBA{R: 255, G: 255, A: 255}
			}
			src.SetNRGBA(x, y, c)
		}
	}
	kp := PlateKeypoints{{0, 0}, {63, 0}, {63, 31}, {0, 31}}
	out := warpPerspectivePlate(src, kp, 32, 16)
	// Center of each quadrant should preserve the dominant color.
	cTL := out.At(7, 3).(color.NRGBA)
	cTR := out.At(24, 3).(color.NRGBA)
	cBL := out.At(7, 12).(color.NRGBA)
	cBR := out.At(24, 12).(color.NRGBA)
	if cTL.R < 200 || cTL.G > 60 || cTL.B > 60 {
		t.Fatalf("top-left expected red, got %+v", cTL)
	}
	if cTR.G < 200 || cTR.R > 60 || cTR.B > 60 {
		t.Fatalf("top-right expected green, got %+v", cTR)
	}
	if cBL.B < 200 || cBL.R > 60 || cBL.G > 60 {
		t.Fatalf("bottom-left expected blue, got %+v", cBL)
	}
	if cBR.R < 200 || cBR.G < 200 || cBR.B > 60 {
		t.Fatalf("bottom-right expected yellow, got %+v", cBR)
	}
}

func TestOrderPlateKeypointsScrambled(t *testing.T) {
	// shuffled order: BR, TL, BL, TR
	raw := [4][2]float64{
		{63, 31},
		{0, 0},
		{0, 31},
		{63, 0},
	}
	out := orderPlateKeypoints(raw)
	if out[0] != [2]float64{0, 0} || out[1] != [2]float64{63, 0} ||
		out[2] != [2]float64{63, 31} || out[3] != [2]float64{0, 31} {
		t.Fatalf("keypoint ordering wrong: %+v", out)
	}
}

func TestKeypointsValidRejectsDegenerate(t *testing.T) {
	good := PlateKeypoints{{0, 0}, {100, 0}, {100, 30}, {0, 30}}
	if !keypointsValid(good, 5) {
		t.Fatalf("expected good polygon to be valid")
	}
	bad := PlateKeypoints{{0, 0}, {2, 0}, {2, 2}, {0, 2}}
	if keypointsValid(bad, 5) {
		t.Fatalf("expected too-small polygon to be invalid")
	}
}
