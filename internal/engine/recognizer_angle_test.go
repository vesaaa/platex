package engine

import (
	"image"
	"testing"
)

func TestMainlandFormatScorePrefersValidLength(t *testing.T) {
	good := []rune("粤L021Y6")
	bad := []rune("浙B9")
	if mainlandFormatScore(good) <= mainlandFormatScore(bad) {
		t.Fatalf("expected valid mainland pattern to score higher")
	}
}

func TestCandidateCropsForSquareInput(t *testing.T) {
	r := &Recognizer{}
	img := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	crops := r.candidateCrops(img)
	if len(crops) < 3 {
		t.Fatalf("expected multiple candidate crops for square input, got %d", len(crops))
	}
}

