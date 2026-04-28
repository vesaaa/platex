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

func TestNeedRecoverySearch(t *testing.T) {
	if !needRecoverySearch("浙B9", 0.63) {
		t.Fatalf("expected short malformed plate to require recovery")
	}
	if !needRecoverySearch("粤B590MF", 0.52) {
		t.Fatalf("expected low-confidence 7-char plate to require recovery")
	}
	if !needRecoverySearch("粤BD12345", 0.58) {
		t.Fatalf("expected low-confidence 8-char plate to require recovery")
	}
	if needRecoverySearch("粤B590MF", 0.81) {
		t.Fatalf("expected high-confidence 7-char plate to skip recovery")
	}
	if !needRecoverySearch("粤LFG060", 0.82) {
		t.Fatalf("expected suspected collapsed new-energy plate to require recovery")
	}
	if needRecoverySearch("粤LFG060", 0.90) {
		t.Fatalf("expected high-confidence suspected collapsed new-energy plate to skip recovery")
	}
	if !needRecoverySearch("粤B590M", 0.40) {
		t.Fatalf("expected irregular length low-conf plate to require recovery")
	}
}

func TestRerankAmbiguousPlate_Y6Tail(t *testing.T) {
	in := "粤L02166"
	confs := []float32{0.95, 0.93, 0.90, 0.89, 0.88, 0.86, 0.91}
	out, _, _ := rerankAmbiguousPlate(in, confs, 0)
	if out != "粤L021Y6" {
		t.Fatalf("expected rerank to produce 粤L021Y6, got %s", out)
	}
}

func TestRerankAmbiguousPlate_TailTransposition(t *testing.T) {
	in := "粤L02Y16"
	confs := []float32{0.95, 0.93, 0.90, 0.89, 0.88, 0.86, 0.91}
	out, _, _ := rerankAmbiguousPlate(in, confs, 0)
	if out != "粤L021Y6" {
		t.Fatalf("expected rerank to produce 粤L021Y6, got %s", out)
	}
}

func TestRerankAmbiguousPlate_NoTranspositionForRepeatedTailDigits(t *testing.T) {
	in := "粤LE7G22"
	confs := []float32{0.97, 0.95, 0.93, 0.92, 0.91, 0.90, 0.90}
	out, _, _ := rerankAmbiguousPlate(in, confs, 0)
	if out != in {
		t.Fatalf("expected unchanged output %s, got %s", in, out)
	}
}

func TestUpscaleImage(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 10, 8))
	up := upscaleImage(src, 2)
	b := up.Bounds()
	if b.Dx() != 20 || b.Dy() != 16 {
		t.Fatalf("unexpected upscaled size %dx%d", b.Dx(), b.Dy())
	}
}

func TestRecoveryVariants(t *testing.T) {
	r := &Recognizer{}
	img := image.NewNRGBA(image.Rect(0, 0, 32, 16))
	variants := r.recoveryVariants(img)
	if len(variants) < 3 {
		t.Fatalf("expected at least 3 recovery variants, got %d", len(variants))
	}
}

func TestStripInnerProvinceNoise(t *testing.T) {
	in := []rune("粤A粤L7G22")
	confs := []float32{0.95, 0.90, 0.88, 0.91, 0.87, 0.86, 0.90, 0.92}
	out, _ := stripInnerProvinceNoise(in, confs)
	if string(out) != "粤AL7G22" {
		t.Fatalf("unexpected stripped result: %s", string(out))
	}
}

func TestIsHighQualityCandidate_GuardsAgainstPrematureStop(t *testing.T) {
	if isHighQualityCandidate("粤LFG060", 99.0) {
		t.Fatalf("expected collapsed-new-energy-like candidate to not early-stop")
	}
	if isHighQualityCandidate("粤L183I1", 102.0) {
		t.Fatalf("expected I/1 ambiguous tail candidate to not early-stop")
	}
	if !isHighQualityCandidate("粤B590MF", 103.0) {
		t.Fatalf("expected strong clean candidate to early-stop")
	}
}
