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
	if !needRecoverySearch("粤LFG060", 0.90) {
		t.Fatalf("expected mid-high-confidence collapsed-new-energy to still require recovery")
	}
	if needRecoverySearch("粤LFG060", 0.95) {
		t.Fatalf("expected high-confidence suspected collapsed new-energy plate to skip recovery")
	}
	if !needRecoverySearch("粤LRA716L", 0.82) {
		t.Fatalf("expected low-confidence 8-char non-NEV output to require recovery")
	}
	if needRecoverySearch("粤LRA716L", 0.93) {
		t.Fatalf("expected high-confidence 8-char non-NEV output to skip recovery")
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

func TestShouldRejectRecoveryResult_AmbiguousLowConfidence(t *testing.T) {
	if !shouldRejectRecoveryResult("粤L183I1", 0.69, 83.0) {
		t.Fatalf("expected ambiguous low-confidence result to be rejected")
	}
	if !shouldRejectRecoveryResult("粤L183I1", 0.75, 105.0) {
		t.Fatalf("expected ambiguous I/1 tail to be rejected even when score is high")
	}
	if shouldRejectRecoveryResult("粤L702D5", 0.94, 108.0) {
		t.Fatalf("expected strong candidate to be accepted")
	}
	if !shouldRejectRecoveryResult("粤LH83S", 0.74, 84.0) {
		t.Fatalf("expected irregular-length low-confidence result to be rejected")
	}
	if !shouldRejectRecoveryResult("粤LRA716L", 0.75, 90.0) {
		t.Fatalf("expected non-NEV 8-char low-confidence result to be rejected")
	}
}

func TestRerankAmbiguousPlate_CollapsedNewEnergyCandidate(t *testing.T) {
	in := "粤LF6064"
	confs := []float32{0.95, 0.93, 0.86, 0.90, 0.89, 0.91, 0.92}
	base := scorePlateCandidate(in, meanConfs(confs))
	out, _, _ := rerankAmbiguousPlate(in, confs, base)
	if out != "粤LFF6064" {
		t.Fatalf("expected rerank candidate 粤LFF6064, got %s", out)
	}
}

func TestSelectConsensusRecoveryCandidate(t *testing.T) {
	agg := map[string]*recoveryCandidateStat{
		"粤L12345": {count: 1, bestScore: 94, bestConf: 0.86, bestConfs: []float32{0.9}},
		"粤L1234S": {count: 3, bestScore: 93, bestConf: 0.84, bestConfs: []float32{0.88}},
	}
	got := selectConsensusRecoveryCandidate(agg)
	if got == nil {
		t.Fatalf("expected non-nil consensus candidate")
	}
	if got.plate != "粤L1234S" {
		t.Fatalf("expected repeated candidate to win, got %s", got.plate)
	}
}

func TestRerankAmbiguousPlate_DataDrivenConfusions(t *testing.T) {
	in := "粤L7890V"
	confs := []float32{0.95, 0.93, 0.92, 0.91, 0.90, 0.62, 0.94}
	base := scorePlateCandidate(in, meanConfs(confs))
	out, _, _ := rerankAmbiguousPlate(in, confs, base)
	if out != "粤L789DV" {
		t.Fatalf("expected data-driven correction to 粤L789DV, got %s", out)
	}
}

func TestPreferConfusionReplacement_D0LowConfidence(t *testing.T) {
	r := []rune("粤LD07111")
	confs := []float32{0.95, 0.93, 0.92, 0.61, 0.91, 0.90, 0.89, 0.88}
	if !preferConfusionReplacement(r, confs, 3, 'D') {
		t.Fatalf("expected low-confidence NEV-slot 0->D to be preferred")
	}
}

func TestPreferConfusionReplacement_D0HighConfidenceNoFlip(t *testing.T) {
	r := []rune("粤LD07111")
	confs := []float32{0.95, 0.93, 0.92, 0.90, 0.91, 0.90, 0.89, 0.88}
	if preferConfusionReplacement(r, confs, 3, 'D') {
		t.Fatalf("expected high-confidence char to keep original")
	}
}

func TestRerankAmbiguousPlate_TailDigitEightToFour(t *testing.T) {
	in := "湘AF55568"
	confs := []float32{0.96, 0.94, 0.92, 0.90, 0.89, 0.88, 0.86, 0.61}
	base := scorePlateCandidate(in, meanConfs(confs))
	out, _, _ := rerankAmbiguousPlate(in, confs, base)
	if out != "湘AF55564" {
		t.Fatalf("expected tail correction to 湘AF55564, got %s", out)
	}
}

func TestRerankLengthMismatchCandidate_CollapsedNEV(t *testing.T) {
	r := []rune("粤LF6064")
	confs := []float32{0.95, 0.93, 0.86, 0.90, 0.89, 0.91, 0.92}
	p, c, ok := rerankLengthMismatchCandidate(r, confs)
	if !ok {
		t.Fatalf("expected collapsed NEV candidate")
	}
	if p != "粤LFF6064" {
		t.Fatalf("expected 粤LFF6064, got %s", p)
	}
	if len(c) != 8 {
		t.Fatalf("expected 8 confidences, got %d", len(c))
	}
}

func TestRerankLengthMismatchCandidate_TrimWeak8th(t *testing.T) {
	r := []rune("粤LRA716L")
	confs := []float32{0.95, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.70}
	p, c, ok := rerankLengthMismatchCandidate(r, confs)
	if !ok {
		t.Fatalf("expected weak-tail trim candidate")
	}
	if p != "粤LRA716" {
		t.Fatalf("expected 粤LRA716, got %s", p)
	}
	if len(c) != 7 {
		t.Fatalf("expected 7 confidences, got %d", len(c))
	}
}

func TestRerankTailI1Candidate(t *testing.T) {
	r := []rune("粤L183I1")
	confs := []float32{0.95, 0.93, 0.90, 0.88, 0.86, 0.62, 0.61}
	p, c, ok := rerankTailI1Candidate(r, confs)
	if !ok {
		t.Fatalf("expected tail I/1 candidate")
	}
	if len([]rune(p)) != 7 {
		t.Fatalf("expected 7-char candidate, got %s", p)
	}
	if len(c) != 7 {
		t.Fatalf("expected 7 confidences, got %d", len(c))
	}
}

func TestRerankTailI1Candidate_SkipHighConfidence(t *testing.T) {
	r := []rune("粤L183I1")
	confs := []float32{0.95, 0.93, 0.90, 0.88, 0.86, 0.90, 0.92}
	if _, _, ok := rerankTailI1Candidate(r, confs); ok {
		t.Fatalf("expected high-confidence I/1 tail to skip rerank")
	}
}
