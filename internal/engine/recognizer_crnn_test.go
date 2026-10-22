package engine

import (
	"image"
	"math"
	"testing"
)

// crnnLogits builds [T, C] logits where step t strongly prefers class seq[t].
func crnnLogits(seq []int, numClasses int) []float32 {
	out := make([]float32, len(seq)*numClasses)
	for t, c := range seq {
		out[t*numClasses+c] = 10
	}
	return out
}

func crnnClass(t *testing.T, ch rune) int {
	t.Helper()
	for i, r := range crnnChars {
		if r == ch {
			return i + 1
		}
	}
	t.Fatalf("char %q not in crnn charset", ch)
	return 0
}

func TestDecodeCRNNKeepsRepeatsSeparatedByBlank(t *testing.T) {
	numClasses := len(crnnChars) + 1
	yue, l, d := crnnClass(t, '粤'), crnnClass(t, 'L'), crnnClass(t, 'D')
	seven := crnnClass(t, '7')
	// 粤 L D _ D D 7 _ 7 : blanks split the repeats, consecutive dups merge.
	seq := []int{yue, l, d, 0, d, d, seven, 0, seven}
	plate, confs, minConf := decodeCRNN(crnnLogits(seq, numClasses), len(seq), numClasses)
	if plate != "粤LDD77" {
		t.Fatalf("plate=%q, want 粤LDD77", plate)
	}
	if len(confs) != 6 {
		t.Fatalf("len(confs)=%d, want 6", len(confs))
	}
	if minConf < 0.99 || minConf > 1 {
		t.Fatalf("minConf=%v, want ~1", minConf)
	}
}

func TestDecodeCRNNEmptyAndShortInput(t *testing.T) {
	numClasses := len(crnnChars) + 1
	if p, _, c := decodeCRNN(crnnLogits([]int{0, 0, 0}, numClasses), 3, numClasses); p != "" || c != 0 {
		t.Fatalf("all-blank: plate=%q conf=%v", p, c)
	}
	if p, _, _ := decodeCRNN(make([]float32, 5), 3, numClasses); p != "" {
		t.Fatalf("short output should decode to empty, got %q", p)
	}
}

func TestDecodeCRNNMinConfIsWeakestChar(t *testing.T) {
	numClasses := len(crnnChars) + 1
	seq := []int{crnnClass(t, '粤'), crnnClass(t, 'B')}
	out := crnnLogits(seq, numClasses)
	// Make step 1 ambiguous between B and 8: ~0.5 each.
	out[1*numClasses+crnnClass(t, '8')] = 10
	_, _, minConf := decodeCRNN(out, len(seq), numClasses)
	if math.Abs(float64(minConf)-0.5) > 0.01 {
		t.Fatalf("minConf=%v, want ~0.5", minConf)
	}
}

func TestExpandKeypointsGrowsQuadOutward(t *testing.T) {
	kp := PlateKeypoints{{100, 100}, {300, 100}, {300, 160}, {100, 160}}
	got := expandKeypoints(kp, 0.1, 0.25)
	want := PlateKeypoints{{80, 85}, {320, 85}, {320, 175}, {80, 175}}
	for i := range want {
		for j := range want[i] {
			if math.Abs(got[i][j]-want[i][j]) > 1e-9 {
				t.Fatalf("expandKeypoints=%v, want %v", got, want)
			}
		}
	}
}

func TestExpandBoxClampsToBounds(t *testing.T) {
	bounds := image.Rect(0, 0, 400, 200)
	got := expandBox(bounds, [4]int{10, 5, 390, 105}, 0.1, 0.2)
	want := [4]int{0, 0, 400, 125}
	if got != want {
		t.Fatalf("expandBox=%v, want %v", got, want)
	}
}
