package engine

import "testing"

func TestNormalizePlateNumberWithConfidence_ProvinceDuplicate(t *testing.T) {
	in := "粤粤LE7G2"
	want := "粤LE7G22"
	confs := []float32{0.95, 0.41, 0.93, 0.90, 0.88, 0.87, 0.90}
	got, _ := normalizePlateNumberWithConfidence(in, confs)
	if got != want {
		t.Fatalf("normalizePlateNumberWithConfidence(%q)=%q, want=%q", in, got, want)
	}
}

func TestNormalizePlateNumberWithConfidence_DigitZeroToD(t *testing.T) {
	in := "粤LD07111"
	want := "粤LDD7111"
	confs := []float32{0.95, 0.93, 0.91, 0.40, 0.92, 0.93, 0.94, 0.95}
	got, _ := normalizePlateNumberWithConfidence(in, confs)
	if got != want {
		t.Fatalf("normalizePlateNumberWithConfidence(%q)=%q, want=%q", in, got, want)
	}
}

func TestNormalizePlateNumberWithConfidence_NoChangeWhenConfident(t *testing.T) {
	in := "粤LD07111"
	confs := []float32{0.95, 0.93, 0.91, 0.93, 0.92, 0.93, 0.94, 0.95}
	got, _ := normalizePlateNumberWithConfidence(in, confs)
	if got != in {
		t.Fatalf("expected unchanged output %q, got %q", in, got)
	}
}

