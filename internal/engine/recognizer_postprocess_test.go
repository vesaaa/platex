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

func TestNormalizePlateNumberWithConfidence_AppendLastDigitForLen6(t *testing.T) {
	in := "粤LE7G2"
	want := "粤LE7G22"
	confs := []float32{0.95, 0.92, 0.89, 0.87, 0.86, 0.70}
	got, _ := normalizePlateNumberWithConfidence(in, confs)
	if got != want {
		t.Fatalf("normalizePlateNumberWithConfidence(%q)=%q, want=%q", in, got, want)
	}
}

func TestNormalizePlateNumberWithConfidence_TrimNonNEVTrailingLetter(t *testing.T) {
	in := "粤LRA716L"
	want := "粤LRA716"
	confs := []float32{0.98, 0.96, 0.95, 0.94, 0.93, 0.94, 0.95, 0.91}
	got, _ := normalizePlateNumberWithConfidence(in, confs)
	if got != want {
		t.Fatalf("normalizePlateNumberWithConfidence(%q)=%q, want=%q", in, got, want)
	}
}

func TestClassifyPlateType_StrictNewEnergyPattern(t *testing.T) {
	if got := classifyPlateType("粤LRA716L"); got != "unknown" {
		t.Fatalf("classifyPlateType(粤LRA716L)=%q, want unknown", got)
	}
	if got := classifyPlateType("粤AD12345"); got != "new_energy" {
		t.Fatalf("classifyPlateType(粤AD12345)=%q, want new_energy", got)
	}
	if got := classifyPlateType("粤LRA716"); got != "standard_7" {
		t.Fatalf("classifyPlateType(粤LRA716)=%q, want standard_7", got)
	}
	if got := classifyPlateType("粤AP00000"); got != "new_energy" {
		t.Fatalf("classifyPlateType(粤AP00000)=%q, want new_energy", got)
	}
}


