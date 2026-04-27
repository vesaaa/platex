package engine

import (
	"image"
	"testing"

	"github.com/vesaa/platex/internal/types"
)

func TestFallbackReason(t *testing.T) {
	square := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	wide := image.NewNRGBA(image.Rect(0, 0, 256, 72))
	tests := []struct {
		name   string
		mode   string
		img    image.Image
		result types.ImageResult
		want   string
	}{
		{
			name:   "non-crop mode never fallback",
			mode:   "full",
			img:    square,
			result: types.ImageResult{Plates: []types.PlateResult{}},
			want:   "",
		},
		{
			name:   "error result no fallback",
			mode:   "crop",
			img:    square,
			result: types.ImageResult{Error: "decode error"},
			want:   "",
		},
		{
			name:   "empty plates should fallback",
			mode:   "crop",
			img:    square,
			result: types.ImageResult{Plates: []types.PlateResult{}},
			want:   "empty",
		},
		{
			name: "single unknown should fallback",
			mode: "crop",
			img:  square,
			result: types.ImageResult{Plates: []types.PlateResult{
				{PlateNumber: "粤AAF564", Type: types.PlateTypeUnknown},
			}},
			want: "unknown",
		},
		{
			name: "recognized non-unknown should not fallback",
			mode: "crop",
			img:  wide,
			result: types.ImageResult{Plates: []types.PlateResult{
				{PlateNumber: "粤LE7G22", Type: types.PlateTypeStandard7, Confidence: 0.93},
			}},
			want: "",
		},
		{
			name: "square image medium confidence should fallback",
			mode: "crop",
			img:  square,
			result: types.ImageResult{Plates: []types.PlateResult{
				{PlateNumber: "辽L021Y6", Type: types.PlateTypeStandard7, Confidence: 0.83},
			}},
			want: "square_medium_conf",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			reason, ok := fallbackReason(tc.mode, tc.img, tc.result)
			if tc.want == "" {
				if ok {
					t.Fatalf("fallbackReason() unexpectedly triggered with reason=%q", reason)
				}
				return
			}
			if !ok || reason != tc.want {
				t.Fatalf("fallbackReason() = (%q, %v), want (%q, true)", reason, ok, tc.want)
			}
		})
	}
}

func TestFilterReliablePlates(t *testing.T) {
	in := []types.PlateResult{
		{PlateNumber: "粤AAF564", Type: types.PlateTypeUnknown},
		{PlateNumber: "粤LE7G22", Type: types.PlateTypeStandard7},
		{PlateNumber: "粤LD021D6", Type: types.PlateTypeNewEnergy},
	}
	out := filterReliablePlates(in)
	if len(out) != 2 {
		t.Fatalf("filterReliablePlates() len = %d, want 2", len(out))
	}
	if out[0].Type == types.PlateTypeUnknown || out[1].Type == types.PlateTypeUnknown {
		t.Fatalf("filterReliablePlates() should remove unknown plates")
	}
}
