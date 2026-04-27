package engine

import (
	"testing"

	"github.com/vesaa/platex/internal/types"
)

func TestShouldFallbackToFull(t *testing.T) {
	tests := []struct {
		name   string
		mode   string
		result types.ImageResult
		want   bool
	}{
		{
			name:   "non-crop mode never fallback",
			mode:   "full",
			result: types.ImageResult{Plates: []types.PlateResult{}},
			want:   false,
		},
		{
			name:   "error result no fallback",
			mode:   "crop",
			result: types.ImageResult{Error: "decode error"},
			want:   false,
		},
		{
			name:   "empty plates should fallback",
			mode:   "crop",
			result: types.ImageResult{Plates: []types.PlateResult{}},
			want:   true,
		},
		{
			name: "single unknown should fallback",
			mode: "crop",
			result: types.ImageResult{Plates: []types.PlateResult{
				{PlateNumber: "粤AAF564", Type: types.PlateTypeUnknown},
			}},
			want: true,
		},
		{
			name: "recognized non-unknown should not fallback",
			mode: "crop",
			result: types.ImageResult{Plates: []types.PlateResult{
				{PlateNumber: "粤LE7G22", Type: types.PlateTypeStandard7},
			}},
			want: false,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := shouldFallbackToFull(tc.mode, tc.result)
			if got != tc.want {
				t.Fatalf("shouldFallbackToFull() = %v, want %v", got, tc.want)
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
