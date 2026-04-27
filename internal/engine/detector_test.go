package engine

import (
	"testing"

	"github.com/vesaa/platex/internal/types"
)

func TestResolveMaxPlatesByMode(t *testing.T) {
	tests := []struct {
		name           string
		defaultMax     int
		defaultFullMax int
		mode           string
		opts           *types.RecognizeOption
		want           int
	}{
		{
			name:           "use crop default",
			defaultMax:     8,
			defaultFullMax: 3,
			mode:           "crop",
			opts:           nil,
			want:           8,
		},
		{
			name:           "use full default",
			defaultMax:     8,
			defaultFullMax: 3,
			mode:           "full",
			opts:           nil,
			want:           3,
		},
		{
			name:           "override by opts",
			defaultMax:     8,
			defaultFullMax: 3,
			mode:           "full",
			opts:           &types.RecognizeOption{MaxPlates: 5},
			want:           5,
		},
		{
			name:           "fallback when invalid default",
			defaultMax:     0,
			defaultFullMax: 0,
			mode:           "crop",
			opts:           nil,
			want:           10,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveMaxPlatesByMode(tc.defaultMax, tc.defaultFullMax, tc.mode, tc.opts)
			if got != tc.want {
				t.Fatalf("resolveMaxPlatesByMode()=%d, want=%d", got, tc.want)
			}
		})
	}
}

func TestDecodeYOLOLikeOutputAndNMS(t *testing.T) {
	// shape: [1, 3, 6], each row [cx,cy,w,h,obj,cls]
	shape := []int64{1, 3, 6}
	output := []float32{
		100, 100, 80, 30, 0.9, 0.9, // strong box A
		102, 100, 80, 30, 0.8, 0.9, // overlapping with A, should be suppressed by NMS
		220, 120, 60, 24, 0.95, 0.9, // distant box B, should be kept
	}

	boxes := decodeYOLOLikeOutput(output, shape, 0.30, 0.45, 10)
	if len(boxes) != 2 {
		t.Fatalf("decodeYOLOLikeOutput() len=%d, want=2", len(boxes))
	}

	// Sorted by score desc; first should be the strongest candidate.
	if boxes[0].score < boxes[1].score {
		t.Fatalf("boxes not sorted by score desc")
	}
}

func TestDecodeYOLOLikeOutputRespectsMaxCandidates(t *testing.T) {
	shape := []int64{1, 3, 6}
	output := []float32{
		100, 100, 30, 10, 0.9, 0.9,
		200, 100, 30, 10, 0.8, 0.9,
		300, 100, 30, 10, 0.7, 0.9,
	}

	boxes := decodeYOLOLikeOutput(output, shape, 0.2, 0.9, 2)
	if len(boxes) != 2 {
		t.Fatalf("decodeYOLOLikeOutput() len=%d, want=2", len(boxes))
	}
}

