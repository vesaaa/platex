package engine

import (
	"testing"

	"github.com/vesaa/platex/internal/types"
)

func TestResolveFullEarlyStopConf_AutoTightenForSinglePlate(t *testing.T) {
	opts := &types.RecognizeOption{MaxPlates: 1}
	got := resolveFullEarlyStopConf(0.65, opts)
	if got != 0.90 {
		t.Fatalf("expected 0.90 for single-plate workload, got %.2f", got)
	}
}

func TestResolveFullEarlyStopConf_RequestOverrideWins(t *testing.T) {
	opts := &types.RecognizeOption{MaxPlates: 1, FullEarlyStopConf: 0.72}
	got := resolveFullEarlyStopConf(0.65, opts)
	if got != 0.72 {
		t.Fatalf("expected request override 0.72, got %.2f", got)
	}
}
