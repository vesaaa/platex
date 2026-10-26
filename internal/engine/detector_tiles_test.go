package engine

import (
	"image"
	"testing"
)

func TestDetectorTilesCoverImageWithOverlap(t *testing.T) {
	b := image.Rect(0, 0, 1944, 2592)
	tiles := detectorTiles(b, 960)
	if len(tiles) != 3*4 {
		t.Fatalf("got %d tiles, want 12", len(tiles))
	}
	// Every pixel row/column is covered and neighbours overlap.
	for _, tl := range tiles {
		if !tl.In(b) || tl.Dx() > 960 || tl.Dy() > 960 {
			t.Fatalf("bad tile %v", tl)
		}
	}
	if tiles[0].Min != b.Min || tiles[len(tiles)-1].Max != b.Max {
		t.Fatalf("tiles don't span the image: first %v last %v", tiles[0], tiles[len(tiles)-1])
	}
	if overlap := tiles[0].Max.X - tiles[1].Min.X; overlap < 200 {
		t.Fatalf("horizontal overlap %d too small", overlap)
	}
}

func TestDetectorTilesSkipsSmallImagesAndZeroSize(t *testing.T) {
	if got := detectorTiles(image.Rect(0, 0, 1100, 800), 960); got != nil {
		t.Fatalf("1100x800 should not be tiled, got %v", got)
	}
	if got := detectorTiles(image.Rect(0, 0, 4000, 3000), 0); got != nil {
		t.Fatalf("tile size 0 should disable tiling, got %v", got)
	}
}

func TestTouchesInnerEdge(t *testing.T) {
	full := image.Rect(0, 0, 2000, 2000)
	tile := image.Rect(500, 0, 1460, 960)
	if !touchesInnerEdge([4]int{500, 100, 600, 140}, tile, full) {
		t.Fatal("box on the tile's inner left edge should be flagged")
	}
	if touchesInnerEdge([4]int{600, 0, 700, 40}, tile, full) {
		t.Fatal("box on the image's top edge is not cut by the tile")
	}
	if touchesInnerEdge([4]int{700, 300, 820, 340}, tile, full) {
		t.Fatal("interior box should not be flagged")
	}
}

func TestMergeDetectionsKeepsBestOfOverlaps(t *testing.T) {
	dets := []PlateDetection{
		{Box: [4]int{100, 100, 220, 140}, Score: 0.6},
		{Box: [4]int{102, 101, 221, 141}, Score: 0.9}, // same plate, higher score
		{Box: [4]int{110, 105, 200, 135}, Score: 0.7}, // inside the first
		{Box: [4]int{400, 400, 520, 440}, Score: 0.5}, // another plate
	}
	got := mergeDetections(dets, 0.45)
	if len(got) != 2 || got[0].Score != 0.9 || got[1].Box != [4]int{400, 400, 520, 440} {
		t.Fatalf("mergeDetections=%+v", got)
	}
}

func TestIsLikelyPlateBoxKeepsSmallPlatesInLargeScenes(t *testing.T) {
	// 150x50 plate in a 1944x2592 roadside frame (area ratio 0.0015).
	if !isLikelyPlateBox(image.Rect(0, 0, 1944, 2592), [4]int{800, 1500, 950, 1550}) {
		t.Fatal("small far plate in a large scene should be kept")
	}
}
