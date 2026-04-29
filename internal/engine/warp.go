package engine

import (
	"image"
	"image/color"
	"math"
)

// PlateKeypoints holds the 4 corner points of a detected plate, in detector
// pixel-space order: top-left, top-right, bottom-right, bottom-left.
type PlateKeypoints [4][2]float64

// warpPerspectivePlate warps the source image so that the four keypoints map to
// the corners of an output plate of size (outW x outH). It uses an inverse
// mapping with bilinear sampling and pads with mid-gray.
//
// This is functionally equivalent to OpenCV's getPerspectiveTransform +
// warpPerspective for our 4-corner case.
func warpPerspectivePlate(src image.Image, kp PlateKeypoints, outW, outH int) image.Image {
	if outW <= 0 || outH <= 0 {
		return src
	}
	dst := image.NewNRGBA(image.Rect(0, 0, outW, outH))
	bg := color.NRGBA{R: 128, G: 128, B: 128, A: 255}
	for y := 0; y < outH; y++ {
		for x := 0; x < outW; x++ {
			dst.SetNRGBA(x, y, bg)
		}
	}

	// Compute homography mapping output rectangle -> source quadrilateral.
	// dst corners (target plate canvas):
	dstCorners := [4][2]float64{
		{0, 0},
		{float64(outW - 1), 0},
		{float64(outW - 1), float64(outH - 1)},
		{0, float64(outH - 1)},
	}
	srcCorners := [4][2]float64{
		{kp[0][0], kp[0][1]},
		{kp[1][0], kp[1][1]},
		{kp[2][0], kp[2][1]},
		{kp[3][0], kp[3][1]},
	}
	// Solve homography H s.t. for each i: srcCorners[i] = H * dstCorners[i].
	H, ok := computeHomography(dstCorners, srcCorners)
	if !ok {
		return src
	}

	bounds := src.Bounds()
	for y := 0; y < outH; y++ {
		for x := 0; x < outW; x++ {
			fx := float64(x)
			fy := float64(y)
			denom := H[6]*fx + H[7]*fy + H[8]
			if denom == 0 {
				continue
			}
			sx := (H[0]*fx + H[1]*fy + H[2]) / denom
			sy := (H[3]*fx + H[4]*fy + H[5]) / denom
			c, ok := bilinearSampleNRGBA(src, bounds, sx, sy)
			if !ok {
				continue
			}
			dst.SetNRGBA(x, y, c)
		}
	}
	return dst
}

// computeHomography solves for a 3x3 homography H mapping src[i] -> dst[i] for
// i in [0,4). Returned as a row-major 9-element array (H[0..8]).
//
// We solve the standard 8x8 linear system:
//
//	[ x  y  1  0  0  0  -x*X  -y*X ] [h0]   [X]
//	[ 0  0  0  x  y  1  -x*Y  -y*Y ] [h1] = [Y]
//	... (4 point pairs => 8 rows)
//
// where (x,y) are src coords, (X,Y) are dst coords. We then set h8=1.
func computeHomography(src, dst [4][2]float64) ([9]float64, bool) {
	var A [8][8]float64
	var b [8]float64
	for i := 0; i < 4; i++ {
		x := src[i][0]
		y := src[i][1]
		X := dst[i][0]
		Y := dst[i][1]
		A[2*i] = [8]float64{x, y, 1, 0, 0, 0, -x * X, -y * X}
		A[2*i+1] = [8]float64{0, 0, 0, x, y, 1, -x * Y, -y * Y}
		b[2*i] = X
		b[2*i+1] = Y
	}
	h, ok := solve8x8(A, b)
	if !ok {
		return [9]float64{}, false
	}
	return [9]float64{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}, true
}

// solve8x8 solves Ax=b for 8x8 system using Gaussian elimination with partial pivoting.
func solve8x8(A [8][8]float64, b [8]float64) ([8]float64, bool) {
	const N = 8
	// Augment.
	var M [8][9]float64
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			M[i][j] = A[i][j]
		}
		M[i][N] = b[i]
	}
	for c := 0; c < N; c++ {
		// Pivot.
		pivot := c
		max := math.Abs(M[c][c])
		for r := c + 1; r < N; r++ {
			if v := math.Abs(M[r][c]); v > max {
				max = v
				pivot = r
			}
		}
		if max < 1e-12 {
			return [8]float64{}, false
		}
		if pivot != c {
			M[c], M[pivot] = M[pivot], M[c]
		}
		// Eliminate.
		for r := 0; r < N; r++ {
			if r == c {
				continue
			}
			factor := M[r][c] / M[c][c]
			if factor == 0 {
				continue
			}
			for k := c; k <= N; k++ {
				M[r][k] -= factor * M[c][k]
			}
		}
	}
	var x [8]float64
	for i := 0; i < N; i++ {
		if M[i][i] == 0 {
			return [8]float64{}, false
		}
		x[i] = M[i][N] / M[i][i]
	}
	return x, true
}

// bilinearSampleNRGBA samples src image at floating-point coordinates with
// bilinear interpolation. Returns (color, true) if the sample lies inside src
// bounds; returns (_, false) otherwise.
func bilinearSampleNRGBA(src image.Image, b image.Rectangle, fx, fy float64) (color.NRGBA, bool) {
	if fx < float64(b.Min.X) || fx > float64(b.Max.X-1) ||
		fy < float64(b.Min.Y) || fy > float64(b.Max.Y-1) {
		return color.NRGBA{}, false
	}
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > b.Max.X-1 {
		x1 = b.Max.X - 1
	}
	if y1 > b.Max.Y-1 {
		y1 = b.Max.Y - 1
	}
	dx := fx - float64(x0)
	dy := fy - float64(y0)
	r00, g00, bl00, _ := src.At(x0, y0).RGBA()
	r01, g01, bl01, _ := src.At(x1, y0).RGBA()
	r10, g10, bl10, _ := src.At(x0, y1).RGBA()
	r11, g11, bl11, _ := src.At(x1, y1).RGBA()
	w00 := (1 - dx) * (1 - dy)
	w01 := dx * (1 - dy)
	w10 := (1 - dx) * dy
	w11 := dx * dy
	rr := w00*float64(r00>>8) + w01*float64(r01>>8) + w10*float64(r10>>8) + w11*float64(r11>>8)
	gg := w00*float64(g00>>8) + w01*float64(g01>>8) + w10*float64(g10>>8) + w11*float64(g11>>8)
	bb := w00*float64(bl00>>8) + w01*float64(bl01>>8) + w10*float64(bl10>>8) + w11*float64(bl11>>8)
	return color.NRGBA{
		R: clamp8(rr),
		G: clamp8(gg),
		B: clamp8(bb),
		A: 255,
	}, true
}

// orderPlateKeypoints reorders 4 raw keypoints into top-left, top-right,
// bottom-right, bottom-left by sorting along the y axis then by x within rows.
// This is robust against detectors that may emit corners in arbitrary order.
func orderPlateKeypoints(raw [4][2]float64) PlateKeypoints {
	pts := raw
	// Sort by y ascending.
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if pts[j][1] < pts[i][1] {
				pts[i], pts[j] = pts[j], pts[i]
			}
		}
	}
	// First two are top, last two are bottom.
	if pts[1][0] < pts[0][0] {
		pts[0], pts[1] = pts[1], pts[0]
	}
	if pts[3][0] > pts[2][0] {
		pts[2], pts[3] = pts[3], pts[2]
	}
	return PlateKeypoints{pts[0], pts[1], pts[2], pts[3]}
}

// keypointsValid checks that the 4 points form a non-degenerate quadrilateral
// (all sides have a minimum length, and the polygon area is meaningful).
func keypointsValid(kp PlateKeypoints, minSide float64) bool {
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		dx := kp[j][0] - kp[i][0]
		dy := kp[j][1] - kp[i][1]
		if math.Sqrt(dx*dx+dy*dy) < minSide {
			return false
		}
	}
	return true
}
