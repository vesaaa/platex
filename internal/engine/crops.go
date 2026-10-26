package engine

import (
	"fmt"
	"image"
)

// PlateCrop is one detected plate region as the recognizer sees it.
type PlateCrop struct {
	Box        [4]int
	Image      image.Image
	Plate      string
	Confidence float32
}

// DetectPlateCrops runs detection on a scene and returns every plate-like
// region warped/cropped exactly as recognizePlateRegion feeds PlateCRNN,
// recognized by PlateCRNN alone. It exists to harvest training crops from
// labeled scene images (cmd/plate-crops).
func (e *Engine) DetectPlateCrops(img image.Image) ([]PlateCrop, error) {
	if e.detector == nil || e.recognizerCRNN == nil {
		return nil, fmt.Errorf("detector and CRNN recognizer are required")
	}
	dets, err := e.detector.DetectPlates(img)
	if err != nil {
		return nil, err
	}
	crops := make([]PlateCrop, 0, len(dets))
	for _, det := range dets {
		if !isLikelyPlateBox(img.Bounds(), det.Box) {
			continue
		}
		det.Keypoints = expandKeypoints(det.Keypoints, crnnWarpMarginX, crnnWarpMarginY)
		det.Box = expandBox(img.Bounds(), det.Box, crnnWarpMarginX, crnnWarpMarginY)
		var region image.Image
		if det.HasKeys && det.Score >= plateWarpScoreFloor && keypointsValid(det.Keypoints, 6.0) {
			region = warpPerspectivePlate(img, det.Keypoints, plateWarpOutW, plateWarpOutH)
		} else {
			region = cropImage(img, det.Box[0], det.Box[1], det.Box[2], det.Box[3])
		}
		plate, _, conf, err := e.recognizerCRNN.Recognize(region)
		if err != nil {
			return nil, err
		}
		crops = append(crops, PlateCrop{Box: det.Box, Image: region, Plate: plate, Confidence: conf})
	}
	return crops, nil
}
