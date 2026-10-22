package engine

import (
	"fmt"
	"image"
	"log/slog"
	"math"
)

// crnnChars is the PlateCRNN charset (plateai/crnn/model.py CHARS); index 0
// of the model output is the CTC blank, so output class i maps to crnnChars[i-1].
var crnnChars = []rune("京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危" +
	"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品应急")

// RecognizerCRNN wraps the plateai PlateCRNN ONNX model: 32x160 BGR input
// normalized with (x/255-0.5)/0.5, 40 CTC steps. It was trained from scratch
// on full labeled plate crops and replaces the 20-step HyperLPR3 recognizer
// as primary when present; its confidence is the minimum per-character
// probability, which is well calibrated enough to route on (see
// crnnTrustConf).
type RecognizerCRNN struct {
	model       *Model
	inputWidth  int
	inputHeight int
	timeSteps   int
	numClasses  int
}

// NewRecognizerCRNN loads the PlateCRNN model. An empty path disables it.
func NewRecognizerCRNN(modelPath string, threads, optLevel int) (*RecognizerCRNN, error) {
	if modelPath == "" {
		return nil, nil
	}
	model, err := loadModel(modelPath, threads, optLevel)
	if err != nil {
		return nil, fmt.Errorf("load crnn recognizer: %w", err)
	}
	r := &RecognizerCRNN{
		model:       model,
		inputWidth:  160,
		inputHeight: 32,
		timeSteps:   40,
		numClasses:  len(crnnChars) + 1,
	}
	if shape := model.GetOutputShape(); len(shape) >= 3 {
		r.timeSteps = int(shape[len(shape)-2])
		r.numClasses = int(shape[len(shape)-1])
	}
	if r.numClasses != len(crnnChars)+1 {
		model.Close()
		return nil, fmt.Errorf("crnn recognizer: model has %d classes, charset expects %d", r.numClasses, len(crnnChars)+1)
	}
	slog.Info("CRNN recognizer loaded", "timesteps", r.timeSteps, "classes", r.numClasses)
	return r, nil
}

// Recognize returns the plate text, per-char probabilities and the minimum
// char probability (0 when nothing was decoded).
func (r *RecognizerCRNN) Recognize(img image.Image) (string, []float32, float32, error) {
	if r == nil || r.model == nil {
		return "", nil, 0, fmt.Errorf("crnn recognizer not initialized")
	}
	out, err := r.model.RunInference(r.preprocess(img))
	if err != nil {
		return "", nil, 0, fmt.Errorf("crnn inference: %w", err)
	}
	plate, confs, minConf := decodeCRNN(out, r.timeSteps, r.numClasses)
	return plate, confs, minConf, nil
}

func (r *RecognizerCRNN) preprocess(img image.Image) []float32 {
	resized := resizeImage(img, r.inputWidth, r.inputHeight)
	channelSize := r.inputHeight * r.inputWidth
	tensor := make([]float32, 3*channelSize)
	for y := 0; y < r.inputHeight; y++ {
		for x := 0; x < r.inputWidth; x++ {
			c := resized.NRGBAAt(x, y)
			idx := y*r.inputWidth + x
			// BGR order, (x/255-0.5)/0.5.
			tensor[0*channelSize+idx] = float32(c.B)/127.5 - 1
			tensor[1*channelSize+idx] = float32(c.G)/127.5 - 1
			tensor[2*channelSize+idx] = float32(c.R)/127.5 - 1
		}
	}
	return tensor
}

// Close releases the underlying ONNX session pool.
func (r *RecognizerCRNN) Close() {
	if r != nil && r.model != nil {
		r.model.Close()
	}
}

// decodeCRNN runs softmax + greedy CTC decode on [T, C] logits.
func decodeCRNN(output []float32, timeSteps, numClasses int) (string, []float32, float32) {
	if timeSteps <= 0 || numClasses <= 0 || len(output) < timeSteps*numClasses {
		return "", nil, 0
	}
	plate := make([]rune, 0, 10)
	confs := make([]float32, 0, 10)
	prev := 0
	for t := 0; t < timeSteps; t++ {
		row := output[t*numClasses : (t+1)*numClasses]
		maxIdx, maxVal := 0, row[0]
		for c := 1; c < numClasses; c++ {
			if row[c] > maxVal {
				maxIdx, maxVal = c, row[c]
			}
		}
		if maxIdx != 0 && maxIdx != prev && maxIdx-1 < len(crnnChars) {
			var sum float64
			for _, v := range row {
				sum += math.Exp(float64(v - maxVal))
			}
			plate = append(plate, crnnChars[maxIdx-1])
			confs = append(confs, float32(1/sum))
		}
		prev = maxIdx
	}
	if len(confs) == 0 {
		return "", nil, 0
	}
	minConf := confs[0]
	for _, c := range confs[1:] {
		minConf = min(minConf, c)
	}
	return string(plate), confs, minConf
}
