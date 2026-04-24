package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
)

type recognizeRequest struct {
	Images []imageInput `json:"images"`
	Mode   string       `json:"mode"`
}
type imageInput struct {
	ID   string `json:"id"`
	Type string `json:"type"`
	Data string `json:"data"`
}

type recognizeResp struct {
	Code int    `json:"code"`
	Msg  string `json:"message"`
	Data struct {
		Results []struct {
			ID     string `json:"id"`
			Error  string `json:"error"`
			Plates []struct {
				PlateNumber string  `json:"plate_number"`
				Confidence  float64 `json:"confidence"`
			} `json:"plates"`
		} `json:"results"`
	} `json:"data"`
}

func main() {
	api := os.Getenv("PLATEX_API")
	if api == "" {
		api = "http://192.168.1.33:8080/api/v1/recognize"
	}
	imgURL := os.Getenv("PLATE_URL")
	if imgURL == "" {
		imgURL = "https://huizhoupark.obs.cn-south-1.myhuaweicloud.com/0D0000189909A7F5/VIDEO_CAR_IN_OUT_PIC/0D0000189909A7F5-1775109123/20260402135203/%E7%B2%A4L021Y6_plate_1775109122.jpg"
	}
	if _, err := url.Parse(imgURL); err != nil {
		panic(err)
	}

	src, err := fetchImage(imgURL)
	if err != nil {
		panic(err)
	}

	angles := []float64{0, 2, 4, 6, 8, 10, 12, 14}
	crops := []float64{0.72, 0.68, 0.64, 0.60}
	xOffsets := []float64{-0.25, -0.15, 0, 0.15}
	for _, crop := range crops {
		for _, xo := range xOffsets {
			for _, a := range angles {
				img := offsetCrop(src, crop, xo, 0)
				if a != 0 {
					img = rotateImageGrayBG(img, a)
				}
				b64, err := toBase64JPEG(img)
				if err != nil {
					fmt.Printf("crop=%.2f xo=%.2f angle=%v err=%v\n", crop, xo, a, err)
					continue
				}
				plate, conf, e := callRecognize(api, b64)
				if e == "" && (len([]rune(plate)) >= 5) {
					fmt.Printf("crop=%.2f xo=%5.2f angle=%6.1f plate=%-14s conf=%.4f\n", crop, xo, a, plate, conf)
				}
			}
		}
	}
}

func fetchImage(u string) (image.Image, error) {
	resp, err := http.Get(u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return img, nil
}

func toBase64JPEG(img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func callRecognize(api, b64 string) (string, float64, string) {
	reqBody := recognizeRequest{
		Images: []imageInput{{ID: "probe", Type: "base64", Data: b64}},
		Mode:   "crop",
	}
	raw, _ := json.Marshal(reqBody)
	resp, err := http.Post(api, "application/json", bytes.NewReader(raw))
	if err != nil {
		return "", 0, err.Error()
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var out recognizeResp
	if err := json.Unmarshal(body, &out); err != nil {
		return "", 0, err.Error()
	}
	if len(out.Data.Results) == 0 {
		return "", 0, "empty_result"
	}
	r := out.Data.Results[0]
	if r.Error != "" {
		return "", 0, r.Error
	}
	if len(r.Plates) == 0 {
		return "", 0, "no_plate"
	}
	return r.Plates[0].PlateNumber, r.Plates[0].Confidence, ""
}

func rotateImageGrayBG(src image.Image, angleDeg float64) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	bg := color.NRGBA{R: 128, G: 128, B: 128, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.SetNRGBA(x, y, bg)
		}
	}
	rad := angleDeg * math.Pi / 180.0
	sinA := math.Sin(rad)
	cosA := math.Cos(rad)
	cx := float64(w-1) / 2.0
	cy := float64(h-1) / 2.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			srcX := cosA*dx + sinA*dy + cx
			srcY := -sinA*dx + cosA*dy + cy
			ix := int(math.Round(srcX))
			iy := int(math.Round(srcY))
			if ix >= 0 && ix < w && iy >= 0 && iy < h {
				dst.Set(x, y, src.At(ix+b.Min.X, iy+b.Min.Y))
			}
		}
	}
	return dst
}

func centerCrop(src image.Image, ratio float64) image.Image {
	return offsetCrop(src, ratio, 0, 0)
}

func offsetCrop(src image.Image, ratio, xOffset, yOffset float64) image.Image {
	if ratio >= 0.999 {
		return src
	}
	if ratio <= 0 {
		return src
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	cw := int(float64(w) * ratio)
	ch := int(float64(h) * ratio)
	if cw < 8 || ch < 8 {
		return src
	}
	maxShiftX := (w - cw) / 2
	maxShiftY := (h - ch) / 2
	shiftX := int(float64(maxShiftX) * xOffset)
	shiftY := int(float64(maxShiftY) * yOffset)
	x0 := b.Min.X + (w-cw)/2 + shiftX
	y0 := b.Min.Y + (h-ch)/2 + shiftY
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x0+cw > b.Max.X {
		x0 = b.Max.X - cw
	}
	if y0+ch > b.Max.Y {
		y0 = b.Max.Y - ch
	}
	rect := image.Rect(x0, y0, x0+cw, y0+ch)
	dst := image.NewNRGBA(image.Rect(0, 0, cw, ch))
	for y := 0; y < ch; y++ {
		for x := 0; x < cw; x++ {
			dst.Set(x, y, src.At(rect.Min.X+x, rect.Min.Y+y))
		}
	}
	return dst
}

