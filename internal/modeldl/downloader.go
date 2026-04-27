package modeldl

import (
	"archive/zip"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
)

// DefaultModelsZIP is the official CDN link for the HyperLPR3 models archive.
const DefaultModelsZIP = "http://hyperlpr.tunm.top/raw/20230229.zip"

// RequiredModels maps the filename inside the ZIP to our target filename.
var RequiredModels = map[string]string{
	"rpv3_mdict_160_r3.onnx":    "plate_rec.onnx",
	"y5fu_320x_sim.onnx":        "plate_detect.onnx",
	"litemodel_cls_96x_r1.onnx": "plate_color.onnx",
}

// DownloadModels downloads the zip archive and extracts the required models.
func DownloadModels(modelsDir string) error {
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return fmt.Errorf("create models directory: %w", err)
	}

	// Check if already downloaded
	recPath := filepath.Join(modelsDir, "plate_rec.onnx")
	if _, err := os.Stat(recPath); err == nil {
		slog.Info("Models already exist. Skip downloading.")
		return nil
	}

	slog.Info("Starting download of HyperLPR3 models archive...")

	// Create temp file for zip
	tempFile, err := os.CreateTemp("", "hyperlpr_models_*.zip")
	if err != nil {
		return fmt.Errorf("create temp zip file: %w", err)
	}
	tempZipPath := tempFile.Name()
	defer os.Remove(tempZipPath)

	// Download zip
	resp, err := http.Get(DefaultModelsZIP)
	if err != nil {
		tempFile.Close()
		return fmt.Errorf("download zip: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tempFile.Close()
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	if _, err := io.Copy(tempFile, resp.Body); err != nil {
		tempFile.Close()
		return fmt.Errorf("save zip file: %w", err)
	}
	tempFile.Close()

	slog.Info("Download complete. Extracting models...")

	// Extract zip
	r, err := zip.OpenReader(tempZipPath)
	if err != nil {
		return fmt.Errorf("open zip file: %w", err)
	}
	defer r.Close()

	extractedCount := 0
	for _, f := range r.File {
		basename := filepath.Base(f.Name)
		targetName, ok := RequiredModels[basename]
		if !ok {
			continue
		}

		slog.Info("Extracting", "model", targetName, "from", basename)

		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("open file in zip: %w", err)
		}

		targetPath := filepath.Join(modelsDir, targetName)
		outFile, err := os.Create(targetPath)
		if err != nil {
			rc.Close()
			return fmt.Errorf("create extracted file: %w", err)
		}

		if _, err := io.Copy(outFile, rc); err != nil {
			outFile.Close()
			rc.Close()
			return fmt.Errorf("write extracted file: %w", err)
		}

		outFile.Close()
		rc.Close()
		extractedCount++
	}

	if extractedCount < len(RequiredModels) {
		return fmt.Errorf("only extracted %d of %d required models from zip", extractedCount, len(RequiredModels))
	}

	slog.Info("All models successfully downloaded and extracted to", "dir", modelsDir)
	return nil
}
