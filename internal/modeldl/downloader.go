package modeldl

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// DefaultModelURLs points to project release assets for model files.
var DefaultModelURLs = map[string]string{
	"plate_rec.onnx":    "https://github.com/vesaaa/platex/releases/download/v1.0.0-models/plate_rec.onnx",
	"plate_detect.onnx": "https://github.com/vesaaa/platex/releases/download/v1.0.0-models/plate_detect.onnx",
	"plate_color.onnx":  "https://github.com/vesaaa/platex/releases/download/v1.0.0-models/plate_color.onnx",
}

// DownloadModels downloads required model files directly from release URLs.
func DownloadModels(modelsDir string) error {
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return fmt.Errorf("create models directory: %w", err)
	}

	client := &http.Client{Timeout: 5 * time.Minute}

	for targetName, modelURL := range DefaultModelURLs {
		targetPath := filepath.Join(modelsDir, targetName)
		if _, err := os.Stat(targetPath); err == nil {
			slog.Info("Model already exists. Skip downloading.", "model", targetName)
			continue
		}

		slog.Info("Downloading model file...", "model", targetName, "url", modelURL)
		if err := downloadToFile(client, modelURL, targetPath); err != nil {
			return fmt.Errorf("download %s: %w", targetName, err)
		}
	}

	slog.Info("All models are ready", "dir", modelsDir)
	return nil
}

func downloadToFile(client *http.Client, modelURL, targetPath string) error {
	resp, err := client.Get(modelURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	tmpPath := targetPath + ".tmp"
	outFile, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer func() {
		outFile.Close()
		_ = os.Remove(tmpPath)
	}()

	if _, err := io.Copy(outFile, resp.Body); err != nil {
		return err
	}
	if err := outFile.Sync(); err != nil {
		return err
	}
	if err := outFile.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, targetPath)
}
