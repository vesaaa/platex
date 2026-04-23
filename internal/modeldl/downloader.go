package modeldl

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
)

// ModelURLs contains the default download URLs for the models.
// These point to the official HyperLPR3 repository or a fast CDN mirror.
var ModelURLs = map[string]string{
	"plate_rec.onnx":    "http://hyperlpr.tunm.top/raw/20230229/onnx/rpv3_mdict_160_r3.onnx",
	"plate_detect.onnx": "http://hyperlpr.tunm.top/raw/20230229/onnx/y5fu_320x_sim.onnx",
	"plate_color.onnx":  "http://hyperlpr.tunm.top/raw/20230229/onnx/litemodel_cls_96x_r1.onnx",
}

// DownloadModels downloads all required models to the specified directory.
func DownloadModels(targetDir string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("create models directory: %w", err)
	}

	for filename, url := range ModelURLs {
		targetPath := filepath.Join(targetDir, filename)
		
		// Check if file already exists
		if _, err := os.Stat(targetPath); err == nil {
			slog.Info("Model already exists, skipping", "file", filename)
			continue
		}

		slog.Info("Downloading model...", "file", filename, "url", url)
		if err := downloadFile(targetPath, url); err != nil {
			slog.Error("Failed to download model", "file", filename, "error", err)
			return err
		}
		slog.Info("Model downloaded successfully", "file", filename)
	}

	slog.Info("All models downloaded successfully to", "dir", targetDir)
	return nil
}

// downloadFile downloads a file from URL to the target path.
func downloadFile(filepath string, url string) error {
	// Create temporary file
	tmpFile := filepath + ".tmp"
	out, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	defer os.Remove(tmpFile) // Clean up if we fail

	// Get data
	resp, err := http.Get(url)
	if err != nil {
		out.Close()
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		out.Close()
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Write data
	if _, err = io.Copy(out, resp.Body); err != nil {
		out.Close()
		return err
	}
	
	// Close explicitly to ensure flush before rename
	if err := out.Close(); err != nil {
		return err
	}

	// Rename temp to final
	return os.Rename(tmpFile, filepath)
}
