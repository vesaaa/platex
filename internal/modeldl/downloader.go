package modeldl

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const defaultDownloadTimeout = 5 * time.Minute

// DefaultModelsZIPs are tried in order (first successful source wins).
// Users can override with PLATEX_MODELS_ZIP_URLS (comma-separated URLs).
var DefaultModelsZIPs = []string{
	"https://github.com/vesaaa/platex/releases/latest/download/20230229.zip",
	"https://github.com/vesaaa/platex/releases/download/v1.0.9/20230229.zip",
	"https://hyperlpr.tunm.top/raw/20230229.zip",
	"http://hyperlpr.tunm.top/raw/20230229.zip",
}

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

	sources := resolveModelZipSources()
	slog.Info("Starting download of HyperLPR3 models archive...", "sources", strings.Join(sources, ","))

	// Create temp file for zip
	tempFile, err := os.CreateTemp("", "hyperlpr_models_*.zip")
	if err != nil {
		return fmt.Errorf("create temp zip file: %w", err)
	}
	tempZipPath := tempFile.Name()
	defer os.Remove(tempZipPath)

	if err := downloadFirstAvailableZip(tempFile, sources); err != nil {
		tempFile.Close()
		return err
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

func resolveModelZipSources() []string {
	if raw, ok := os.LookupEnv("PLATEX_MODELS_ZIP_URLS"); ok {
		parts := strings.Split(raw, ",")
		out := make([]string, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				out = append(out, p)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return DefaultModelsZIPs
}

func downloadFirstAvailableZip(tempFile *os.File, sources []string) error {
	client := &http.Client{Timeout: defaultDownloadTimeout}
	var lastErr error

	for _, src := range sources {
		slog.Info("Trying model source", "url", src)
		if _, err := tempFile.Seek(0, 0); err != nil {
			return fmt.Errorf("seek temp zip file: %w", err)
		}
		if err := tempFile.Truncate(0); err != nil {
			return fmt.Errorf("truncate temp zip file: %w", err)
		}

		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, src, nil)
		if err != nil {
			lastErr = fmt.Errorf("create request for %s: %w", src, err)
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("download zip from %s: %w", src, err)
			slog.Warn("Model source failed", "url", src, "error", err)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("download zip from %s: unexpected status code %d", src, resp.StatusCode)
			resp.Body.Close()
			slog.Warn("Model source failed", "url", src, "status", resp.StatusCode)
			continue
		}

		if _, err := io.Copy(tempFile, resp.Body); err != nil {
			resp.Body.Close()
			lastErr = fmt.Errorf("save zip file from %s: %w", src, err)
			slog.Warn("Model source failed", "url", src, "error", err)
			continue
		}
		resp.Body.Close()
		slog.Info("Model source succeeded", "url", src)
		return nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no model sources configured")
	}
	return fmt.Errorf("download zip failed from all sources: %w", lastErr)
}
