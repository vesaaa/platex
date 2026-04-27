// Package api implements the HTTP API server for the LPR system.
package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/vesaa/platex/internal/engine"
	"github.com/vesaa/platex/internal/types"
)

// Server is the HTTP API server.
type Server struct {
	engine    *engine.Engine
	mux       *http.ServeMux
	startTime time.Time
	version   string
}

// NewServer creates a new API server.
func NewServer(eng *engine.Engine, version string) *Server {
	s := &Server{
		engine:    eng,
		mux:       http.NewServeMux(),
		startTime: time.Now(),
		version:   version,
	}
	s.registerRoutes()
	return s
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	return s.withMiddleware(s.mux)
}

// registerRoutes sets up the API routes.
func (s *Server) registerRoutes() {
	s.mux.HandleFunc("POST /api/v1/recognize", s.handleRecognize)
	s.mux.HandleFunc("GET /api/v1/health", s.handleHealth)
	s.mux.HandleFunc("GET /api/v1/stats", s.handleStats)
	s.mux.HandleFunc("GET /api/v1/info", s.handleInfo)
}

// withMiddleware wraps the handler with logging and recovery middleware.
func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Recovery
		defer func() {
			if err := recover(); err != nil {
				slog.Error("Panic recovered", "error", err, "path", r.URL.Path)
				writeJSON(w, http.StatusInternalServerError, types.RecognizeResponse{
					Code:    500,
					Message: "internal server error",
				})
			}
		}()

		// CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)

		slog.Info("Request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}

// handleRecognize handles POST /api/v1/recognize
func (s *Server) handleRecognize(w http.ResponseWriter, r *http.Request) {
	// Read body with size limit (100MB)
	body, err := io.ReadAll(io.LimitReader(r.Body, 100*1024*1024))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, types.RecognizeResponse{
			Code:    400,
			Message: fmt.Sprintf("read body: %v", err),
		})
		return
	}

	var req types.RecognizeRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, types.RecognizeResponse{
			Code:    400,
			Message: fmt.Sprintf("invalid JSON: %v", err),
		})
		return
	}

	if len(req.Images) == 0 {
		writeJSON(w, http.StatusBadRequest, types.RecognizeResponse{
			Code:    400,
			Message: "no images provided",
		})
		return
	}

	// Set default mode
	mode := req.Mode
	if mode == "" {
		mode = "auto"
	}

	start := time.Now()

	// Process images
	results := s.engine.RecognizeBatch(req.Images, mode, req.Options)

	writeJSON(w, http.StatusOK, types.RecognizeResponse{
		Code:    0,
		Message: "success",
		Data: &types.RecognizeData{
			Results:        results,
			TotalElapsedMs: time.Since(start).Milliseconds(),
			Mode:           mode,
		},
	})
}

// handleHealth handles GET /api/v1/health
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	uptime := int64(time.Since(s.startTime).Seconds())

	writeJSON(w, http.StatusOK, types.HealthResponse{
		Code:    0,
		Message: "ok",
		Data: &types.HealthData{
			Status:        "running",
			Version:       s.version,
			ModelsLoaded:  true,
			UptimeSeconds: uptime,
		},
	})
}

// handleStats handles GET /api/v1/stats
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := s.engine.GetStats()

	writeJSON(w, http.StatusOK, types.StatsResponse{
		Code:    0,
		Message: "ok",
		Data:    stats,
	})
}

// handleInfo handles GET /api/v1/info
func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	info := map[string]interface{}{
		"version": s.version,
		"supported_plate_types": []string{
			"standard_7 - 标准7位车牌 (蓝/黄/黑/白)",
			"new_energy  - 新能源8位车牌 (绿)",
		},
		"supported_colors": map[int]string{
			0: "其他", 1: "白色", 2: "黑色", 3: "蓝色", 4: "黄色", 5: "绿色",
			6: "红色", 7: "橙色", 8: "紫色", 9: "灰色", 10: "银色", 11: "棕色", 12: "粉色",
		},
		"input_types": []string{"base64", "path", "url"},
		"modes":       []string{"auto", "crop", "full"},
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"code":    0,
		"message": "ok",
		"data":    info,
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("Failed to write response", "error", err)
	}
}
