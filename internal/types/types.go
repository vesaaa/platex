// Package types defines all data structures for the LPR system.
package types

// PlateColor represents the color of a license plate.
type PlateColor int

const (
	ColorBlue  PlateColor = 0 // 蓝色 - 普通小型车
	ColorYellow PlateColor = 1 // 黄色 - 大型车辆/教练车
	ColorGreen PlateColor = 2 // 绿色 - 新能源
	ColorBlack PlateColor = 3 // 黑色 - 使馆/港澳
	ColorWhite PlateColor = 4 // 白色 - 军用/警用
	ColorOther PlateColor = 5 // 其他
)

// ColorNames maps PlateColor to Chinese name.
var ColorNames = map[PlateColor]string{
	ColorBlue:   "蓝色",
	ColorYellow: "黄色",
	ColorGreen:  "绿色",
	ColorBlack:  "黑色",
	ColorWhite:  "白色",
	ColorOther:  "其他",
}

// PlateType represents the type of license plate.
type PlateType string

const (
	PlateTypeStandard7 PlateType = "standard_7" // 标准7位
	PlateTypeNewEnergy PlateType = "new_energy"  // 新能源8位
	PlateTypeUnknown   PlateType = "unknown"
)

// PlateResult represents a single recognized license plate.
type PlateResult struct {
	PlateNumber     string     `json:"plate_number"`
	Color           PlateColor `json:"color"`
	ColorName       string     `json:"color_name"`
	Confidence      float32    `json:"confidence"`
	CharConfidences []float32  `json:"char_confidences,omitempty"`
	Type            PlateType  `json:"type"`
	BBox            [4]int     `json:"bbox,omitempty"` // x1, y1, x2, y2 (full image mode only)
}

// ImageResult represents the recognition result for a single image.
type ImageResult struct {
	ID        string        `json:"id"`
	Plates    []PlateResult `json:"plates"`
	ElapsedMs int64         `json:"elapsed_ms"`
	Error     string        `json:"error,omitempty"`
}

// RecognizeRequest represents the HTTP request body.
type RecognizeRequest struct {
	Images  []ImageInput     `json:"images"`
	Mode    string           `json:"mode,omitempty"` // "crop" (default) or "full"
	Options *RecognizeOption `json:"options,omitempty"`
}

// ImageInput represents a single image input.
type ImageInput struct {
	ID   string `json:"id"`
	Type string `json:"type"` // "base64", "path", "url"
	Data string `json:"data"`
}

// RecognizeOption represents optional parameters.
type RecognizeOption struct {
	MaxPlates     int     `json:"max_plates,omitempty"`
	MinConfidence float32 `json:"min_confidence,omitempty"`
}

// RecognizeResponse represents the HTTP response body.
type RecognizeResponse struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Data    *RecognizeData   `json:"data,omitempty"`
}

// RecognizeData contains the recognition results.
type RecognizeData struct {
	Results       []ImageResult `json:"results"`
	TotalElapsedMs int64        `json:"total_elapsed_ms"`
	Mode          string        `json:"mode"`
}

// HealthResponse represents the health check response.
type HealthResponse struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    *HealthData `json:"data,omitempty"`
}

// HealthData contains health check details.
type HealthData struct {
	Status        string `json:"status"`
	Version       string `json:"version"`
	ModelsLoaded  bool   `json:"models_loaded"`
	WorkerCount   int    `json:"worker_count"`
	UptimeSeconds int64  `json:"uptime_seconds"`
}

// StatsResponse represents the stats response.
type StatsResponse struct {
	Code    int        `json:"code"`
	Message string     `json:"message"`
	Data    *StatsData `json:"data,omitempty"`
}

// StatsData contains runtime statistics.
type StatsData struct {
	TotalRequests    int64   `json:"total_requests"`
	TotalImages      int64   `json:"total_images"`
	TotalPlates      int64   `json:"total_plates"`
	AvgLatencyMs     float64 `json:"avg_latency_ms"`
	SuccessRate      float64 `json:"success_rate"`
	CurrentQPS       float64 `json:"current_qps"`
}
