// Package types defines all data structures for the LPR system.
package types

// PlateColor represents the color of a license plate.
type PlateColor int

const (
	ColorOther  PlateColor = 0  // 0 其他
	ColorWhite  PlateColor = 1  // 1 白色
	ColorBlack  PlateColor = 2  // 2 黑色
	ColorBlue   PlateColor = 3  // 3 蓝色
	ColorYellow PlateColor = 4  // 4 黄色
	ColorGreen  PlateColor = 5  // 5 绿色
	ColorRed    PlateColor = 6  // 6 红色
	ColorOrange PlateColor = 7  // 7 橙色
	ColorPurple PlateColor = 8  // 8 紫色
	ColorGrey   PlateColor = 9  // 9 灰色
	ColorSilver PlateColor = 10 // 10 银色
	ColorBrown  PlateColor = 11 // 11 棕色
	ColorPink   PlateColor = 12 // 12 粉色
)

// ColorNames maps PlateColor to Chinese name.
var ColorNames = map[PlateColor]string{
	ColorOther:  "其他",
	ColorWhite:  "白色",
	ColorBlack:  "黑色",
	ColorBlue:   "蓝色",
	ColorYellow: "黄色",
	ColorGreen:  "绿色",
	ColorRed:    "红色",
	ColorOrange: "橙色",
	ColorPurple: "紫色",
	ColorGrey:   "灰色",
	ColorSilver: "银色",
	ColorBrown:  "棕色",
	ColorPink:   "粉色",
}

// PlateType represents the type of license plate.
type PlateType string

const (
	PlateTypeStandard7 PlateType = "standard_7" // 标准7位
	PlateTypeNewEnergy PlateType = "new_energy" // 新能源8位
	PlateTypeUnknown   PlateType = "unknown"
)

// PlateResult represents a single recognized license plate.
type PlateResult struct {
	PlateNumber string     `json:"plate_number"`
	Color       PlateColor `json:"color"`
	ColorName   string     `json:"color_name"`
	Confidence  float32    `json:"confidence"`
	Type        PlateType  `json:"type"`
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
	Mode    string           `json:"mode,omitempty"` // "auto" (default), "crop", or "full"
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
	MaxPlates         int     `json:"max_plates,omitempty"`
	MinConfidence     float32 `json:"min_confidence,omitempty"`
	ResizeMode        string  `json:"resize_mode,omitempty"`          // "letterbox" (default) or "stretch"
	FullEarlyStopConf float32 `json:"full_early_stop_conf,omitempty"` // full模式直识别命中即停阈值
}

// RecognizeResponse represents the HTTP response body.
type RecognizeResponse struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Data    *RecognizeData `json:"data,omitempty"`
}

// RecognizeData contains the recognition results.
type RecognizeData struct {
	Results        []ImageResult `json:"results"`
	TotalElapsedMs int64         `json:"total_elapsed_ms"`
	Mode           string        `json:"mode"`
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
	TotalRequests int64   `json:"total_requests"`
	TotalImages   int64   `json:"total_images"`
	TotalPlates   int64   `json:"total_plates"`
	AvgLatencyMs  float64 `json:"avg_latency_ms"`
	SuccessRate   float64 `json:"success_rate"`
	CurrentQPS    float64 `json:"current_qps"`
}
