package engine

import "github.com/vesaa/platex/internal/types"

// weChars is the character table used by the we0091234 plate_rec_color model.
// Index 0 is the CTC blank token. The order MUST match the upstream
// onnx_infer.py exactly (see we0091234/Chinese_license_plate_detection_recognition).
var weChars = []string{
	"#", // 0: CTC blank
	"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
	"苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
	"桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
	"学", "警", "港", "澳", "挂", "使", "领", "民", "航", "危",
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
	"P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
	"险", "品",
}

// weColorLabels maps the 5-class output of plate_rec_color to our internal
// PlateColor codes. Order is fixed by the upstream model definition.
var weColorLabels = []int{
	int(types.ColorBlack),  // 0
	int(types.ColorBlue),   // 1
	int(types.ColorGreen),  // 2
	int(types.ColorWhite),  // 3
	int(types.ColorYellow), // 4
}
