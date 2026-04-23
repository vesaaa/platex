package engine

// Chinese license plate character set used by HyperLPR3 CRNN model.
// Index 0 is the CTC blank token.
// This charset covers standard 7-char and 8-char (new energy) plates.
var plateChars = []string{
	"#",  // 0: CTC blank
	"京", "沪", "津", "渝", "冀", "豫", "云", "辽", "黑", "湘", // 1-10
	"皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋", // 11-20
	"蒙", "陕", "吉", "闽", "贵", "粤", "川", "青", "藏", "琼", // 21-30
	"宁",                                                         // 31
	"A", "B", "C", "D", "E", "F", "G", "H", "J", "K",           // 32-41
	"L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",           // 42-51
	"W", "X", "Y", "Z",                                          // 52-55
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",           // 56-65
	"警", "学", "挂", "港", "澳", "使", "领", "应", "急",          // 66-74
}

// plateColorLabels maps model output index to color code.
var plateColorLabels = []int{
	0, // Blue
	1, // Yellow
	2, // Green
	3, // Black
	4, // White
	5, // Other
}
