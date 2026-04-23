package engine

// HyperLPR3 token list - must match the exact order from hyperlpr3/common/tokenize.py
// Index 0 is the CTC blank token.
// The model output has 6625 classes, but only the first 78 are valid characters.
var plateChars = []string{
	"",   // 0: CTC blank
	"'",  // 1: separator (not used in plates)
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", // 2-11
	"A", "B", "C", "D", "E", "F", "G", "H", "J", // 12-20
	"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", // 21-36
	"云", "京", "冀", "吉", "学", "宁", // 37-42
	"川", "挂", "新", "晋", "桂", "民", "沪", "津", "浙", "渝", // 43-52
	"港", "湘", "琼", "甘", "皖", "粤", "航", "苏", "蒙", "藏", "警", "豫", // 53-64
	"贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", "黑", // 65-73
	"领", "使", "澳", // 74-76
}

// plateColorLabels maps model output index to color code.
var plateColorLabels = []int{
	3, // Blue (Model output 0 -> ColorBlue 3)
	4, // Yellow (Model output 1 -> ColorYellow 4)
	5, // Green (Model output 2 -> ColorGreen 5)
	2, // Black (Model output 3 -> ColorBlack 2)
	1, // White (Model output 4 -> ColorWhite 1)
	0, // Other (Model output 5 -> ColorOther 0)
}
