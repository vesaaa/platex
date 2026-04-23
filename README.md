# PlateX 高性能中国车牌识别服务

## 快速开始

### 1. 编译

```bash
# 本机编译
bash scripts/build.sh native

# Linux AMD64
bash scripts/build.sh linux-amd64

# Linux ARM64 (需要交叉编译工具链)
bash scripts/build.sh linux-arm64
```

### 2. 下载模型

**方法一：一键自动下载（推荐）**
如果你部署的服务器有外网环境，可以直接运行以下命令，程序会自动从 GitHub 拉取所需模型并存入 `models/` 目录：
```bash
./lpr-server -download
```

**方法二：无网环境手动下载**
如果部署环境是内网，请手动下载以下 HyperLPR3 预训练模型并放置到运行目录的 `models/` 文件夹下：
- [plate_rec.onnx](https://raw.githubusercontent.com/szad670401/HyperLPR/master/Prj-Python/hyperlpr3/resource/models/r2_mobile/plate_rec.onnx) (字符识别模型 - 必需)
- [plate_detect.onnx](https://raw.githubusercontent.com/szad670401/HyperLPR/master/Prj-Python/hyperlpr3/resource/models/r2_mobile/plate_detect.onnx) (全图检测模型 - 全图模式必需)
- [plate_color.onnx](https://raw.githubusercontent.com/szad670401/HyperLPR/master/Prj-Python/hyperlpr3/resource/models/r2_mobile/plate_color.onnx) (颜色分类模型 - 可选，若缺失则使用内置的HSV色彩推算作为回退)

最终的目录结构应为：
```
models/
├── plate_rec.onnx       
├── plate_detect.onnx    
└── plate_color.onnx     
```

### 3. 运行

```bash
# 使用默认配置
./lpr-server

# 指定配置文件
./lpr-server -config configs/config.yaml

# 命令行覆盖
./lpr-server -port 9090 -workers 8 -log-level debug
```

### 4. 测试

```bash
# 健康检查
curl http://localhost:8080/api/v1/health

# 车牌识别 (Base64)
curl -X POST http://localhost:8080/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "id": "test_001",
        "type": "base64",
        "data": "<base64_encoded_image>"
      }
    ],
    "mode": "crop"
  }'

# 车牌识别 (文件路径)
curl -X POST http://localhost:8080/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "id": "test_001",
        "type": "path",
        "data": "/data/plates/car001.jpg"
      }
    ]
  }'
```

## API 文档

### POST /api/v1/recognize

识别车牌号码和颜色。

**请求体:**
```json
{
  "images": [
    {"id": "img_001", "type": "base64", "data": "..."},
    {"id": "img_002", "type": "path", "data": "/path/to/image.jpg"}
  ],
  "mode": "crop",
  "options": {
    "min_confidence": 0.6
  }
}
```

**响应体:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "results": [
      {
        "id": "img_001",
        "plates": [
          {
            "plate_number": "粤B590MF",
            "color": 3,
            "color_name": "蓝色",
            "confidence": 0.97,
            "type": "standard_7"
          }
        ],
        "elapsed_ms": 6
      }
    ],
    "total_elapsed_ms": 12,
    "mode": "crop"
  }
}
```

### GET /api/v1/health

健康检查。

### GET /api/v1/stats

运行时统计 (QPS、延迟、成功率)。

### GET /api/v1/info

服务信息 (版本、支持的车牌类型、颜色编码)。

## 颜色编码

| 编码 | 颜色 |
|------|------|
| 0 | 其他 |
| 1 | 白色 |
| 2 | 黑色 |
| 3 | 蓝色 |
| 4 | 黄色 |
| 5 | 绿色 |
| 6 | 红色 |
| 7 | 橙色 |
| 8 | 紫色 |
| 9 | 灰色 |
| 10 | 银色 |
| 11 | 棕色 |
| 12 | 粉色 |

## 部署

### systemd 服务

```bash
sudo cp lpr-server /opt/lpr/
sudo cp -r configs models /opt/lpr/
sudo cp deploy/lpr-server.service /etc/systemd/system/
sudo systemctl enable lpr-server
sudo systemctl start lpr-server
```

## 技术栈

- **语言**: Go
- **推理引擎**: ONNX Runtime
- **模型**: HyperLPR3 (检测 + 识别 + 颜色)
- **HTTP**: Go 标准库 net/http
