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

将 HyperLPR3 的 ONNX 模型文件放到 `models/` 目录:

```
models/
├── plate_rec.onnx       # 字符识别模型 (必需)
├── plate_detect.onnx    # 车牌检测模型 (全图模式需要)
└── plate_color.onnx     # 颜色分类模型 (可选，有HSV回退)
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
            "color": 0,
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

| 编码 | 颜色 | 车牌类型 |
|------|------|---------|
| 0 | 蓝色 | 普通小型车 |
| 1 | 黄色 | 大型车辆 |
| 2 | 绿色 | 新能源 |
| 3 | 黑色 | 使馆/港澳 |
| 4 | 白色 | 军用/警用 |
| 5 | 其他 | 无法分类 |

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
