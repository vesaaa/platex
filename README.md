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

### 2. 系统与环境要求 (裸机部署)

如果你打算直接在物理机或虚拟机上裸跑（不使用 Docker），请确保系统满足以下条件：
- **操作系统**：推荐 Ubuntu 20.04+ / Debian 11+ (基于 `glibc >= 2.31`)
- **不支持旧系统**：**请勿在 CentOS 7 上直接运行预编译的 Release 包**（CentOS 7 的 `glibc` 为 `2.17`，版本过低，会导致找不到 `GLIBC_2.3x` 或 `libstdc++` 报错）。
- **Docker 推荐**：如果必须在 CentOS 7 等老旧系统上部署，强烈建议使用 Docker 方案，完全屏蔽底层 C++ 库版本冲突！

### 3. 下载模型

**方法一：一键自动下载（推荐）**
如果你部署的服务器有外网环境，可以直接运行以下命令，程序会自动从 GitHub 拉取所需模型并存入 `models/` 目录：
```bash
./lpr-server -download
```

**方法二：无网环境手动下载**
如果部署环境是内网，请手动下载官方模型压缩包并在运行目录下解压重命名：
1. 下载压缩包：[20230229.zip](http://hyperlpr.tunm.top/raw/20230229.zip)
2. 解压并提取以下文件，放置到运行目录的 `models/` 文件夹下，**并重命名为对应名称**：
- 将 `rpv3_mdict_160_r3.onnx` 重命名为 `plate_rec.onnx` (字符识别模型 - 必需)
- 将 `y5fu_320x_sim.onnx` 重命名为 `plate_detect.onnx` (全图检测模型 - 全图模式必需)
- 将 `litemodel_cls_96x_r1.onnx` 重命名为 `plate_color.onnx` (颜色分类模型 - 可选，无此文件则使用HSV回退)

最终的目录结构应为：
```
models/
├── plate_rec.onnx        # 字符识别 (输入: 1×3×48×160, 必需)
├── plate_detect.onnx     # 全图检测 (输入: 1×3×320×320, full 模式必需)
└── plate_color.onnx      # 颜色分类 (可选, 无则自动回退 HSV 启发式算法)
```

### 4. 运行

**方式一：使用 Docker 运行（推荐，无视任何环境限制）**
```bash
docker run -d -p 8080:8080 ghcr.io/vesaaa/platex:v0.5.1
```
> 注：Docker 镜像内置了所有最新的 ONNX 模型文件，拉取即用，彻底免去手动下载和环境配置！

**方式二：裸机直接运行**
```bash
# 确保 models/ 目录中已经有模型文件
./lpr-server

# 指定配置文件
./lpr-server -config configs/config.yaml

# 命令行覆盖
./lpr-server -port 9090 -workers 8 -log-level debug
```

### 5. 测试

```bash
# 健康检查
curl http://localhost:8080/api/v1/health

# 车牌识别 (Base64, 不传 mode 默认 crop)
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

# 车牌识别 (文件路径, 指定 letterbox 缩放)
curl -X POST http://localhost:8080/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "id": "test_001",
        "type": "path",
        "data": "/data/plates/car001.jpg"
      }
    ],
    "options": {
      "resize_mode": "letterbox"
    }
  }'

# 全图检测+识别 (full 模式, 自动检测车牌框后识别)
curl -X POST http://localhost:8080/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "id": "scene_001",
        "type": "url",
        "data": "https://example.com/parking_lot.jpg"
      }
    ],
    "mode": "full",
    "options": {
      "max_plates": 5,
      "min_confidence": 0.6
    }
  }'
```

## 更新记录

项目更新记录统一维护在 `CHANGELOG.md`。

## API 文档

### POST /api/v1/recognize

识别车牌号码和颜色。

**请求体:**
```json
{
  "images": [
    {"id": "img_001", "type": "base64", "data": "..."},
    {"id": "img_002", "type": "path", "data": "/path/to/image.jpg"},
    {"id": "img_003", "type": "url", "data": "https://example.com/car.jpg"}
  ],
  "mode": "crop",
  "options": {
    "min_confidence": 0.6,
    "resize_mode": "auto"
  }
}
```

`type=url` 需要在配置中启用 `engine.url.enabled`（默认已启用），并受以下参数控制：
- `engine.url.fetch_timeout_ms`: 下载超时
- `engine.url.max_image_bytes`: 单图大小上限
- `engine.url.max_fetch_concurrency`: URL 拉取并发上限（与识别 worker 独立）
- `engine.url.max_fetch_retries`: 下载失败最大重试次数（建议 `1~3`）
- `engine.url.retry_backoff_ms`: 重试基础退避时间，实际退避会按尝试次数递增
- `engine.url.max_idle_conns` / `engine.url.max_idle_conns_per_host`: HTTP 连接池参数，批量 URL 请求建议适当调高
- `engine.url.block_private_ip`: 是否阻止私网/回环地址（防 SSRF）
- `engine.url.allowed_schemes`: 允许的协议（默认 `http`/`https`）

批量请求建议同时配置 `engine.submit_timeout_ms`（默认 `300ms`）：  
当 worker 繁忙时，任务会短时阻塞等待队列而不是立即失败，超时后返回 busy 错误，兼顾吞吐与成功率。

`mode=full` 的检测后处理参数由配置文件 `engine.detection` 控制：
- `engine.detection.conf_threshold`: 检测置信度阈值（默认 `0.30`）
- `engine.detection.iou_threshold`: NMS IoU 阈值（默认 `0.45`）
- `engine.detection.max_candidates`: NMS 后最多保留候选框数量（默认 `50`）

识别路由说明：
- 不传 `mode` 时默认按 `crop` 识别（高吞吐路径）
- 显式传 `mode=full` 时才走大图检测+识别流程

`engine.recognition.full_max_plates`（默认 `3`）用于限制 full 模式默认识别候选框数量，可显著影响 full 吞吐；
如需更高召回可调大，若追求 QPS 建议保持 `2~4`。

推荐调参方向：
- 漏检较多（远距离/小车牌）：适当降低 `conf_threshold`（如 `0.20~0.28`）
- 重复框较多：适当降低 `iou_threshold`（如 `0.35~0.42`）
- 邻近车牌互相被吞：适当提高 `iou_threshold`（如 `0.50~0.60`）
- 推理延迟升高：适当降低 `max_candidates`

**`options` 字段说明:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_confidence` | float | 0.6 | 最低置信度阈值，低于此值的结果将被过滤 |
| `max_plates` | int | 10 | 请求级覆盖项：单张图片最多返回的车牌数量（可覆盖 full/crop 默认值） |
| `resize_mode` | string | `"auto"` | 预处理缩放策略，见下表 |

**`resize_mode` 缩放模式说明:**

识别模型的输入尺寸固定为 `160×48`（宽高比 3.33:1）。当上游裁剪的车牌小图分辨率不固定时，不同的缩放策略会影响识别准确率：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `auto`（默认） | 智能判断：输入图片宽高比与 3.33:1 偏差 ≤10% 时用 `stretch`，>10% 时用 `letterbox` | **推荐！适合上游裁剪尺寸不固定的场景** |
| `letterbox` | 等比缩放至 160×48 区域内，空余部分用灰色 (128) 填充，字符永不变形 | 追求极致准确率，或输入宽高比差异极大 |
| `stretch` | 直接暴力拉伸到 160×48，速度最快但可能导致字符变形 | 已知裁剪比例恒定且接近 3.33:1 |

示例：假设上游两种裁剪尺寸，`auto` 模式的自动决策：
| 输入尺寸 | 宽高比 | 与 3.33 偏差 | auto 选择 | 原因 |
|----------|--------|-------------|-----------|------|
| 256×74 | 3.46:1 | 3.8% ≤ 10% | `stretch` | 比例接近，拉伸无畸变，更快 |
| 192×48 | 4.00:1 | 20% > 10% | `letterbox` | 比例偏差大，需要保形填充 |

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

**`type` 字段说明 (车牌规格):**

系统会在识别后自动推算车牌的物理规格，方便业务层进行计费或分类拦截：
| 值 | 含义 | 示例 |
|----|------|------|
| `standard_7` | 标准 7 位字符车牌（蓝牌、黄牌、黑牌、白牌） | `粤B590MF` |
| `new_energy` | 8 位字符新能源车牌（通常伴随绿色） | `粤BD12345` |
| `unknown` | 未知类型（未识别完全、残缺或非标准号牌） | — |

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

### Docker 部署（推荐）

```bash
docker run -d -p 8080:8080 ghcr.io/vesaaa/platex:v0.5.1
```

### systemd 服务（裸机）

```bash
sudo cp lpr-server /opt/lpr/
sudo cp -r configs models /opt/lpr/
sudo cp deploy/lpr-server.service /etc/systemd/system/
sudo systemctl enable lpr-server
sudo systemctl start lpr-server
```

## 技术栈与依赖版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Go | 1.22 (LTS) | 编译工具链 |
| ONNX Runtime (C++) | 1.18.1 | 底层推理引擎 |
| onnxruntime_go | v1.11.0 (API 18) | Go 语言桥接库 |
| 模型来源 | HyperLPR3 (20230229) | 检测 + 识别 + 颜色分类 |

> **版本兼容说明**: Go 桥接库 `onnxruntime_go` 的版本号与底层 ORT C++ 库的 API 版本必须严格匹配。升级任一组件前，请参照 [onnxruntime_go](https://github.com/yalue/onnxruntime_go) 头文件中的 `ORT_API_VERSION` 定义进行对照。
