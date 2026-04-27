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
docker run -d -p 8080:8080 ghcr.io/vesaaa/platex:v0.5.43
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

# 车牌识别 (Base64, 不传 mode 默认 auto 路由)
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
    "mode": "auto",
    "options": {
      "min_confidence": 0.6,
      "max_plates": 10,
      "resize_mode": "auto"
    }
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
      "min_confidence": 0.6,
      "full_early_stop_conf": 0.65
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
  "mode": "auto",
  "options": {
    "min_confidence": 0.6,
    "max_plates": 10,
    "resize_mode": "auto"
  }
}
```

`type=url` 需要在配置中启用 `engine.url.enabled`（默认已启用），并受以下参数控制：
- `engine.url.fetch_timeout_ms`：下载超时（默认 `1200`）
- `engine.url.max_image_bytes`：单图大小上限（默认 `5242880`）
- `engine.url.max_fetch_concurrency`：URL 拉图并发上限（默认 `16`，与识别 worker 独立）
- `engine.url.max_fetch_retries`：下载失败最大重试次数（默认 `2`）
- `engine.url.retry_backoff_ms`：重试基础退避时间（默认 `120`）
- `engine.url.max_idle_conns` / `engine.url.max_idle_conns_per_host`：HTTP 连接池（默认 `256` / `64`）
- `engine.url.block_private_ip`：是否阻止私网/回环地址（默认 `true`，防 SSRF）
- `engine.url.allowed_schemes`：允许协议（默认 `http`/`https`）

批量请求建议同时配置 `engine.submit_timeout_ms`（默认 `300ms`）：  
当 worker 繁忙时，任务会短时阻塞等待队列而不是立即失败，超时后返回 busy 错误，兼顾吞吐与成功率。

并行性能调优建议（Linux）：
- `engine.workers=0` 时自动使用 `CPU核心数`；也可通过启动参数 `-workers N` 手动覆盖
- `PLATEX_MODEL_POOL_SIZE`：ONNX Session 池大小（默认自动 `max(2, min(6, CPU/4))`，可手动设 `4~6`）
- `engine.onnx.threads_per_session` 建议保持 `1`，优先通过 `workers` 与 `PLATEX_MODEL_POOL_SIZE` 提升并行
- URL 批量场景可适当提高 `engine.url.max_fetch_concurrency`（如 `16 -> 32`）

Docker 场景可直接通过环境变量覆盖关键并发参数（无需改配置文件）：
- `PLATEX_WORKERS`：覆盖 `engine.workers`
- `PLATEX_MODEL_POOL_SIZE`：覆盖 ONNX session pool 大小
- `PLATEX_ONNX_THREADS_PER_SESSION`：覆盖 `engine.onnx.threads_per_session`
- `PLATEX_URL_MAX_FETCH_CONCURRENCY`：覆盖 `engine.url.max_fetch_concurrency`
- `PLATEX_SUBMIT_TIMEOUT_MS`：覆盖 `engine.submit_timeout_ms`
- `PLATEX_FULL_EARLY_STOP_CONF`：覆盖 `engine.recognition.full_early_stop_conf`（默认 `0.65`，用于 full 模式直识别命中即停）

按容器内存规格的建议参数（CPU 充足、以吞吐优先）：
- 8G 容器：
  - `PLATEX_WORKERS=8`
  - `PLATEX_MODEL_POOL_SIZE=4`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=16`
  - `PLATEX_SUBMIT_TIMEOUT_MS=200`
- 16G 容器：
  - `PLATEX_WORKERS=12`
  - `PLATEX_MODEL_POOL_SIZE=6`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=24`
  - `PLATEX_SUBMIT_TIMEOUT_MS=200`
- 32G 容器：
  - `PLATEX_WORKERS=18`
  - `PLATEX_MODEL_POOL_SIZE=8`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=32`
  - `PLATEX_SUBMIT_TIMEOUT_MS=150`

按 CPU 核心数的建议参数（Docker）：
- 4C：
  - `PLATEX_WORKERS=4`
  - `PLATEX_MODEL_POOL_SIZE=2`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=8`
  - `PLATEX_SUBMIT_TIMEOUT_MS=250`
- 8C：
  - `PLATEX_WORKERS=8`
  - `PLATEX_MODEL_POOL_SIZE=4`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=16`
  - `PLATEX_SUBMIT_TIMEOUT_MS=200`
- 16C：
  - `PLATEX_WORKERS=16`
  - `PLATEX_MODEL_POOL_SIZE=6`
  - `PLATEX_ONNX_THREADS_PER_SESSION=1`
  - `PLATEX_URL_MAX_FETCH_CONCURRENCY=24`
  - `PLATEX_SUBMIT_TIMEOUT_MS=150`

说明：
- 上述值是安全起点，最终以你的压测结果（QPS、P95、错误率）为准。
- 调优顺序建议：先调 `PLATEX_MODEL_POOL_SIZE`，再调 `PLATEX_WORKERS`，`threads_per_session` 通常保持 `1`。
- 如果出现 CPU 抢占明显、P95 抖动增大，优先降低 `PLATEX_WORKERS` 或 `PLATEX_MODEL_POOL_SIZE`。

`mode=full` 的检测后处理参数由配置文件 `engine.detection` 控制：
- `engine.detection.conf_threshold`: 检测置信度阈值（默认 `0.30`）
- `engine.detection.iou_threshold`: NMS IoU 阈值（默认 `0.45`）
- `engine.detection.max_candidates`: NMS 后最多保留候选框数量（默认 `50`）

识别路由说明（`mode=auto`，默认）：
- 输入图宽高比在 `3.33 ±10%`（约 `3.0 ~ 3.66`）范围内，自动按 `crop` 流程识别（更快）
- 不在该范围内，自动按 `full` 流程识别（先检测再识别）

`engine.recognition.full_max_plates`（默认 `3`）用于限制 full 模式默认识别候选框数量，可显著影响 full 吞吐；
如需更高召回可调大，若追求 QPS 建议保持 `2~4`。

`engine.recognition.full_early_stop_conf`（默认 `0.65`）用于 full 模式“直识别命中即停”阈值：
- 仅在 `mode=full` 的直识别短路阶段生效
- 识别置信度达到该阈值后可直接返回，避免继续走检测与重搜索
- 覆盖优先级：请求 `options.full_early_stop_conf` > 环境变量 `PLATEX_FULL_EARLY_STOP_CONF` > 配置文件默认值

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
| `full_early_stop_conf` | float | 0.65 | full 模式直识别“命中即停”阈值（越低越激进，越高越保守） |

`mode` 参数说明（请求顶层）：
- `auto`（默认）：按图像比例自动路由到 `crop` 或 `full`
- `crop`：仅走车牌小图识别路径
- `full`：先检测再识别

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
docker run -d -p 8080:8080 ghcr.io/vesaaa/platex:v0.5.43
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
