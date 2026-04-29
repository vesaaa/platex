# Changelog

All notable changes to this project will be documented in this file.

## v0.6.15

- Major accuracy lever: parse the 4 plate corner keypoints emitted by the detector
  (yolov5face-style 15-column head) and apply a 4-point perspective warp before
  recognizer inference, fixing the long-standing tilt/perspective failure class.
- Implement pure-Go homography solver and bilinear-sampling warp without
  introducing any new C/C++ dependency.
- Two-tier full-mode pipeline:
  - high-confidence detection (score >= 0.65) uses a single warp-only recognition
    short-circuit to keep the path inexpensive
  - lower-confidence cases run both warp and axis-aligned crop, picking the best
    candidate by structural score
- Add tests covering homography correctness, warp orientation, keypoint ordering,
  and degenerate-quad rejection.
- Benchmark on 1000-image test set under the dual gate
  (must not regress on accuracy or QPS):
  - pass rate: `92.2%` -> `92.7%` (+0.5pt), 6 tilted/perspective samples recovered
  - QPS: stays in baseline `~56` band
- Document model architecture for future iterations:
  - recognizer: PaddleOCR PP-OCR v3 SVTR-LCNet, fixed `1x3x48x160` input,
    `1x20x78` output (CTC) — wider input to grow timesteps is not an option
    without retraining
  - detector: yolov5face-style head exposes 4-corner keypoints in columns 5..12
  - color model: actual input is `96x96` and output has 3 classes — current code
    passes `96x24` and assumes 6 classes (kept on heuristic fallback for now;
    fix scheduled for the next iteration)

## v0.6.14

- Keep the best current baseline under the dual gate (accuracy must not drop, speed must not regress):
  pass rate stays at `92.2%`, while throughput remains in the `57.5~58.8 QPS` range on the 1000-image benchmark.
- Upgrade recovery rerank with low-cost, low-confidence-only binary-feature corrections:
  - data-driven confusion scorer (`D/0`, `7/1`, `E/L`) with conservative feature gating
  - 7<->8 length mismatch rerank candidate (collapsed-NEV expand / weak 8th-char trim)
  - tail `I/1` ambiguity candidate for hard tilted samples
  - tail `8/4` micro-correction (last-char low-confidence only)
- Add/expand unit tests for the above rerank paths and confidence guards.
- Record attempted but rejected strategies (kept out of serving path due gate failure):
  - full-mode broader recovery trigger (`collapsed NEV conf < 0.97`): no stable gain, QPS dropped to ~53
  - wider collapsed-NEV rerank gate (`mean conf < 0.995`): no accuracy gain, QPS lower than baseline
  - online multi-angle/contrast fallback in full pipeline: accuracy unchanged (`92.2%`), QPS dropped to ~39.6
  - narrow online fallback (2 variants): accuracy unchanged (`92.2%`), QPS dropped to ~51.3

## v0.5.23

- Tighten recovery trigger for normal 7/8-char mainland-prefixed plates.
- Keep clear standard outputs on fast path to avoid over-correction regressions.
- Keep recovery path for malformed or clearly abnormal low-confidence cases.

## v0.5.7

- Add `type=url` input support for direct HTTP/HTTPS image recognition.
- Add URL fetch concurrency control (`engine.url.max_fetch_concurrency`) separated from OCR workers.
- Add URL fetch timeout and max image bytes limits.
- Add basic SSRF protection for private/loopback addresses with scheme allowlist.
