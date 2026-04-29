# Changelog

All notable changes to this project will be documented in this file.

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
