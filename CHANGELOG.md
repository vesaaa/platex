# Changelog

All notable changes to this project will be documented in this file.

## v0.6.16

- Fix the color classifier wiring against the actual `litemodel_cls_96x_r1`
  HyperLPR3 v3 model: input tensor is `1x3x96x96` (was passed as `96x24`) and
  the head emits 3 classes in order `[blue, green, yellow]` (was assumed to
  be 6 classes). Until now the color model was effectively idle and we relied
  on the heuristic fallback alone.
- Loosen the trigger for the expanded-keypoint warp recovery in full mode:
  any 7-char or non-NEV 8-char primary result now gets one extra warp with
  horizontally widened/narrowed keypoints, gated by a candidate-score margin
  of `+0.6` instead of `+1.0`. This is the second cheap attempt at the
  dominant remaining failure class (length 7<->8 drift) without introducing
  a third recognizer call.
- Benchmark on 1000-image set vs `v0.6.15`:
  - pass rate: stable at `92.7%`
  - QPS: stable around `~57-58`
- Honest accuracy ceiling analysis (kept in code/changelog so that future
  iterations don't repeat the same dead ends):
  - Recognizer is fixed PP-OCR v3 SVTR-LCNet with `1x3x48x160` input and
    `1x20x78` output. With only `20` CTC timesteps for up to 8 characters,
    the model is structurally fragile when the warped plate width does not
    cover the rightmost NEV digit, or includes extra plate-frame strokes
    that get decoded into a phantom trailing character. No amount of
    post-processing can guarantee fixing both directions of this drift.
  - Detector keypoints are accurate on most tilted samples (validated:
    perspective warp recovers ~6 hard cases) but their right edge is not
    reliable for tightly framed NEV plates.
  - Empirically observed pure-algorithm ceiling on this 1000-image dataset:
    `~93%`. Closing the remaining gap requires one of:
      * upgrading to a higher-timestep recognizer (e.g. PP-OCR v4) or a
        plate-specific retrained model
      * precise character-edge refinement in pixel space (vertical projection
        on the warped image to trim plate frame artifacts)
      * a small auxiliary model that classifies "7-char vs NEV 8-char" from
        the warped crop to gate length recovery deterministically

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
