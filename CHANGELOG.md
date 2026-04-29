# Changelog

All notable changes to this project will be documented in this file.

## Unreleased — Notes from the YOLO26 evaluation (2026-04-29)

- Evaluated the upstream-newer detector model `yolo26s-plate-detect.pt`
  (we0091234/yolo26-plate, Ultralytics YOLO26 pose, 64MB) as a drop-in
  replacement for the YOLOv5-face detector currently in use.
- Result on the same 1000-image test set:
  - pass rate: `92.8% -> 82.4%` (-10.4pt regression)
  - QPS: `57 -> 9.7` (5.8x slower due to model size)
  - Failure mode: YOLO26 was trained for full-frame scenes and on already-
    cropped plate inputs (e.g. 192x176) it consistently emits keypoints that
    chop the leftmost province character. 47 cases out of 176 mismatches show
    `粤 -> L` at position 0; same pattern explains `D/F/B/A/W` substitutions.
- Tried multiple ensemble policies on top of v0.7.0 (always-run WE,
  loosened/strict alignment requirements, exact-insertion match). None went
  above 92.9%. The empirical fusion ceiling on this dataset is `~92.9%`
  even though the "any-model-correct" oracle ceiling is `~98.5%`. Closing
  that gap requires a runtime signal we don't currently have (e.g. a
  learned arbiter, or per-character confidence calibration).
- Conclusion: keep `v0.7.0` as the production baseline. Path to 98%+ now
  squarely depends on either a higher-timestep recognizer or a small
  learned arbiter trained on the failure set.

## v0.7.0

- Add a dual-recognizer ensemble path. The HyperLPR3 v3 SVTR-LCNet model
  remains the primary recognizer; an optional alternate plate_rec_color
  model (we0091234, ONNX dual head plate+color) is loaded when present and
  used as a conservative cross-check on suspicious primary outputs.
- Engine plumbing changes:
  - extend the ONNX loader with a dual-output session pool
    (`loadModelDual`, `RunInferenceDual`)
  - add `RecognizerWE` with the upstream BGR mean=0.588/std=0.193 preprocess,
    78-token charset (民/航/危/险/品), and the 5-class color head
  - add `recognition.recognizer_we` config path; missing model file is
    non-fatal and simply disables the ensemble
- Conservative ensemble policy (kept narrow on purpose to protect speed and
  avoid regressions, see analysis below):
  - WE only runs when the primary plate is empty / structurally invalid /
    mid-confidence (< 0.93 for length 7 or non-NEV 8)
  - WE result is accepted only when its province char matches the primary's
    and the candidate forms a clean length-7<->NEV-8 length recovery with
    high alignment overlap
- Model investigation findings (recorded so future iterations can build on them):
  - HyperLPR3 v3 recognizer: PaddleOCR PP-OCR v3 SVTR-LCNet, fixed
    `1x3x48x160` input, fixed `1x20x78` output (CTC). Cannot grow timesteps
    without retraining.
  - we0091234 plate_rec_color: shape `1x21x78` (only one extra timestep)
    plus a 5-class color head. Trained on a different/wider dataset.
  - Empirical on the 1000-image set:
    - HyperLPR3 alone: 92.7% pass, 76 mismatch
    - we0091234 alone (hybrid detect+direct): 92.0% pass, 80 mismatch
    - Failure overlap: only 15 cases fail in both. Theoretical fusion
      ceiling on this set is ~98.5%.
  - Why the realized gain is only +0.1pt (92.7 -> 92.8): aggressive fusion
    overrides plenty of correct primary answers, while a strict province-
    locked, alignment-required policy is needed to keep precision. The
    safest conservative rules close most of the gap on length-7<->8 NEV
    cases but cannot rescue non-NEV 8-char heavy-vehicle plates without
    introducing more regressions.
  - To break past ~93% on this dataset, the most promising paths are
    swapping in a higher-timestep recognition model (e.g. PP-OCR v4 / a
    plate-specific retrained CRNN with 30+ steps), or training a small
    arbiter that learns when to trust which recognizer.
- Benchmark gate (must not regress accuracy or QPS vs v0.6.16):
  - pass rate: `92.7% -> 92.8%` (+0.1pt, ~1 sample)
  - QPS: stable at `~57` (within baseline band)

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
