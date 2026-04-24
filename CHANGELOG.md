# Changelog

All notable changes to this project will be documented in this file.

## v0.5.23

- Tighten recovery trigger for normal 7/8-char mainland-prefixed plates.
- Keep clear standard outputs on fast path to avoid over-correction regressions.
- Keep recovery path for malformed or clearly abnormal low-confidence cases.

## v0.5.7

- Add `type=url` input support for direct HTTP/HTTPS image recognition.
- Add URL fetch concurrency control (`engine.url.max_fetch_concurrency`) separated from OCR workers.
- Add URL fetch timeout and max image bytes limits.
- Add basic SSRF protection for private/loopback addresses with scheme allowlist.
