#!/bin/sh
set -eu

CPU_DIR="/app/ort/cpu"
CUDA_DIR="/app/ort/cuda"
WANT_PROVIDER="${PLATEX_ONNX_PROVIDER:-cpu}"

link_onnx_libs() {
  src_dir="$1"
  base="${src_dir}/libonnxruntime.so"
  ver="$(ls "${src_dir}"/libonnxruntime.so.* 2>/dev/null | head -n 1 || true)"
  if [ ! -f "${base}" ] || [ -z "${ver}" ]; then
    return 1
  fi
  ln -sf "${base}" /app/libonnxruntime.so
  ln -sf "${ver}" /app/libonnxruntime.so.1.18.1
  return 0
}

use_cpu() {
  if ! link_onnx_libs "${CPU_DIR}"; then
    echo "[entrypoint] ERROR: CPU ONNX runtime libs not found in ${CPU_DIR}"
    exit 1
  fi
  export LD_LIBRARY_PATH="${CPU_DIR}:/app:${LD_LIBRARY_PATH:-}"
  export PLATEX_ONNX_PROVIDER="cpu"
  echo "[entrypoint] ONNX provider: cpu"
}

cuda_ready() {
  [ -f "${CUDA_DIR}/libonnxruntime_providers_cuda.so" ] || return 1
  ldd "${CUDA_DIR}/libonnxruntime_providers_cuda.so" 2>/dev/null | grep -q "not found" && return 1
  return 0
}

use_cuda() {
  if ! link_onnx_libs "${CUDA_DIR}"; then
    return 1
  fi
  export LD_LIBRARY_PATH="${CUDA_DIR}:/app:${LD_LIBRARY_PATH:-}"
  export PLATEX_ONNX_PROVIDER="cuda"
  echo "[entrypoint] ONNX provider: cuda (device ${PLATEX_ONNX_CUDA_DEVICE_ID:-0})"
  return 0
}

if [ "${WANT_PROVIDER}" = "cuda" ]; then
  if [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
    if cuda_ready && use_cuda; then
      :
    else
      echo "[entrypoint] WARN: CUDA runtime dependency check failed; fallback to cpu"
      use_cpu
    fi
  else
    echo "[entrypoint] WARN: NVIDIA device not detected; fallback to cpu"
    use_cpu
  fi
else
  use_cpu
fi

exec "$@"
