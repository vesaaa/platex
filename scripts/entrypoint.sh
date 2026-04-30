#!/bin/sh
set -eu

CPU_DIR="/app/ort/cpu"
CUDA_DIR="/app/ort/cuda"
WANT_PROVIDER="${PLATEX_ONNX_PROVIDER:-cpu}"

use_cpu() {
  ln -sf "${CPU_DIR}/libonnxruntime.so" /app/libonnxruntime.so
  ln -sf "${CPU_DIR}/libonnxruntime.so.1.18.1" /app/libonnxruntime.so.1.18.1
  export LD_LIBRARY_PATH="${CPU_DIR}:/app:${LD_LIBRARY_PATH:-}"
  export PLATEX_ONNX_PROVIDER="cpu"
  echo "[entrypoint] ONNX provider: cpu"
}

use_cuda() {
  ln -sf "${CUDA_DIR}/libonnxruntime.so" /app/libonnxruntime.so
  ln -sf "${CUDA_DIR}/libonnxruntime.so.1.18.1" /app/libonnxruntime.so.1.18.1
  export LD_LIBRARY_PATH="${CUDA_DIR}:/app:${LD_LIBRARY_PATH:-}"
  export PLATEX_ONNX_PROVIDER="cuda"
  echo "[entrypoint] ONNX provider: cuda (device ${PLATEX_ONNX_CUDA_DEVICE_ID:-0})"
}

if [ "${WANT_PROVIDER}" = "cuda" ]; then
  if [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
    if [ -f "${CUDA_DIR}/libonnxruntime.so" ]; then
      use_cuda
    else
      echo "[entrypoint] WARN: CUDA runtime libs missing; fallback to cpu"
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
