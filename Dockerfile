# --- Builder Stage ---
FROM golang:1.22-bookworm AS builder

WORKDIR /app
COPY . .

# Run the build script to compile the binary and download the ONNX library
RUN bash scripts/build.sh linux-amd64

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app

# Optional WE recognizer model download URL (can be overridden at build time).
# Pinned tag (reproducible):
#   https://github.com/vesaaa/plateai/releases/download/v2026.5/plate_rec_color.onnx
# Floating "Latest" release (same file name; tracks whatever GitHub marks Latest):
#   https://github.com/vesaaa/plateai/releases/latest/download/plate_rec_color.onnx
# Example override:
#   docker build --build-arg WE_REC_MODEL_URL=https://.../plate_rec_color.onnx .
ARG WE_REC_MODEL_URL="https://github.com/vesaaa/plateai/releases/download/v2026.5/plate_rec_color.onnx"
# Optional PlateCRNN primary recognizer (plateai/crnn). Empty = skip.
ARG CRNN_REC_MODEL_URL="https://github.com/vesaaa/plateai/releases/download/crnn-2026.10-ft1/plate_rec_crnn.onnx"
ARG MODELS_ZIP_URLS="https://github.com/vesaaa/platex/releases/download/v1.0.0-models/20230229.zip"
ARG ORT_VERSION="1.18.1"
ARG ORT_GPU_VARIANT="cuda12"
ARG ORT_GPU11_VERSION="1.18.1"
ARG ORT_GPU12_VERSION="1.20.1"

# Install certificates and any required dependencies for ONNX C++ runtime
RUN apt-get update && \
    apt-get install -y ca-certificates libgomp1 libstdc++6 libgcc-s1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy configs
COPY configs/ /app/configs/

# Copy the built Go binary and the ONNX shared library
COPY --from=builder /app/build/linux-amd64/lpr-server /app/
COPY --from=builder /app/build/linux-amd64/libonnxruntime.so* /app/ort/cpu/
COPY scripts/entrypoint.sh /app/entrypoint.sh

# Download CUDA-enabled ONNX Runtime so a single image supports both CPU/CUDA.
RUN mkdir -p /app/ort/cuda && \
    if [ "$ORT_GPU_VARIANT" = "cpu" ]; then \
      echo "CPU image selected: skip GPU runtime bundle"; \
    else \
      if [ "$ORT_GPU_VARIANT" = "cuda11" ]; then \
        ORT_GPU_VERSION="$ORT_GPU11_VERSION"; \
      elif [ "$ORT_GPU_VARIANT" = "cuda12" ]; then \
        ORT_GPU_VERSION="$ORT_GPU12_VERSION"; \
      else \
        echo "Unsupported ORT_GPU_VARIANT: $ORT_GPU_VARIANT (expected cpu|cuda11|cuda12)" && exit 1; \
      fi; \
      echo "Downloading ONNX Runtime GPU package v${ORT_GPU_VERSION} (${ORT_GPU_VARIANT})"; \
      curl -fL --retry 3 --retry-delay 2 \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_GPU_VERSION}/onnxruntime-linux-x64-gpu-${ORT_GPU_VERSION}.tgz" \
        -o /tmp/ort-gpu.tgz && \
      tar -xzf /tmp/ort-gpu.tgz -C /tmp && \
      cp /tmp/onnxruntime-linux-x64-gpu-${ORT_GPU_VERSION}/lib/libonnxruntime*.so* /app/ort/cuda/ && \
      rm -rf /tmp/ort-gpu.tgz /tmp/onnxruntime-linux-x64-gpu-${ORT_GPU_VERSION}; \
    fi

# Grant execution permission and run the automatic model download step during image build!
# This ensures the image comes with the models out-of-the-box.
RUN chmod +x /app/lpr-server /app/entrypoint.sh && \
    ln -sf /app/ort/cpu/libonnxruntime.so /app/libonnxruntime.so && \
    ln -sf /app/ort/cpu/libonnxruntime.so.1.18.1 /app/libonnxruntime.so.1.18.1 && \
    PLATEX_MODELS_ZIP_URLS="$MODELS_ZIP_URLS" /app/lpr-server -download

# Optional: fetch WE plate_rec_color model from GitHub Release.
# Keep this non-fatal so CI still succeeds even if URL is unavailable.
RUN mkdir -p /app/models && \
    if [ -n "$WE_REC_MODEL_URL" ]; then \
      echo "Downloading optional WE model from: $WE_REC_MODEL_URL"; \
      curl -fL --retry 3 --retry-delay 2 "$WE_REC_MODEL_URL" -o /app/models/plate_rec_color.onnx || \
      echo "WARN: optional WE model download failed; continuing without plate_rec_color.onnx"; \
    fi

# Optional: fetch PlateCRNN primary recognizer. Non-fatal like the WE model;
# without it the engine keeps the legacy recognizers.
RUN if [ -n "$CRNN_REC_MODEL_URL" ]; then \
      echo "Downloading optional CRNN model from: $CRNN_REC_MODEL_URL"; \
      curl -fL --retry 3 --retry-delay 2 "$CRNN_REC_MODEL_URL" -o /app/models/plate_rec_crnn.onnx || \
      echo "WARN: optional CRNN model download failed; continuing without plate_rec_crnn.onnx"; \
    fi

# Expose API port
EXPOSE 8080

# Run the server
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["./lpr-server"]
