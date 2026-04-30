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
# Example:
#   docker build --build-arg WE_REC_MODEL_URL=https://.../plate_rec_color.onnx .
ARG WE_REC_MODEL_URL="https://github.com/vesaaa/plateai/releases/download/v2026.4/plate_rec_color.onnx"
ARG ORT_VERSION="1.18.1"

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
    curl -fL --retry 3 --retry-delay 2 \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
      -o /tmp/ort-gpu.tgz && \
    tar -xzf /tmp/ort-gpu.tgz -C /tmp && \
    cp /tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime*.so* /app/ort/cuda/ && \
    rm -rf /tmp/ort-gpu.tgz /tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}

# Grant execution permission and run the automatic model download step during image build!
# This ensures the image comes with the models out-of-the-box.
RUN chmod +x /app/lpr-server /app/entrypoint.sh && \
    ln -sf /app/ort/cpu/libonnxruntime.so /app/libonnxruntime.so && \
    ln -sf /app/ort/cpu/libonnxruntime.so.1.18.1 /app/libonnxruntime.so.1.18.1 && \
    /app/lpr-server -download

# Optional: fetch WE plate_rec_color model from GitHub Release.
# Keep this non-fatal so CI still succeeds even if URL is unavailable.
RUN mkdir -p /app/models && \
    if [ -n "$WE_REC_MODEL_URL" ]; then \
      echo "Downloading optional WE model from: $WE_REC_MODEL_URL"; \
      curl -fL --retry 3 --retry-delay 2 "$WE_REC_MODEL_URL" -o /app/models/plate_rec_color.onnx || \
      echo "WARN: optional WE model download failed; continuing without plate_rec_color.onnx"; \
    fi

# Expose API port
EXPOSE 8080

# Run the server
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["./lpr-server"]
