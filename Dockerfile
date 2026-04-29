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

# Install certificates and any required dependencies for ONNX C++ runtime
RUN apt-get update && \
    apt-get install -y ca-certificates libgomp1 libstdc++6 libgcc-s1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy configs
COPY configs/ /app/configs/

# Copy the built Go binary and the ONNX shared library
COPY --from=builder /app/build/linux-amd64/lpr-server /app/
COPY --from=builder /app/build/linux-amd64/libonnxruntime.so* /app/

# Grant execution permission and run the automatic model download step during image build!
# This ensures the image comes with the models out-of-the-box.
RUN chmod +x /app/lpr-server && /app/lpr-server -download

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
CMD ["./lpr-server"]
