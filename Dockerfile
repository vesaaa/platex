# --- Builder Stage ---
FROM golang:1.25-bookworm AS builder

WORKDIR /app
COPY . .

# Run the build script to compile the binary and download the ONNX library
RUN bash scripts/build.sh linux-amd64

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app

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

# Expose API port
EXPOSE 8080

# Run the server
CMD ["./lpr-server"]
