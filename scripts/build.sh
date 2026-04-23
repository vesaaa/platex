#!/bin/bash
# JSAI-LPR Build Script
# Supports: linux/amd64, linux/arm64

set -e

VERSION=${VERSION:-"0.1.0"}
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

LDFLAGS="-s -w -X main.version=${VERSION} -X main.buildTime=${BUILD_TIME} -X main.gitCommit=${GIT_COMMIT}"

echo "🚗 Building JSAI-LPR Server v${VERSION}"
echo "   Build time: ${BUILD_TIME}"
echo "   Git commit: ${GIT_COMMIT}"
echo ""

build() {
    local os=$1
    local arch=$2
    local output="build/${os}-${arch}/lpr-server"

    echo "📦 Building for ${os}/${arch}..."
    
    mkdir -p "build/${os}-${arch}"
    
    # Download ONNX Runtime shared library for the target architecture
    ORT_VERSION="1.17.1"
    ORT_URL=""
    if [ "${arch}" = "amd64" ]; then
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    elif [ "${arch}" = "arm64" ]; then
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
    fi

    if [ ! -z "${ORT_URL}" ]; then
        echo "   Downloading ONNX Runtime..."
        curl -sL "${ORT_URL}" -o ort.tgz
        tar -xzf ort.tgz
        ORT_DIR=$(tar -tf ort.tgz | head -1 | cut -f1 -d"/")
        
        # Link against the downloaded library
        export CGO_CFLAGS="-I$(pwd)/${ORT_DIR}/include"
        export CGO_LDFLAGS="-L$(pwd)/${ORT_DIR}/lib -lonnxruntime"
        
        # Copy library to output for runtime
        cp ${ORT_DIR}/lib/libonnxruntime.so* "build/${os}-${arch}/"
        rm -rf ort.tgz ${ORT_DIR}
    fi
    
    CGO_ENABLED=1 GOOS=${os} GOARCH=${arch} \
        go build -tags "${os}" -ldflags "${LDFLAGS}" \
        -o "${output}" \
        ./cmd/lpr-server/

    echo "   ✅ Output: ${output}"
    
    # Copy configs and create models dir
    cp -r configs "build/${os}-${arch}/"
    mkdir -p "build/${os}-${arch}/models"
    
    # Copy systemd service file
    if [ -f "deploy/lpr-server.service" ]; then
        cp deploy/lpr-server.service "build/${os}-${arch}/"
    fi
}

# Parse arguments
TARGET=${1:-"native"}

case ${TARGET} in
    "native")
        echo "Building for current platform..."
        mkdir -p build/native
        CGO_ENABLED=1 go build -ldflags "${LDFLAGS}" -o build/native/lpr-server ./cmd/lpr-server/
        cp -r configs build/native/
        mkdir -p build/native/models
        echo "✅ Build complete: build/native/lpr-server"
        ;;
    "linux-amd64")
        build linux amd64
        ;;
    "linux-arm64")
        # Requires aarch64-linux-gnu-gcc for cross-compilation
        export CC=aarch64-linux-gnu-gcc
        build linux arm64
        ;;
    "all")
        build linux amd64
        # ARM64 cross-compile requires toolchain
        if command -v aarch64-linux-gnu-gcc &> /dev/null; then
            export CC=aarch64-linux-gnu-gcc
            build linux arm64
        else
            echo "⚠️  Skipping arm64: aarch64-linux-gnu-gcc not found"
            echo "   Install with: sudo apt install gcc-aarch64-linux-gnu"
        fi
        ;;
    *)
        echo "Usage: $0 [native|linux-amd64|linux-arm64|all]"
        exit 1
        ;;
esac

echo ""
echo "🎉 Build complete!"
