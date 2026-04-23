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
    
    CGO_ENABLED=1 GOOS=${os} GOARCH=${arch} \
        go build -ldflags "${LDFLAGS}" \
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
