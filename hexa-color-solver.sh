#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# Save original arguments for re-exec
ORIG_ARGS=("$@")

# Parse arguments
POSITIONAL=()
USE_ROCM=false
REBUILD=false

for arg in "$@"; do
    case "$arg" in
        --rocm)
            USE_ROCM=true
            ;;
        --rebuild|-b)
            REBUILD=true
            ;;
        --help|-h)
            echo "Hexa Color Solver v3.0 - OKLCH + APCA Terminal Palette Optimizer"
            echo
            echo "Usage: $0 [OPTIONS] [GENERATIONS] [POPULATION] [THEME_FILE]"
            echo
            echo "Options:"
            echo "  --rocm              Build for ROCm/HIP instead of CUDA"
            echo "  --rebuild, -b       Force rebuild (auto-rebuilds if sources changed)"
            echo "  --help, -h          Show this help"
            echo
            echo "Positional arguments:"
            echo "  GENERATIONS         Number of generations (default: 5000)"
            echo "  POPULATION          Population size (default: 200000)"
            echo "  THEME_FILE          Output theme file (default: themes/theme-YYMMDD-HHMMSS)"
            exit 0
            ;;
        *)
            POSITIONAL+=("$arg")
            ;;
    esac
done

set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

# Determine which nix shell we need
if [ "$USE_ROCM" = true ]; then
    REQUIRED_SHELL="rocm"
else
    REQUIRED_SHELL="cuda"
fi

# Re-exec under nix develop if not already in the right shell
if [ "${HEXA_NIX_SHELL:-}" != "$REQUIRED_SHELL" ]; then
    if [ "$USE_ROCM" = true ]; then
        exec nix develop .#rocm -c env HEXA_NIX_SHELL=rocm "$0" "${ORIG_ARGS[@]}"
    else
        exec nix develop -c env HEXA_NIX_SHELL=cuda "$0" "${ORIG_ARGS[@]}"
    fi
fi

# From here on, we're inside the correct nix shell

GENERATIONS=${1:-5000}
POPULATION=${2:-200000}

if [ "$USE_ROCM" = true ]; then
    BUILD_DIR="build-rocm"
    MODE="ROCm (HIP)"
else
    BUILD_DIR="build-cuda"
    MODE="CUDA"
fi

BINARY="$SCRIPT_DIR/$BUILD_DIR/hexa-color-solver"

echo "Hexa Color Solver v3.0"
echo "Backend: $MODE"
echo "Parameters: generations=$GENERATIONS, population=$POPULATION"

# Theme output
THEMES_DIR="$SCRIPT_DIR/themes"
mkdir -p "$THEMES_DIR"

THEME_FILE="${3:-$THEMES_DIR/theme-$(date +%y%m%d-%H%M%S)}"
[[ "$THEME_FILE" != /* ]] && THEME_FILE="$SCRIPT_DIR/$THEME_FILE"

echo "Output: $THEME_FILE"
echo

# Check if rebuild is needed
needs_rebuild() {
    [ ! -x "$BINARY" ] && return 0

    # Source files to check
    local sources=(
        "$SCRIPT_DIR/hexa-color-solver.cu"
        "$SCRIPT_DIR/color.cuh"
        "$SCRIPT_DIR/output.hpp"
        "$SCRIPT_DIR/CMakeLists.txt"
        "$SCRIPT_DIR/flake.nix"
    )

    for src in "${sources[@]}"; do
        [ -f "$src" ] && [ "$src" -nt "$BINARY" ] && return 0
    done

    return 1
}

# Build if needed
if [ "$REBUILD" = true ] || needs_rebuild; then
    echo "Building ($MODE)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ "$USE_ROCM" = true ]; then
        cmake .. -DUSE_ROCM=ON -DCMAKE_CXX_COMPILER=hipcc \
            -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=$HIP_DEVICE_LIB_PATH $NIX_CFLAGS_COMPILE"
    else
        cmake .. -DUSE_ROCM=OFF
    fi

    cmake --build . -j
    cd "$SCRIPT_DIR"
    echo
else
    echo "Binary up-to-date, skipping build"
    echo
fi

# Run
echo "Running optimizer..."
echo

"$BINARY" -g "$GENERATIONS" -p "$POPULATION" -o "$THEME_FILE"
