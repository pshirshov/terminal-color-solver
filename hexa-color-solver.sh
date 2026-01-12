#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# Parse arguments
POSITIONAL=()
USE_ROCM=false
REBUILD=false
NO_FM_PAIRS=""
NO_GB_EXCLUSIONS=""

for arg in "$@"; do
    case "$arg" in
        --rocm)
            USE_ROCM=true
            ;;
        --rebuild|-b)
            REBUILD=true
            ;;
        --no-fm-pairs)
            NO_FM_PAIRS="--no-fm-pairs"
            ;;
        --no-gb-exclusions)
            NO_GB_EXCLUSIONS="--no-gb-exclusions"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [GENERATIONS] [POPULATION] [THEME_FILE]"
            echo
            echo "Options:"
            echo "  --rocm              Build for ROCm/HIP instead of CUDA"
            echo "  --rebuild, -b       Force rebuild even if binary exists"
            echo "  --no-fm-pairs       Disable FM pairs constraints"
            echo "  --no-gb-exclusions  Disable green/blue exclusions"
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

GENERATIONS=${1:-5000}
POPULATION=${2:-200000}

# Set up paths based on mode
if [ "$USE_ROCM" = true ]; then
    BUILD_DIR="build-rocm"
    NIX_SHELL=".#rocm"
    MODE="ROCm (HIP)"
else
    BUILD_DIR="build-cuda"
    NIX_SHELL=""
    MODE="CUDA"
fi

BINARY="$SCRIPT_DIR/$BUILD_DIR/hexa-color-solver"

echo "Hexa Color Solver"
echo "Mode: $MODE"
echo "Parameters: generations=$GENERATIONS, population=$POPULATION"
[ -n "$NO_FM_PAIRS" ] && echo "Flag: --no-fm-pairs (FM pairs disabled)"
[ -n "$NO_GB_EXCLUSIONS" ] && echo "Flag: --no-gb-exclusions (G/B exclusions disabled)"

# Theme output
THEMES_DIR="$SCRIPT_DIR/themes"
mkdir -p "$THEMES_DIR"

THEME_FILE="${3:-$THEMES_DIR/theme-$(date +%y%m%d-%H%M%S)}"
[[ "$THEME_FILE" != /* ]] && THEME_FILE="$SCRIPT_DIR/$THEME_FILE"

echo "Output: $THEME_FILE"
echo

# Build function
do_build() {
    echo "Building ($MODE)..."

    if [ "$USE_ROCM" = true ]; then
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake .. -DUSE_ROCM=ON -DCMAKE_CXX_COMPILER=hipcc \
            -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=$HIP_DEVICE_LIB_PATH $NIX_CFLAGS_COMPILE"
    else
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake .. -DUSE_ROCM=OFF
    fi

    cmake --build . -j
    cd "$SCRIPT_DIR"
    echo
}

# Check if build is needed
if [ "$REBUILD" = true ] || [ ! -x "$BINARY" ]; then
    if [ -n "$NIX_SHELL" ]; then
        nix develop "$NIX_SHELL" -c bash -c "$(declare -f do_build); do_build"
    else
        nix develop -c bash -c "$(declare -f do_build); do_build"
    fi
else
    echo "Binary exists, skipping build (use --rebuild to force)"
    echo
fi

# Run
echo "Running optimizer..."
echo

if [ -n "$NIX_SHELL" ]; then
    nix develop "$NIX_SHELL" -c "$BINARY" \
        -g "$GENERATIONS" -p "$POPULATION" -o "$THEME_FILE" $NO_FM_PAIRS $NO_GB_EXCLUSIONS
else
    nix develop -c "$BINARY" \
        -g "$GENERATIONS" -p "$POPULATION" -o "$THEME_FILE" $NO_FM_PAIRS $NO_GB_EXCLUSIONS
fi
