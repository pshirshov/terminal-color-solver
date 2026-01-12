#!/usr/bin/env bash
set -euo pipefail

# Terminal Screenshot Capture Tool
# Runs a program in tmux with specific dimensions and captures its output with ANSI colors preserved.

usage() {
    cat <<'EOF'
Usage: term-capture.sh [OPTIONS] -- COMMAND [ARGS...]

Capture terminal output of a program with colors preserved.

Options:
    -w, --width N       Terminal width (default: 80)
    -h, --height N      Terminal height (default: 24)
    -d, --delay N       Delay in seconds before capture (default: 1)
    -o, --output FILE   Output file (default: stdout)
    -k, --keys KEYS     Send keys after startup (can be repeated)
    -i, --interactive   Wait for Enter instead of delay
    --help              Show this help

Examples:
    # Capture mc with default size
    term-capture.sh -- mc

    # Capture htop at 120x30, wait 2 seconds
    term-capture.sh -w 120 -h 30 -d 2 -- htop

    # Capture mc, navigate down twice, then capture
    term-capture.sh -k Down -k Down -- mc

    # Interactive: press Enter when ready to capture
    term-capture.sh -i -- mc

    # Save to file
    term-capture.sh -o screenshot.txt -- mc
EOF
    exit 0
}

# Defaults
WIDTH=80
HEIGHT=24
DELAY=1
OUTPUT=""
INTERACTIVE=false
KEYS=()
SESSION_NAME="term-capture-$$"
SOCKET_NAME="term-capture-$$"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -k|--keys)
            KEYS+=("$2")
            shift 2
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: No command specified" >&2
    echo "Use --help for usage" >&2
    exit 1
fi

# Store command as array to preserve quoting
COMMAND=("$@")

cleanup() {
    tmux -L "$SOCKET_NAME" kill-server 2>/dev/null || true
}
trap cleanup EXIT

# Start session with specified dimensions
# Wrap command to keep session alive for capture (sleep after command exits)
tmux -L "$SOCKET_NAME" new-session -d -s "$SESSION_NAME" -x "$WIDTH" -y "$HEIGHT" \
    bash -c "$(printf '%q ' "${COMMAND[@]}"); sleep 3600"

# Small initial delay to let the program start
sleep 0.3

# Send any requested keys
for key in "${KEYS[@]}"; do
    tmux -L "$SOCKET_NAME" send-keys -t "$SESSION_NAME" "$key"
    sleep 0.1
done

# Wait for capture
if [[ "$INTERACTIVE" == "true" ]]; then
    echo "Press Enter to capture (Ctrl+C to abort)..." >&2
    read -r
else
    sleep "$DELAY"
fi

# Capture the pane with ANSI escapes preserved
if [[ -n "$OUTPUT" ]]; then
    tmux -L "$SOCKET_NAME" capture-pane -t "$SESSION_NAME" -p -e > "$OUTPUT"
    echo "Captured to: $OUTPUT" >&2
else
    tmux -L "$SOCKET_NAME" capture-pane -t "$SESSION_NAME" -p -e
fi
