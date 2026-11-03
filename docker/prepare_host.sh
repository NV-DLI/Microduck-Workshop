#!/bin/bash
# One-time host preparation before starting the course container.
# Creates the output and cache folders the container mounts and checks the
# desktop session variables. It never changes course models or data.
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs "$HOME/.cache/huggingface" \
    "$HOME/.cache/microduck/ov" "$HOME/.cache/microduck/warp" \
    "$HOME/.cache/microduck/nv" "$HOME/.cache/microduck/kit" \
    "$HOME/.cache/microduck/kit-cache"
case "$(uname -m)" in
    x86_64|aarch64) ;;
    *) echo "Unsupported CPU architecture $(uname -m); the course image supports x86_64 and aarch64." >&2; exit 1 ;;
esac
[ -n "$DISPLAY" ] || { echo "DISPLAY is empty: run this in a terminal on the desktop." >&2; exit 1; }
if [ ! -f "${XAUTHORITY:-}" ]; then
    echo "XAUTHORITY does not name a file: ${XAUTHORITY:-<unset>} (echo \$XAUTHORITY in the desktop session)" >&2
    exit 1
fi
DATA="${MICRODUCK_COURSE_DATA:-course-data}"
mkdir -p "$DATA"
echo "Host ready ($(uname -m)): DISPLAY=$DISPLAY"
