#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSIONS="${VERSIONS:-1 2 3 4}"
SCENES="${SCENES:-rgb rgby rand10k}"
if [[ -n "${IMAGE_SIZE:-}" ]]; then
    IMAGE_SIZE="$IMAGE_SIZE"
elif [[ "${SIZE:-}" =~ ^[0-9]+$ ]]; then
    IMAGE_SIZE="$SIZE"
else
    IMAGE_SIZE="1024"
fi
SEED="${SEED:-0}"

for version in $VERSIONS; do
    executable="render_v${version}"
    echo "------------------------------------------------"
    echo "Building RENDER_VERSION=$version -> $executable"
    echo "------------------------------------------------"
    make RENDER_VERSION="$version" EXECUTABLE="$executable"

    for scene in $SCENES; do
        echo "Checking version=$version scene=$scene"
        ./"$executable" -c -s "$IMAGE_SIZE" -S "$SEED" -f "" "$scene"
    done
done

echo "Correctness checks complete"
