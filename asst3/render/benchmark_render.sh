#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
VERSIONS="${VERSIONS:-1 2 3 4}"
SCENES="${SCENES:-rgb rgby rand10k rand100k biglittle littlebig pattern}"
if [[ -n "${IMAGE_SIZE:-}" ]]; then
    IMAGE_SIZE="$IMAGE_SIZE"
elif [[ "${SIZE:-}" =~ ^[0-9]+$ ]]; then
    IMAGE_SIZE="$SIZE"
else
    IMAGE_SIZE="1024"
fi
FRAMES="${FRAMES:-0:5}"
RUNS="${RUNS:-5}"
SEED="${SEED:-0}"

mkdir -p "$LOG_DIR"

echo "Starting CUDA renderer benchmark"
echo "Logs: $LOG_DIR"
echo "Versions: $VERSIONS"
echo "Scenes: $SCENES"
echo "Image size: $IMAGE_SIZE"
echo "Frames: $FRAMES"
echo "Runs: $RUNS"
echo "Seed: $SEED"

for version in $VERSIONS; do
    executable="render_v${version}"
    echo "------------------------------------------------"
    echo "Building RENDER_VERSION=$version -> $executable"
    echo "------------------------------------------------"
    make RENDER_VERSION="$version" EXECUTABLE="$executable"

    for scene in $SCENES; do
        log_file="$LOG_DIR/v${version}_${scene}_s${IMAGE_SIZE}_seed${SEED}.log"
        echo "Running version=$version scene=$scene -> $log_file"

        {
            echo "Version: $version"
            echo "Scene: $scene"
            echo "Size: $IMAGE_SIZE"
            echo "Frames: $FRAMES"
            echo "Runs: $RUNS"
            echo "Seed: $SEED"
            echo "Timestamp: $(date)"
            echo "------------------------------------"
        } > "$log_file"

        ./"$executable" -r cuda -s "$IMAGE_SIZE" -b "$FRAMES" -S "$SEED" -f "" "$scene" > /dev/null 2>&1

        for run in $(seq 1 "$RUNS"); do
            echo "Run #$run" >> "$log_file"
            ./"$executable" -r cuda -s "$IMAGE_SIZE" -b "$FRAMES" -S "$SEED" -f "" "$scene" >> "$log_file" 2>&1
            echo "------------------------------------" >> "$log_file"
        done
    done
done

echo "Benchmark complete"
