#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


LOG_NAME_RE = re.compile(r"v(?P<version>\d+)_(?P<scene>.+)_s(?P<size>[^_]+)_seed(?P<seed>[^.]+)\.log$")
FIELD_RE = re.compile(r"^(Clear|Advance|Render|Total):\s+(?P<value>[\d.]+)\s+ms$", re.MULTILINE)
OVERALL_RE = re.compile(r"^Overall:\s+(?P<value>[\d.]+)\s+sec", re.MULTILINE)


def read_header(content):
    header = {}
    for line in content.splitlines():
        if line.startswith("------------------------------------"):
            break
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()
    return header


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metadata_value(header, name_match, key):
    header_value = header.get(key)
    if header_value:
        return header_value
    if name_match:
        return name_match.group(key)
    return None


def metadata_int(header, name_match, key):
    header_value = parse_int(header.get(key))
    if header_value is not None:
        return header_value
    if name_match:
        return parse_int(name_match.group(key))
    return None


def parse_log(path):
    content = path.read_text(errors="replace")
    header = read_header(content)
    name_match = LOG_NAME_RE.match(path.name)

    version = metadata_int(header, name_match, "version")
    scene = metadata_value(header, name_match, "scene")
    size = metadata_int(header, name_match, "size")
    seed = metadata_int(header, name_match, "seed")

    if version is None or not scene or size is None or seed is None:
        print(f"Skipping {path.name}: invalid benchmark metadata")
        return []

    rows = []
    current = {}
    run = 0

    for line in content.splitlines():
        if line.startswith("Run #"):
            if current:
                rows.append(current)
                current = {}
            run += 1
            current = {
                "version": version,
                "scene": scene,
                "size": size,
                "seed": seed,
                "run": run,
            }
            continue

        metric = FIELD_RE.match(line)
        if metric and current:
            current[metric.group(1).lower() + "_ms"] = float(metric.group("value"))
            continue

        overall = OVERALL_RE.match(line)
        if overall and current:
            current["overall_sec"] = float(overall.group("value"))

    if current:
        rows.append(current)

    return [row for row in rows if "render_ms" in row]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["version"], row["scene"], row["size"], row["seed"])
        groups[key].append(row)

    summaries = []
    for (version, scene, size, seed), items in sorted(groups.items()):
        summary = {
            "version": version,
            "scene": scene,
            "size": size,
            "seed": seed,
            "runs": len(items),
        }
        for field in ["clear_ms", "advance_ms", "render_ms", "total_ms", "overall_sec"]:
            values = [item[field] for item in items if field in item]
            if values:
                summary[field + "_mean"] = mean(values)
                summary[field + "_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)

    v1_by_scene = {
        row["scene"]: row["render_ms_mean"]
        for row in summaries
        if row["version"] == 1 and "render_ms_mean" in row
    }
    for row in summaries:
        baseline = v1_by_scene.get(row["scene"])
        if baseline and row.get("render_ms_mean"):
            row["render_speedup_vs_v1"] = baseline / row["render_ms_mean"]

    return summaries


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV files were generated without plots.")
        return

    scenes = sorted({row["scene"] for row in summary_rows})
    versions = sorted({row["version"] for row in summary_rows})
    lookup = {(row["scene"], row["version"]): row for row in summary_rows}

    x = list(range(len(scenes)))
    width = 0.8 / max(1, len(versions))

    plt.figure(figsize=(max(10, len(scenes) * 1.2), 6))
    for offset, version in enumerate(versions):
        values = [
            lookup.get((scene, version), {}).get("render_ms_mean", 0.0)
            for scene in scenes
        ]
        positions = [base - 0.4 + width / 2 + offset * width for base in x]
        plt.bar(positions, values, width=width, label=f"v{version}")

    plt.xticks(x, scenes, rotation=30, ha="right")
    plt.ylabel("Render time (ms)")
    plt.title("CUDA Renderer Render Time by Version")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "render_time_by_version.png", dpi=180)
    plt.close()

    speedup_rows = [row for row in summary_rows if "render_speedup_vs_v1" in row]
    if not speedup_rows:
        return

    plt.figure(figsize=(max(10, len(scenes) * 1.2), 6))
    for offset, version in enumerate(versions):
        values = [
            lookup.get((scene, version), {}).get("render_speedup_vs_v1", 0.0)
            for scene in scenes
        ]
        positions = [base - 0.4 + width / 2 + offset * width for base in x]
        plt.bar(positions, values, width=width, label=f"v{version}")

    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    plt.xticks(x, scenes, rotation=30, ha="right")
    plt.ylabel("Speedup vs v1")
    plt.title("CUDA Renderer Speedup by Version")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "render_speedup_vs_v1.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze CUDA renderer benchmark logs.")
    parser.add_argument("--log-dir", default="logs", help="Directory containing benchmark logs.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for path in sorted(log_dir.glob("v*_*.log")):
        rows.extend(parse_log(path))

    if not rows:
        raise SystemExit(f"No benchmark rows found in {log_dir}")

    summary_rows = summarize(rows)

    raw_fields = [
        "version", "scene", "size", "seed", "run",
        "clear_ms", "advance_ms", "render_ms", "total_ms", "overall_sec",
    ]
    summary_fields = [
        "version", "scene", "size", "seed", "runs",
        "clear_ms_mean", "clear_ms_std",
        "advance_ms_mean", "advance_ms_std",
        "render_ms_mean", "render_ms_std",
        "total_ms_mean", "total_ms_std",
        "overall_sec_mean", "overall_sec_std",
        "render_speedup_vs_v1",
    ]

    write_csv(log_dir / "render_raw.csv", rows, raw_fields)
    write_csv(log_dir / "render_summary.csv", summary_rows, summary_fields)
    plot_summary(summary_rows, log_dir)

    print(f"Parsed {len(rows)} runs from {log_dir}")
    print(f"Wrote {log_dir / 'render_raw.csv'}")
    print(f"Wrote {log_dir / 'render_summary.csv'}")


if __name__ == "__main__":
    main()
