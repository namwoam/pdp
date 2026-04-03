#!/usr/bin/env python3
"""Simple benchmark runner for SPMV binaries with hardware reporting and plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

TIME_RE = re.compile(r"spmv_[a-z0-9_]+_time_ms=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
OK_RE = re.compile(r"\bOK\b")


@dataclass
class RunResult:
    binary: str
    dataset: str
    run_index: int
    threads: int
    time_ms: float
    ok: bool


def run_cmd(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return (proc.stdout or "") + (proc.stderr or "")


def detect_hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }

    if platform.system() == "Darwin":
        for key, label in [
            ("machdep.cpu.brand_string", "cpu_brand"),
            ("hw.physicalcpu", "cpu_physical"),
            ("hw.logicalcpu", "cpu_logical"),
            ("hw.memsize", "mem_bytes"),
        ]:
            out = run_cmd(["sysctl", "-n", key]).strip()
            if out:
                info[label] = out
    elif platform.system() == "Linux":
        lscpu = shutil.which("lscpu")
        if lscpu:
            info["lscpu"] = run_cmd([lscpu]).strip()

    return info


def benchmark_one(binary: Path, dataset_prefix: Path, runs: int, threads: int) -> list[RunResult]:
    results: list[RunResult] = []
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    mtx = dataset_prefix.with_suffix(".mtx")
    vec = dataset_prefix.with_suffix(".vec")

    for run_idx in range(1, runs + 1):
        proc = subprocess.run(
            [str(binary), str(mtx), str(vec)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        t_match = TIME_RE.search(text)
        if not t_match:
            raise RuntimeError(
                f"Could not parse timing from {binary.name} on {dataset_prefix.name} (run {run_idx})."
            )
        time_ms = float(t_match.group(1))
        ok = bool(OK_RE.search(text))
        results.append(
            RunResult(
                binary=binary.name,
                dataset=dataset_prefix.name,
                run_index=run_idx,
                threads=threads,
                time_ms=time_ms,
                ok=ok,
            )
        )
    return results


def write_csv(rows: list[RunResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["binary", "dataset", "run_index", "threads", "time_ms", "ok"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: list[RunResult]) -> dict:
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        grouped.setdefault((r.binary, r.dataset), []).append(r.time_ms)

    summary = {}
    for (binary, dataset), vals in grouped.items():
        summary.setdefault(dataset, {})[binary] = {
            "runs": len(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "mean_ms": statistics.mean(vals),
            "median_ms": statistics.median(vals),
            "stdev_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return summary


def write_plot(summary: dict, out_png: Path) -> None:
    datasets = sorted(summary.keys())
    binaries = sorted({b for d in datasets for b in summary[d].keys()})

    x = list(range(len(datasets)))
    width = 0.8 / max(1, len(binaries))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, binary in enumerate(binaries):
        vals = []
        for ds in datasets:
            metric = summary[ds].get(binary)
            vals.append(metric["median_ms"] if metric else float("nan"))
        offset = (i - (len(binaries) - 1) / 2) * width
        ax.bar([v + offset for v in x], vals, width=width, label=binary)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("SPMV benchmark comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_markdown_report(
    payload: dict[str, Any],
    out_md: Path,
    out_png: Path,
    out_csv: Path,
    out_json: Path,
) -> None:
    summary = payload["summary"]
    hardware = payload["hardware"]
    config = payload["config"]

    datasets = sorted(summary.keys())
    binaries = sorted({b for d in datasets for b in summary[d].keys()})

    lines: list[str] = []
    lines.append("# SPMV Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): {payload['generated_at_utc']}")
    lines.append(f"- All runs OK: {payload['all_ok']}")
    lines.append(f"- Runs per case: {config['runs']}")
    lines.append(f"- OMP_NUM_THREADS: {config['threads']}")
    lines.append("")

    lines.append("## Hardware")
    lines.append("")
    for key in sorted(hardware.keys()):
        lines.append(f"- {key}: {hardware[key]}")
    lines.append("")

    lines.append("## Median Runtime (ms)")
    lines.append("")
    header = ["Dataset"] + binaries
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for ds in datasets:
        row = [ds]
        for b in binaries:
            metric = summary[ds].get(b)
            row.append(f"{metric['median_ms']:.6f}" if metric else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Plot")
    lines.append("")
    lines.append(f"![Benchmark plot]({out_png.name})")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Raw runs CSV: {out_csv.name}")
    lines.append(f"- Summary JSON: {out_json.name}")
    lines.append(f"- Plot PNG: {out_png.name}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SPMV binaries and plot results.")
    parser.add_argument(
        "--binaries",
        nargs="+",
        default=["./spmv_openmp", "./spmv_serial", "./spmv_not_csr"],
        help="Binary paths to benchmark.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset prefixes (without extension). If omitted, all .mtx files under "
            "./testcases/testcases are used."
        ),
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per binary/dataset.")
    parser.add_argument("--threads", type=int, default=8, help="OMP_NUM_THREADS value.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <script_dir>/bench_results).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    def _resolve_from_script(path_str: str) -> Path:
        p = Path(path_str)
        return p.resolve() if p.is_absolute() else (script_dir / p).resolve()

    def _discover_all_datasets() -> list[Path]:
        testcases_dir = (script_dir / "testcases" / "testcases").resolve()
        if not testcases_dir.exists():
            raise FileNotFoundError(f"Testcases directory not found: {testcases_dir}")

        # Use every Matrix Market testcase and map to prefix path (strip .mtx).
        dataset_prefixes = [
            p.with_suffix("")
            for p in sorted(testcases_dir.glob("*.mtx"))
            if not p.name.endswith(".mtx.gold")
        ]
        if not dataset_prefixes:
            raise FileNotFoundError(f"No .mtx datasets found in: {testcases_dir}")
        return dataset_prefixes

    binaries = [_resolve_from_script(b) for b in args.binaries]
    datasets = [_resolve_from_script(d) for d in args.datasets] if args.datasets else _discover_all_datasets()
    out_dir = (
        _resolve_from_script(args.out_dir)
        if args.out_dir
        else (script_dir / "bench_results").resolve()
    )

    for b in binaries:
        if not b.exists():
            raise FileNotFoundError(f"Binary not found: {b}")
    for d in datasets:
        if not d.with_suffix(".mtx").exists():
            raise FileNotFoundError(f"Dataset mtx not found: {d.with_suffix('.mtx')}")

    all_rows: list[RunResult] = []
    for b in binaries:
        for d in datasets:
            all_rows.extend(benchmark_one(b, d, args.runs, args.threads))

    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"benchmark_runs_{timestamp}.csv"
    out_json = out_dir / f"benchmark_summary_{timestamp}.json"
    out_png = out_dir / f"benchmark_plot_{timestamp}.png"
    out_md = out_dir / f"benchmark_report_{timestamp}.md"

    write_csv(all_rows, out_csv)
    summary = summarize(all_rows)

    payload = {
        "generated_at_utc": timestamp,
        "hardware": detect_hardware(),
        "config": {
            "runs": args.runs,
            "threads": args.threads,
            "binaries": [str(b) for b in binaries],
            "datasets": [str(d) for d in datasets],
        },
        "summary": summary,
        "all_ok": all(r.ok for r in all_rows),
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    write_plot(summary, out_png)
    write_markdown_report(payload, out_md, out_png, out_csv, out_json)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_md}")
    print(f"All runs OK: {payload['all_ok']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise
