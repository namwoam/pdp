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
        default=[
            "./testcases/testcases/test_sparse_1000",
            "./testcases/testcases/large_sparse_5000",
            "./testcases/testcases/huge_200k_100",
        ],
        help="Dataset prefixes (without extension).",
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per binary/dataset.")
    parser.add_argument("--threads", type=int, default=8, help="OMP_NUM_THREADS value.")
    parser.add_argument("--out-dir", default="./bench_results", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    binaries = [Path(b).resolve() for b in args.binaries]
    datasets = [Path(d).resolve() for d in args.datasets]
    out_dir = Path(args.out_dir).resolve()

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

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_png}")
    print(f"All runs OK: {payload['all_ok']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise
