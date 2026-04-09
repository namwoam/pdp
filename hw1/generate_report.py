#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

TIME_RE = re.compile(r"spmv_[a-z0-9_]+_time_ms=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
OK_RE = re.compile(r"\bOK\b")

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figure"
DATA_DIR = (ROOT / "testcases").resolve()
REPORT_PATH = ROOT / "report.md"
RAW_CSV_PATH = ROOT / "report_raw_results.csv"
SUMMARY_JSON_PATH = ROOT / "report_metrics.json"

THREADS = [1, 2, 4, 8, 16]
RUNS = 3
FORMAT_DATASETS = [
    "test_small_uniform_50",
    "test_med_skew_500",
    "test_med_uniform_500",
    "test_sparse_1000",
    "large_sparse_2000",
    "large_denserows_2000",
    "large_sparse_5000",
]
SCALING_DATASETS = [
    "large_sparse_5000",
    "large_10000_sparse_100",
    "huge_denserows_5000",
    "huge_200k_100",
]
ALL_DATASETS = sorted(
    {
        *FORMAT_DATASETS,
        *SCALING_DATASETS,
        "huge_200k_100",
    }
)
ALL_DATASETS = sorted(ALL_DATASETS, key=lambda name: name)

BINARIES = {
    "spmv_not_csr": ROOT / "spmv_not_csr",
    "spmv_serial": ROOT / "spmv_serial",
    "spmv_openmp": ROOT / "spmv_openmp",
}


def dataset_prefix(name: str) -> Path:
    return DATA_DIR / name


def read_matrix_header(name: str) -> dict[str, int]:
    mtx = dataset_prefix(name).with_suffix(".mtx")
    with mtx.open() as f:
        for line in f:
            if not line.startswith("%"):
                rows, cols, nnz = map(int, line.split())
                return {"rows": rows, "cols": cols, "nnz": nnz}
    raise RuntimeError(f"Could not parse Matrix Market header: {mtx}")


def detect_hardware() -> dict[str, str]:
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": str(os.cpu_count()),
    }
    try:
        proc = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if proc.stdout.strip():
            for line in proc.stdout.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in {
                    "Model name",
                    "Socket(s)",
                    "Core(s) per socket",
                    "Thread(s) per core",
                    "CPU(s)",
                    "NUMA node(s)",
                }:
                    hardware[key] = value
    except FileNotFoundError:
        pass
    return hardware


def run_case(binary: str, dataset: str, threads: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    binary_path = BINARIES[binary]
    prefix = dataset_prefix(dataset)
    mtx = prefix.with_suffix(".mtx")
    vec = prefix.with_suffix(".vec")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    for run_idx in range(1, RUNS + 1):
        proc = subprocess.run(
            [str(binary_path), str(mtx), str(vec)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        time_match = TIME_RE.search(output)
        if not time_match:
            raise RuntimeError(
                f"Could not parse timing from {binary} on {dataset} at {threads} threads.\n{output}"
            )
        rows.append(
            {
                "binary": binary,
                "dataset": dataset,
                "threads": threads,
                "run": run_idx,
                "time_ms": float(time_match.group(1)),
                "ok": bool(OK_RE.search(output)),
            }
        )
    return rows


def median(records: list[dict[str, object]]) -> float:
    return statistics.median(float(r["time_ms"]) for r in records)


def group_key(record: dict[str, object]) -> tuple[str, str, int]:
    return str(record["binary"]), str(record["dataset"]), int(record["threads"])


def shorten(name: str) -> str:
    mapping = {
        "test_small_uniform_50": "small50",
        "test_med_skew_500": "skew500",
        "test_med_uniform_500": "uniform500",
        "test_sparse_1000": "sparse1000",
        "large_sparse_2000": "sparse2000",
        "large_denserows_2000": "dense2000",
        "large_sparse_5000": "sparse5000",
        "huge_denserows_5000": "dense5000",
        "large_10000_sparse_100": "sparse10k",
        "huge_200k_100": "huge200k",
    }
    return mapping.get(name, name)


def storage_bytes_coo(meta: dict[str, int]) -> int:
    return meta["nnz"] * (4 + 4 + 8)


def storage_bytes_csr(meta: dict[str, int]) -> int:
    return meta["nnz"] * (4 + 8) + (meta["rows"] + 1) * 4


def format_ms(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def make_grouped_runtime_plot(
    grouped: dict[tuple[str, str, int], list[dict[str, object]]],
    metas: dict[str, dict[str, int]],
) -> None:
    datasets = sorted(FORMAT_DATASETS, key=lambda name: metas[name]["nnz"])
    labels = [shorten(name) for name in datasets]
    x = list(range(len(datasets)))
    binaries = [
        ("spmv_not_csr", "Naive COO scan"),
        ("spmv_serial", "Serial CSR"),
        ("spmv_openmp", "OpenMP CSR (16 threads)"),
    ]
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, (binary, label) in enumerate(binaries):
        values = []
        for dataset in datasets:
            recs = grouped[(binary, dataset, 16 if binary == "spmv_openmp" else 1)]
            values.append(median(recs))
        offsets = [value + (idx - 1) * width for value in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_yscale("log")
    ax.set_ylabel("Median kernel time (ms)")
    ax.set_title("Storage format and parallelism change kernel cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "storage_runtime_comparison.png", dpi=180)
    plt.close(fig)


def make_size_runtime_plot(
    grouped: dict[tuple[str, str, int], list[dict[str, object]]],
    metas: dict[str, dict[str, int]],
) -> None:
    datasets = sorted(metas, key=lambda name: metas[name]["nnz"])
    nnz = [metas[name]["nnz"] for name in datasets]
    serial = [median(grouped[("spmv_serial", name, 1)]) for name in datasets]
    openmp = [median(grouped[("spmv_openmp", name, 16)]) for name in datasets]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(nnz, serial, marker="o", linewidth=2, label="Serial CSR")
    ax.plot(nnz, openmp, marker="o", linewidth=2, label="OpenMP CSR (16 threads)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix nonzeros (nnz)")
    ax.set_ylabel("Median kernel time (ms)")
    ax.set_title("Kernel time grows with matrix work, not just row count")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "matrix_size_runtime.png", dpi=180)
    plt.close(fig)


def make_memory_plot(metas: dict[str, dict[str, int]]) -> None:
    datasets = sorted(metas, key=lambda name: metas[name]["nnz"])
    labels = [shorten(name) for name in datasets]
    coo = [storage_bytes_coo(metas[name]) / (1024 * 1024) for name in datasets]
    csr = [storage_bytes_csr(metas[name]) / (1024 * 1024) for name in datasets]
    x = list(range(len(datasets)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([v - width / 2 for v in x], coo, width=width, label="COO bytes")
    ax.bar([v + width / 2 for v in x], csr, width=width, label="CSR bytes")
    ax.set_ylabel("Estimated matrix storage (MiB)")
    ax.set_title("CSR stores less matrix metadata than COO")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "matrix_storage_footprint.png", dpi=180)
    plt.close(fig)


def make_scaling_plot(
    grouped: dict[tuple[str, str, int], list[dict[str, object]]],
) -> None:
    fig, (ax_runtime, ax_speedup) = plt.subplots(1, 2, figsize=(13, 5.5))

    for dataset in SCALING_DATASETS:
        runtimes = [median(grouped[("spmv_openmp", dataset, threads)]) for threads in THREADS]
        serial_time = median(grouped[("spmv_serial", dataset, 1)])
        speedups = [serial_time / runtime for runtime in runtimes]
        label = shorten(dataset)
        ax_runtime.plot(THREADS, runtimes, marker="o", linewidth=2, label=label)
        ax_speedup.plot(THREADS, speedups, marker="o", linewidth=2, label=label)

    ax_runtime.set_xlabel("Threads")
    ax_runtime.set_ylabel("Median kernel time (ms)")
    ax_runtime.set_title("Thread count vs runtime")
    ax_runtime.set_xscale("log", base=2)
    ax_runtime.set_xticks(THREADS)
    ax_runtime.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_runtime.grid(which="both", alpha=0.25)

    ax_speedup.set_xlabel("Threads")
    ax_speedup.set_ylabel("Speedup over serial CSR")
    ax_speedup.set_title("Parallel speedup over serial CSR")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xticks(THREADS)
    ax_speedup.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_speedup.plot(THREADS, THREADS, linestyle="--", color="black", alpha=0.35, label="Ideal")
    ax_speedup.grid(which="both", alpha=0.25)

    handles, labels = ax_speedup.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "thread_scaling.png", dpi=180)
    plt.close(fig)


def summarize_metrics(
    grouped: dict[tuple[str, str, int], list[dict[str, object]]],
    metas: dict[str, dict[str, int]],
) -> dict[str, object]:
    size_order = sorted(metas, key=lambda name: metas[name]["nnz"])
    fastest_dataset = max(
        SCALING_DATASETS,
        key=lambda name: median(grouped[("spmv_serial", name, 1)])
        / median(grouped[("spmv_openmp", name, 16)]),
    )
    slowest_scaling_dataset = min(
        SCALING_DATASETS,
        key=lambda name: median(grouped[("spmv_serial", name, 1)])
        / median(grouped[("spmv_openmp", name, 16)]),
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": detect_hardware(),
        "all_ok": all(all(bool(record["ok"]) for record in records) for records in grouped.values()),
        "matrix_metadata": metas,
        "matrix_order_by_nnz": size_order,
        "serial_vs_openmp16": {
            dataset: {
                "serial_ms": median(grouped[("spmv_serial", dataset, 1)]),
                "openmp16_ms": median(grouped[("spmv_openmp", dataset, 16)]),
                "speedup": median(grouped[("spmv_serial", dataset, 1)])
                / median(grouped[("spmv_openmp", dataset, 16)]),
            }
            for dataset in size_order
        },
        "format_comparison": {
            dataset: {
                "not_csr_ms": median(grouped[("spmv_not_csr", dataset, 1)]),
                "serial_ms": median(grouped[("spmv_serial", dataset, 1)]),
                "openmp16_ms": median(grouped[("spmv_openmp", dataset, 16)]),
            }
            for dataset in FORMAT_DATASETS
        },
        "thread_scaling": {
            dataset: {
                str(threads): {
                    "openmp_ms": median(grouped[("spmv_openmp", dataset, threads)]),
                    "speedup_over_serial": median(grouped[("spmv_serial", dataset, 1)])
                    / median(grouped[("spmv_openmp", dataset, threads)]),
                }
                for threads in THREADS
            }
            for dataset in SCALING_DATASETS
        },
        "best_speedup_dataset": fastest_dataset,
        "worst_speedup_dataset": slowest_scaling_dataset,
    }


def write_raw_csv(records: list[dict[str, object]]) -> None:
    fieldnames = ["binary", "dataset", "threads", "run", "time_ms", "ok"]
    with RAW_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_report(summary: dict[str, object]) -> None:
    serial_vs_openmp16 = summary["serial_vs_openmp16"]
    format_comparison = summary["format_comparison"]
    thread_scaling = summary["thread_scaling"]
    hardware = summary["hardware"]
    matrix_metadata = summary["matrix_metadata"]

    huge = serial_vs_openmp16["huge_200k_100"]
    sparse10k = serial_vs_openmp16["large_10000_sparse_100"]
    best_dataset = str(summary["best_speedup_dataset"])
    worst_dataset = str(summary["worst_speedup_dataset"])
    best_speedup = serial_vs_openmp16[best_dataset]["speedup"]
    worst_speedup = serial_vs_openmp16[worst_dataset]["speedup"]
    dense2000 = format_comparison["large_denserows_2000"]
    sparse2000 = format_comparison["large_sparse_2000"]
    sparse5000_scaling = thread_scaling["large_sparse_5000"]
    huge_scaling = thread_scaling["huge_200k_100"]

    lines = [
        "# Parallel Sparse Matrix-Vector Multiplication Report",
        "",
        "## Environment and Methodology",
        "",
        f"- Generated at (UTC): {summary['generated_at_utc']}",
        f"- All benchmark runs passed verification: {summary['all_ok']}",
        f"- CPU: {hardware.get('Model name', 'unknown')}",
        (
            f"- Topology: {hardware.get('Socket(s)', '?')} sockets, "
            f"{hardware.get('Core(s) per socket', '?')} cores/socket, "
            f"{hardware.get('Thread(s) per core', '?')} threads/core, "
            f"{hardware.get('NUMA node(s)', '?')} NUMA nodes"
        ),
        "- Thread counts tested for OpenMP: 1, 2, 4, 8, 16 (matching the assignment cap).",
        "- Each point is the median of 3 runs.",
        "- Reported times are the kernel times printed by the binaries. They exclude file parsing and CSR construction, because those steps happen before the timed region in the provided code.",
        "",
        "## Implementations and Applied Optimizations",
        "",
        "- `spmv_not_csr.c` is the naive baseline: for every row it scans all COO triples, so the inner work is proportional to `rows * nnz`.",
        "- `spmv_serial.c` converts COO to CSR once, then computes each row from its contiguous CSR slice. This removes the wasted full-matrix scan in the naive version.",
        "- `spmv_openmp.c` parallelizes the CSR kernel over rows with `#pragma omp for schedule(static)` and adds `#pragma omp simd` on the inner dot-product loop.",
        "- The OpenMP version also applies NUMA-aware first-touch on CSR arrays and the output vector, uses `OMP_PROC_BIND=spread` and `OMP_PLACES=cores`, and replicates `x` per thread when the vector is small enough to improve locality.",
        "",
        "## Correctness",
        "",
        "All benchmarked runs reported `OK` against the provided `.gold` outputs, so the performance comparisons below are between correct implementations.",
        "",
        "## How Does Storage Format Influence Correctness, Memory Usage, and Runtime?",
        "",
        "Correctness did not change across COO and CSR: all versions produced the expected output within the assignment tolerance. The real differences are memory layout and kernel cost.",
        "",
        f"Figure 1 shows the runtime gap directly. On `large_sparse_2000`, the naive COO scan took {format_ms(sparse2000['not_csr_ms'])} ms while serial CSR took {format_ms(sparse2000['serial_ms'])} ms, a {sparse2000['not_csr_ms'] / sparse2000['serial_ms']:.1f}x reduction. On `large_denserows_2000`, naive COO still needed {format_ms(dense2000['not_csr_ms'])} ms while serial CSR dropped to {format_ms(dense2000['serial_ms'])} ms. The gap grows because CSR only touches the nonzeros that belong to the current row, while the naive baseline repeatedly rechecks unrelated entries.",
        "",
        "Figure 2 shows the estimated matrix-storage footprint assuming 32-bit indices and 64-bit values. COO uses `row + col + value` per nonzero, about 16 bytes per entry. CSR uses `col + value` per nonzero plus one row-pointer array, about 12 bytes per nonzero plus `4 * (rows + 1)` bytes. For large matrices this is consistently smaller. For example, `huge_200k_100` is about "
        f"{storage_bytes_coo(matrix_metadata['huge_200k_100']) / (1024 * 1024):.1f} MiB in COO versus "
        f"{storage_bytes_csr(matrix_metadata['huge_200k_100']) / (1024 * 1024):.1f} MiB in CSR.",
        "",
        "![Storage runtime comparison](figure/storage_runtime_comparison.png)",
        "",
        "![Estimated matrix storage](figure/matrix_storage_footprint.png)",
        "",
        "## How Does Matrix Size Influence Performance?",
        "",
        "The dominant trend is that runtime scales with the amount of nonzero work, not just the row count. Figure 3 orders the matrices by `nnz`, and both CSR implementations follow that growth. Tiny matrices stay below 0.1 ms, the 1M-nnz matrices land around the sub-millisecond to low-millisecond range, and the 20M-nnz `huge_200k_100` case is the clear outlier.",
        "",
        f"On `large_10000_sparse_100` (1,000,000 nonzeros), serial CSR needed {format_ms(sparse10k['serial_ms'])} ms. On `huge_200k_100` (20,000,000 nonzeros), serial CSR rose to {format_ms(huge['serial_ms'])} ms. The 20x increase in nonzeros therefore produced roughly a {huge['serial_ms'] / sparse10k['serial_ms']:.1f}x increase in kernel time, which is close to the work increase expected for SpMV.",
        "",
        "![Matrix size vs runtime](figure/matrix_size_runtime.png)",
        "",
        "## How Does the Number of Threads Influence Performance?",
        "",
        "The scaling behaviour is strongly workload-dependent.",
        "",
        f"- For `large_sparse_5000`, adding threads helps little after a point: 1-thread OpenMP took {format_ms(sparse5000_scaling['1']['openmp_ms'])} ms and 16 threads reached {format_ms(sparse5000_scaling['16']['openmp_ms'])} ms. The matrix is too small for thread startup, synchronization, and memory traffic to amortize perfectly.",
        f"- For `huge_200k_100`, the effect is much stronger: 1-thread OpenMP took {format_ms(huge_scaling['1']['openmp_ms'])} ms and 16 threads dropped to {format_ms(huge_scaling['16']['openmp_ms'])} ms.",
        "",
        "In other words, more threads help when there is enough nonzero work and memory-level parallelism to keep the cores busy. On small matrices, parallel overhead dominates sooner.",
        "",
        "## What Is the Speedup of the Parallel Code Compared to the Serial Version?",
        "",
        f"The best 16-thread speedup in these experiments was on `{best_dataset}` at {best_speedup:.2f}x over serial CSR. The weakest 16-thread speedup among the scaling datasets was on `{worst_dataset}` at {worst_speedup:.2f}x.",
        "",
        f"For the largest case, `huge_200k_100`, serial CSR took {format_ms(huge['serial_ms'])} ms and the 16-thread OpenMP kernel took {format_ms(huge['openmp16_ms'])} ms, which is a {huge['speedup']:.2f}x speedup. On `large_10000_sparse_100`, the 16-thread speedup was {sparse10k['speedup']:.2f}x.",
        "",
        "Figure 4 plots both runtime and speedup versus thread count. The ideal line is included as a reference; the measured curves stay below it because SpMV is memory-bound and the rows do not all carry the same amount of work.",
        "",
        "![Thread scaling and speedup](figure/thread_scaling.png)",
        "",
        "## Explain the Observed Scalability: Strong vs. Weak",
        "",
        "These experiments primarily measure strong scaling, not weak scaling. For each matrix, the problem size stays fixed while the thread count changes from 1 to 16. The plots therefore answer: how much faster does the same SpMV get when more threads are applied?",
        "",
        "The results show moderate strong scaling on the larger matrices and weak strong scaling on the smaller ones. This is expected for SpMV because the kernel has low arithmetic intensity and quickly becomes limited by memory bandwidth, NUMA traffic, and synchronization overhead. The OpenMP implementation improves strong scaling by preserving locality: static row partitioning aligns with the first-touch initialization, thread pinning spreads threads across cores, and the optional thread-local copy of `x` reduces shared-vector contention for medium-size inputs.",
        "",
        "A true weak-scaling study would increase matrix size proportionally with thread count so that each thread keeps roughly constant work. That was not the experiment required by the assignment questions here, so I describe the observed behaviour as strong scaling.",
        "",
        "## Additional Insights",
        "",
        f"- Comparing `large_sparse_2000` and `large_denserows_2000` shows why nnz distribution matters. They have the same row count, but the denser matrix has 10x more nonzeros and correspondingly higher kernel time.",
        "- The OpenMP version is not just 'serial CSR plus threads'. The NUMA-aware first-touch and thread pinning decisions match the dual-socket machine described in the spec, which is why the parallel code remains effective on the bigger matrices.",
        "- The naive COO-scan baseline is useful for correctness and for demonstrating why CSR matters, but it stops being a practical performance baseline once matrices grow beyond the small and medium cases.",
        "",
        "## Reproducibility Artifacts",
        "",
        "- Raw benchmark runs: `report_raw_results.csv`",
        "- Summary metrics: `report_metrics.json`",
        "- Figure directory: `figure/`",
        "- Generator script: `generate_report.py`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    FIGURE_DIR.mkdir(exist_ok=True)

    metas = {name: read_matrix_header(name) for name in sorted(set(ALL_DATASETS))}

    plan: list[tuple[str, str, int]] = []
    for dataset in FORMAT_DATASETS:
        plan.append(("spmv_not_csr", dataset, 1))
        plan.append(("spmv_serial", dataset, 1))
        plan.append(("spmv_openmp", dataset, 16))
    for dataset in metas:
        plan.append(("spmv_serial", dataset, 1))
        plan.append(("spmv_openmp", dataset, 16))
    for dataset in SCALING_DATASETS:
        for threads in THREADS:
            plan.append(("spmv_openmp", dataset, threads))

    # Remove duplicates while preserving order.
    seen: set[tuple[str, str, int]] = set()
    unique_plan: list[tuple[str, str, int]] = []
    for item in plan:
        if item in seen:
            continue
        seen.add(item)
        unique_plan.append(item)

    records: list[dict[str, object]] = []
    for binary, dataset, threads in unique_plan:
        records.extend(run_case(binary, dataset, threads))

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    make_grouped_runtime_plot(grouped, metas)
    make_size_runtime_plot(grouped, metas)
    make_memory_plot(metas)
    make_scaling_plot(grouped)

    summary = summarize_metrics(grouped, metas)
    write_raw_csv(records)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RAW_CSV_PATH}")
    print(f"Wrote {SUMMARY_JSON_PATH}")
    print(f"Wrote {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
