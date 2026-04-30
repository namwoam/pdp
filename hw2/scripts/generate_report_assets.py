from __future__ import annotations

import csv
import math
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"
TESTCASE_DIR = Path("/home/Team12/testcases")


# Median wall times from: REPS=10 NPROCS="1 4 8 16" ./bench.sh
# Collected on 2026-05-01 in the local HW3 MPI environment.
BENCHMARK_ROWS = [
    ("imbalance_c100000", 1, 0.225, 1.00, "OK"),
    ("imbalance_c100000", 4, 0.092, 1.04, "OK"),
    ("imbalance_c100000", 8, 0.087, 1.32, "OK"),
    ("imbalance_c100000", 16, 0.068, 1.34, "OK"),
    ("large_c1000000", 1, 1.715, 1.00, "OK"),
    ("large_c1000000", 4, 0.552, 1.00, "OK"),
    ("large_c1000000", 8, 0.393, 1.15, "OK"),
    ("large_c1000000", 16, 0.298, 1.21, "OK"),
    ("large_c2000000", 1, 3.407, 1.00, "OK"),
    ("large_c2000000", 4, 1.114, 1.02, "OK"),
    ("large_c2000000", 8, 0.667, 1.07, "OK"),
    ("large_c2000000", 16, 0.480, 1.11, "OK"),
    ("large_c4000000", 1, 4.506, 1.00, "OK"),
    ("large_c4000000", 4, 2.126, 1.11, "OK"),
    ("large_c4000000", 8, 1.254, 1.03, "OK"),
    ("large_c4000000", 16, 0.860, 1.05, "OK"),
    ("medium_c200000", 1, 0.368, 1.00, "OK"),
    ("medium_c200000", 4, 0.147, 1.02, "OK"),
    ("medium_c200000", 8, 0.119, 1.17, "OK"),
    ("medium_c200000", 16, 0.092, 1.22, "OK"),
]


def read_testcase_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(TESTCASE_DIR.glob("*.bin")):
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != b"CRDR":
                raise ValueError(f"{path} has bad magic {magic!r}")
            version = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            bbox = struct.unpack("<6f", f.read(24))

            radius_sum = 0.0
            radius_sq_sum = 0.0
            radius_min = math.inf
            radius_max = 0.0
            for _ in range(count):
                rec = f.read(15)
                if len(rec) != 15:
                    raise ValueError(f"{path} ended inside circle records")
                _, _, radius = struct.unpack("<fff", rec[:12])
                radius_sum += radius
                radius_sq_sum += radius * radius
                radius_min = min(radius_min, radius)
                radius_max = max(radius_max, radius)

        width = round(bbox[3] - bbox[0])
        height = round(bbox[4] - bbox[1])
        rows.append(
            {
                "testcase": path.stem,
                "version": version,
                "circles": count,
                "width": width,
                "height": height,
                "pixels": width * height,
                "mean_radius": radius_sum / count if count else 0.0,
                "min_radius": radius_min if count else 0.0,
                "max_radius": radius_max,
                "sum_radius_sq": radius_sq_sum,
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_df(metadata: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        BENCHMARK_ROWS,
        columns=["testcase", "processes", "time_s", "imbalance_max_mean", "verify"],
    )
    base = df[df["processes"] == 1][["testcase", "time_s"]].rename(
        columns={"time_s": "serial_time_s"}
    )
    df = df.merge(base, on="testcase", how="left").merge(metadata, on="testcase", how="left")
    df["speedup"] = df["serial_time_s"] / df["time_s"]
    df["efficiency"] = df["speedup"] / df["processes"]
    df["circles_per_process"] = df["circles"] / df["processes"]
    df["throughput_mcircles_s"] = df["circles"] / df["time_s"] / 1_000_000
    return df.sort_values(["testcase", "processes"])


def write_csvs(metadata: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    metadata.to_csv(DATA_DIR / "testcase_metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    benchmark.to_csv(DATA_DIR / "benchmark_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = (
        benchmark[benchmark["processes"] == 16]
        .loc[
            :,
            [
                "testcase",
                "circles",
                "time_s",
                "speedup",
                "efficiency",
                "imbalance_max_mean",
                "throughput_mcircles_s",
                "verify",
            ],
        ]
        .sort_values("circles")
    )
    summary.to_csv(DATA_DIR / "summary_16proc.csv", index=False)


def save_figures(benchmark: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", n_colors=benchmark["testcase"].nunique())

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=benchmark,
        x="processes",
        y="time_s",
        hue="testcase",
        marker="o",
        palette=palette,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Median wall time (s, log scale)")
    ax.set_title("Strong Scaling Runtime")
    ax.legend(title="testcase", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "strong_scaling_runtime.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=benchmark,
        x="processes",
        y="speedup",
        hue="testcase",
        marker="o",
        palette=palette,
    )
    ax.plot([1, 16], [1, 16], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Speedup vs. 1 process")
    ax.set_title("Speedup")
    ax.legend(title="testcase", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "speedup.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=benchmark,
        x="processes",
        y="efficiency",
        hue="testcase",
        marker="o",
        palette=palette,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Parallel efficiency")
    ax.set_title("Efficiency")
    ax.legend(title="testcase", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "efficiency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=benchmark,
        x="processes",
        y="imbalance_max_mean",
        hue="testcase",
        marker="o",
        palette=palette,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Render imbalance (max / mean)")
    ax.set_title("Per-Rank Render Load Balance")
    ax.legend(title="testcase", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "imbalance.png", dpi=180)
    plt.close()

    proc16 = benchmark[benchmark["processes"] == 16].sort_values("circles")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=proc16,
        x="circles_per_process",
        y="time_s",
        marker="o",
        color="#356859",
    )
    for _, row in proc16.iterrows():
        ax.annotate(
            row["testcase"].replace("_", "\n"),
            (row["circles_per_process"], row["time_s"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Circles per process at 16 MPI processes")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title("Problem Size per Process")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "problem_size_per_process.png", dpi=180)
    plt.close()


def main() -> None:
    metadata = read_testcase_metadata()
    benchmark = build_benchmark_df(metadata)
    write_csvs(metadata, benchmark)
    save_figures(benchmark)

    print(f"wrote {DATA_DIR / 'testcase_metadata.csv'}")
    print(f"wrote {DATA_DIR / 'benchmark_results.csv'}")
    print(f"wrote {DATA_DIR / 'summary_16proc.csv'}")
    for figure in sorted(FIGURE_DIR.glob("*.png")):
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
