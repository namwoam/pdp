"""Generate report figures and CSVs for HW3 (CUDA SGEMM).

Parses the raw stdout dumps written by ./bench.sh into data/raw_*.txt,
emits data/results.csv, data/summary_4096.csv, and the figure/*.png set
used in the report.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"

LINE_RE = re.compile(
    r"Running size:\s*(?P<size>\d+).*?"
    r"avg time:\s*(?P<time>[0-9.eE+-]+)s,\s*"
    r"performance:\s*(?P<gflops>[0-9.eE+-]+)\s*GFLOPS"
)

# Absolute GFLOPS thresholds at size=4096 from the HW3 spec §6.
TIER_THRESHOLDS = [("T1", 200), ("T2", 2050), ("T3", 3700), ("T4", 7600), ("T5", 9500)]

# Per-kernel labels for figures. Add an entry here when you add a
# new kernel id in kernels/runner.cu.
KERNEL_LABELS = {
    "cublas": "cuBLAS (FP32, reference)",
    "student": "Student kernel",
}

# Optimization progression: hand-maintained list of (stage, GFLOPS @ 4096).
# Update as you iterate so the report has the table the spec asks for.
# Numbers here are placeholders; replace with measurements from each step.
OPTIMIZATION_STAGES = [
    ("naive", 200.0),
    ("+ shared-mem tiling", 1800.0),
    ("+ 1D blocktiling", 3500.0),
    ("+ 2D blocktiling + regs", 6200.0),
    ("+ vectorized float4", 7400.0),
    ("+ warp tiling", 8200.0),
    ("+ double buffering", 8800.0),
    ("+ block swizzle (final)", 9000.0),
]


def parse_raw(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"raw_{name}.txt"
    if not path.exists():
        return pd.DataFrame(columns=["kernel", "size", "avg_time_s", "gflops"])
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            rows.append(
                {
                    "kernel": name,
                    "size": int(m["size"]),
                    "avg_time_s": float(m["time"]),
                    "gflops": float(m["gflops"]),
                }
            )
    return pd.DataFrame(rows)


def load_results() -> pd.DataFrame:
    frames = [parse_raw(name) for name in KERNEL_LABELS]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise SystemExit(
            f"no parseable runs in {DATA_DIR}/raw_*.txt — run ./bench.sh first"
        )
    df["label"] = df["kernel"].map(KERNEL_LABELS).fillna(df["kernel"])
    return df.sort_values(["kernel", "size"])


def write_csvs(df: pd.DataFrame) -> None:
    df.to_csv(DATA_DIR / "results.csv", index=False)

    wide = (
        df.pivot_table(index="size", columns="kernel", values="gflops")
        .reset_index()
        .sort_values("size")
    )
    if {"cublas", "student"}.issubset(wide.columns):
        wide["percent_cublas"] = 100.0 * wide["student"] / wide["cublas"]
    wide.to_csv(DATA_DIR / "results_by_size.csv", index=False)

    at_4096 = df[df["size"] == 4096][["kernel", "avg_time_s", "gflops"]]
    if not at_4096.empty:
        at_4096.to_csv(DATA_DIR / "summary_4096.csv", index=False)


def plot_gflops_vs_size(df: pd.DataFrame) -> None:
    sizes = sorted(df["size"].unique())
    plt.figure(figsize=(9, 5.5))
    ax = sns.lineplot(
        data=df, x="size", y="gflops", hue="label", marker="o", linewidth=2
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size M = N = K")
    ax.set_ylabel("GFLOPS")
    ax.set_title("SGEMM throughput vs. matrix size")
    ax.legend(title="kernel", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gflops_vs_size.png", dpi=180)
    plt.close()


def plot_runtime_vs_size(df: pd.DataFrame) -> None:
    sizes = sorted(df["size"].unique())
    plt.figure(figsize=(9, 5.5))
    ax = sns.lineplot(
        data=df, x="size", y="avg_time_s", hue="label", marker="o", linewidth=2
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size M = N = K")
    ax.set_ylabel("Avg kernel time (s, log scale)")
    ax.set_title("Kernel runtime vs. matrix size")
    ax.legend(title="kernel", fontsize="small")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "runtime_vs_size.png", dpi=180)
    plt.close()


def plot_percent_cublas(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="size", columns="kernel", values="gflops")
    if not {"cublas", "student"}.issubset(pivot.columns):
        return
    pivot = pivot.dropna(subset=["cublas", "student"]).sort_index()
    pct = 100.0 * pivot["student"] / pivot["cublas"]

    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(x=[str(s) for s in pct.index], y=pct.values, color="#356859")
    for i, v in enumerate(pct.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Matrix size M = N = K")
    ax.set_ylabel("Student GFLOPS / cuBLAS GFLOPS (%)")
    ax.set_title("Fraction of cuBLAS achieved")
    ax.set_ylim(0, max(pct.max() * 1.15, 100))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "percent_cublas.png", dpi=180)
    plt.close()


def plot_tier_thresholds(df: pd.DataFrame) -> None:
    student_4096 = df[(df["kernel"] == "student") & (df["size"] == 4096)]
    if student_4096.empty:
        return
    g_student = float(student_4096["gflops"].iloc[0])
    cublas_4096 = df[(df["kernel"] == "cublas") & (df["size"] == 4096)]
    g_cublas = float(cublas_4096["gflops"].iloc[0]) if not cublas_4096.empty else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(["Student @ 4096"], [g_student], color="#356859", width=0.45)
    if g_cublas is not None:
        ax.bar(["cuBLAS @ 4096"], [g_cublas], color="#7f7f7f", width=0.45)

    palette = sns.color_palette("flare", n_colors=len(TIER_THRESHOLDS))
    for (tier, thr), color in zip(TIER_THRESHOLDS, palette):
        ax.axhline(thr, color=color, linestyle="--", linewidth=1)
        ax.text(
            1.01,
            thr,
            f"{tier} ({thr})",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=color,
        )

    ax.set_ylabel("GFLOPS @ size = 4096")
    ax.set_title("Performance tiers (spec §6) vs. measured")
    ax.set_ylim(0, max(g_student, g_cublas or 0, 10000) * 1.1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tier_thresholds.png", dpi=180)
    plt.close()


def plot_optimization_progression() -> None:
    if not OPTIMIZATION_STAGES:
        return
    labels = [s for s, _ in OPTIMIZATION_STAGES]
    values = [g for _, g in OPTIMIZATION_STAGES]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette("viridis", n_colors=len(labels))
    ax.barh(labels, values, color=palette)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.0f}", va="center", fontsize=9)

    for tier, thr in TIER_THRESHOLDS:
        ax.axvline(thr, color="#888", linestyle=":", linewidth=1)
        ax.text(thr, len(labels) - 0.4, tier, fontsize=8, color="#555", ha="center")

    ax.invert_yaxis()
    ax.set_xlabel("GFLOPS @ size = 4096")
    ax.set_title("Optimization progression (hand-recorded)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "optimization_progression.png", dpi=180)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    df = load_results()
    write_csvs(df)

    plot_gflops_vs_size(df)
    plot_runtime_vs_size(df)
    plot_percent_cublas(df)
    plot_tier_thresholds(df)
    plot_optimization_progression()

    print(f"wrote {DATA_DIR / 'results.csv'}")
    print(f"wrote {DATA_DIR / 'results_by_size.csv'}")
    for fig in sorted(FIGURE_DIR.glob("*.png")):
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()
