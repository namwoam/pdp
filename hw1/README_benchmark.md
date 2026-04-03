# SPMV Benchmark Script (Poetry)

This project includes a simple Python benchmark utility for:

- Running SPMV binaries repeatedly
- Capturing hardware information for cross-machine comparison
- Writing raw results (CSV) and summary (JSON)
- Generating a bar chart of median runtime per dataset
- Generating a Markdown report that embeds the visualization and hardware info

## 1) Install Poetry and dependencies

From `hw1/`:

```bash
poetry install
```

## 2) Build binaries

```bash
/opt/homebrew/opt/llvm/bin/clang -I/opt/homebrew/opt/libomp/include -fopenmp -O3 spmv_openmp.c -L/opt/homebrew/opt/libomp/lib -lomp -o spmv_openmp
/opt/homebrew/opt/llvm/bin/clang -O3 spmv_serial.c -o spmv_serial
/opt/homebrew/opt/llvm/bin/clang -O3 spmv_not_csr.c -o spmv_not_csr
```

## 3) Run benchmark

```bash
poetry run python benchmark_spmv.py --runs 7 --threads 8
```

If `--datasets` is omitted, the script runs on all `.mtx` files under `testcases/testcases/`.

Optional custom binaries/datasets:

```bash
poetry run python benchmark_spmv.py \
  --binaries ./spmv_openmp ./spmv_serial ./spmv_not_csr \
  --datasets ./testcases/testcases/test_sparse_1000 ./testcases/testcases/huge_200k_100 \
  --runs 10 \
  --threads 8
```

## 4) Outputs

By default outputs are written to `hw1/bench_results/` (stored with source code):

- `benchmark_runs_<timestamp>.csv`
- `benchmark_summary_<timestamp>.json`
- `benchmark_plot_<timestamp>.png`
- `benchmark_report_<timestamp>.md`

The JSON file includes a `hardware` section (OS, CPU identifiers, core counts, memory where available) to make cross-machine results comparable.
The Markdown report includes hardware details, a median runtime table, and the plot image.
