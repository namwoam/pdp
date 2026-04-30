# HW3 MPI Circle Renderer Report

## Summary

This project implements the required CRDR circle renderer in `renderer.c` and the optimized MPI implementation in `renderer_mpi.c`. The final MPI renderer preserves the file order semantics specified in `spec.pdf`: circles are stored back-to-front, so rendering records in input order lets later circles overwrite earlier circles exactly.

All benchmarked outputs were byte-identical to the provided golden PNG files. The benchmark data used in this report were generated with:

```bash
REPS=10 NPROCS="1 4 8 16 64" ./bench.sh
poetry run python scripts/generate_report_assets.py
```

The generated data and figures are stored in:

- `data/testcase_metadata.csv`
- `data/benchmark_results.csv`
- `data/summary_16proc.csv`
- `data/summary_64proc.csv`
- `figure/strong_scaling_runtime.png`
- `figure/speedup.png`
- `figure/efficiency.png`
- `figure/imbalance.png`
- `figure/problem_size_per_process.png`

## Implementation

The final MPI strategy is cyclic row decomposition. Rank `r` owns image rows `r, r + P, r + 2P, ...`, where `P` is the number of MPI processes. Rank 0 reads the CRDR file, broadcasts the header and all circle records, and each rank rasterizes every circle only into its owned rows. At the end, `MPI_Gatherv` collects compact local row slices and rank 0 re-interleaves them into the final PNG.

The main optimizations are:

- Cyclic row ownership instead of contiguous row blocks, which avoids severe spatial imbalance when circles cluster in one image region.
- Direct opaque RGB writes instead of alpha compositing, matching the assignment format where records have only `uint8 r, g, b`.
- Per-row analytic x-range trimming using `sqrtf`, followed by the exact circle-inside test for correctness.
- Local compact row buffers, so each process stores only its assigned rows instead of a full image.
- In-source GCC optimization pragmas because the official compile command uses `mpicc renderer_mpi.c -o renderer_mpi -lm -march=native` without an explicit `-O` flag.

## Test Cases

The binary metadata in `data/testcase_metadata.csv` shows that the large cases use a 640x480 image with 1M, 2M, and 4M circles. The special `imbalance_c100000` case is smaller, 320x240 with 100K circles, but has much larger radius variation: max radius is about 1998 pixels, compared with about 20 pixels for the regular large cases. This explains why it is useful for load-balance stress testing.

| Testcase | Circles | Image | Mean radius | Max radius |
|---|---:|---:|---:|---:|
| `imbalance_c100000` | 100,000 | 320x240 | 25.62 | 1998.10 |
| `medium_c200000` | 200,000 | 640x480 | 10.49 | 20.00 |
| `large_c1000000` | 1,000,000 | 640x480 | 10.50 | 20.00 |
| `large_c2000000` | 2,000,000 | 640x480 | 10.50 | 20.00 |
| `large_c4000000` | 4,000,000 | 640x480 | 10.50 | 20.00 |

## Benchmark Results

The benchmark uses the median wall time across 10 repetitions for each `(testcase, process count)` pair. The one-process MPI run is used as the serial baseline for speedup and efficiency because it exercises the same optimized renderer with `P=1`.

![Strong scaling runtime](figure/strong_scaling_runtime.png)

![Speedup](figure/speedup.png)

| Testcase | 1 proc (s) | 4 proc (s) | 8 proc (s) | 16 proc (s) | 64 proc (s) | 64-proc speedup | 64-proc efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `imbalance_c100000` | 0.165 | 0.098 | 0.088 | 0.068 | 0.065 | 2.54x | 4.0% |
| `medium_c200000` | 0.368 | 0.163 | 0.127 | 0.098 | 0.090 | 4.09x | 6.4% |
| `large_c1000000` | 1.780 | 0.325 | 0.224 | 0.163 | 0.153 | 11.63x | 18.2% |
| `large_c2000000` | 3.336 | 0.609 | 0.375 | 0.284 | 0.252 | 13.24x | 20.7% |
| `large_c4000000` | 4.534 | 1.186 | 0.656 | 0.470 | 0.445 | 10.19x | 15.9% |

The best 64-process strong-scaling result is `large_c2000000`, which improves from 3.336 s on one process to 0.252 s on 64 processes, a 13.24x speedup. On a single local node, the same case improves to 0.284 s on 16 processes, an 11.75x speedup with 73.4% efficiency. The smaller cases still improve, but their efficiency drops because fixed costs such as record broadcast, gather, PNG writing, MPI launch overhead, and multi-node communication become a larger fraction of total runtime.

![Efficiency](figure/efficiency.png)

## Problem Size Per Process

At 64 processes, throughput increases with larger scenes: 1.54 million circles/s for `imbalance_c100000`, 2.22 million circles/s for `medium_c200000`, 6.54 million circles/s for `large_c1000000`, 7.94 million circles/s for `large_c2000000`, and 8.99 million circles/s for `large_c4000000`. This is shown in `data/summary_64proc.csv` and `figure/problem_size_per_process.png`.

![Problem size per process](figure/problem_size_per_process.png)

The weak-scaling conclusion is mixed. More circles per process amortize fixed MPI and output costs, so throughput improves as the problem gets larger. However, wall time still grows from 0.153 s at 1M circles to 0.445 s at 4M circles on 64 processes because each process still loops over every circle and only skips pixels outside its owned rows. The algorithm scales well enough for strong scaling, but it is not true weak scaling because broadcast size and per-rank circle iteration both grow with total circle count.

## Load Balance

`renderer_mpi.c` logs per-rank render time statistics and reports imbalance as `max / mean`. The measured imbalance values are in `data/benchmark_results.csv` and plotted below.

![Render imbalance](figure/imbalance.png)

The regular large cases are reasonably balanced at 16 processes: `large_c2000000` has imbalance 1.23 and `large_c4000000` has imbalance 1.10. At 64 processes, the regular large cases remain controlled, with imbalance values from 1.27 to 1.48. The cyclic row decomposition is effective here because each rank receives rows distributed across the full image instead of one contiguous vertical region.

The worst 64-process imbalance is `imbalance_c100000` at 1.64. The special imbalance case contains extremely large circles and a much smaller image, so some rows receive more work even after cyclic distribution. The imbalance is still controlled: the renderer continues to improve from 0.165 s on one process to 0.065 s on 64 processes.

## Conclusion

The MPI renderer is correct on all benchmarked test cases and shows useful strong scaling through the 64-process competition configuration. The best measured result is 13.24x speedup on `large_c2000000`; the 4M-circle case reaches the highest throughput at 8.99 million circles/s but lower speedup because the one-process baseline is already more throughput-efficient for that large input.

The main performance limitation is that every rank still scans every circle record. This avoids ordering and compositing complexity and makes correctness straightforward, but it means broadcast and per-rank loop overhead grow with scene size. A future improvement would combine row decomposition with a spatial binning or circle-to-row-range index so each process skips circles that cannot affect any owned row, while preserving global back-to-front order for overlapping pixels.
