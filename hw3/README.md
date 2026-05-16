# HW4: CUDA Matmul Optimization — Student Skeleton

Implement your SGEMM kernel in [kernels/student_kernel.cu](kernels/student_kernel.cu).
Do **not** modify any other file. TA grades with the official skeleton; only your
`student_kernel.cu` is copied in.

## Build
```sh
module load cuda
make                # default CUDA_ARCH=70 (V100)
```

## Run
```sh
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 ./main
```

## Submit
Zip:
```
Team_<N>_HW4.zip
  ├── kernels/student_kernel.cu
  └── Team_<N>_HW4_report.pdf
```

See the assignment spec (`HW4_2026Spring.md`) for full grading rules and tiers.
