/* Student stub: spmv_openmp.c
 * Keeps IO, timing, verification. Students implement CSR conversion and OpenMP SpMV.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double time_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* Minimal MTX reader (coordinate) */
int read_mtx(const char *path, int *M, int *N, int *nnz,
             int **rows, int **cols, double **vals) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[1024]; do { if (!fgets(line,sizeof(line),f)){fclose(f);return-1;} } while(line[0]=='%');
    int m,n,k; if (sscanf(line, "%d %d %d", &m,&n,&k)!=3) { fclose(f); return -1; }
    *M=m; *N=n; *nnz=k;
    *rows=malloc(sizeof(int)*k); *cols=malloc(sizeof(int)*k); *vals=malloc(sizeof(double)*k);
    for (int i=0;i<k;i++){int r,c; double v; if (fscanf(f, "%d %d %lf", &r,&c,&v)!=3){fclose(f);return-1;} (*rows)[i]=r-1; (*cols)[i]=c-1; (*vals)[i]=v; }
    fclose(f); return 0;
}

int read_vec(const char *path, int N, double **x) {
    FILE *f = fopen(path, "r"); *x = malloc(sizeof(double)*N);
    if (!f) { for(int i=0;i<N;i++) (*x)[i]=1.0; return 0; }
    for (int i=0;i<N;i++) { if (fscanf(f, "%lf", &(*x)[i])!=1) (*x)[i]=1.0; }
    fclose(f); return 0;
}

int read_gold(const char *path, int M, double **ygold) { FILE *f=fopen(path,"r"); if(!f) return -1; *ygold=malloc(sizeof(double)*M); for(int i=0;i<M;i++) if(fscanf(f,"%lf",&(*ygold)[i])!=1) (*ygold)[i]=0.0; fclose(f); return 0; }

int verify(int M, double *y, double *ygold) { double tol=0.02; for(int i=0;i<M;i++) if (fabs(y[i]-ygold[i])>tol) return 0; return 1; }

/* Build CSR from COO. Parallel first-touch for NUMA locality,
 * then sort each row by column for cache-friendly x[] gather.
 */
void build_csr(int M, int N, int nnz, int *rows, int *cols, double *vals,
               int **row_ptr_out, int **col_idx_out, double **vals_csr_out) {
    int    *row_ptr  = calloc(M+1, sizeof(int));
    int    *col_idx  = malloc(sizeof(int)*nnz);
    double *vals_csr = malloc(sizeof(double)*nnz);

    for (int k = 0; k < nnz; k++) row_ptr[rows[k] + 1]++;
    for (int i = 0; i < M; i++)   row_ptr[i+1] += row_ptr[i];

    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (int i = 0; i < M; i++) {
        for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
            col_idx[k]  = 0;
            vals_csr[k] = 0.0;
        }
    }

    int *tmp = calloc(M > 0 ? (size_t)M : 0, sizeof(int));
    for (int k = 0; k < nnz; k++) {
        int r = rows[k];
        int dest = row_ptr[r] + tmp[r];
        col_idx[dest]  = cols[k];
        vals_csr[dest] = vals[k];
        tmp[r]++;
    }
    free(tmp);

    /* Sort each row by column index for sequential x[] access. */
    #pragma omp parallel for schedule(dynamic, 64) proc_bind(spread)
    for (int i = 0; i < M; i++) {
        int lo = row_ptr[i], hi = row_ptr[i+1];
        for (int a = lo + 1; a < hi; a++) {
            int    ck = col_idx[a];
            double vk = vals_csr[a];
            int b = a - 1;
            while (b >= lo && col_idx[b] > ck) {
                col_idx[b+1]  = col_idx[b];
                vals_csr[b+1] = vals_csr[b];
                b--;
            }
            col_idx[b+1]  = ck;
            vals_csr[b+1] = vk;
        }
    }

    *row_ptr_out  = row_ptr;
    *col_idx_out  = col_idx;
    *vals_csr_out = vals_csr;
}

/* Per-thread replica of x to avoid cross-socket reads. */
static double *build_thread_local_x(int N, const double *x, int *stride_out) {
#ifdef _OPENMP
    const int nth = omp_get_max_threads();
#else
    const int nth = 1;
#endif
    const size_t max_bytes = 512u * 1024u * 1024u;

    *stride_out = N;
    if (N <= 0 || nth <= 1 || N < 8192 || N > 65536) return NULL;

    const size_t stride = (size_t)N;
    if (stride > SIZE_MAX / (size_t)nth) return NULL;
    const size_t elems = stride * (size_t)nth;
    if (elems > SIZE_MAX / sizeof(double)) return NULL;
    const size_t bytes = elems * sizeof(double);
    if (bytes > max_bytes) return NULL;

    double *x_local = malloc(bytes);
    if (!x_local) return NULL;

#ifdef _OPENMP
    #pragma omp parallel proc_bind(spread)
    {
        const int tid = omp_get_thread_num();
        double *dst = x_local + (size_t)tid * stride;
        for (int i = 0; i < N; i++) dst[i] = x[i];
    }
#else
    memcpy(x_local, x, bytes);
#endif
    return x_local;
}

/* First-touch y to avoid page faults in the timed region. */
static void prepare_output(int M, double *y) {
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (int i = 0; i < M; i++) y[i] = 0.0;
}

/* SpMV kernel. Static schedule matches NUMA first-touch in build_csr. */
void spmv_csr_openmp(int M,
                     const int    * restrict row_ptr,
                     const int    * restrict col_idx,
                     const double * restrict vals_csr,
                     const double * restrict x_shared,
                     const double * restrict x_local,
                     int x_stride,
                     double       * restrict y) {
    #pragma omp parallel proc_bind(spread)
    {
        const double * restrict x = x_shared;
#ifdef _OPENMP
        if (x_local) x = x_local + (size_t)omp_get_thread_num() * (size_t)x_stride;
#endif

        #pragma omp for schedule(static)
        for (int i = 0; i < M; i++) {
            const int kbeg = row_ptr[i];
            const int kend = row_ptr[i+1];
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = kbeg; k < kend; k++)
                sum += vals_csr[k] * x[col_idx[k]];
            y[i] = sum;
        }
    }
}

int main(int argc, char **argv) {
    /* Spread threads across sockets. "0" preserves user-set values. */
    setenv("OMP_PROC_BIND", "spread", 0);
    setenv("OMP_PLACES",    "cores",  0);

    if (argc < 2) { fprintf(stderr,"Usage: %s matrix.mtx [vector.txt]\n", argv[0]); return 1; }
    const char *mtx = argv[1]; const char *vec = (argc>2?argv[2]:NULL);
    int M,N,nnz; int *rows=NULL,*cols=NULL; double *vals=NULL;
    if (read_mtx(mtx,&M,&N,&nnz,&rows,&cols,&vals)!=0) { fprintf(stderr,"Failed to read mtx\n"); return 1; }
    double *x=NULL; read_vec(vec,N,&x);
    int x_stride = N;
    double *x_local = build_thread_local_x(N, x, &x_stride);

    int *row_ptr=NULL,*col_idx=NULL; double *vals_csr=NULL;
    build_csr(M,N,nnz,rows,cols,vals,&row_ptr,&col_idx,&vals_csr);

    double *y = malloc(sizeof(double)*M);
    prepare_output(M, y);
    double t0 = time_ms();
    spmv_csr_openmp(M,row_ptr,col_idx,vals_csr,x,x_local,x_stride,y);
    double t1 = time_ms();
    fprintf(stderr,"spmv_openmp_time_ms=%.3f\n", t1-t0);

    char goldpath[1024]; snprintf(goldpath,sizeof(goldpath),"%s.gold", mtx);
    double *ygold=NULL; if (read_gold(goldpath,M,&ygold)==0) {
        if (verify(M,y,ygold)) fprintf(stderr,"OK\n"); else fprintf(stderr,"WRONG\n"); free(ygold);
    } else fprintf(stderr,"No gold found (%s) — skipping verify\n", goldpath);

    free(rows); free(cols); free(vals); free(row_ptr); free(col_idx); free(vals_csr); free(x); free(x_local); free(y);
    return 0;
}
