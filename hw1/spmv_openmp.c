/* Student stub: spmv_openmp.c
 * Keeps IO, timing, verification. Students implement CSR conversion and OpenMP SpMV.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
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

/* Build CSR with parallel first-touch on col_idx/vals_csr.
 * On this 2-socket Xeon, a serial build leaves all CSR pages on socket 0
 * and socket-1 threads pay cross-UPI traffic during SpMV. By letting each
 * worker zero its own rows' slice first, Linux pins those pages to the
 * worker's local NUMA node, so the later SpMV reads are all node-local.
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

    *row_ptr_out  = row_ptr;
    *col_idx_out  = col_idx;
    *vals_csr_out = vals_csr;
}

/* SpMV: static schedule matches the NUMA first-touch done in build_csr.
 * restrict lets the compiler assume no aliasing between arrays.
 */
void spmv_csr_openmp(int M,
                     const int    * restrict row_ptr,
                     const int    * restrict col_idx,
                     const double * restrict vals_csr,
                     const double * restrict x,
                     double       * restrict y) {
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (int i = 0; i < M; i++) {
        const int kbeg = row_ptr[i];
        const int kend = row_ptr[i+1];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int k = kbeg;
        const int kend4 = kbeg + ((kend - kbeg) & ~3);
        for (; k < kend4; k += 4) {
            s0 += vals_csr[k  ] * x[col_idx[k  ]];
            s1 += vals_csr[k+1] * x[col_idx[k+1]];
            s2 += vals_csr[k+2] * x[col_idx[k+2]];
            s3 += vals_csr[k+3] * x[col_idx[k+3]];
        }
        for (; k < kend; k++)
            s0 += vals_csr[k] * x[col_idx[k]];
        y[i] = (s0 + s1) + (s2 + s3);
    }
}

int main(int argc, char **argv) {
    /* Pin OpenMP threads across both sockets before any parallel region
     * runs. libgomp reads these lazily, so setenv here still takes effect.
     * The "0" override flag leaves any user-supplied value alone.
     */
    setenv("OMP_PROC_BIND", "spread", 0);
    setenv("OMP_PLACES",    "cores",  0);

    if (argc < 2) { fprintf(stderr,"Usage: %s matrix.mtx [vector.txt]\n", argv[0]); return 1; }
    const char *mtx = argv[1]; const char *vec = (argc>2?argv[2]:NULL);
    int M,N,nnz; int *rows=NULL,*cols=NULL; double *vals=NULL;
    if (read_mtx(mtx,&M,&N,&nnz,&rows,&cols,&vals)!=0) { fprintf(stderr,"Failed to read mtx\n"); return 1; }
    double *x=NULL; read_vec(vec,N,&x);

    int *row_ptr=NULL,*col_idx=NULL; double *vals_csr=NULL;
    build_csr(M,N,nnz,rows,cols,vals,&row_ptr,&col_idx,&vals_csr);

    double *y = malloc(sizeof(double)*M);
    double t0 = time_ms();
    spmv_csr_openmp(M,row_ptr,col_idx,vals_csr,x,y);
    double t1 = time_ms();
    fprintf(stderr,"spmv_openmp_time_ms=%.3f\n", t1-t0);

    char goldpath[1024]; snprintf(goldpath,sizeof(goldpath),"%s.gold", mtx);
    double *ygold=NULL; if (read_gold(goldpath,M,&ygold)==0) {
        if (verify(M,y,ygold)) fprintf(stderr,"OK\n"); else fprintf(stderr,"WRONG\n"); free(ygold);
    } else fprintf(stderr,"No gold found (%s) — skipping verify\n", goldpath);

    free(rows); free(cols); free(vals); free(row_ptr); free(col_idx); free(vals_csr); free(x); free(y);
    return 0;
}
