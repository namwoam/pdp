/* Student stub: spmv_serial.c
 * Keeps IO, timing, verification. Students implement CSR conversion and serial SpMV.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

int read_gold(const char *path, int M, double **ygold) {
    FILE *f = fopen(path, "r"); if (!f) return -1; *ygold = malloc(sizeof(double)*M);
    for (int i=0;i<M;i++) if (fscanf(f, "%lf", &(*ygold)[i])!=1) (*ygold)[i]=0.0;
    fclose(f); return 0;
}

int verify(int M, double *y, double *ygold) { double tol=0.02; for(int i=0;i<M;i++){ if (fabs(y[i]-ygold[i])>tol) return 0;} return 1; }

/* TODO: Student implements CSR builder. Provide row_ptr, col_idx, vals_csr arrays. */
void build_csr(int M, int N, int nnz, int *rows, int *cols, double *vals,
               int **row_ptr, int **col_idx, double **vals_csr) {
    fprintf(stderr, "[STUDENT] build_csr() not implemented — fill this in.\n");
    *row_ptr = calloc(M+1, sizeof(int)); *col_idx = malloc(sizeof(int)*nnz); *vals_csr = malloc(sizeof(double)*nnz);
}

/* TODO: Student implements serial CSR SpMV (y = A*x). Current stub zeroes y. */
void spmv_csr_serial(int M, int *row_ptr, int *col_idx, double *vals_csr, double *x, double *y) {
    fprintf(stderr, "[STUDENT] spmv_csr_serial() not implemented — fill this in.\n");
    for (int i=0;i<M;i++) y[i]=0.0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,"Usage: %s matrix.mtx [vector.txt]\n", argv[0]); return 1; }
    const char *mtx = argv[1]; const char *vec = (argc>2?argv[2]:NULL);
    int M,N,nnz; int *rows=NULL,*cols=NULL; double *vals=NULL;
    if (read_mtx(mtx,&M,&N,&nnz,&rows,&cols,&vals)!=0) { fprintf(stderr,"Failed to read mtx\n"); return 1; }
    double *x=NULL; read_vec(vec,N,&x);

    int *row_ptr=NULL,*col_idx=NULL; double *vals_csr=NULL;
    build_csr(M,N,nnz,rows,cols,vals,&row_ptr,&col_idx,&vals_csr);

    double *y = malloc(sizeof(double)*M);
    double t0 = time_ms();
    spmv_csr_serial(M,row_ptr,col_idx,vals_csr,x,y);
    double t1 = time_ms();
    fprintf(stderr,"spmv_serial_time_ms=%.3f\n", t1-t0);

    char goldpath[1024]; snprintf(goldpath,sizeof(goldpath),"%s.gold", mtx);
    double *ygold=NULL; if (read_gold(goldpath,M,&ygold)==0) {
        if (verify(M,y,ygold)) fprintf(stderr,"OK\n"); else fprintf(stderr,"WRONG\n"); free(ygold);
    } else fprintf(stderr,"No gold found (%s) — skipping verify\n", goldpath);

    free(rows); free(cols); free(vals); free(row_ptr); free(col_idx); free(vals_csr); free(x); free(y);
    return 0;
}
