/* Student stub: spmv_not_csr.c
 * Keeps IO, timing, verification. Students implement the naive COO-scan compute.
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

/* Simple Matrix Market reader (coordinate real) */
int read_mtx(const char *path, int *M, int *N, int *nnz,
             int **rows, int **cols, double **vals) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    /* skip header/comments */
    do { if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; } } while(line[0] == '%');
    int m,n,k;
    if (sscanf(line, "%d %d %d", &m, &n, &k) != 3) { fclose(f); return -1; }
    *M = m; *N = n; *nnz = k;
    *rows = malloc(sizeof(int)*k);
    *cols = malloc(sizeof(int)*k);
    *vals = malloc(sizeof(double)*k);
    for (int i=0;i<k;i++) {
        int r,c; double v; if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) { fclose(f); return -1; }
        (*rows)[i] = r-1; (*cols)[i] = c-1; (*vals)[i] = v;
    }
    fclose(f); return 0;
}

int read_vec(const char *path, int N, double **x) {
    FILE *f = fopen(path, "r");
    *x = malloc(sizeof(double)*N);
    if (!f) { for(int i=0;i<N;i++) (*x)[i]=1.0; return 0; }
    for (int i=0;i<N;i++) { if (fscanf(f, "%lf", &(*x)[i])!=1) { (*x)[i]=1.0; } }
    fclose(f); return 0;
}

int read_gold(const char *path, int M, double **ygold) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    *ygold = malloc(sizeof(double)*M);
    for (int i=0;i<M;i++) { if (fscanf(f, "%lf", &(*ygold)[i])!=1) { (*ygold)[i]=0.0;} }
    fclose(f); return 0;
}

int verify(int M, double *y, double *ygold) {
    double tol = 0.02;
    for (int i=0;i<M;i++) {
        double a = y[i], b = ygold[i];
        double diff = fabs(a-b);
        if (diff > tol) return 0;
    }
    return 1;
}

/* TODO: Student implements this naive COO-scan compute. For now it zeroes y and prints a note. */
void compute_not_csr(int M, int N, int nnz, int *rows, int *cols, double *vals, double *x, double *y) {
    /* STUDENT TASK: implement naive COO-scan that computes y = A*x
       Example pseudo: for each k in [0,nnz): y[rows[k]] += vals[k] * x[cols[k]];
    */
    fprintf(stderr, "[STUDENT] compute_not_csr() not implemented — fill this in.\n");
    for (int i=0;i<M;i++) y[i]=0.0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s matrix.mtx [vector.txt]\n", argv[0]); return 1; }
    const char *mtx = argv[1]; const char *vec = (argc>2?argv[2]:NULL);
    int M,N,nnz; int *rows=NULL,*cols=NULL; double *vals=NULL;
    if (read_mtx(mtx,&M,&N,&nnz,&rows,&cols,&vals)!=0) { fprintf(stderr,"Failed to read mtx\n"); return 1; }
    double *x=NULL; read_vec(vec,N,&x);
    double *y = malloc(sizeof(double)*M);
    double t0 = time_ms();
    compute_not_csr(M,N,nnz,rows,cols,vals,x,y);
    double t1 = time_ms();
    fprintf(stderr,"spmv_not_csr_time_ms=%.3f\n", t1-t0);

    /* verify against gold file alongside matrix (matrix.mtx.gold) */
    char goldpath[1024]; snprintf(goldpath,sizeof(goldpath),"%s.gold", mtx);
    double *ygold=NULL; if (read_gold(goldpath,M,&ygold)==0) {
        if (verify(M,y,ygold)) fprintf(stderr,"OK\n"); else fprintf(stderr,"WRONG\n");
        free(ygold);
    } else {
        fprintf(stderr,"No gold found (%s) — skipping verify\n", goldpath);
    }

    free(rows); free(cols); free(vals); free(x); free(y);
    return 0;
}
