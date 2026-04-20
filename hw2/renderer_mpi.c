/*
 * renderer_mpi.c
 * MPI-parallel renderer: root reads records and scatters them to ranks;
 * each rank renders its chunk into an RGB buffer (alpha removed, colors opaque);
 * root receives per-rank RGB buffers in rank order and composites by overwriting
 * to reproduce serial ordering. Per-record format is: float32 x,y,radius + uint8 r,g,b.
 * Usage: mpirun -n <procs> ./renderer_mpi <input.bin> <output.png>
 */

#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RECSZ (sizeof(float)*3 + 3)

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input.bin> <output.png>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    uint32_t version = 0;
    uint64_t count = 0;
    float bbox[6] = {0};
    int W = 640, H = 480;
    /* overall_start preserved for optional wall-time measurement (root only) */
    double overall_start = 0.0;

    unsigned char *all_records = NULL;

    if (rank == 0) {
        /* record overall start just before opening/reading the input */
        overall_start = MPI_Wtime();
        FILE *f = fopen(inpath, "rb");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        char magic[5] = {0};
        if (fread(magic, 1, 4, f) != 4) { fprintf(stderr, "failed read magic\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (strncmp(magic, "CRDR", 4) != 0) { fprintf(stderr, "bad magic: %.4s\n", magic); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&version, sizeof(version), 1, f) != 1) { fprintf(stderr, "failed read version\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&count, sizeof(count), 1, f) != 1) { fprintf(stderr, "failed read count\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(bbox, sizeof(float), 6, f) != 6) { fprintf(stderr, "failed read bbox\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }

        W = (int)roundf(bbox[3] - bbox[0]);
        H = (int)roundf(bbox[4] - bbox[1]);
        if (W <= 0) W = 640;
        if (H <= 0) H = 480;

        fprintf(stderr, "rank0: magic=CRDR version=%u count=%llu image %dx%d\n", version, (unsigned long long)count, W, H);

        size_t totsz = (size_t)count * RECSZ;
        all_records = malloc(totsz);
        if (!all_records) { perror("malloc all_records"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(all_records, 1, totsz, f) != totsz) { fprintf(stderr, "failed read records\n"); free(all_records); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        fclose(f);
    }

    // STUDENT TODO: Broadcast header and scatter records (MPI_Scatterv).
    // Decide which fields to broadcast and how to partition raw bytes.
    int *sendcounts = malloc(sizeof(int) * nprocs);
    int *displs = malloc(sizeof(int) * nprocs);
    for (int i = 0; i < nprocs; ++i) { sendcounts[i] = 0; displs[i] = 0; }

    if (rank == 0) {
        free(all_records);
        all_records = NULL;
    }

    // allocate local image buffer
    size_t npix = (size_t)W * H;
    float *img = calloc(npix * 3, sizeof(float));
    if (!img) { perror("alloc img"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // STUDENT TODO: Receive raw-record bytes into `mybuf` and rasterize into `img`.
    // Implement per-record parsing and per-pixel writes here.

    // STUDENT TODO: Gather per-rank images and composite into `acc_img` on root.
    // Ensure ordering and overwrite semantics match the serial renderer.
    float *acc_img = calloc(npix * 3, sizeof(float));
    if (!acc_img) { perror("alloc acc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* STUDENT TODO: Implement proper gather/composite.
     */
    memcpy(acc_img, img, npix * 3 * sizeof(float));

    /* STUDENT TODO: Compose final image.
     */
    if (rank == 0) {

    }

    // Provided converter (kept for convenience)
    if (rank == 0) {
        size_t stride = (size_t)W * 3;
        unsigned char *pixels = malloc((size_t)H * stride);
        if (!pixels) { perror("malloc pixels"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t p = (size_t)y * W + x;
                size_t idx = p * 3;
                float fr = acc_img[idx + 0];
                float fg = acc_img[idx + 1];
                float fb = acc_img[idx + 2];
                int ir = (int)roundf(fmaxf(0.0f, fminf(1.0f, fr)) * 255.0f);
                int ig = (int)roundf(fmaxf(0.0f, fminf(1.0f, fg)) * 255.0f);
                int ib = (int)roundf(fmaxf(0.0f, fminf(1.0f, fb)) * 255.0f);
                pixels[(size_t)y * stride + x*3 + 0] = (unsigned char)ir;
                pixels[(size_t)y * stride + x*3 + 1] = (unsigned char)ig;
                pixels[(size_t)y * stride + x*3 + 2] = (unsigned char)ib;
            }
        }
        if (!stbi_write_png(outpath, W, H, 3, pixels, (int)stride)) {
            fprintf(stderr, "stbi_write_png failed\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* report overall wall time (root only) */
        {
            double overall_end = MPI_Wtime();
            double overall_elapsed = overall_end - overall_start;
            fprintf(stderr, "rank0: Total wall time (before read -> after write): %.6f s\n", overall_elapsed);
        }
        free(pixels);
        free(acc_img);
        fprintf(stderr, "rank0: Wrote PNG %s\n", outpath);
    }

    free(img);
    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}
