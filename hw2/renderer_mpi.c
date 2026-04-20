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

static void rasterize_circle(float *img, unsigned char *mask, int W, int H,
                             float cx, float cy, float radius, unsigned char rgb[3]) {
    int xmin = (int)floorf(cx - radius);
    int xmax = (int)floorf(cx + radius);
    int ymin = (int)floorf(cy - radius);
    int ymax = (int)floorf(cy + radius);
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax >= W) xmax = W - 1;
    if (ymax >= H) ymax = H - 1;

    float Cr = rgb[0] / 255.0f;
    float Cg = rgb[1] / 255.0f;
    float Cb = rgb[2] / 255.0f;
    float r2 = radius * radius;

    for (int y = ymin; y <= ymax; ++y) {
        float py = (float)y + 0.5f;
        float dy = py - cy;
        for (int x = xmin; x <= xmax; ++x) {
            float px = (float)x + 0.5f;
            float dx = px - cx;
            if (dx * dx + dy * dy <= r2) {
                size_t p = (size_t)y * W + x;
                size_t idx = p * 3;
                img[idx + 0] = Cr;
                img[idx + 1] = Cg;
                img[idx + 2] = Cb;
                mask[p] = 1;
            }
        }
    }
}

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

    MPI_Bcast(&version, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(bbox, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast header and scatter records as raw bytes with contiguous ranges per rank.
    int *sendcounts = malloc(sizeof(int) * nprocs);
    int *displs = malloc(sizeof(int) * nprocs);
    for (int i = 0; i < nprocs; ++i) { sendcounts[i] = 0; displs[i] = 0; }

    if (!sendcounts || !displs) { perror("alloc sendcounts/displs"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < nprocs; ++i) {
        uint64_t recs_i = count / (uint64_t)nprocs + ((uint64_t)i < (count % (uint64_t)nprocs) ? 1u : 0u);
        uint64_t bytes_i = recs_i * (uint64_t)RECSZ;
        if (bytes_i > (uint64_t)INT32_MAX) {
            if (rank == 0) fprintf(stderr, "record chunk too large for MPI_Scatterv int count\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sendcounts[i] = (int)bytes_i;
        if (i > 0) displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    int mybytes = sendcounts[rank];
    unsigned char *mybuf = NULL;
    if (mybytes > 0) {
        mybuf = malloc((size_t)mybytes);
        if (!mybuf) { perror("alloc mybuf"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Scatterv(all_records, sendcounts, displs, MPI_BYTE,
                 mybuf, mybytes, MPI_BYTE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(all_records);
        all_records = NULL;
    }

    // allocate local image buffer
    size_t npix = (size_t)W * H;
    float *img = calloc(npix * 3, sizeof(float));
    if (!img) { perror("alloc img"); MPI_Abort(MPI_COMM_WORLD, 1); }
    unsigned char *mask = calloc(npix, 1);
    if (!mask) { perror("alloc mask"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // Parse local raw records and rasterize in local draw order.
    int myrecs = mybytes / RECSZ;
    for (int i = 0; i < myrecs; ++i) {
        const unsigned char *rec = mybuf + (size_t)i * RECSZ;
        float cx, cy, radius;
        unsigned char rgb[3];
        memcpy(&cx, rec + 0, sizeof(float));
        memcpy(&cy, rec + sizeof(float), sizeof(float));
        memcpy(&radius, rec + sizeof(float) * 2, sizeof(float));
        rgb[0] = rec[sizeof(float) * 3 + 0];
        rgb[1] = rec[sizeof(float) * 3 + 1];
        rgb[2] = rec[sizeof(float) * 3 + 2];

        rasterize_circle(img, mask, W, H, cx, cy, radius, rgb);
    }
    free(mybuf);

    float *all_imgs = NULL;
    unsigned char *all_masks = NULL;
    float *acc_img = NULL;
    if (rank == 0) {
        all_imgs = malloc((size_t)nprocs * npix * 3 * sizeof(float));
        all_masks = malloc((size_t)nprocs * npix);
        acc_img = calloc(npix * 3, sizeof(float));
        if (!all_imgs || !all_masks || !acc_img) { perror("alloc gather buffers"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Gather(img, (int)(npix * 3), MPI_FLOAT,
               all_imgs, (int)(npix * 3), MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(mask, (int)npix, MPI_UNSIGNED_CHAR,
               all_masks, (int)npix, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Composite in rank order because rank chunks preserve global draw order.
        for (int r = 0; r < nprocs; ++r) {
            const float *rimg = all_imgs + (size_t)r * npix * 3;
            const unsigned char *rmask = all_masks + (size_t)r * npix;
            for (size_t p = 0; p < npix; ++p) {
                if (rmask[p]) {
                    size_t idx = p * 3;
                    acc_img[idx + 0] = rimg[idx + 0];
                    acc_img[idx + 1] = rimg[idx + 1];
                    acc_img[idx + 2] = rimg[idx + 2];
                }
            }
        }
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
        free(all_imgs);
        free(all_masks);
        free(acc_img);
        fprintf(stderr, "rank0: Wrote PNG %s\n", outpath);
    }

    free(img);
    free(mask);
    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}
