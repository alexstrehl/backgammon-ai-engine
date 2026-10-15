/*
 * backend_cblas.c -- OpenBLAS forward backend.
 *
 * Per layer: out[n x R] = x[n x C] . W^T via cblas_sgemm (W is [R x C]
 * row-major, so B = W with Trans), then a fused bias + activation pass.
 * Batches of at most GEMV_MAX rows go row by row through cblas_sgemv.
 * Rows are processed in chunks of CHUNK so scratch is fixed at create;
 * a chunk is padded with zero rows to a multiple of ROW_PAD, since
 * OpenBLAS sgemm is markedly slower on ragged row counts.
 * OpenBLAS is forced to one thread at create.
 */

#include "puretd_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CBLAS prototypes declared here so no cblas.h is needed (any LP64
 * CBLAS with standard symbol names links). */
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111
#define CBLAS_TRANS     112
extern void cblas_sgemm(int order, int ta, int tb, int m, int n, int k,
                        float alpha, const float *a, int lda,
                        const float *b, int ldb, float beta, float *c, int ldc);
extern void cblas_sgemv(int order, int ta, int m, int n, float alpha,
                        const float *a, int lda, const float *x, int incx,
                        float beta, float *y, int incy);

/* OpenBLAS extensions: weak, absent in other CBLAS libraries. */
extern void openblas_set_num_threads(int n) __attribute__((weak));
extern int openblas_get_num_threads(void) __attribute__((weak));
extern char *openblas_get_config(void) __attribute__((weak));
extern char *openblas_get_corename(void) __attribute__((weak));

#define GEMV_MAX 6              /* n <= this: per-row sgemv */
#define CHUNK 512               /* rows per sgemm pass, a multiple of ROW_PAD */
#define ROW_PAD 4

struct bgc_backend {
    int num_hidden;
    int activation;
    int output_mode;
    int out_dim;
    int layer_in[NN_MAX_LAYERS + 1];
    int layer_out[NN_MAX_LAYERS + 1];
    float *weight[NN_MAX_LAYERS + 1];   /* [out x in] row-major, 64-byte aligned */
    float *bias[NN_MAX_LAYERS + 1];
    int input_size;
    float *xpad;                        /* CHUNK x input_size: padded input rows */
    float *buf_a, *buf_b;               /* CHUNK rows of the widest layer */
    char desc[160];
};

static float *alloc_f(size_t n) {
    size_t bytes = (n * sizeof(float) + 63) & ~(size_t)63;
    return aligned_alloc(64, bytes ? bytes : 64);
}

/* Each of nrows rows of y (rows floats): y = act(y + bias). kind -1 linear,
 * -2 output sigmoid, else the hidden activation. */
static void epilogue(int kind, int rows, int nrows, const float *restrict bias,
                     float *restrict y) {
    for (int s = 0; s < nrows; s++, y += rows) {
        switch (kind) {
        case -1:
            for (int i = 0; i < rows; i++) y[i] += bias[i];
            break;
        case -2:
        case NN_ACTIVATION_SIGMOID:
            for (int i = 0; i < rows; i++) y[i] = 1.0f / (1.0f + expf(-(y[i] + bias[i])));
            break;
        case NN_ACTIVATION_TANH:
            for (int i = 0; i < rows; i++) y[i] = tanhf(y[i] + bias[i]);
            break;
        case NN_ACTIVATION_LEAKY_RELU:
            for (int i = 0; i < rows; i++) {
                float v = y[i] + bias[i];
                y[i] = v > 0.0f ? v : 0.01f * v;
            }
            break;
        case NN_ACTIVATION_HARDSIGMOID:
            for (int i = 0; i < rows; i++) {
                float v = y[i] + bias[i];
                y[i] = v <= -3.0f ? 0.0f : v >= 3.0f ? 1.0f : v / 6.0f + 0.5f;
            }
            break;
        default:   /* relu */
            for (int i = 0; i < rows; i++) {
                float v = y[i] + bias[i];
                y[i] = v > 0.0f ? v : 0.0f;
            }
            break;
        }
    }
}

bgc_backend *bgc_backend_create(const NNModel *m) {
    bgc_backend *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->num_hidden = m->num_hidden;
    b->activation = m->activation;
    b->output_mode = m->output_mode;
    const int total = m->num_hidden + 1;
    int max_dim = 1;
    for (int L = 0; L < total; L++) {
        const int rows = m->layer_out[L], cols = m->layer_in[L];
        b->layer_in[L] = cols;
        b->layer_out[L] = rows;
        if (rows > max_dim) max_dim = rows;
        b->weight[L] = alloc_f((size_t)rows * cols);
        b->bias[L] = alloc_f((size_t)rows);
        if (!b->weight[L] || !b->bias[L]) goto fail;
        memcpy(b->weight[L], m->weight[L], (size_t)rows * cols * sizeof(float));
        memcpy(b->bias[L], m->bias[L], (size_t)rows * sizeof(float));
    }
    b->out_dim = b->layer_out[total - 1];
    b->input_size = m->input_size;
    b->xpad = alloc_f((size_t)CHUNK * m->input_size);
    b->buf_a = alloc_f((size_t)CHUNK * max_dim);
    b->buf_b = alloc_f((size_t)CHUNK * max_dim);
    if (!b->xpad || !b->buf_a || !b->buf_b) goto fail;

    /* One thread; a library already at one (or without the call) is left as is. */
    if (openblas_set_num_threads &&
        !(openblas_get_num_threads && openblas_get_num_threads() == 1))
        openblas_set_num_threads(1);

    const char *cfg = openblas_get_config ? openblas_get_config() : "CBLAS";
    const char *core = openblas_get_corename ? openblas_get_corename() : NULL;
    int threads = openblas_get_num_threads ? openblas_get_num_threads() : 1;
    snprintf(b->desc, sizeof b->desc, "cblas %s%s%s, %d thread%s", cfg,
             core ? " core=" : "", core ? core : "", threads, threads == 1 ? "" : "s");
    return b;
fail:
    fprintf(stderr, "backend_cblas: out of memory\n");
    bgc_backend_destroy(b);
    return NULL;
}

void bgc_backend_destroy(bgc_backend *b) {
    if (!b) return;
    for (int L = 0; L <= b->num_hidden; L++) { free(b->weight[L]); free(b->bias[L]); }
    free(b->xpad);
    free(b->buf_a);
    free(b->buf_b);
    free(b);
}

static inline int layer_kind(const bgc_backend *b, int L) {
    if (L < b->num_hidden) return b->activation;
    return b->output_mode == NN_OUTPUT_EQUITY ? -1 : -2;
}

/* n rows (n <= CHUNK) through all layers into out[n x out_dim]. */
static void forward_gemm(bgc_backend *b, const float *x, int ldx, int n, float *out) {
    const int total = b->num_hidden + 1;
    const int np = (n + ROW_PAD - 1) / ROW_PAD * ROW_PAD;
    const float *in = x;
    int ld_in = ldx;
    if (np != n) {
        const int C = b->input_size;
        for (int s = 0; s < n; s++)
            memcpy(b->xpad + (size_t)s * C, x + (size_t)s * ldx, (size_t)C * sizeof(float));
        memset(b->xpad + (size_t)n * C, 0, (size_t)(np - n) * C * sizeof(float));
        in = b->xpad;
        ld_in = C;
    }
    float *o = b->buf_a;
    for (int L = 0; L < total; L++) {
        const int R = b->layer_out[L], C = b->layer_in[L];
        float *dst = (L == total - 1 && np == n) ? out : o;
        cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS, np, R, C, 1.0f,
                    in, ld_in, b->weight[L], C, 0.0f, dst, R);
        epilogue(layer_kind(b, L), R, L == total - 1 ? n : np, b->bias[L], dst);
        in = dst;
        ld_in = R;
        o = (o == b->buf_a) ? b->buf_b : b->buf_a;
    }
    if (np != n) memcpy(out, in, (size_t)n * b->out_dim * sizeof(float));
}

/* One row through all layers via sgemv. */
static void forward_gemv(bgc_backend *b, const float *x, float *out) {
    const int total = b->num_hidden + 1;
    const float *in = x;
    float *o = b->buf_a;
    for (int L = 0; L < total; L++) {
        const int R = b->layer_out[L], C = b->layer_in[L];
        float *dst = (L == total - 1) ? out : o;
        cblas_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, R, C, 1.0f, b->weight[L], C,
                    in, 1, 0.0f, dst, 1);
        epilogue(layer_kind(b, L), R, 1, b->bias[L], dst);
        in = dst;
        o = (o == b->buf_a) ? b->buf_b : b->buf_a;
    }
}

void bgc_backend_forward(bgc_backend *b, const float *x, int ldx, int n,
                         float *out) {
    if (n <= GEMV_MAX) {
        for (int s = 0; s < n; s++)
            forward_gemv(b, x + (size_t)s * ldx, out + (size_t)s * b->out_dim);
        return;
    }
    for (int s = 0; s < n; s += CHUNK) {
        const int k = n - s < CHUNK ? n - s : CHUNK;
        forward_gemm(b, x + (size_t)s * ldx, ldx, k, out + (size_t)s * b->out_dim);
    }
}

const char *bgc_backend_name(void) { return "cblas"; }

int bgc_backend_row_exact(void) { return 0; }

void bgc_backend_forward_delta(bgc_backend *b, const float *parent, const float *x, int ldx,
                               int n, float *out) {
    (void)parent;
    bgc_backend_forward(b, x, ldx, n, out);
}

const char *bgc_backend_describe(const bgc_backend *b) { return b->desc; }
