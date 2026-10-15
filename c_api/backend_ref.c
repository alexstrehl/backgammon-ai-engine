/*
 * backend_ref.c -- scalar reference backend (no SIMD, no libraries).
 *
 * The parity anchor for the other backends and the portable fallback.
 * Plain per-row dot products in the order j = 0..cols-1.
 */

#include "puretd_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct bgc_backend {
    int num_hidden;
    int activation;
    int output_mode;
    int out_dim;
    int layer_in[NN_MAX_LAYERS + 1];
    int layer_out[NN_MAX_LAYERS + 1];
    float *weight[NN_MAX_LAYERS + 1];   /* [out x in] row-major */
    float *bias[NN_MAX_LAYERS + 1];
    float *buf_a, *buf_b;               /* one row of the widest layer */
};

static inline float act(int kind, float x) {
    switch (kind) {
        case NN_ACTIVATION_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case NN_ACTIVATION_TANH:        return tanhf(x);
        case NN_ACTIVATION_LEAKY_RELU:  return x > 0.0f ? x : 0.01f * x;
        case NN_ACTIVATION_HARDSIGMOID:
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f)  return 1.0f;
            return x / 6.0f + 0.5f;
        case NN_ACTIVATION_RELU:
        default:                        return x > 0.0f ? x : 0.0f;
    }
}

bgc_backend *bgc_backend_create(const NNModel *m) {
    bgc_backend *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->num_hidden = m->num_hidden;
    b->activation = m->activation;
    b->output_mode = m->output_mode;
    int total = m->num_hidden + 1;
    int max_dim = m->input_size;
    for (int L = 0; L < total; L++) {
        int rows = m->layer_out[L], cols = m->layer_in[L];
        b->layer_in[L] = cols;
        b->layer_out[L] = rows;
        if (rows > max_dim) max_dim = rows;
        b->weight[L] = malloc((size_t)rows * cols * sizeof(float));
        b->bias[L] = malloc((size_t)rows * sizeof(float));
        if (!b->weight[L] || !b->bias[L]) { bgc_backend_destroy(b); return NULL; }
        memcpy(b->weight[L], m->weight[L], (size_t)rows * cols * sizeof(float));
        memcpy(b->bias[L], m->bias[L], (size_t)rows * sizeof(float));
    }
    b->out_dim = b->layer_out[total - 1];
    b->buf_a = malloc((size_t)max_dim * sizeof(float));
    b->buf_b = malloc((size_t)max_dim * sizeof(float));
    if (!b->buf_a || !b->buf_b) { bgc_backend_destroy(b); return NULL; }
    return b;
}

void bgc_backend_destroy(bgc_backend *b) {
    if (!b) return;
    for (int L = 0; L <= b->num_hidden; L++) { free(b->weight[L]); free(b->bias[L]); }
    free(b->buf_a);
    free(b->buf_b);
    free(b);
}

void bgc_backend_forward(bgc_backend *b, const float *x, int ldx, int n,
                         float *out) {
    const int total = b->num_hidden + 1;
    const int equity = (b->output_mode == NN_OUTPUT_EQUITY);
    for (int s = 0; s < n; s++) {
        const float *in = x + (size_t)s * ldx;
        float *o = b->buf_a;
        for (int L = 0; L < total; L++) {
            const int rows = b->layer_out[L], cols = b->layer_in[L];
            const float *W = b->weight[L];
            const float *bias = b->bias[L];
            const int hidden = (L < b->num_hidden);
            for (int i = 0; i < rows; i++) {
                const float *w = W + (size_t)i * cols;
                float sum = bias[i];
                for (int j = 0; j < cols; j++) sum += w[j] * in[j];
                if (hidden)      o[i] = act(b->activation, sum);
                else if (equity) o[i] = sum;
                else             o[i] = 1.0f / (1.0f + expf(-sum));
            }
            in = o;
            o = (o == b->buf_a) ? b->buf_b : b->buf_a;
        }
        memcpy(out + (size_t)s * b->out_dim, in, (size_t)b->out_dim * sizeof(float));
    }
}

const char *bgc_backend_name(void) { return "ref"; }

int bgc_backend_row_exact(void) { return 1; }

void bgc_backend_forward_delta(bgc_backend *b, const float *parent, const float *x, int ldx,
                               int n, float *out) {
    (void)parent;
    bgc_backend_forward(b, x, ldx, n, out);
}

const char *bgc_backend_describe(const bgc_backend *b) {
    (void)b;
    return "ref scalar";
}
