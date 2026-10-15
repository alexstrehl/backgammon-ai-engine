/*
 * backend_shim.c -- test/bench entry points over a bare backend
 * (libbgt_<name>.so). Lets bench_forward.py and the forward parity
 * tests drive a backend without the core.
 */

#include "puretd_backend.h"

#include <stdlib.h>
#include <time.h>

typedef struct {
    bgc_backend *b;
    int in_size;
    int out_dim;
    long long params;
} bgt_handle;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Load a BGNN .bin and pack it into the backend. NULL on failure. */
void *bgt_load(const char *path) {
    NNModel m;
    if (nn_load(&m, path) != 0) return NULL;
    bgt_handle *h = calloc(1, sizeof(*h));
    if (!h) { nn_free(&m); return NULL; }
    h->b = bgc_backend_create(&m);
    h->in_size = m.input_size;
    h->out_dim = m.layer_out[m.num_hidden];
    for (int L = 0; L <= m.num_hidden; L++)
        h->params += (long long)m.layer_in[L] * m.layer_out[L] + m.layer_out[L];
    nn_free(&m);
    if (!h->b) { free(h); return NULL; }
    return h;
}

void bgt_free(void *hp) {
    bgt_handle *h = hp;
    if (!h) return;
    bgc_backend_destroy(h->b);
    free(h);
}

int bgt_in_size(const void *hp) { return ((const bgt_handle *)hp)->in_size; }
int bgt_out_dim(const void *hp) { return ((const bgt_handle *)hp)->out_dim; }
long long bgt_params(const void *hp) { return ((const bgt_handle *)hp)->params; }
const char *bgt_name(void) { return bgc_backend_name(); }
const char *bgt_describe(const void *hp) {
    return bgc_backend_describe(((const bgt_handle *)hp)->b);
}

void bgt_forward(void *hp, const float *x, int ldx, int n, float *out) {
    bgc_backend_forward(((bgt_handle *)hp)->b, x, ldx, n, out);
}

/* `reps` forwards of the same n-row batch, timed inside C. Returns the
 * total seconds (CLOCK_MONOTONIC). */
double bgt_time_forward(void *hp, const float *x, int ldx, int n, int reps,
                        float *out) {
    bgc_backend *b = ((bgt_handle *)hp)->b;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) bgc_backend_forward(b, x, ldx, n, out);
    return now_s() - t0;
}
