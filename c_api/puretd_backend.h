/*
 * puretd_backend.h -- forward-pass backend contract for the PureTD C API.
 *
 * The core (puretd.c) does move generation, encoding, move selection,
 * cube logic, 1-ply search and the game loop. A backend does only the
 * net forward. One backend is linked per library variant:
 * libpuretd_<name>.so (full API) and libbgt_<name>.so (backend + test
 * shim, see backend_shim.c).
 *
 * Contract
 *   - bgc_backend_create copies or repacks the weights of a loaded BGNN
 *     model (c_inference/nn_eval.h). The NNModel may be freed afterwards.
 *   - bgc_backend_forward computes out[n * out_dim] = net(x) in fp32:
 *       hidden layers: model->activation (relu, sigmoid, tanh,
 *                      leaky_relu, hardsigmoid; formulas as nn_eval.c);
 *       output layer:  linear for NN_OUTPUT_EQUITY, sigmoid for
 *                      NN_OUTPUT_PROBABILITY and NN_OUTPUT_PROB5.
 *     No prob5 clamp or equity reduction: the core does that, so every
 *     backend feeds the same post-processing.
 *   - x rows are `ldx` floats apart, any ldx >= input_size. ldx may
 *     EQUAL input_size (then there are no pad columns). A backend reads
 *     only the first input_size columns of each row; one that wants a
 *     padded or aligned K packs rows into scratch sized at create.
 *     Rows are NOT guaranteed to be 32- or 64-byte aligned. The core
 *     passes ldx = input_size rounded up to 16 with zero pad columns
 *     and row 0 64-byte aligned; bench_forward.py tests both layouts.
 *   - out rows are contiguous, exactly out_dim floats apart (1, or 5
 *     for prob5). The core's prob5 clamp/reduce runs on them, so a
 *     backend must not reduce.
 *   - n >= 0; the backend chunks a large n internally.
 *   - Single-threaded. A backend never spawns or uses worker threads
 *     (openblas_set_num_threads(1), at::set_num_threads(1), ...).
 *   - After create, forward does no heap allocation.
 *   - Compile without -ffast-math: sigmoid's expf may overflow to inf,
 *     which -ffinite-math-only turns into NaN.
 */

#ifndef PURETD_BACKEND_H
#define PURETD_BACKEND_H

#include "nn_eval.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bgc_backend bgc_backend;   /* opaque, backend-defined */

/* Pack `m` into a backend instance. NULL on failure (message on stderr). */
bgc_backend *bgc_backend_create(const NNModel *m);

void bgc_backend_destroy(bgc_backend *b);

/* out[n * out_dim] = net(x), x row-major with row stride ldx. */
void bgc_backend_forward(bgc_backend *b, const float *x, int ldx, int n,
                         float *out);

/* forward of rows that are successors of one position: each differs from
 * `parent` (that position, turn switched, the same cube columns; row
 * stride as x) in a few input columns. A backend may compute layer 0 as
 * the parent's pre-activation plus the differing columns' terms (simd:
 * NNUE-style, unless BGC_SIMD_DELTA0=0); the result then equals
 * bgc_backend_forward up to layer 0's summation order, and a row's output
 * depends only on the row and the parent (so the row-exact reuse below
 * holds between calls with the same parent). The others run
 * bgc_backend_forward. */
void bgc_backend_forward_delta(bgc_backend *b, const float *parent, const float *x,
                               int ldx, int n, float *out);

#ifdef BGC_BF16
/* bf16 builds (CFLAGS_EXTRA=-DBGC_BF16, simd only; OFF by default): the
 * forward with bf16 weights and activations (VDPBF16PS), fp32 accumulation.
 * NOT within the parity bounds; with a parent (as forward_delta) layer 0
 * stays fp32 by deltas. bgc_backend_bf16: 1 when usable (the CPU
 * has avx512_bf16, relu net, BGC_BF16 != "0" at create); otherwise
 * bgc_backend_forward_bf16 is the fp32 forward. The core declares both weak:
 * other backends do not define them. */
int bgc_backend_bf16(const bgc_backend *b);
void bgc_backend_forward_bf16(bgc_backend *b, const float *parent /* nullable */,
                              const float *x, int ldx, int n, float *out);
#endif

/* 1 when a row's outputs do not depend on the other rows of the call nor
 * on n (bitwise): the core then reuses a row's value computed in one batch
 * for the same row in another (bgc_value_1ply_play). ref (per-row loops)
 * and simd (exact zero skipping, fixed k order per output) are; BLAS
 * and ATen kernels pick blockings by shape, so cblas and torch say 0. */
int bgc_backend_row_exact(void);

/* Short identifier: "ref", "simd", "cblas", "torch". */
const char *bgc_backend_name(void);

/* One line for benchmark headers, e.g. "simd avx512 4x16" or
 * "cblas OpenBLAS 0.3.28 1 thread". Static or instance-owned storage. */
const char *bgc_backend_describe(const bgc_backend *b);

#ifdef __cplusplus
}
#endif

#endif /* PURETD_BACKEND_H */
