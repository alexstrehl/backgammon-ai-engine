/*
 * backend_torch.cpp -- libtorch/ATen forward backend (MKL sgemm via addmm).
 *
 * ATen ops only: no torch::nn, no TorchScript, no autograd. Weights are
 * held pre-transposed [in, out] contiguous fp32; each layer is one
 * addmm_out into a preallocated [BGC_TORCH_CHUNK, out] buffer, the last
 * layer writes straight into the caller's `out`. Single-threaded: create
 * sets ATen/MKL/OpenMP intra-op threads to 1 (and inter-op to 1 when the
 * process has not fixed it yet).
 *
 * ATen still allocates TensorImpl headers (from_blob/narrow views) per
 * call; tensor data is never allocated after create.
 */

#include "puretd_backend.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

/* Rows per addmm; larger n is chunked. */
#define BGC_TORCH_CHUNK 4096

struct bgc_backend {
    int num_layers;                         /* hidden + output */
    int input_size;
    int activation;
    int output_mode;
    int out_dim;
    at::Tensor wt[NN_MAX_LAYERS + 1];       /* [in, out] contiguous */
    at::Tensor bias[NN_MAX_LAYERS + 1];     /* [out] */
    at::Tensor buf[NN_MAX_LAYERS];          /* hidden outputs [CHUNK, out] */
    at::Tensor view[NN_MAX_LAYERS];         /* buf[L][:view_rows], cached */
    int view_rows;
    std::string desc;
};

static void hidden_act(int kind, at::Tensor &h) {
    switch (kind) {
        case NN_ACTIVATION_SIGMOID:     at::sigmoid_(h); break;
        case NN_ACTIVATION_TANH:        at::tanh_(h); break;
        case NN_ACTIVATION_LEAKY_RELU:  at::leaky_relu_(h, 0.01); break;
        case NN_ACTIVATION_HARDSIGMOID: at::hardsigmoid_(h); break;
        default:                        at::relu_(h); break;
    }
}

extern "C" bgc_backend *bgc_backend_create(const NNModel *m) {
    bgc_backend *b = nullptr;
    try {
        at::set_num_threads(1);
        try { at::set_num_interop_threads(1); } catch (const std::exception &) {}
        c10::InferenceMode guard;
        b = new bgc_backend();
        b->num_layers = m->num_hidden + 1;
        b->input_size = m->input_size;
        b->activation = m->activation;
        b->output_mode = m->output_mode;
        const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        for (int L = 0; L < b->num_layers; L++) {
            const int rows = m->layer_out[L], cols = m->layer_in[L];
            /* nn.Linear [out, in] -> owned [in, out]; copy_ always copies
               (.contiguous() would alias m's buffer when out == 1). */
            b->wt[L] = at::empty({cols, rows}, f32)
                           .copy_(at::from_blob(m->weight[L], {rows, cols}, f32).t());
            b->bias[L] = at::from_blob(m->bias[L], {rows}, f32).clone();
            if (L < m->num_hidden) b->buf[L] = at::empty({BGC_TORCH_CHUNK, rows}, f32);
        }
        b->out_dim = m->layer_out[m->num_hidden];
        b->view_rows = -1;
        b->desc = std::string("torch ") + TORCH_VERSION + " mkl=" +
                  (at::hasMKL() ? "yes" : "no") +
                  " threads=" + std::to_string(at::get_num_threads());
        return b;
    } catch (const std::exception &e) {
        fprintf(stderr, "backend_torch: create failed: %s\n", e.what());
        delete b;
        return nullptr;
    }
}

extern "C" void bgc_backend_destroy(bgc_backend *b) { delete b; }

extern "C" void bgc_backend_forward(bgc_backend *b, const float *x, int ldx, int n,
                                    float *out) {
    try {
        c10::InferenceMode guard;
        const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const int last = b->num_layers - 1;
        const bool equity = (b->output_mode == NN_OUTPUT_EQUITY);
        for (int r0 = 0; r0 < n; r0 += BGC_TORCH_CHUNK) {
            const int m = n - r0 < BGC_TORCH_CHUNK ? n - r0 : BGC_TORCH_CHUNK;
            if (m != b->view_rows) {
                for (int L = 0; L < last; L++) b->view[L] = b->buf[L].narrow(0, 0, m);
                b->view_rows = m;
            }
            at::Tensor h = at::from_blob(const_cast<float *>(x) + (size_t)r0 * ldx,
                                         {m, b->input_size}, {ldx, 1}, f32);
            for (int L = 0; L < last; L++) {
                at::addmm_out(b->view[L], b->bias[L], h, b->wt[L]);
                hidden_act(b->activation, b->view[L]);
                h = b->view[L];
            }
            at::Tensor o = at::from_blob(out + (size_t)r0 * b->out_dim,
                                         {m, b->out_dim}, f32);
            at::addmm_out(o, b->bias[last], h, b->wt[last]);
            if (!equity) at::sigmoid_(o);
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "backend_torch: forward failed: %s\n", e.what());
        for (size_t i = 0; i < (size_t)n * b->out_dim; i++) out[i] = NAN;
    }
}

extern "C" const char *bgc_backend_name(void) { return "torch"; }

extern "C" int bgc_backend_row_exact(void) { return 0; }

extern "C" void bgc_backend_forward_delta(bgc_backend *b, const float *parent, const float *x,
                                          int ldx, int n, float *out) {
    (void)parent;
    bgc_backend_forward(b, x, ldx, n, out);
}

extern "C" const char *bgc_backend_describe(const bgc_backend *b) {
    return b->desc.c_str();
}
