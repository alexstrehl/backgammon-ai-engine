/*
 * nn_eval.c -- Neural network inference for backgammon models.
 *
 * Loads BGNN binary format, runs forward pass on CPU.
 * No external dependencies beyond stdlib and math.
 */

#include "nn_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Activation functions ──────────────────────────────────────── */

static inline float act_relu(float x)        { return x > 0.0f ? x : 0.0f; }
static inline float act_sigmoid(float x)     { return 1.0f / (1.0f + expf(-x)); }
static inline float act_tanh_(float x)       { return tanhf(x); }
static inline float act_leaky_relu(float x)  { return x > 0.0f ? x : 0.01f * x; }

typedef float (*act_fn)(float);

static act_fn get_activation(int act) {
    switch (act) {
        case NN_ACTIVATION_RELU:        return act_relu;
        case NN_ACTIVATION_SIGMOID:     return act_sigmoid;
        case NN_ACTIVATION_TANH:        return act_tanh_;
        case NN_ACTIVATION_LEAKY_RELU:  return act_leaky_relu;
        default:                        return act_relu;
    }
}

/* ── Load ──────────────────────────────────────────────────────── */

int nn_load(NNModel *model, const char *path) {
    memset(model, 0, sizeof(*model));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nn_load: cannot open '%s'\n", path);
        return -1;
    }

    /* Magic */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BGNN", 4) != 0) {
        fprintf(stderr, "nn_load: bad magic in '%s'\n", path);
        fclose(f);
        return -1;
    }

    /* Header */
    if (fread(&model->num_hidden, 4, 1, f) != 1 ||
        fread(&model->input_size, 4, 1, f) != 1 ||
        fread(&model->activation, 4, 1, f) != 1) {
        fprintf(stderr, "nn_load: truncated header\n");
        fclose(f);
        return -1;
    }

    if (model->num_hidden < 1 || model->num_hidden > NN_MAX_LAYERS) {
        fprintf(stderr, "nn_load: bad num_hidden=%d\n", model->num_hidden);
        fclose(f);
        return -1;
    }

    for (int i = 0; i < model->num_hidden; i++) {
        if (fread(&model->hidden_sizes[i], 4, 1, f) != 1) {
            fprintf(stderr, "nn_load: truncated hidden sizes\n");
            fclose(f);
            return -1;
        }
    }

    /* Set up layer dimensions */
    int total_layers = model->num_hidden + 1;  /* hidden + output */
    int prev = model->input_size;
    int max_dim = model->input_size;

    for (int i = 0; i < model->num_hidden; i++) {
        model->layer_in[i] = prev;
        model->layer_out[i] = model->hidden_sizes[i];
        prev = model->hidden_sizes[i];
        if (prev > max_dim) max_dim = prev;
    }
    /* Output layer: 1 neuron */
    model->layer_in[model->num_hidden] = prev;
    model->layer_out[model->num_hidden] = 1;

    /* Allocate and read weights/biases */
    for (int i = 0; i < total_layers; i++) {
        int rows = model->layer_out[i];
        int cols = model->layer_in[i];

        model->weight[i] = (float *)malloc((size_t)rows * cols * sizeof(float));
        model->bias[i] = (float *)malloc((size_t)rows * sizeof(float));

        if (!model->weight[i] || !model->bias[i]) {
            fprintf(stderr, "nn_load: allocation failed for layer %d\n", i);
            fclose(f);
            nn_free(model);
            return -1;
        }

        if (fread(model->weight[i], sizeof(float), (size_t)rows * cols, f)
                != (size_t)rows * cols ||
            fread(model->bias[i], sizeof(float), rows, f) != (size_t)rows) {
            fprintf(stderr, "nn_load: truncated weights at layer %d\n", i);
            fclose(f);
            nn_free(model);
            return -1;
        }
    }

    fclose(f);

    /* Scratch buffers */
    model->buf_a = (float *)malloc((size_t)max_dim * sizeof(float));
    model->buf_b = (float *)malloc((size_t)max_dim * sizeof(float));
    if (!model->buf_a || !model->buf_b) {
        fprintf(stderr, "nn_load: scratch buffer allocation failed\n");
        nn_free(model);
        return -1;
    }

    return 0;
}

/* ── Free ──────────────────────────────────────────────────────── */

void nn_free(NNModel *model) {
    int total_layers = model->num_hidden + 1;
    for (int i = 0; i < total_layers; i++) {
        free(model->weight[i]);
        free(model->bias[i]);
        model->weight[i] = NULL;
        model->bias[i] = NULL;
    }
    free(model->buf_a);
    free(model->buf_b);
    model->buf_a = NULL;
    model->buf_b = NULL;
}

/* ── Forward pass ──────────────────────────────────────────────── */

float nn_forward(const NNModel *model, const float *input) {
    act_fn activate = get_activation(model->activation);
    const float *in = input;
    float *out = model->buf_a;
    int total_layers = model->num_hidden + 1;

    for (int L = 0; L < total_layers; L++) {
        int rows = model->layer_out[L];
        int cols = model->layer_in[L];
        const float *W = model->weight[L];
        const float *b = model->bias[L];

        for (int i = 0; i < rows; i++) {
            float sum = b[i];
            const float *w_row = W + i * cols;
            for (int j = 0; j < cols; j++) {
                sum += w_row[j] * in[j];
            }
            if (L < model->num_hidden) {
                out[i] = activate(sum);
            } else {
                /* Output layer: always sigmoid */
                out[i] = act_sigmoid(sum);
            }
        }

        in = out;
        out = (out == model->buf_a) ? model->buf_b : model->buf_a;
    }

    return in[0];
}
