/*
 * nn_eval.h -- Neural network inference for backgammon models.
 *
 * Loads models exported by export_weights.py (BGNN binary format).
 * Pure C, no external dependencies beyond math.h.
 */

#ifndef NN_EVAL_H
#define NN_EVAL_H

#define NN_MAX_LAYERS 8

#define NN_ACTIVATION_RELU        0
#define NN_ACTIVATION_SIGMOID     1
#define NN_ACTIVATION_TANH        2
#define NN_ACTIVATION_LEAKY_RELU  3

typedef struct {
    int num_hidden;                     /* number of hidden layers */
    int input_size;
    int activation;
    int hidden_sizes[NN_MAX_LAYERS];

    /* Per-layer weights and biases (num_hidden + 1 for output layer) */
    float *weight[NN_MAX_LAYERS + 1];
    float *bias[NN_MAX_LAYERS + 1];
    int layer_in[NN_MAX_LAYERS + 1];
    int layer_out[NN_MAX_LAYERS + 1];

    /* Scratch buffers for forward pass (pre-allocated) */
    float *buf_a;
    float *buf_b;
} NNModel;

/*
 * Load model from binary file (.bin exported by export_weights.py).
 * Returns 0 on success, -1 on error.
 */
int nn_load(NNModel *model, const char *path);

/*
 * Free all allocated memory in the model.
 */
void nn_free(NNModel *model);

/*
 * Forward pass: input[input_size] -> returns scalar in [0,1].
 * For backgammon: output = P(on-roll player wins).
 */
float nn_forward(const NNModel *model, const float *input);

#endif /* NN_EVAL_H */
