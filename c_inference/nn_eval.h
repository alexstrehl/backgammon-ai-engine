/*
 * nn_eval.h -- Neural network inference for backgammon models.
 *
 * Sample code illustrating how the trained model can be run from pure C
 * for inclusion in other projects. No external dependencies beyond math.h.
 *
 * Loads models exported by export_weights.py (BGNN binary format).
 * See play_models.c for a complete usage example.
 */

#ifndef NN_EVAL_H
#define NN_EVAL_H

#define NN_MAX_LAYERS 8

#define NN_ACTIVATION_RELU        0
#define NN_ACTIVATION_SIGMOID     1
#define NN_ACTIVATION_TANH        2
#define NN_ACTIVATION_LEAKY_RELU  3
#define NN_ACTIVATION_HARDSIGMOID 4

#define NN_OUTPUT_PROBABILITY     0   /* sigmoid output in [0,1] */
#define NN_OUTPUT_EQUITY          1   /* linear output (unbounded) */
#define NN_OUTPUT_PROB5           2   /* 5 sigmoid outputs -> money equity */

#define NN_PROB5_OUTPUTS          5

typedef struct {
    int num_hidden;                     /* number of hidden layers */
    int input_size;
    int activation;
    int output_mode;                    /* 0=probability, 1=equity, 2=prob5 */
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
 * Forward pass: input[input_size] -> returns scalar.
 * For probability mode: output in [0,1] (sigmoid), P(on-roll player wins).
 * For equity mode: output is unbounded (linear), expected equity.
 * For prob5 mode: returns cubeless money equity derived from the 5
 * postprocessed sigmoid outputs (2*Pw + Pwg + Pwbg - Plg - Plbg - 1).
 */
float nn_forward(const NNModel *model, const float *input);

/*
 * prob5 only: forward pass writing the 5 postprocessed probabilities
 * (P(win), P(win gammon), P(win bg), P(lose gammon), P(lose bg)) into
 * probs[5] and returning the derived money equity. On a non-prob5
 * model, fills probs with zeros and returns nn_forward(model, input).
 */
float nn_forward_prob5(const NNModel *model, const float *input, float probs[NN_PROB5_OUTPUTS]);

#endif /* NN_EVAL_H */
