/*
 * nn_eval.h -- Neural-net inference for backgammon models in plain C
 * (no dependencies beyond math.h), written to be reusable elsewhere.
 *
 * Loads models exported by export_weights.py (BGNN binary format; the
 * layout is documented there).
 *
 * Two header formats:
 *   format 2, magic "BGNV" + int32 version 2 + the 64-byte encoder name,
 *     then the format-1 fields;
 *   format 1, magic "BGNN" (LEGACY: no encoder name; encoder_name
 *     reads "").
 * A format-1-only reader rejects a format-2 file at its magic ("bad
 * magic"), so a format-2 export is never misread. nn_load rejects an
 * unknown "BGNV" version.
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

#define NN_FORMAT_LEGACY          1   /* "BGNN": no encoder name           */
#define NN_FORMAT_VERSION         2   /* "BGNV" version 2: encoder name    */
#define NN_ENCODER_NAME_MAX       64  /* the header's name field, bytes,
                                         NUL padded (<= 63 characters)    */

typedef struct {
    int format_version;                 /* NN_FORMAT_LEGACY or NN_FORMAT_VERSION */
    char encoder_name[NN_ENCODER_NAME_MAX];  /* the checkpoint's encoder_name
                                           (encoding.get_encoder name); ""
                                           in a format-1 file */
    int num_hidden;                     /* number of hidden layers */
    int input_size;
    int activation;
    int output_mode;                    /* 0=probability, 1=equity, 2=prob5 */
    int prob5_board_sanity;             /* prob5: 1 = gnubg SanityCheck zeroing
                                           (Prob5Jan); nn_load sets 0 (plain prob5) */
    int hidden_sizes[NN_MAX_LAYERS];

    /* Per-layer weights and biases (num_hidden + 1 for output layer) */
    float *weight[NN_MAX_LAYERS + 1];
    float *bias[NN_MAX_LAYERS + 1];
    int layer_in[NN_MAX_LAYERS + 1];
    int layer_out[NN_MAX_LAYERS + 1];

    /* Scratch buffers for the forward pass */
    float *buf_a;
    float *buf_b;
} NNModel;

/* Load a .bin from export_weights.py (format 1 or 2). Returns 0 on
 * success, -1 on error (message on stderr). The encoder name is read,
 * not interpreted (c_api/puretd.c resolves it). */
int nn_load(NNModel *model, const char *path);

/* Free the model's allocations. */
void nn_free(NNModel *model);

/*
 * Forward pass: input[input_size] -> scalar.
 * probability mode: P(on-roll player wins), sigmoid in [0,1].
 * equity mode: expected equity, linear.
 * prob5 mode: money equity from the 5 sigmoid outputs after the nesting
 * clamps (2*Pw + Pwg + Pwbg - Plg - Plbg - 1); with prob5_board_sanity
 * (input must be perspective196), gnubg's SanityCheck "impossible"
 * zeroing first.
 */
float nn_forward(const NNModel *model, const float *input);

/*
 * prob5 only: writes the 5 probabilities as nn_forward computes them
 * (P(win), P(win gammon), P(win bg), P(lose gammon), P(lose bg)) into
 * probs[5] and returns the money equity. On a non-prob5 model, zero-fills
 * probs and returns nn_forward(model, input).
 */
float nn_forward_prob5(const NNModel *model, const float *input, float probs[NN_PROB5_OUTPUTS]);

#endif /* NN_EVAL_H */
