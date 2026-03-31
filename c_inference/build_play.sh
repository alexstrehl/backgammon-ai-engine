#!/bin/bash
# Build the C play_models binary and test_nn verifier.
# Run from the c_inference/ directory.

set -e

cd "$(dirname "$0")"

ENGINE=../c_engine

echo "Building play_models..."
gcc -O2 -o play_models play_models.c $ENGINE/bg_engine.c nn_eval.c -I$ENGINE -lm -Wall
echo "  -> play_models built"

echo "Building test_nn..."
gcc -O2 -o test_nn test_nn.c $ENGINE/bg_engine.c nn_eval.c -I$ENGINE -lm -Wall
echo "  -> test_nn built"

echo "Done. Usage:"
echo "  # Export a model first:"
echo "  python ../export_weights.py ../best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt model.bin"
echo ""
echo "  # Play two models:"
echo "  ./play_models model1.bin model2.bin 1000"
echo ""
echo "  # Verify C matches Python:"
echo "  ./test_nn model.bin"
