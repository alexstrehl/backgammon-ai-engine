#!/bin/bash
# Build a server that can be used as an externl player in GNU Backgammon.

ENGINE=../c_engine
ENGINE_LIB=$ENGINE/libbg_engine.so
NEURALNET_LIB=../c_inference/nn_eval.o
set -e 

echo "Building backgammon server for GNU Backgammon external player. (Requires libevent)"
gcc -std=gnu99 -Wall -Wextra -O3 -I$ENGINE/ -I../c_inference/ -c server.c

if [[ -f $ENGINE_LIB && -f $NEURALNET_LIB ]]; then
	gcc -o server server.o $NEURALNET_LIB -L$ENGINE/ -lbg_engine -levent -lm
else
	echo "Please build objectes in $ENGINE and ../c_inference first"
	exit
fi
echo "Success: server created."

echo "  # Export a model first:"
echo "  python ../export_weights.py ../best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt model.bin"
echo ""
echo "  # Start server and open GNU Backgammon:"
echo "  ./server --modelfile=model.bin"
echo ""
echo "  # Open GNU Backgammon and in the Settings->Players... menu"
echo "  For one of the players select 'External' and type in 'localhost:9876' and click the OK button."
echo "  This should then connect GNU Backgammon to the server and you can play a game. "
echo "  ... and you may have to set LD_LIBRARY_PATH=$ENGINE before running). "

