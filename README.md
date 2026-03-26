# BG-Engine: Neural Network Backgammon via Self-Play

A backgammon neural network trained entirely through self-play, using TD(0) temporal difference learning with 1-ply value iteration. The best model beats GNU Backgammon's 0-ply neural network at DMP (Double Match Point) play.

**Key result:** A 528k-parameter network achieves 51.2% win rate against gnubg 0-ply (p=0.0004, 20k games) — trained purely from self-play TD learning using Tesauro's original encoding.

## Results

| Model | Architecture | Params | vs gnubg 0-ply | Training |
|-------|-------------|--------|----------------|----------|
| Best (1-ply VI) | [512,512,256,128] | 528k | **51.2%** (p=0.0004) | 1-ply value iteration fine-tune |
| Strong (1-ply VI) | [256,256,256] | 182k | 50.1% (tied) | 1-ply value iteration fine-tune |

The mEMG error rate (gnubg 2-ply analysis) is ~2.5, placing the model in the "world class" range.

## How It Works

### Training Pipeline

1. **Batch TD(0) self-play**: Play 1000 games with the current network, collecting (position, target) pairs
2. **Compute targets**: Either standard 0-ply bootstrap (`target = 1 - V(next_state)`) or 1-ply bootstrap (`target = E_dice[max_move(1 - V(next))]`)
3. **Train**: One epoch of minibatch SGD (Adam) on the collected data
4. **Repeat**: Collect new games with updated weights

Standard 0-ply training is ~30x faster and effective for building a strong base model.  For large networks training on the GPU is faster and supported.
1-ply value estimation is slower when initialized with weights from standard 0-ply training, it often improves.  Currently GPU is not supported.

### 1-Ply Bootstrapped value estimates

Instead of 
```

target = 1 - V(next_state)          # high variance: depends on one dice roll
```
We compute:
```
target = E_dice[max_move(1 - V(next))]  # lower variance: averages over all 21 dice
```

This is the exact Bellman backup — an unbiased estimator of V(s) that eliminates dice variance from the training signal. Terminal states are handled explicitly (exact value 0/1 instead of network estimate).

The 1-ply value iteration approach was inspired by and validated against [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon). Key differences: we use pure batch TD (not experience replay), 0-ply move selection during self-play (not 1-ply), MSE loss (not cross-entropy), and 196-feature perspective encoding.  Has Hinton mentions one can think of this as similar to Alpha_0 except we are using a much cheaper estimate of the target but still more expensive that orginal TD-gammon.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch
- GCC (optional, for the C engine — ~20x faster training)
- gnubg-nn (`pip install gnubg-nn`, optional, for evaluation vs GNU Backgammon)

### Play the best model against gnubg

```bash
pip install gnubg-nn
python play_models.py --model1 best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt --gnubg --games 1000
```

### Training

The training scripts support training from scratch, resuming from a checkpoint, and expanding models to larger architectures (width or depth expansion).

The best model was built through progressive expansion — starting small and gradually increasing capacity:

```bash
# Build C engine first (optional but ~20x faster)
cd c_engine && bash build_unix.sh && cd ..

# 1. Train small model from scratch with online TD
python td_train.py --episodes 2000000 --hidden 80 --activation relu \
  --lr 0.1 --end-lr 0.01 --fast --save-path models/td_80

# 2. Width-expand to [150,150], batch TD fine-tune
python td_batch_train.py --episodes 300000 --hidden 150 150 --activation relu \
  --expand models/td_80_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_150_150 --workers 8

# 3. Width-expand to [256,256], fine-tune
python td_batch_train.py --episodes 1000000 --hidden 256 256 --activation relu \
  --expand models/batch_150_150_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_256_256 --workers 8

# 4. Width-expand to [512,512], fine-tune
python td_batch_train.py --episodes 2500000 --hidden 512 512 --activation relu \
  --expand models/batch_256_256_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_512_512 --workers 8

# 5. Depth-expand to [512,512,256], fine-tune
python td_batch_train.py --episodes 3000000 \
  --expand-depth models/batch_512_512_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_512_512_256 --workers 8

# 6. Depth-expand to [512,512,256,128], fine-tune with LR warmup
python td_batch_train.py --episodes 1000000 \
  --expand-depth models/batch_512_512_256_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --save-path models/batch_512_512_256_128 --workers 8

# 7. 1-ply value iteration fine-tune (the key step)
python td_batch_train.py --episodes 500000 \
  --resume models/batch_512_512_256_128_final.pt \
  --fast --adam --lr 1e-4 --end-lr 5e-6 --warmup-cycles 20 \
  --oneply --save-path models/1ply_final --workers 8 \
  --eval-every 100000 --eval-games 5000
```

### Evaluate

```bash
# Head-to-head vs gnubg 0-ply (DMP)
python play_models.py --model1 models/1ply_final.pt --gnubg --games 10000 --workers 8

# Head-to-head between two models
python play_models.py --model1 model_a.pt --model2 model_b.pt --games 10000

# mEMG analysis with GNU Backgammon CLI
python gnubg_eval.py --model models/1ply_final.pt --games 1000 --gnubg /usr/games/gnubg
```

## File Overview

| File | Description |
|------|-------------|
| `backgammon_engine.py` | Board representation and move generation (Python) |
| `encoding.py` | Perspective encoding (196 features) |
| `model.py` | PyTorch network: configurable hidden layers, sigmoid output for P(win) |
| `td_batch_train.py` | **Main training script**: batch TD(0) with optional 1-ply value iteration |
| `td_train.py` | Online TD(0) training (simpler, slower) |
| `td_agent.py` | Agent wrapper for trained models |
| `agents.py` | Agent interface, RandomAgent, GnubgNNAgent |
| `play_models.py` | Head-to-head evaluation with parallel workers |
| `gnubg_eval.py` | Export games to .mat format, analyze with GNU Backgammon |
| `evaluate.py` | Simple evaluation harness |
| `c_engine/` | C implementation of move generation (~20x faster) |

## Pretrained Models

| Model | File | Win% vs gnubg 0-ply |
|-------|------|---------------------|
| Best | `best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt` | 51.2% |
| Lighter | `best_models/td_batch_relu_256_256_256_1ply_vi_final.pt` | 50.1% |

## Training History

The best model's lineage:
1. [80] online TD from scratch, 2M episodes (lr=0.1→0.01)
2. [150,150] batch TD fine-tune, 300k episodes → 4.9 mEMG
3. [256,256] width-expanded, 1M episodes
4. [512,512] width-expanded, ~2.5M episodes
5. [512,512,256] depth-expanded, 3M episodes → 3.7 mEMG
6. [512,512,256,128] depth-expanded with LR warmup, 1M episodes → 3.6 mEMG
7. **1-ply fine-tune, 500k episodes → beats gnubg 0-ply (51.2%)**

## Key Findings

**1-ply targets dramatically reduce training noise.** Under standard 0-ply TD, the loss floor is ~0.004 regardless of network size — this is irreducible variance from single-dice-roll bootstrap targets. With 1-ply targets (averaging over all 21 dice outcomes), the loss floor drops to ~0.0001, enabling the network to learn finer positional distinctions.

**Progressive expansion beats training from scratch.** Width and depth expansion (warm-starting larger models from smaller trained ones) is far more efficient than training large models from scratch. A depth-expanded [256,256,128] (72k params) beats a from-scratch [512,256,256] (263k params) despite being 3.6x smaller.

**LR warmup helps depth expansion.** Ramping the learning rate over 20 cycles when fine-tuning a depth-expanded model improves results.

**Extra input features don't help.** We tested extended encodings (pip counts, game-phase gating, gnubg's 25 expert contact features) — none improved over the base 196-feature perspective encoding. The bottleneck is training signal quality, not input features.

**Supervised learning optimizations don't help.** Weight decay ({1e-3, 1e-4, 1e-5}), cosine annealing, and other regularization techniques had no effect on loss or playing strength.

**gnubg's error analysis underestimates DMP play quality.** Some moves flagged as errors by gnubg's static analysis are confirmed correct by deeper analysis (XG+, gnubg rollouts). Win rate vs gnubg 0-ply is a more reliable metric for DMP-optimized models.

## TODO

- Extend to match play and cube decisions (double/take/pass)
- Multi-output network (P(win), P(gammon), P(backgammon)) for money game play
- Experiment with explicit exploration (epsilon-greedy during self-play)
- Replace 1-ply targets with deeper lookahead (similar to AlphaZero's MCTS or jacobhilton's "amplified" equity estimator)
- GPU-accelerated 1-ply target computation (batch across positions)

## Attribution

Written by the author, with coding assistance from Claude (Anthropic).

Inspired by and references:
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — 1-ply value iteration approach, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg) — Supervised network topology and Discussions
-Gerry Tesauro's TDGammon and related work.
-GNUbg for all there wonderful work and especially to have a basline to compare to.

## License

MIT
