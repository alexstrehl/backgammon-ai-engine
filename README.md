# BG-Engine: Self-Play Reinforcement Learning for Backgammon

A backgammon AI trained entirely through self-play [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning).  The main approach is temporal difference learning with a neural network using Tesauro's original feature encoding (see [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon)).  In addition to sampled Bellman backups, we implemented exact Bellman backups (sometimes called 1-ply backups), moving closer to an AlphaZero-style approach.  The framework supports multiprocessing and GPU acceleration to speed up training.  Currently it only covers 1-point matches (DMP) but we plan on extending to money games and match play.

**Key result:** A 528k-parameter network achieves 51.2% win rate against gnubg 0-ply (p=0.0004, 20k games) in 1-point matches

## Current Models and Results

| Model | Architecture | Params | vs gnubg 0-ply | Training | Date |
|-------|-------------|--------|----------------|----------|------|
| Best (1-ply VI) | [512,512,256,128] | 528k | **51.2%** (p=0.0004) | 1-ply value iteration refinement | 2026-03-26 |
| Strong (1-ply VI) | [256,256,256] | 182k | 50.1% (tied) | 1-ply value iteration refinement | 2026-03-26 |

The mEMG error rate (gnubg 2-ply analysis) is ~2.5, placing the model in the "world class" range.

## How It Works

### Training Pipeline

1. **Batch TD(0) self-play**: Play N (typically 1000) games with the current network, collecting (position, target) pairs
2. **Compute targets**: Either a sampled Bellman backup (`target = 1 - V(next_state)`, one dice roll) or an exact Bellman backup (`target = E_dice[max_move(1 - V(next))]`, averaging over all 21 dice)
3. **Train**: One epoch of minibatch SGD (Adam) on the collected data
4. **Repeat**: Collect new games with updated weights

### Sampled vs Exact Bellman Backups

Standard TD(0) uses a **sampled Bellman backup** — the target depends on a single dice roll:
```
target = 1 - V(next_state)          # high variance: one dice roll
```

The 1-ply method computes the **exact Bellman backup**, averaging over all 21 dice outcomes:
```
target = E_dice[max_move(1 - V(next))]  # lower variance: all 21 dice
```

This eliminates the variance due to dice from the training signal. Terminal states are handled explicitly (exact value 0/1 instead of network estimate).

The 1-ply value iteration approach was inspired by and validated against the "1-ply amplified equity estimate" from [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon).  As Hinton notes, this sits between TD-Gammon (sampled backups) and AlphaZero (deep search) in terms of target quality and cost.

Sampled backups (0-ply) are ~30x faster and effective for building a strong base model.
Exact backups (1-ply) are slower but when initialized with 0-ply-trained weights, they often improve results.
GPU training is currently only supported for the 0-ply sampled backup mode.

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

The best model was built through progressive expansion — starting small and gradually increasing capacity:

```bash
# Build C engine first (optional but ~20x faster)
cd c_engine && bash build_unix.sh && cd ..

# 1. Train small model from scratch with online TD
python td_train.py --episodes 2000000 --hidden 80 --activation relu \
  --lr 0.1 --end-lr 0.01 --save-path models/td_80

# 2. Width-expand [80] to [150], then depth-expand to [150,150]
python td_batch_train.py --episodes 300000 --hidden 150 --activation relu \
  --expand models/td_80_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_150 --workers 8
python td_batch_train.py --episodes 300000  --activation relu \
  --expand-depth models/batch_150_final.pt \
  --fast --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_150_150 --workers 8

# 3. Width-expand [150,150] to [256,256], continue training
python td_batch_train.py --episodes 1000000 --hidden 256 256 --activation relu \
  --expand models/batch_150_150_final.pt \
  --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_256_256 --workers 8

# 4. Width-expand [256,256] to [512,512] (GPU supported)
python td_batch_train.py --episodes 250000 --hidden 512 512 --activation relu \
  --expand models/batch_256_256_final.pt \
  --adam --lr 5e-5 --end-lr 5e-6 --eval-every 50000 \
  --save-path models/batch_512_512 --workers 8 --device cuda

# 5. Depth-expand to [512,512,256]
python td_batch_train.py --episodes 3000000  --activation relu \
  --expand-depth models/batch_512_512_final.pt \
  --adam --lr 5e-5 --end-lr 5e-6 \
  --save-path models/batch_512_512_256 --workers 8

# 6. Depth-expand to [512,512,256,128] with LR warmup
python td_batch_train.py --episodes 1000000 \
  --expand-depth models/batch_512_512_256_final.pt \
  --adam --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --save-path models/batch_512_512_256_128 --workers 8

# 7. Refine with exact Bellman backups (the key step)
python td_batch_train.py --episodes 500000 --activation relu \
  --resume models/batch_512_512_256_128_final.pt \
  --adam --lr 1e-4 --end-lr 5e-6 --warmup-cycles 20 \
  --oneply --save-path models/1ply_final --workers 8 \
  --eval-every 100000 --eval-games 5000
```

Training supports multiprocessing via the `--workers` flag. The full pipeline above took approximately 5 hours on a modern 64-core machine.

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

## Key Findings

**Exact Bellman backups dramatically reduce training noise.** Under sampled backups (0-ply), the loss floor is ~0.004 regardless of network size — irreducible variance from single-dice-roll targets. With exact backups (1-ply), the loss floor drops to ~0.0001, enabling the network to learn finer positional distinctions.

**Progressive expansion beats training from scratch.** Width and depth expansion (warm-starting larger models from smaller trained ones) is far more efficient than training large models from scratch. A depth-expanded [256,256,128] (72k params) beats a from-scratch [512,256,256] (263k params) despite being 3.6x smaller.

**LR warmup helps depth expansion.** Ramping the learning rate over 20 cycles when training a depth-expanded model improves results.

**Extra input features don't help.** We tested extended encodings (pip counts, game-phase gating, gnubg's 25 expert contact features) — none improved over the base 196-feature perspective encoding. The bottleneck is training signal quality, not input features.

**Supervised learning optimizations don't help.** Weight decay ({1e-3, 1e-4, 1e-5}), cosine annealing, and other regularization techniques had no effect on loss or playing strength.

## TODO

- Extend to match play and cube decisions (double/take/pass)
- Experiment with explicit exploration (epsilon-greedy during self-play)
- Replace 1-ply targets with deeper lookahead (similar to AlphaZero's MCTS or Jacob Hilton's "amplified" equity estimator)
- GPU-accelerated 1-ply target computation (batch across positions)

## Attribution

Written by Alexander Strehl, with coding assistance from Claude (Anthropic).

Inspired by and references:
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — 1-ply value iteration approach, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg) — Supervised-learning settings, topology, and approaches, as well as extended and helpful discussions        