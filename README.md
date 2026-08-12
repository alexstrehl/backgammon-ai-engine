# PureTD: Self-Play Reinforcement Learning for Backgammon

A backgammon AI trained entirely through self-play [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning).  The main approach is temporal difference learning with a neural network using Tesauro's original feature encoding (see [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon)).  In addition to sampled Bellman backups, we implemented exact Bellman backups (sometimes called 1-ply backups), moving closer to an AlphaZero-style approach.  The framework supports multiprocessing and GPU acceleration to speed up training.  Currently it covers 1-point matches (DMP) and money games, but we plan to extend to match play.

**Key results:** A 562k-parameter network achieves +20.7 mEq/game against gnubg 1-ply¹ (10M games, 95% CI [+18.9, +22.6]) in cubeful money games. XG++ analysis of 1000 self-play games gives this model a Performance Rating (PR) of 0.94 (95% CI [0.89, 0.99]).

Two important findings:
1) For money games, simple RL techniques are sufficient to get a base-model that is nearly as good as or better than gnubg's base-model.
2) Cube action for money games can be learned via RL in the very natural way of simply introducing new actions (offer double, take or drop) to the agent and then learning as usual from self-play.  No formulas based on take-points were used.  We suspect this approach will also work for cube action in match play but that remains future work.

As far as we're aware this is the first open-source backgammon AI trained entirely through self-play reinforcement learning (with the complete training pipeline included) to achieve near-SOTA 0-ply playing strength in DMP and cubeful money games.  It is also the first open-source implementation to learn cube action directly through self-play RL, although the approach is the same as described (but not evaluated for money games) by [Andrew Lin](https://ieeexplore.ieee.org/document/9382451/).

[BGBlitz](https://bgblitz.com/) is a notable non-open-source example of an extremely strong (competitive with or ahead of gnubg) backgammon AI trained with pure RL techniques (see their [technical presentation](https://bgblitz.com/download/blog/Aachen_BGBlitz.pdf)).

## Current Models and Results

| Model | Type | Output | Architecture | Params | vs gnubg 0-ply | mEMG | XG++ PR (0-ply) | File |
|-------|------|--------|-------------|--------|----------------|------|-----------------|------|
| Best cubeful | cubeful-money | equity | [512,512,256,256] | 562k | **+79.9 mEq/game** [+78.2, +81.7] | **1.94** [1.87, 2.01] | **0.94** [0.89, 0.99] | `best_models/cubeful_money_512_512_256_256.pt` |
| Best cubeful 3L | cubeful-money | equity | [512,512,256] | 497k | **+53.6 mEq/game** [+51.9, +55.4] | **2.6** [2.5, 2.7] | — | `best_models/cubeful_money_512_512_256.pt` |
| Best cubeless | cubeless-money | equity | [512,512,256,256] | 561k | **+44.2 mEq/game** [+43.4, +45.1] | N/A | — | `best_models/cubeless_money_512_512_256_256.pt` |
| Best cubeless 3L | cubeless-money | equity | [512,512,256] | 495k | **+28.6 mEq/game** [+27.8, +29.5] | N/A | — | `best_models/cubeless_money_512_512_256.pt` |
| Cubeless prob5 | cubeless-money | prob5 | [512,512,256,128] | 528k | **+44.7 mEq/game** [+43.9, +45.6] | **1.34** [1.31, 1.38] | — | `best_models/cubeless_prob5_512_512_256_128.pt` |
| Best DMP | DMP | win prob | [512,512,256,256] | 561k | **51.80%** [51.77%, 51.83%] | **1.5** [1.5, 1.6] | **1.19** [1.14, 1.24] | `best_models/dmp_512_512_256_256.pt` |

_All "vs gnubg 0-ply" results are from 10M-game evaluations with non-parametric (bootstrap) 95% confidence intervals, every row measured on the corrected move generator — the Best cubeful and Cubeless prob5 rows in `experiments/head_to_head/results/h2h_redo_validation_20260808.csv`, the other four in `experiments/head_to_head/results/h2h_model_table_rows_20260810.csv`._ _Both cubeful rows and the Cubeless prob5 row play the real gnubg 1.08.003 binary over its external-player socket, with gnubg's native money cube (Jacoby) for cubeful._ _The remaining rows use the gnubg-nn wrapper², which selects the same checker moves as the real binary at 0-ply (decision-gate checks: 232/250 identical picks cubeless, 294/300 DMP, with every remaining difference an exact evaluation tie)._ _The Best cubeful 3L mEMG was measured on the prior checkpoint (since trained 2M further episodes); its H-H figure is the committed checkpoint._ _Cubeful rows report the capped (±128) mean equity — native-cube games include rare runaway redouble wars that make the raw mean unstable._ _mEMG is scored by gnubg's 4-ply analysis on 4000 self-play games per model, with bootstrap 95% CIs._ _XG++ PR is the eXtreme Gammon (XG++) Performance Rating — lower is better; the values shown are 0-ply, from self-play games with bootstrap 95% CIs (1000 games each)._

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

The 1-ply value iteration approach was inspired by and validated against the "1-ply amplified equity estimate" from [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon).  As Hilton notes, this sits between TD-Gammon (sampled backups) and AlphaZero (deep search) in terms of target quality and cost.

Sampled backups (0-ply) are ~5x faster and effective for building a strong base model.
Exact backups (1-ply) are slower but when initialized with 0-ply-trained weights, they often improve results.

### Cubeful and cubeless money models

For money games, the model is trained to output equity (+1 for winning 1 point or dollar and -1 for losing 1 point). Undoubled gammons/backgammons are worth +2/+3 equity.

For cubeful play we add 4 additional binary inputs (cube_centered, cube_own, cube_opponent_own, and is_cube_action). The first three, (cube_centered, cube_own, cube_opponent_own) form a 1-hot vector encoding the cube ownership. The last, is_cube_action is set to 1 when the agent is faced with a cube decision and to 0 when it's faced with a checker decision. All cube decisions are treated as proper state-action-next_state transitions and trained with bootstrapped targets.

For numeric stability of the outputs we normalize the network so that it predicts the expected equity given the current cube ownership and board-state assuming the cube value is 1. So the estimated equity of a position is the model output times the current cube value. A similar approach is described by Andrew Lin (see above). However, published results from that paper are only on very small networks for match play (and specifically not money play) and do not treat cube decisions as proper state-action transitions as our approach (nor include an is_cube_action feature).

## Quick Start

### Requirements

- Python 3.10+
- PyTorch
- GCC (optional, for the C engine — ~20x faster training)
- gnubg-nn (`pip install gnubg-nn`, optional, for evaluation vs GNU Backgammon)

### Play the best model against gnubg

```bash
pip install gnubg-nn
python play_models.py --model1 best_models/dmp_512_512_256_256.pt --gnubg --games 1000
```

### DMP Training

The best DMP model was built through progressive expansion — starting small and gradually increasing capacity:

```bash
# Build C engine first (optional but ~20x faster)
cd c_engine && bash build_unix.sh && cd ..

# Create the models/ directory (saves fail silently if missing)
mkdir -p models

# 1. Train small model from scratch with online TD
python train_online.py --game-mode dmp --num-episodes 2000000 \
  --hidden 80 --lr 0.1 \
  --save models/td_80.pt

# 2. Width-expand [80] -> [150], then depth-expand to [150,150]
python train_batch.py --game-mode dmp \
  --expand models/td_80.pt --hidden 150 \
  --num-episodes 300000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --save models/batch_150.pt
python train_batch.py --game-mode dmp \
  --expand-depth models/batch_150.pt \
  --num-episodes 300000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --save models/batch_150_150.pt

# 3. Width-expand [150,150] -> [256,256]
python train_batch.py --game-mode dmp \
  --expand models/batch_150_150.pt --hidden 256,256 \
  --num-episodes 1000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --save models/batch_256_256.pt

# 4. Width-expand [256,256] -> [512,512] (GPU)
python train_batch.py --game-mode dmp \
  --expand models/batch_256_256.pt --hidden 512,512 \
  --num-episodes 250000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --device cuda --save models/batch_512_512.pt

# 5. Depth-expand to [512,512,256], train 3M episodes
python train_batch.py --game-mode dmp \
  --expand-depth models/batch_512_512.pt --expand-depth-size 256 \
  --num-episodes 3000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --save models/batch_512_512_256.pt

# 6. Depth-expand to [512,512,256,256] with LR warmup
python train_batch.py --game-mode dmp \
  --expand-depth models/batch_512_512_256.pt --expand-depth-size 256 \
  --num-episodes 3000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --warmup-cycles 20 --workers 8 --save models/batch_512_512_256_256.pt

# 7. 1-ply refinement (exact Bellman backups)
python train_batch.py --game-mode dmp \
  --resume models/batch_512_512_256_256.pt \
  --num-episodes 500000 --oneply \
  --optimizer adam --lr 5e-5 --end-lr 2.5e-6 \
  --workers 8 --save models/dmp_1ply.pt

```

Training supports multiprocessing via `--workers` and GPU via `--device cuda`.

### Cubeless Money Training

Train a cubeless money model from an existing DMP model. The `--warm-start-equity` flag
converts the DMP model's probability output to equity output (re-initializing the output
layer) while preserving the hidden layer weights:

```bash
mkdir -p models

# Warm-start from DMP [512,512,256] and train cubeless money
python train_batch.py --game-mode cubeless-money \
  --warm-start-equity best_models/dmp_512_512_256.pt \
  --num-episodes 10000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 48 --episodes-per-round 2000 \
  --device cuda --save models/cubeless_money_512_512_256.pt
```

The DMP warm-start significantly accelerates cubeless money training.  Training from scratch is possible but we found it can lead to getting stuck in apparent plateaus (likely resolvable with enough training).

### 5-output probability model (prob5)

As an alternative to the scalar-equity output, cubeless money can also be trained with a **5-output probability head**: `P(win)`, `P(win gammon)`, `P(win backgammon)`, `P(lose gammon)`, `P(lose backgammon)`. Equity is derived as `2·P(win) + P(wg) + P(wbg) − P(lg) − P(lbg) − 1`. The `ProbNetwork` class and the `ProbAgent` play-time wrapper live in `model.py` / `prob_agent.py`; `play_models.py` auto-detects prob5 checkpoints (via the saved `model_type`) and routes them to `ProbAgent`.

```bash
# Train a prob5 model (TD(0) self-play; cubeless money), then 1-ply refinement
python train_prob5.py --hidden 512,512,256,128 \
  --num-episodes 2000000 --lr 1e-3 --save models/cubeless_prob5_512_512_256_128.pt

python train_prob5.py --oneply \
  --resume models/cubeless_prob5_512_512_256_128.pt \
  --num-episodes 500000 --lr 1.5e-4 --save models/cubeless_prob5_512_512_256_128.pt

# Play the prob5 model against gnubg (cubeless money)
python play_models.py --model1 best_models/cubeless_prob5_512_512_256_128.pt \
  --gnubg --game-mode cubeless-money --games 1000
```

Prob5 is cubeless-money only (no cube policy).

### Cubeful Money Training

Train a cubeful money model from a cubeless money model. The `--warm-start-cubeful` flag
extends the 196-input model to 200 inputs (adding the 3-way cube-ownership one-hot plus the `is_cube_action` flag):

```bash
mkdir -p models

# 1. Warm-start from cubeless money model and train cubeful
python train_batch.py --game-mode cubeful-money \
  --warm-start-cubeful models/cubeless_money_512_512_256.pt \
  --num-episodes 10000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --warmup-cycles 20 --workers 48 --episodes-per-round 2000 \
  --device cuda --save models/cubeful_512_512_256.pt

# 2. 1-ply refinement (exact Bellman backups for cubeful play)
python train_batch.py --game-mode cubeful-money \
  --resume models/cubeful_512_512_256.pt \
  --num-episodes 500000 --oneply \
  --optimizer adam --lr 1e-5 --end-lr 5e-6 \
  --workers 48 --episodes-per-round 1000 \
  --save models/cubeful_512_512_256_1ply.pt

# 3. Low-LR polish to recover network strength after 1-ply
python train_batch.py --game-mode cubeful-money \
  --resume models/cubeful_512_512_256_1ply.pt \
  --num-episodes 2000000 --optimizer adam --lr 2e-5 --end-lr 1e-6 \
  --workers 48 --episodes-per-round 2000 \
  --device cuda --save models/cubeful_512_512_256_final.pt
```

The full pipeline is: **DMP → cubeless money → cubeful money → 1-ply → polish**.
Each stage warm-starts from the previous, so learned checker play transfers through.

We observed that 1-ply training alone sometimes degrades model strength, but following
it with a low-LR 0-ply polish reliably recovers and improves equity.

### Evaluate

> **Note on the gnubg cubeful eval.** Cubeful evaluation plays the real
> GNU Backgammon 1.08 binary over its external-player socket
> (`--model2 gnubg-ext`, see `gnubg_external.py`): gnubg picks its own
> checker plays and makes its native money-cube decisions (Jacoby) — no
> approximation. (An earlier gnubg-nn-based cubeful baseline used a
> 21-point match-equity approximation for the cube; it has been removed.)

```bash
# 1. Best cubeful model vs gnubg (real 1.08 binary, cubeful money, mEq/game)
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 gnubg-ext --plies 0 --games 50000 --workers 32 \
    --game-mode cubeful-money --jacoby

# 2. Best cubeful model self-play with .mat game files
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 best_models/cubeful_money_512_512_256_256.pt \
    --game-mode cubeful-money --jacoby \
    --games 100 --save-games cubeful_selfplay_games

# 3. Same as above but with 1-ply checker play (stronger but ~200x slower)
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 best_models/cubeful_money_512_512_256_256.pt \
    --game-mode cubeful-money --jacoby --oneply1 --oneply2 \
    --games 100 --save-games cubeful_selfplay_1ply_games

# 4. Best cubeless money model vs gnubg (cubeless money, mEq/game)
python play_models.py \
    --model1 best_models/cubeless_money_512_512_256.pt \
    --gnubg --games 50000 --workers 32 \
    --game-mode cubeless-money

# 5. Best DMP model vs gnubg (win rate)
python play_models.py \
    --model1 best_models/dmp_512_512_256_256.pt \
    --gnubg --games 10000 --workers 32

# mEMG analysis with GNU Backgammon CLI
python gnubg_eval.py --model best_models/dmp_512_512_256_256.pt \
    --games 1000 --gnubg /usr/games/gnubg
```

### Inspect a model

```bash
python tools/describe_model.py best_models/dmp_512_512_256_256.pt
```

Prints architecture, parameter count, encoder, and output mode for any saved `.pt` checkpoint.

## File Overview

| File | Description |
|------|-------------|
| `backgammon_engine.py` | Board representation and move generation (Python) |
| `encoding.py` | Perspective encoding (196 features) |
| `model.py` | PyTorch network: configurable hidden layers, sigmoid/linear scalar output or a 5-output prob5 head |
| `train_batch.py` | **Main training script**: batch TD(0) with optional 1-ply value iteration |
| `train_online.py` | Online TD(0) training (simpler, slower) |
| `train_prob5.py` | Train a 5-output prob5 model (cubeless money, TD(0)) |
| `trainer.py` | Trainer class: optimizer, round-based and online training loops |
| `modes.py` | Game modes (DMP, Money) with terminal handling |
| `td_agent.py` | Agent wrapper for trained models |
| `prob_agent.py` | Play-time wrapper for prob5 models |
| `agents.py` | Agent interface, RandomAgent, GnubgNNAgent |
| `play_models.py` | Head-to-head evaluation with parallel workers |
| `gnubg_eval.py` | Export games to .mat format, analyze with GNU Backgammon |
| `tools/describe_model.py` | Print architecture and training info for a saved model |
| `c_engine/` | C implementation of move generation (~20x faster) |
| `c_inference/` | Standalone C inference (loads exported weights; supports prob5) |
| `experiments/benchmarks/` | Single-core per-decision compute benchmarks (PureTD vs official gnubg binary vs Open Sage); see `experiments/benchmarks/README.md` |

## Key Findings

**Exact Bellman backups dramatically reduce training noise.** Under sampled backups (0-ply), the loss floor is ~0.004 regardless of network size — irreducible variance from single-dice-roll targets. With exact backups (1-ply), the loss floor drops to ~0.0001, enabling the network to learn finer positional distinctions.

**Progressive expansion beats training from scratch.** Width and depth expansion (warm-starting larger models from smaller trained ones) is far more efficient than training large models from scratch. A depth-expanded [256,256,128] (72k params) beats a from-scratch [512,256,256] (263k params) despite being 3.6x smaller. Our approach follows [Net2Net](https://arxiv.org/abs/1511.05641) (Chen, Goodfellow & Shlens, 2015): both width and depth expansion preserve the network's function at initialization.

**LR warmup helps depth expansion.** Ramping the learning rate over 20 cycles when training a depth-expanded model improves results.

**Extra input features don't help.** We tested extended encodings (pip counts, game-phase gating, gnubg's 25 expert contact features) — none improved over the base 196-feature perspective encoding. The bottleneck is training signal quality, not input features.

**Reusing weights from existing models helps.** Initializing the cubeless money model from DMP-trained weights produces faster convergence and fewer plateaus than training from scratch, despite the different output representations (probability vs. equity). Similarly when training a cubeful money model, initializing the weights based on a mature cubeless money model is helpful.

**Supervised learning optimizations don't help.** Weight decay ({1e-3, 1e-4, 1e-5}), cosine annealing, and other regularization techniques had no effect on loss or playing strength.

## TODO

- Extend to match play
- Experiment with explicit exploration (epsilon-greedy during self-play)
- Replace 1-ply targets with deeper lookahead (similar to AlphaZero's MCTS or Jacob Hilton's "amplified" equity estimator)
- GPU-accelerated 1-ply target computation (batch across positions)

## Attribution

Written by Alexander Strehl, with coding assistance from Claude (Anthropic).

Inspired by and references:
- **Øystein Schønning-Johansen** — helpful discussions, identifying bugs in the backgammon engine, and helped identify a conceptual bug in our cubeful approach, which led to treating cube actions as proper state transitions.
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — 1-ply value iteration approach, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg) — Supervised-learning settings, topology, and approaches, as well as extended and helpful discussions
- Gerry Tesauro's TDGammon and related work.
- GNUbg for all their wonderful work and especially for providing a baseline to compare to.

## License

MIT

---

¹ GNU Backgammon 1.08.003 (neural-net weights `gnubg.wd`, md5 `14184acc9c60ef67be0fad88548fd51a`), played over its external-player socket at full width — pruning nets disabled, keep-all move filter (`gnubg-ext-full`), native money cube (Jacoby). This is the configuration all gnubg **1-ply** rows use; the "vs gnubg 0-ply" rows play gnubg as shipped — prune nets on, Normal move filter (`gnubg-ext`). Against gnubg's default pruned 1-ply the edge is +20.8 [+18.9, +22.6].

² Evaluated using [gnubg-nn](https://github.com/StonesAndDice/gnubg-nn-pypi) 1.1.0a6, which wraps GNU Backgammon's neural network (weights version 1.01, 1,097,867 bytes).