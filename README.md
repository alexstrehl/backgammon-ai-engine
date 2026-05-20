# BG-Engine: Self-Play Reinforcement Learning for Backgammon

A backgammon AI trained entirely through self-play [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning).  The main approach is temporal difference learning with a neural network using Tesauro's original feature encoding (see [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon)).  In addition to sampled Bellman backups, we implemented exact Bellman backups (sometimes called 1-ply backups), moving closer to an AlphaZero-style approach.  The framework supports multiprocessing and GPU acceleration to speed up training.  Currently it covers 1-point matches (DMP) and money games, but we plan on extending to match play.

**Key results:** A 561k-parameter network achieves 51.84% win rate against gnubg 0-ply¹ (10M games, 95% CI [51.8%, 51.9%]) in 1-point matches. A 562k-parameter network achieves +58.5 mEq/game (0.059 points per game advantage, 10M games, 95% CI [+56.5, +60.4]) vs 0-ply in cubeful money matches.

Two important findings:
1) Simple RL techniques are sufficient to get a base-model that is nearly as good as or better than gnubg's base-model.
2) Cube action for money games can be learned via RL in the very natural way of simply introducing new actions (offer double, take or drop) to the agent and then learning as usual from self-play.  No formulas based on take-points were used.  We suspect this approach will also work for cube action in match play but that remains future work.

As far as we're aware this is the first open-source backgammon AI trained entirely through self-play reinforcement learning (with the complete training pipeline included) to achieve near-SOTA 0-ply playing strength in DMP and cubeful money games.  It is also the first open-source implementation to learn cube action directly through self-play RL, although the approach is the same as described (but not evaluated for money games) by [Andrew Lin](https://ieeexplore.ieee.org/document/9382451/).

The models we've trained show a small but significant advantage over gnubg when neither is allowed any search (so 0-ply vs 0-ply).  For our DMP model, experiments with naive 1-ply and 2-ply search implementations have shown that the advantage holds for 1-ply but our models become tied when both are allowed 2-ply search.  For the cubeful-money model, a 1-ply vs 1-ply head-to-head against gnubg gives +26.9 mEq/game (95% CI [+9.5, +45.0], p=0.0038) — our model maintains an advantage at 1-ply but that advantage is cut in half.  This may be an indication that gnubg's base networks are more tuned for a deep search than ours or that gnubg is using a more sophisticated search.  Given the goals of this project to experiment with AI techniques and to develop a SOTA or near-SOTA backgammon AI, the next major avenue seems evidently to be methods that do a deeper search and optimize the model for searching (including MCTS and AlphaZero-like approaches).

[BGBlitz](https://bgblitz.com/) is a notable non-open-source example of an extremely strong (competitive with or ahead of gnubg) backgammon AI trained with pure RL techniques (see their [technical presentation](https://bgblitz.com/download/blog/Aachen_BGBlitz.pdf)).

## Current Models and Results

| Model | Type | Architecture | Params | vs gnubg 0-ply | mEMG | XG++ PR (0-ply / 1-ply) | File |
|-------|------|-------------|--------|----------------|------|--------------------------|------|
| Best cubeful | cubeful-money | [512,512,256,256] | 562k | **+58.5 mEq/game** [+56.5, +60.4] | **2.1** [2.0, 2.2] | **1.06** [1.01, 1.11] / **0.48** | `best_models/cubeful_money_512_512_256_256.pt` |
| Best cubeful 3L | cubeful-money | [512,512,256] | 497k | **+40.7 mEq/game** [+38.9, +42.5] | **2.6** [2.5, 2.7] | — | `best_models/cubeful_money_512_512_256.pt` |
| Best cubeless | cubeless-money | [512,512,256] | 495k | **+27.9 mEq/game** [+27.0, +28.7] | N/A | — | `best_models/cubeless_money_512_512_256.pt` |
| Best DMP | DMP | [512,512,256,256] | 561k | **51.84%** [51.8%, 51.9%] | **1.5** [1.5, 1.6] | **1.19** [1.14, 1.24] / — | `best_models/dmp_512_512_256_256.pt` |
| DMP efficient | DMP | [256,256,256] | 182k | **50.80%** [50.8%, 50.9%] | 2.3 [2.2, 2.3] | — | `best_models/dmp_256_256_256.pt` |
| DMP (original) | DMP | [512,512,256,128] | 528k | 51.2% (p=0.0004, 20k games) | 2.1 [2.0, 2.1] | — | `best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt` |

_All "vs gnubg 0-ply" results are from 10M-game evaluations with non-parametric (bootstrap) 95% confidence intervals (except DMP original, which is the earlier 20k-game figure)._ _Cubeful results report the raw mean equity in mEq/game._ _mEMG is scored by gnubg's 4-ply analysis on 4000 self-play games per model, with bootstrap 95% CIs._ _XG++ PR is the eXtreme Gammon (XG++) Performance Rating — lower is better. 0-ply values are from self-play games with bootstrap 95% CIs (1500 games for the cubeful 4L, 1000 for the DMP); the cubeful 4L 1-ply value (0.48) is the earlier 200-game point estimate._

**Money-mode caveat for cubeful eval.** gnubg-nn does not appear to implement money-game equity for cube decisions (calls return "Not implemented for money"). To approximate, we evaluate cube decisions inside a simulated 21-point match at 0-0 score, with cube value hardcoded to 2 for any redouble (and Jacoby rule is enforced by doubling whenever gnubg recommends "no double, too good" on a centered cube). A small bug in this approximation that occasionally fed `0-ply` cube decisions through a no-op evaluator was fixed in commit `c7eccde`; figures above are post-fix and the raw and capped means now agree within ~0.5 mEq.

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

Sampled backups (0-ply) are ~30x faster and effective for building a strong base model.
Exact backups (1-ply) are slower but when initialized with 0-ply-trained weights, they often improve results.
GPU training is currently only supported for the 0-ply sampled backup mode.

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

### Cubeful Money Training

Train a cubeful money model from a cubeless money model. The `--warm-start-cubeful` flag
extends the 196-input model to 199 inputs (adding 3 cube one-hot features):

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
  --device cuda --save models/cubeful_512_512_256_1ply.pt

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

> **Note on the gnubg cubeful eval (`--gnubg-cubeful`).** The cubeful
> baseline implemented in `GnubgNNCubefulAgent` picks checker plays
> by calling `gnubg_nn.probabilities` on each candidate next-state and
> converting the 5-tuple to cubeless money equity, then taking the
> argmin over the opponent's equity. gnubg's desktop "Cubeful checker
> evaluations" GUI option does take cube ownership into account, but
> we could not find a way to enable that path through `gnubg-nn` —
> empirically both `probabilities` and `best_move` are cube-blind at
> every ply/score we tried. There is a `best_move` API we could call
> instead, but in a 400k-game H-H at 0-ply, using it either at money
> score `(0,0)` or at the 21-point 0-0 match-equity approximation was
> statistically inferior to the equity approach above, so we leave
> that path unused.

```bash
# 1. Best cubeful model vs gnubg (cubeful money, mEq/game)
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --gnubg-cubeful --games 50000 --workers 32 \
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
python describe_model.py best_models/dmp_512_512_256_256.pt
```

Prints architecture, parameter count, encoder, and output mode for any saved `.pt` checkpoint.

## File Overview

| File | Description |
|------|-------------|
| `backgammon_engine.py` | Board representation and move generation (Python) |
| `encoding.py` | Perspective encoding (196 features) |
| `model.py` | PyTorch network: configurable hidden layers, sigmoid or linear output |
| `train_batch.py` | **Main training script**: batch TD(0) with optional 1-ply value iteration |
| `train_online.py` | Online TD(0) training (simpler, slower) |
| `trainer.py` | Trainer class: optimizer, round-based and online training loops |
| `modes.py` | Game modes (DMP, Money) with terminal handling |
| `td_agent.py` | Agent wrapper for trained models |
| `agents.py` | Agent interface, RandomAgent, GnubgNNAgent |
| `play_models.py` | Head-to-head evaluation with parallel workers |
| `gnubg_eval.py` | Export games to .mat format, analyze with GNU Backgammon |
| `describe_model.py` | Print architecture and training info for a saved model |
| `c_engine/` | C implementation of move generation (~20x faster) |

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
- **Øystein Schønning-Johansen** — helpful discussions, identifying bugs in the backgammon engine, and helped identify a conceptual bug in our cubeful approach, which lead to treating cube actions as proper state transitions.
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — 1-ply value iteration approach, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg) — Supervised-learning settings, topology, and approaches, as well as extended and helpful discussions
- Gerry Tesauro's TDGammon and related work.
- GNUbg for all their wonderful work and especially for providing a baseline to compare to.

## License

MIT

---

¹ Evaluated using [gnubg-nn](https://github.com/StonesAndDice/gnubg-nn-pypi) 1.1.0a6, which wraps GNU Backgammon's neural network (weights version 1.01, 1,097,867 bytes).