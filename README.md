# PureTD: Self-Play Reinforcement Learning for Backgammon

A backgammon AI trained entirely through self-play [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning). The core method is temporal-difference learning with a neural network over Tesauro's original feature encoding ([TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon)). Besides sampled Bellman backups, the trainer implements exact Bellman backups (1-ply backups), a step towards AlphaZero-style targets. Training runs multiprocess and on the GPU. Supported games: 1-point matches (DMP) and money games; match play is planned (partial work on the experimental `matchplay` branch).

**Key results:** A 563k-parameter network scores +27.2 mEq/game against gnubg 1-ply¹ (10M games, 95% CI [+25.3, +29.0]) in cubeful money games. XG++ analysis of 1000 self-play games gives this model a Performance Rating (PR) of 0.80 (95% CI [0.76, 0.84]).

Two findings:
1. For money games, simple RL techniques produce a base model nearly as good as or better than gnubg's base model.
2. Cube action for money games can be learned by RL by adding the cube actions (offer double, take, drop) to the agent's action set and learning from self-play as usual. No take-point formulas are used. The same approach is expected to work for match play; that is future work.

To our knowledge this is the first open-source backgammon AI trained entirely through self-play RL, with the complete training pipeline included, to reach near-SOTA 0-ply strength in DMP and cubeful money games, and the first open-source implementation to learn cube action directly through self-play RL. The approach is the one described, but not evaluated for money games, by [Andrew Lin](https://ieeexplore.ieee.org/document/9382451/).

[BGBlitz](https://bgblitz.com/) is a closed-source backgammon AI trained with pure RL that is competitive with or ahead of gnubg ([technical presentation](https://bgblitz.com/download/blog/Aachen_BGBlitz.pdf)).

`best_models/experimental/` holds larger cubeful and prob5 models that evaluate better but run slower. The experimental `matchplay` branch holds code and preliminary models for a match-play agent that, like the money-game models, uses no Janowski formulas for cube action and cubeful play.

## Current Models and Results

| Model | Type | Output | Architecture | Params | vs gnubg 0-ply | XG++ PR (0-ply) | md5 | File |
|-------|------|--------|-------------|--------|----------------|-----------------|-----|------|
| Best 4L cubeful | cubeful-money | equity | [512,512,256,256] | 563k | **+85.2 mEq/game** [+83.5, +87.0] | **0.80** [0.76, 0.84] | `6afda0f8` | `best_models/cubeful_money_512_512_256_256.pt` |
| Wide 5L cubeful | cubeful-money | equity | [768,768,256,256,256] | 1,074k | **+102.8 mEq/game** [+101.1, +104.5]² | — | `de9e6e84` | `best_models/experimental/cubeful_money_768_768_256_256_256.pt` |
| Best 3L cubeful | cubeful-money | equity | [512,512,256] | 497k | **+53.6 mEq/game** [+51.9, +55.4] | — | `799f8c75` | `best_models/cubeful_money_512_512_256.pt` |
| 4L cubeless prob5 | cubeless-money | prob5 | [512,512,256,256] | 562k | **+45.8 mEq/game** [+45.0, +46.7] | — | `ac95a983` | `best_models/cubeless_prob5_512_512_256_256.pt` |
| Wide 5L cubeless prob5 | cubeless-money | prob5 | [768,768,256,256,256] | 1,072k | **+54.2 mEq/game** [+53.4, +55.0]² | — | `b3fad4eb` | `best_models/experimental/cubeless_prob5_768_768_256_256_256.pt` |

*All "vs gnubg 0-ply" results are 10M-game evaluations with non-parametric (bootstrap) 95% confidence intervals. Cubeful rows report the capped (±128) mean equity. All rows play the real gnubg 1.08.003 binary over its external-player socket. XG++ PR is the eXtreme Gammon (XG++) Performance Rating, lower is better, measured at 0-ply on 1000 self-play games with bootstrap 95% CIs. Backing data: `experiments/head_to_head/results/` (per-row provenance in the CSV headers).*

*² Paired dice (5M pairs, CIs bootstrap over pairs): [h2h_wide5L_vs_gnubg0_paired.csv](experiments/head_to_head/results/h2h_wide5L_vs_gnubg0_paired.csv).*

## How It Works

### Training Pipeline

1. **Batch TD(0) self-play**: play N (typically 1000) games with the current network, collecting (position, target) pairs.
2. **Compute targets**: a sampled Bellman backup (`target = 1 - V(next_state)`, one dice roll) or an exact Bellman backup (`target = E_dice[max_move(1 - V(next))]`, averaged over all 21 dice).
3. **Train**: one epoch of minibatch SGD (Adam) on the collected data.
4. **Repeat** with the updated weights.

### Sampled vs Exact Bellman Backups

Standard TD(0) uses a **sampled Bellman backup**; the target depends on a single dice roll:
```
target = 1 - V(next_state)          # high variance: one dice roll
```

The 1-ply method computes the **exact Bellman backup**, averaging over all 21 dice outcomes:
```
target = E_dice[max_move(1 - V(next))]  # lower variance: all 21 dice
```

This removes the dice variance from the training signal. Terminal states take their exact value (0/1) instead of the network estimate.

The 1-ply value iteration was inspired by and validated against the "1-ply amplified equity estimate" of [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon). In target quality and cost it sits between TD-Gammon (sampled backups) and AlphaZero (deep search).

Sampled backups (0-ply) are ~5x faster and build a strong base model. Exact backups (1-ply) are slower; initialised from 0-ply-trained weights they usually improve it.

### Cubeful and cubeless money models

Money-game models output equity (+1 for winning one point, -1 for losing one). Undoubled gammons and backgammons are worth +2/+3.

Cubeful play adds 4 binary inputs: cube_centered, cube_own and cube_opponent_own form a one-hot of cube ownership; is_cube_action is 1 for a cube decision and 0 for a checker decision. Cube decisions are ordinary state-action-next_state transitions trained with bootstrapped targets.

The network predicts the expected equity for the current cube ownership and board, with the cube value normalised to 1; the equity of a position is the model output times the cube value. Andrew Lin (above) describes a similar normalisation, with published results only on small match-play networks that neither treat cube decisions as state-action transitions nor use an is_cube_action input.

## Quick Start

### Requirements

- Python 3.10–3.13 (gnubg-nn has no wheels for newer interpreters)
- PyTorch + NumPy (+ gnubg-nn for evaluation vs GNU Backgammon), pinned in `requirements.txt`
- GCC (optional): builds the C engine (~20x faster training) and the C API (`c_api/`) that training and `play_models.py` use by default, on demand at first use. Without it they run the Python path after a NOTE; `--python-api` selects that path.
- GNU Backgammon 1.08.003 for the `gnubg-ext` opponent: install per [`experiments/head_to_head/README.md`](experiments/head_to_head/README.md)

```bash
# one-time setup; every `python` below is this venv's interpreter
python3 -m venv .venv        # python3 must be 3.10-3.13
. .venv/bin/activate
pip install -r requirements.txt
```

### Play the cubeful model against gnubg

```bash
python play_models.py --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 gnubg-ext --plies 0 --games 1000 --game-mode cubeful-money --jacoby
```

### DMP Training

The DMP model is a shared parent of every model. Recipes for training 4L models from scratch are in [`experiments/retraining/README.md`](experiments/retraining/README.md).

Training supports multiprocessing via `--workers` and GPU via `--device cuda`.

### Cubeless Money Training

Train a cubeless money model from a DMP model. `--warm-start-equity` converts the DMP model's probability output to an equity output (re-initialising the output layer) and keeps the hidden-layer weights:

```bash
mkdir -p models

# Warm-start from DMP [512,512,256] and train cubeless money
python train_batch.py --game-mode cubeless-money \
  --warm-start-equity best_models/dmp_512_512_256.pt \
  --num-episodes 10000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 48 --episodes-per-round 2000 \
  --device cuda --save models/cubeless_money_512_512_256.pt
```

The DMP warm start accelerates cubeless money training considerably. Training from scratch works but can stall on apparent plateaus.

### 5-output probability model (prob5)

Cubeless money can also be trained with a **5-output probability head**: `P(win)`, `P(win gammon)`, `P(win backgammon)`, `P(lose gammon)`, `P(lose backgammon)`. Equity is `2·P(win) + P(wg) + P(wbg) − P(lg) − P(lbg) − 1`. `ProbNetwork` (`model.py`) is the network, `ProbAgent` (`prob_agent.py`) the play-time wrapper. `play_models.py` detects prob5 checkpoints by the saved `model_type` and plays them through the C API's prob5 kind (`ProbAgent` with `--python-api`).

```bash
# Train a prob5 model (TD(0) self-play; cubeless money), then 1-ply refinement
python train_prob5.py --hidden 512,512,256,256 \
  --num-episodes 2000000 --lr 1e-3 --save models/cubeless_prob5_512_512_256_256.pt

# 1-ply refinement (GPU: see experiments/gpu_1ply_bench/README.md)
python train_prob5.py --oneply --device cuda --workers 32 \
  --resume models/cubeless_prob5_512_512_256_256.pt \
  --num-episodes 500000 --lr 1.5e-4 --save models/cubeless_prob5_512_512_256_256_1ply.pt

# Play the prob5 model against gnubg-nn (cubeless money)
python play_models.py --model1 best_models/cubeless_prob5_512_512_256_256.pt \
  --gnubg-nn --game-mode cubeless-money --games 1000
```

Prob5 is cubeless-money only (no cube policy).

### Cubeful Money Training

Train a cubeful money model from a cubeless money model. `--warm-start-cubeful` extends the 196-input model to 200 inputs (the 3-way cube-ownership one-hot plus `is_cube_action`):

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
```

The full pipeline is **DMP → cubeless money → cubeful money → 1-ply**. Each stage warm-starts from the previous one, so learned checker play carries over. Leg sizes, LR schedules and the closing 1-ply anneal are in [`experiments/retraining/README.md`](experiments/retraining/README.md).

### Evaluate

Cubeful evaluation plays the real GNU Backgammon 1.08 binary over its external-player socket (`--model2 gnubg-ext`, `third_party_opponents/gnubg/gnubg_external.py`): gnubg picks its own checker plays and makes its native money-cube decisions (Jacoby).

Checkpoint sides play through the C API (simd backend), matching the Python agents' decisions up to float near-ties. 2-ply sides and `--bf16` prob5 sides play on the Python path after a NOTE; `--python-api` selects it, `--c-api` makes the fallback an error.

```bash
# 1. Best 4L cubeful model vs gnubg (real 1.08 binary, cubeful money, mEq/game)
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 gnubg-ext --plies 0 --games 50000 --workers 32 \
    --game-mode cubeful-money --jacoby --paired-dice

# 2. Best 4L cubeful model self-play with .mat game files
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 best_models/cubeful_money_512_512_256_256.pt \
    --game-mode cubeful-money --jacoby \
    --games 100 --save-games cubeful_selfplay_games

# 3. Same as above but with 1-ply checker and cube play (stronger; ~200x slower per game: 2.2 s vs 11 ms)
python play_models.py \
    --model1 best_models/cubeful_money_512_512_256_256.pt \
    --model2 best_models/cubeful_money_512_512_256_256.pt \
    --game-mode cubeful-money --jacoby --oneply1 --oneply2 \
    --games 100 --save-games cubeful_selfplay_1ply_games

# 4. 4L cubeless prob5 model vs gnubg-nn (cubeless money, mEq/game)
python play_models.py \
    --model1 best_models/cubeless_prob5_512_512_256_256.pt \
    --gnubg-nn --games 50000 --workers 32 \
    --game-mode cubeless-money

# 5. Best DMP model vs gnubg-nn (win rate)
python play_models.py \
    --model1 best_models/dmp_512_512_256_256.pt \
    --gnubg-nn --games 10000 --workers 32
```

`--paired-dice` plays each dice sequence twice with the sides swapped; CIs are bootstrapped over pairs. `--games` must be even.

### Inspect a model

```bash
python tools/describe_model.py best_models/dmp_512_512_256_256.pt
```

Prints architecture, parameter count, encoder and output mode of a saved `.pt` checkpoint.

### Tests

```bash
pytest           # default tier, ~8 min on 4 threads; no CUDA needed
pytest --slow    # everything (or BG_TESTS_SLOW=1): training runs, all C-API backends, full sizes
```

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
| `agents.py` | Agent interface, RandomAgent |
| `play_models.py` | Head-to-head evaluation with parallel workers |
| `third_party_opponents/` | Opponent adapters: gnubg (game recording, .mat export, binary preflight), HedgeHog, Open Sage; see `third_party_opponents/README.md` |
| `tools/describe_model.py` | Print architecture and training info of a saved model |
| `tools/collect_positions.py` | Self-play position banks (npz) for distillation, via the trainers' episode collectors |
| `tools/distill.py` | Distil a student net from a teacher's 0-ply labels on those banks (any net type) |
| `tools/gpu_eval_server_validity.py` | Checks for the prob5 GPU eval server: 1-ply CPU vs GPU targets, silent-corruption paths, dead-server and SIGTERM liveness (needs CUDA) |
| `c_engine/` | C move generation (~20x faster) |
| `c_inference/` | C loader and forward pass for exported weights (`nn_eval.c/h`, prob5 included); used by `c_api/` and `c_engine/bg_nply.c` |
| `c_api/` | PureTD C API (libpuretd): 0/1-ply checker play, money cube, self-play, Python binding; see `c_api/README.md` |
| `experiments/benchmarks/` | Single-core per-decision compute benchmarks (PureTD vs official gnubg binary vs Open Sage); see `experiments/benchmarks/README.md` |
| `experiments/gpu_1ply_bench/` | GPU 1-ply training collector: topology/VRAM benchmark, NVIDIA setup; see `experiments/gpu_1ply_bench/README.md` |
| `experiments/retraining/` | Training recipe for the current `best_models/` (chain order, leg sizes, LR schedules) |

Paper tables → code, commands and results: [EXPERIMENTS.md](EXPERIMENTS.md)

## Key Findings

**Exact Bellman backups reduce training noise.** Under sampled backups (0-ply) the loss floor is ~0.004 regardless of network size, the irreducible variance of single-dice-roll targets. Under exact backups (1-ply) it drops to ~0.0001, and the network learns finer positional distinctions.

**Progressive expansion beats training from scratch.** Width and depth expansion (warm-starting larger models from smaller trained ones) is far more efficient than training large models from scratch: an expanded [256,256,128] (149k params) beat a from-scratch [512,256,256] (298k params) at half the size. Both expansions follow [Net2Net](https://arxiv.org/abs/1511.05641) (Chen, Goodfellow & Shlens, 2015) and preserve the network's function at initialisation. Depth expansion appends a square layer only; widen with `--expand`.

**Reusing weights helps across tasks.** Initialising the cubeless money model from DMP weights converges faster with fewer plateaus than training from scratch, despite the different outputs (probability vs equity). Initialising the cubeful money model from a mature cubeless money model helps likewise.

**LR warmup on task transfers.** Warm-starting a new task (DMP → cubeless equity, cubeless → cubeful) puts a freshly initialised head on a trained trunk. `--warmup-cycles 20` ramps the learning rate from 0.1× the scheduled rate to the full rate over the first 20 rounds, on top of any anneal.

**Extra input features do not improve evaluation accuracy.** Extended encodings (pip counts, game-phase gating, gnubg's 25 expert contact features) produced no model stronger than the 196-feature perspective encoding. The bottleneck is training-signal quality, not input features. Expert features may still reduce run time.

## TODO

- Extend to match play (work in progress on branch `matchplay`)
- Replace 1-ply targets with deeper lookahead (AlphaZero-style MCTS or Jacob Hilton's "amplified" equity estimator)
- Evaluate the models inside a full engine with expectimax search

## Attribution

Written by Alexander Strehl, with coding assistance from Claude (Anthropic).

Inspired by and references:
- **Øystein Schønning-Johansen**: discussions, bugs found in the backgammon engine, and the conceptual bug in the cubeful approach whose fix treats cube actions as state transitions
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon): 1-ply value iteration, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg): supervised-learning settings, topology and approaches, and extended discussions
- Gerry Tesauro's TD-Gammon and related work
- GNU Backgammon, in particular as the baseline to compare against
- The Janowski cube layer (`janowski.py`) implements R. Janowski, "Take-Points in Money Games" (1993), with GNU Backgammon 1.08 as the reference implementation

## License

MIT ([LICENSE](LICENSE)), except [third_party_opponents/gnubg/gnubg_bench/benchpool.patch](third_party_opponents/gnubg/gnubg_bench/benchpool.patch), which modifies GNU Backgammon and is GPL-3.0-or-later (see its header).

---

¹ GNU Backgammon 1.08.003 (neural-net weights `gnubg.wd`, md5 `14184acc9c60ef67be0fad88548fd51a`), played over its external-player socket as shipped: pruning nets on, Normal move filter (`gnubg-ext`), native money cube (Jacoby). Against unpruned gnubg 1-ply (`gnubg-ext-full`) the edge is +27.8 [+25.9, +29.6], statistically indistinguishable.
