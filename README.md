# PureTD: Self-Play Reinforcement Learning for Backgammon

## Match play

This branch contains work in progress: applying the PureTD approach to
match play. The network takes the match score and cube ownership as
inputs and outputs a single number — the match-winning chance for the
player on roll.

Against a baseline built the standard way (**p5jan**: our cubeless
prob5 model with Janowski's formulae for the cubeful adjustment), our
best model wins **50.10% [50.05, 50.15]** of 11-point matches
(n = 4M), so it is slightly better but close. Offline analysis with
gnubg and XG suggests its entire advantage is in cube action (cube
decisions).

A couple of important details on our approach follow. **First**, we hand
the network MET-derived features (Kazaross-XG2): the match equity for
winning or losing a single, gammon or backgammon at the current score,
at both the current cube value and twice it, plus the value of the
opponent dropping (13 new features). We also incorporate, in the
learning procedure itself, the exact match equity of the resulting score
at terminal states, in four places:

1. When deciding whether to take or drop a cube, the agent compares its
   network's evaluation of *taking* against the exact match equity for
   *dropping*.
2. When deciding whether to offer a cube, it compares its network's
   evaluation of *not* doubling against the value of doubling, which is
   the opponent's best reply from (1). Thus, the exact match equity
   enters this decision too.
3. When a game ends, the TD target for that state is the exact match
   equity at the new score, rather than a bootstrap from the network's
   own estimate.
4. When assessing a checker play that ends the game, the resulting
   position is priced by the exact match equity at the score that play
   produces.

**Second**, early in training we hit apparent plateaus, and breaking
through seemed to require some *oracle* rounds, which regress the model
directly toward the outputs of an oracle built on our p5jan model. While
that was only true early on, the champion's last 60+ million training
episodes were pure TD training with exact Bellman updates (`--oneply`).

### Components

| File | Purpose |
|---|---|
| `modes.py:MatchState` | score/cube/Crawford state machine; games chain via `start_game`/`after_game` |
| `encoding.py` | match encoders; current: `matchplay1h64met_perspective196` (337 in: one-hot aways to 64, one-hot cube, `post_crawford`) + 13 MET features |
| `td_agent.py` | match-play decisions in Python: 0-ply move selection, 1-ply checker values for targets, `offer_double_matchplay` |
| `trainer.py`, `train_batch.py`, `train_cli.py` | TD(0) self-play training; `--game-mode matchplay`, `--oneply`, `--met`, `--c-collect`, oracle-distillation flags |
| `c_api/puretd_match.c` | the match-play agent in C: encoder, cube rule, 0-ply play, 1-ply values, MET lookups — ports of the `td_agent.py` methods, checked row for row against them |
| `c_api/puretd_match_collect.c` | the C match collector (`bgc_collect_match`): plays whole matches and emits the rows and targets of the Python collector; `train_batch.py --c-collect` |
| `c_api/puretd_janowski.c` | p5jan's whole decision in C (prob5 forward, Janowski conversion, checker and cube choice); `eval_matchplay.py --fast-forward simd` runs a `p5jan:` side on it |
| `c_api/fast_forward.py` | the C forward backends as a drop-in for a net's torch forward (`--fast-forward simd`, `BG_FAST_FORWARD`); decisions stay in Python |
| `c_api/puretd_py.py` | ctypes binding of the C API (`Engine`, `collect_match`, `JanowskiC`); see `c_api/README.md` |
| `vectorized_matchplay.py` | lockstep batched Python self-play collector (`--vectorized-collect`) |
| `gpu_eval_server.py` | single-CUDA-context eval server for many CPU workers (`--gpu-eval-server`, `--oracle-eval-server`) |
| `cuda_graph_step.py` | CUDA-graph replay of the training step (`BG_TRAIN_CUDA_GRAPH`; on by default on a CUDA device) |
| `pool_guard.py`, `proc_utils.py` | worker-pool hygiene: a dead collection worker is detected and the round redone instead of hanging; workers die with their parent |
| `met.py` | match equity tables from gnubg's XML (`mwc`, `mwc_after`, `outcome_block`); `--met auto` is the bundled `met_tables/Kazaross-XG2.xml` |
| `janowski.py`, `janowski_match.py` | analytic cubeless→cubeful MWC conversion; port of gnubg's `Cl2CfMatch`/`GetPoints`; `JanowskiMatchAgent` = the `p5jan:` side |
| `gnubg_ids.py` | gnubg Position ID / Match ID encoders — paste any (state, match_state) into gnubg/XG |
| `gnubg_external.py` | real gnubg over its external socket (money and match play) |
| `eval_matchplay.py` | head-to-head match eval (`p5jan:`/`gnubg-ext[:N]`/`model:`/`dmp:` sides), `--paired` common-random-number dice, `--fast-forward`, `--raw-out` npz and `--combine` for pooling runs |
| `gnubg_eval_matchplay.py` | self-play mEMG at real match scores via gnubg batch analysis; multi-game `.mat` export |
| `tools/run_gnubg_mats.sh`, `tools/gnubg_vs_xg.py` | gnubg analysis of recorded `.mat` matches, one process per match, and the comparison with XG's per-match exports (the offline analysis above) |
| `verify_targets.py`, `audit_collector.py` | mechanical verification of collector targets/encodings |
| `c_api/matchplay_parity.py`, `c_api/match_collect_parity.py` | C-vs-Python parity on real match positions (decisions, collector rows) |
| `stats.py` | bootstrap confidence intervals and p-values |

**Design in brief.** An episode is one full match. The network outputs
P(win MATCH) from the on-roll player's view. Unlike the money setup,
`cube_value` never scales a target — it only drives score transitions.
Cube rule: double iff `min(v_dt, v_dp) > v_nd`; take iff
`v_dp > v_dt`. Targets: a sample-Bellman backup (0-ply, the roll that
happened) or an exact-Bellman backup (`--oneply`, the expectation over
all 21 rolls); game boundaries grounded at the new-game MWC (real MET
via `--met`, else the net's own).

### Models

| File | md5 (first 8) | Size | What |
|---|---|---|---|
| `best_models/matchplay_1h64met_512_512_256_256.pt` | `dbe2579a` | 2.44 MiB, 639,745 params, 350 inputs, hidden 512/512/256/256 | trained match-play champion (1h64 encoder + MET features) |
| `best_models/experimental/cubeless_prob5_512_512_256_256_256.pt` | `bade7d51` | 2.40 MiB, 627,717 params, 196 inputs, hidden 512/512/256/256/256 | money prob5 net one layer deeper, size-matched to the champion; with `janowski_match.py` on top = **p5jan**, the baseline the champion is compared against |
| `best_models/cubeless_prob5_512_512_256_128.pt` | `c8b2b265` | 2.02 MiB, 528,389 params, 196 inputs, hidden 512/512/256/128 | money prob5 net; with `janowski_match.py` on top = the earlier **p5jan**: the oracle/teacher of the oracle legs and the p5jan of the gnubg comparisons below |

The match-play agent's extra inputs, on top of the 196-feature money
encoding:

| feature | slots | |
|---|---|---|
| cube perspective | 3 | one-hot: centred / mine / theirs — inherited from the money cubeful encoder |
| `is_cube_action` | 1 | set at a cube decision — inherited |
| cube value | 7 | one-hot: 1, 2, 4, 8, 16, 32, >=64 |
| `my_away` | 64 | one-hot, thus supporting match lengths up to 64 |
| `opp_away` | 64 | one-hot |
| `crawford` | 1 | binary flag |
| `post_crawford` | 1 | binary flag |
| MET outcome prices | 13 | 6 at the current cube, 6 at twice it, 1 for the opponent dropping |
| **total** | **154** | 196 + 154 = 350 inputs |

### Results

- **Trained champion vs p5jan** (on the 5-layer prob5): **50.10%
  [50.05, 50.15]** of 11-point matches (n = 4M).
- **p5jan** (4-layer prob5 net + Janowski match cube layer, no match-play
  training at all) **beats gnubg-ext 0-ply 52.1% [51.4, 52.8]** and
  **ties gnubg 1-ply 50.2% [49.5, 50.9]** at 3pt (n=20k each, pinned
  source-built gnubg 1.08.003 with both bearoff DBs).
- The Janowski match cube conversion is verified against gnubg's own
  printed probabilities to **5.7e-7** (incl. Crawford, post-Crawford,
  2-away/2-away, redoubles): `tests/test_janowski_match_gnubg.py`.
  Cube decision quality vs a gnubg 2-ply referee is statistically
  indistinguishable from gnubg's own 0-ply (0.125 vs 0.107
  mMWC/decision, n=793).
- A p5jan-1-ply **oracle-distillation leg, in isolation, degrades**
  the model: −0.310pp [−0.465, −0.155], p = 0.0001 (400k matches). A
  following TD leg recovers it by ~200k episodes, and this combined
  training (oracle + TD) may help escape plateaus.
- **Speed** of the Python paths (single-threaded, 0-ply acting,
  identical positions): checker decisions are a wash — champion 1816/s
  vs p5jan 1773/s. On cube decisions p5jan is ~33% faster (8279/s vs
  6214/s): the champion needs two network forwards (`v_nd`, `v_dt`)
  plus a MET lookup, where p5jan takes one cubeless forward and derives
  all three branches analytically. Full 11-point matches run at ~66/s
  on 16 workers. The C paths are several times faster: `--fast-forward
  simd` moves the champion's net forward to C and runs p5jan's whole
  decision in C (about 4x per side), and the C collector
  (`--c-collect`) trains about 4x faster than the Python lockstep
  collector: 180-204 vs 48 episodes per second on a 64-core CPU with
  one GPU, each collector at its best worker count.

### Reproduce

```bash
. .venv/bin/activate   # per Quick Start below
export GNUBG_CMD=/usr/games/gnubg   # GNU Backgammon 1.08.003 (unset: /usr/local/bin/gnubg, then /usr/games/gnubg); only the gnubg commands need it

# champion vs p5jan (5-layer prob5), 11pt, paired dice, net forward and
# p5jan decisions in C; 20k matches resolve about +/-0.7pp (the Results
# figure pools 4M)
python eval_matchplay.py \
    --model1 best_models/matchplay_1h64met_512_512_256_256.pt \
    --model2 p5jan:best_models/experimental/cubeless_prob5_512_512_256_256_256.pt \
    --met auto --met2 auto --paired --fast-forward simd \
    --matches 20000 --match-length 11 --workers 8 --seed 1

# p5jan (4-layer prob5) vs gnubg (0-ply / 1-ply referee)
python eval_matchplay.py \
    --model1 p5jan:best_models/cubeless_prob5_512_512_256_128.pt \
    --model2 gnubg-ext --matches 20000 --match-length 3 --workers 8 --seed 7
#   ... --model2 gnubg-ext:1 for the 1-ply tie

# champion self-play scored by gnubg at the real match scores (mEMG per
# unforced decision); --keep-mats DIR also keeps the .mat files
python gnubg_eval_matchplay.py \
    --model best_models/matchplay_1h64met_512_512_256_256.pt --met auto \
    --matches 100 --match-length 11 --plies 2 --workers 8 --gnubg-workers 8

# collector audit
python verify_targets.py --model best_models/matchplay_1h64met_512_512_256_256.pt
```

### Training

The commands of the champion's lineage. They ran on earlier versions of
this code; where an option has changed since, the command is written for
this version with the same settings.

**From scratch, 11-point matches.** The run used the Python lockstep
collector. The C collector of the last legs (below) produces the same
training rows (the tests check it exactly against the Python collectors)
and is about 4x faster.

```bash
python train_batch.py --game-mode matchplay --match-length 11 \
    --hidden 512,512,256,256 --encoder matchplay1h64met_perspective196 \
    --output-mode probability \
    --oneply --met auto --met-scope all \
    --vectorized-collect --workers 16 --vectorized-batch 32 \
    --num-episodes 2000000 --episodes-per-round 2000 \
    --lr 1e-4 --end-lr 2.5e-6 --save models/mp.pt --save-every 100000
```

**Options worth knowing.**

| flag | effect |
|---|---|
| `--oneply` | exact-Bellman targets (21-dice enumeration). Drop it for sampled backups: ~10x faster per episode, noisier targets. |
| `--c-collect --c-collect-backend simd --c-collect-rng vectorized` | collect in C (`c_api/puretd_match_collect.c`): the rows and targets of the Python collector, about 4x faster. `simd` is the AVX2/AVX-512 forward with runtime dispatch (`torch`, `cblas`, `ref` are the others); `vectorized` draws each match's dice from its own stream, as `--vectorized-collect` does, so both play the same matches for the same `--seed` and `--workers` (`serial` = one stream per worker, the plain Python collector's order). A missing library is built on demand (`bash c_api/build.sh`). Requires `--boltzmann-temp 0` (the default) and `--met`. |
| `--oracle-targets NET` (with `--oracle-move-eps EPS`, `--oracle-plies`) | supervised distillation from the p5jan teacher, DAgger-style (Ross, Gordon & Bagnell 2011): states come from the student's own self-play trajectory and the student's value function still picks the moves — only the *targets* come from the teacher. `--oracle-move-eps` additionally supervises every action that **either** the teacher or the student rates within EPS of its own best, so moves the two disagree about still train even though greedy play never visits them. `--oracle-plies 1` prices targets by the teacher's 1-ply value; `--oracle-1ply-eps` gates which successors get that deeper pricing. For mature models, it tends to degrade the model on its own — see Results. |
| `--td-move-eps EPS` | also emit a training row for every legal successor within EPS (MWC) of the best 0-ply value, each targeted at its own 1-ply value. Does not change which move is played, so the state distribution is untouched. Used in the champion's lineage; never validated head-to-head. Python collectors only. |
| `--met-scope cube,terminal,boundary,all` | which consumers read the real MET. `cube`: the value of being dropped, in offer/take decisions. `terminal`: what a win/gammon/backgammon is worth at this score, when pricing game-ending moves. `boundary`: the TD target when a game ends. `all` enables all three. Any scope left out falls back to the network's own estimate. |
| `TRAINER_SCORE_CURRICULUM_P=P` (environment variable) | fraction of matches that start at a random away-away score rather than 0-0; default 0.5. There is no command-line flag for it. |
| `--vectorized-collect --vectorized-batch B` | the lockstep batched Python collector: each worker plays B matches at once and fuses their network forwards into one GPU call. Keep `--episodes-per-round >= workers x B` or the fused batches stay under-filled. |
| `--gpu-eval-server --gpu-eval-route-moves` | one CUDA context shared by all workers (~2-3x faster than the CPU Python collector). Not bitwise reproducible. |
| `--device cuda` | the training step on the GPU; collection stays on the CPU workers. `BG_TRAIN_CUDA_GRAPH` (default on) replays the step as a CUDA graph; `=0` runs the plain eager step. |
| `--lr` / `--end-lr` | set equal for a flat rate. The early oracle-cycle legs used flat 2.5e-6; the champion's later pure-TD legs annealed linearly. |
| `--save-every N` | a checkpoint at the first round boundary after every N episodes (`{--save}_ep{N}.pt`); the final net goes to `--save`. |

**Oracle round, then pure TD** — the cycle used early in the champion's
lineage, from an existing net. The oracle leg regresses toward p5jan
and is incompatible with `--oneply`/`--met`; it degrades the model on
its own (see Results), and the TD leg that follows recovers it:

```bash
# leg 1 -- p5jan 1-ply oracle. --oracle-1ply-eps gates which successors
# get the expensive deep pricing (~3x more episodes for the same wall time).
python train_batch.py --game-mode matchplay --match-length 11 \
    --resume models/mp.pt \
    --oracle-targets best_models/cubeless_prob5_512_512_256_128.pt \
    --oracle-plies 1 --oracle-move-eps 0.01 --oracle-1ply-eps 0.002 \
    --oracle-eval-server --workers 16 \
    --num-episodes 35000 --episodes-per-round 1000 \
    --lr 2.5e-6 --end-lr 2.5e-6 --save models/mp_o.pt --save-every 5000

# leg 2 -- pure TD on top, ~20x the episodes of leg 1
python train_batch.py --game-mode matchplay --match-length 11 \
    --resume models/mp_o.pt \
    --oneply --met auto --met-scope all \
    --gpu-eval-server --gpu-eval-route-moves --workers 16 \
    --num-episodes 600000 --episodes-per-round 2000 \
    --lr 2.5e-6 --end-lr 2.5e-6 --save models/mp_t.pt --save-every 100000
```

**Pure TD legs** — the recipe of the champion's later legs: 60+ million
episodes over 15 legs of 0.8M-20M episodes, each resuming from the one
before. Shown is the last leg, as run: 8.8M episodes with the C
collector, resuming from the previous leg's final checkpoint; its final
checkpoint is the champion file. Every match started at 0-0
(`TRAINER_SCORE_CURRICULUM_P=0.0`; the default 0.5 starts half of them
at a random score), and the training step ran as a CUDA graph
(`BG_TRAIN_CUDA_GRAPH=1`, which is the default on a CUDA device).

```bash
TRAINER_SCORE_CURRICULUM_P=0.0 BG_TRAIN_CUDA_GRAPH=1 \
python train_batch.py --game-mode matchplay --match-length 11 \
    --resume models/mp_prev.pt \
    --oneply --met auto --met-scope all \
    --c-collect --c-collect-backend simd --c-collect-rng vectorized \
    --boltzmann-temp 0 --workers 60 --episodes-per-round 4000 --device cuda \
    --num-episodes 8800000 --lr 2.88e-6 --end-lr 9.35e-7 --seed 8000 \
    --save models/mp_td.pt --save-every 20000
```

### Match equity table

`--met auto` resolves to the copy bundled at
`met_tables/Kazaross-XG2.xml` (GNU Backgammon's table, unmodified), so
no gnubg install is required; it falls back to gnubg's system table and
raises if neither is present. Train and evaluate against the same table
— scoring a model against a MET it was not trained on silently changes
every cube decision.

## Project README

A backgammon AI trained entirely through self-play [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning).  The main approach is temporal difference learning with a neural network using Tesauro's original feature encoding (see [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon)).  In addition to sampled Bellman backups, we implemented exact Bellman backups (sometimes called 1-ply backups), moving closer to an AlphaZero-style approach.  The framework supports multiprocessing and GPU acceleration to speed up training.  It covers 1-point matches (DMP), money games and, on this branch, N-point match play (above).

Cube action is learned via RL in the very natural way of simply introducing new actions (offer double, take or drop) to the agent and then learning as usual from self-play.  No formulas based on take-points are used by the trained agents; the Janowski formulae appear only in the p5jan baseline they are compared against.

As far as we're aware this is the first open-source backgammon AI trained entirely through self-play reinforcement learning (with the complete training pipeline included) to achieve near-SOTA 0-ply playing strength in DMP and cubeful money games.  It is also the first open-source implementation to learn cube action directly through self-play RL, although the approach is the same as described (but not evaluated for money games) by [Andrew Lin](https://ieeexplore.ieee.org/document/9382451/).

[BGBlitz](https://bgblitz.com/) is a notable non-open-source example of an extremely strong (competitive with or ahead of gnubg) backgammon AI trained with pure RL techniques (see their [technical presentation](https://bgblitz.com/download/blog/Aachen_BGBlitz.pdf)).

### How It Works

#### Training Pipeline

1. **Batch TD(0) self-play**: Play N (typically 1000) games with the current network, collecting (position, target) pairs
2. **Compute targets**: Either a sampled Bellman backup (`target = 1 - V(next_state)`, one dice roll) or an exact Bellman backup (`target = E_dice[max_move(1 - V(next))]`, averaging over all 21 dice)
3. **Train**: One epoch of minibatch SGD (Adam) on the collected data
4. **Repeat**: Collect new games with updated weights

#### Sampled vs Exact Bellman Backups

Standard TD(0) uses a **sampled Bellman backup** — the target depends on a single dice roll:
```
target = 1 - V(next_state)          # high variance: one dice roll
```

The 1-ply method computes the **exact Bellman backup**, averaging over all 21 dice outcomes:
```
target = E_dice[max_move(1 - V(next))]  # lower variance: all 21 dice
```

This eliminates the variance due to dice from the training signal. Terminal states are handled explicitly (exact value 0/1 instead of network estimate; in match play, the exact match equity at the new score).

The 1-ply value iteration approach was inspired by and validated against the "1-ply amplified equity estimate" from [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon).  As Hilton notes, this sits between TD-Gammon (sampled backups) and AlphaZero (deep search) in terms of target quality and cost.

Sampled backups (0-ply) are ~5x faster and effective for building a strong base model.
Exact backups (1-ply) are slower but when initialized with 0-ply-trained weights, they often improve results.

#### Cube decisions as actions

For cube play we add binary inputs to the 196-feature board encoding: a 3-way one-hot of the cube ownership (centered, mine, the opponent's) and `is_cube_action`, set to 1 when the agent faces a cube decision and 0 when it faces a checker decision. All cube decisions are treated as proper state-action-next_state transitions and trained with bootstrapped targets, exactly like checker plays. The match-play encoder inherits these four inputs and adds the match context (score, cube value, Crawford flags and the MET features listed above).

A similar approach is described by Andrew Lin (see above). However, published results from that paper are only on very small networks for match play and do not treat cube decisions as proper state-action transitions as our approach does (nor include an `is_cube_action` feature).

### Quick Start

#### Requirements

- Python 3.10–3.12 (the pinned gnubg-nn package, used only by the optional gnubg-nn sanity-check agents, has no wheels for newer interpreters)
- PyTorch + NumPy — pinned in `requirements.txt` to the versions that trained the champion
- GCC, for the C move generator (`c_engine/`) and the C API (`c_api/`: the match-play agent, the C collector and p5jan in C). Both build on demand on first use; the commands below build them explicitly.
- Optional: GNU Backgammon 1.08.003 (`gnubg` on PATH or `GNUBG_CMD`) for the `gnubg-ext` opponent and the mEMG analysis; its match equity table is bundled, so plain training and evaluation need no gnubg. OpenBLAS for the `cblas` backend.

```bash
# one-time setup; every `python` below is this venv's interpreter
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash c_engine/build_unix.sh && bash c_api/build.sh
```

#### Tests

```bash
BGC_MATCH_SHM=0 python -m pytest          # the default tier
BGC_MATCH_SHM=0 python -m pytest --slow   # also the long ones (training runs, full-size parity)
```

Tests marked `cuda` skip without a CUDA device. `BGC_MATCH_SHM=0` turns
off the C match engine's shared memo (a POSIX shared-memory table of
1-ply values that the worker processes of one training round share), so
that unrelated test processes never share one.

#### Inspect a model

```bash
python describe_model.py best_models/matchplay_1h64met_512_512_256_256.pt
```

Prints architecture, parameter count, encoder, and output mode for any saved `.pt` checkpoint.

### File Overview

| File | Description |
|------|-------------|
| `backgammon_engine.py` | Board representation and move generation (Python) |
| `c_engine/` | C implementation of move generation and the 196-feature encoding (~20x faster); built on import |
| `encoding.py` | Perspective encoding (196 features) and the cubeful and match-play encoders on top of it |
| `model.py` | PyTorch networks: configurable hidden layers; a sigmoid/linear scalar output (`TDNetwork`) or a 5-output prob5 head (`ProbNetwork`) |
| `modes.py` | Game modes (DMP, money, match play) with terminal handling; `MatchState` |
| `td_agent.py` | Agent wrapper for trained scalar-output models: checker play, cube decisions, 1-ply values |
| `agents.py` | Agent interface and RandomAgent |
| `trainer.py` | Trainer class: optimizer, round-based and online training loops, the collectors |
| `train_batch.py` | **Main training script**: batch TD(0) with optional exact (1-ply) backups |
| `train_cli.py` | Shared CLI arguments and network construction (fresh, resume, expand, warm-start) |
| `train_online.py` | Online TD(0) training (simpler, slower) |
| `vectorized_matchplay.py`, `gpu_eval_server.py`, `cuda_graph_step.py` | the GPU-side collection and training-step paths (see Components) |
| `pool_guard.py`, `proc_utils.py` | worker-pool supervision |
| `eval_matchplay.py` | Head-to-head match evaluation with parallel workers |
| `play_models.py` | Head-to-head evaluation for single games (DMP and money) |
| `gnubg_eval.py`, `gnubg_eval_matchplay.py` | Game recording, `.mat` export and gnubg batch analysis (money; match play) |
| `gnubg_external.py`, `gnubg_ids.py` | gnubg as an opponent over its external socket; gnubg Position/Match IDs |
| `met.py`, `met_tables/` | Match equity tables; the bundled Kazaross-XG2 table |
| `janowski.py`, `janowski_match.py` | Janowski cube formulae (money; match play) and the p5jan agent |
| `stats.py` | Bootstrap confidence intervals and p-values |
| `describe_model.py` | Print architecture and training info for a saved model |
| `export_weights.py`, `c_inference/` | Flat weight export and standalone C inference |
| `surgery_1h64.py`, `surgery_1h64met.py` | Function-preserving input surgery between the match encoders (used in the champion's lineage) |
| `audit_collector.py`, `verify_targets.py` | Independent re-verification of every collected training row |
| `mine_cube_errors.py`, `regate_cube_errors.py`, `train_from_position.py`, `probe_matchplay_cube.py` | Cube-decision diagnostics against gnubg's score-aware verdicts |
| `mps.sh` | Optional CUDA MPS helper for many-worker 0-ply collection |
| `c_api/` | PureTD C API (libpuretd): 0/1-ply checker play, money cube, self-play, the money and match collectors, p5jan in C, Python binding; see `c_api/README.md` |
| `tools/` | gnubg/XG offline analysis of recorded matches; benchmark helpers and the fixed position pools the tests use |
| `tests/` | pytest suite (`pytest.ini`, `conftest.py`) |

### Attribution

Inspired by and references:
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — 1-ply value iteration approach, OCaml implementation with experience replay
- [carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg) — Supervised-learning settings, topology, and approaches
- Gerry Tesauro's TDGammon and related work.
- GNUbg for all their wonderful work and especially for providing a baseline to compare to.

### License

MIT
