# Match play (N-point matches)

This branch contains work in progress: applying the PureTD approach to
match play. The network takes the match score and cube ownership as
inputs and outputs a single number — the match-winning chance for the
player on roll. Money play is documented in README.md; this file covers
only the match-play additions.

Against a baseline built the standard way (**p5jan**: our cubeless
prob5 model with Janowski's formulae for the cubeful adjustment), our
best model wins **49.91% [49.84, 49.98]** of 11-point matches
(n = 1.9M), so it is inferior but close (it is also ~25% slower than
the Janowski-based approach when making cube decisions). Nevertheless,
we believe it is meaningful progress toward learning cubeful play
directly. We have not detected convergence, and training of the
champion continues.

A couple of important details on our approach follow. **First**, we hand
the network MET-derived features (Kazaross-XG2): the match equity for
winning or losing a single, gammon or backgammon at the current score,
at both the current cube value and twice it. We also supply the exact match equity of the
resulting score at terminal states, in four places:

1. deciding to take or drop — the value of dropping;
2. deciding whether to offer — the opponent's take/drop, via (1);
3. when a game ends — the true match equity at the new score, used in
   the TD update;
4. assessing a game-winning checker play — the true match equity at the
   resulting score.

**Second**, training hit apparent plateaus, and breaking through seemed
to require some *oracle* rounds that regress the model directly toward
the outputs of an oracle based on our p5jan model.

## Components

| File | Purpose |
|---|---|
| `modes.py:MatchState` | score/cube/Crawford state machine; games chain via `start_game`/`after_game` |
| `encoding.py` | match encoders; current: `matchplay1h64met_perspective196` (337 in: one-hot aways to 64, one-hot cube, `post_crawford`) + 13 MET features |
| `td_agent.py` | match-play decisions: 0-ply move selection, 1-ply checker values for targets, `offer_double_matchplay` |
| `trainer.py`, `train_batch.py`, `train_cli.py` | TD(0) self-play training; `--game-mode matchplay`, `--oneply`, `--met`, oracle-distillation flags |
| `vectorized_matchplay.py` | lockstep batched self-play collector (`--vectorized-collect`) |
| `gpu_eval_server.py` | single-CUDA-context eval server for many CPU workers (`--gpu-eval-server`, `--oracle-eval-server`) |
| `met.py` | match equity tables from gnubg XML (`mwc`, `mwc_after`, `outcome_block`) |
| `janowski.py`, `janowski_match.py` | analytic cubeless→cubeful MWC conversion; port of gnubg's `Cl2CfMatch`/`GetPoints`; `JanowskiMatchAgent` = the `p5jan:` side |
| `gnubg_ids.py` | gnubg Position ID / Match ID encoders — paste any (state, match_state) into gnubg/XG |
| `gnubg_external.py` | real gnubg over its external socket (money and match play) |
| `eval_matchplay.py` | head-to-head match eval (`p5jan:`/`gnubg-ext[:N]`/`gnubg[:N]`/`model:`/`dmp:` sides), implicit-MET dump, `--raw-out` npz |
| `gnubg_eval_matchplay.py` | self-play mEMG at real match scores via gnubg batch analysis |
| `verify_targets.py`, `audit_collector.py` | mechanical verification of collector targets/encodings |

**Design in brief.** An episode is one full match. The network outputs
P(win MATCH) from the on-roll player's view. Unlike the money setup,
`cube_value` never scales a target — it only drives score transitions.
Cube rule: double iff `min(v_dt, v_dp) > v_nd`; take iff
`v_dp > v_dt`. Targets: a sample-Bellman backup (0-ply, the roll that
happened) or an exact-Bellman backup (`--oneply`, the expectation over
all 21 rolls); game boundaries grounded at the new-game MWC (real MET
via `--met`, else the net's own).

## Models

| File | md5 (first 8) | Size | What |
|---|---|---|---|
| `best_models/matchplay_1h64met_512_512_256_256.pt` | `aee85bd4` | 2.44 MB, 639,745 params, 350 inputs, hidden 512/512/256/256 | trained match-play champion (1h64 encoder + MET features) |
| `best_models/cubeless_prob5_512_512_256_128.pt` | `c8b2b265` | 2.02 MB, 528,389 params, 196 inputs, hidden 512/512/256/128 | money prob5 net; with `janowski_match.py` on top = **p5jan**, the strongest match player here and the oracle/teacher |

The champion's extra inputs are the match state: one-hot away scores
(64 slots per side), one-hot cube, `post_crawford`, and the 13 MET
outcome prices.

## Results (measured, current)

- **Trained champion vs p5jan**: **49.91% [49.84, 49.98]** of 11-point
  matches (n = 1.9M).
- **p5jan** (prob5 net + Janowski match cube layer, no match-play
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
  following TD leg recovers it by ~200k episodes but does not overtake
  plain TD.
- **Speed** (single-threaded, 0-ply acting, identical positions):
  checker decisions are a wash — champion 1816/s vs p5jan 1773/s. On
  cube decisions p5jan is ~33% faster (8279/s vs 6214/s): the champion
  needs two network forwards (`v_nd`, `v_dt`) plus a MET lookup, where
  p5jan takes one cubeless forward and derives all three branches
  analytically. So the two are near-equivalent in cost and the standing
  above is not bought with compute. Full 11-point matches run at ~66/s
  on 16 workers.

## Reproduce

```bash
. .venv/bin/activate   # per README Quick Start
export GNUBG_CMD=/usr/local/bin/gnubg   # pinned 1.08.003, both bearoff DBs

# p5jan vs gnubg (0-ply / 1-ply referee)
python eval_matchplay.py \
    --model1 p5jan:best_models/cubeless_prob5_512_512_256_128.pt \
    --model2 gnubg-ext --matches 20000 --match-length 3 --workers 8 --seed 7
#   ... --model2 gnubg-ext:1 for the 1-ply tie

# champion vs p5jan, 11pt
python eval_matchplay.py \
    --model1 best_models/matchplay_1h64met_512_512_256_256.pt --met auto \
    --model2 p5jan:best_models/cubeless_prob5_512_512_256_128.pt \
    --matches 20000 --match-length 11 --workers 8

# collector audit
python verify_targets.py --model best_models/matchplay_1h64met_512_512_256_256.pt
```

## Training

All commands below were run as shown.

**From scratch, 11-point matches.**

```bash
python train_batch.py --game-mode matchplay --match-length 11 \
    --hidden 512,512,256,256 --encoder matchplay1h64met_perspective196 \
    --output-mode probability \
    --oneply --met auto --met-scope all --score-curriculum 0.5 \
    --vectorized-collect --workers 16 --vectorized-batch 32 \
    --num-episodes 2000000 --episodes-per-round 2000 \
    --lr 1e-4 --end-lr 2.5e-6 --save models/mp.pt --save-every 100000
```

**Options worth knowing.**

| flag | effect |
|---|---|
| `--oneply` | exact-Bellman targets (21-dice enumeration). Drop it for sampled backups: ~10x faster per episode, noisier targets. |
| `--oracle-targets NET` (with `--oracle-move-eps EPS`, `--oracle-plies`) | supervised distillation from the p5jan teacher, DAgger-style (Ross, Gordon & Bagnell 2011): states come from the student's own self-play trajectory and the student's value function still picks the moves — only the *targets* come from the teacher. `--oracle-move-eps` additionally supervises every action that **either** the teacher or the student rates within EPS of its own best, so moves the two disagree about still train even though greedy play never visits them. `--oracle-plies 1` prices targets by the teacher's 1-ply value; `--oracle-1ply-eps` gates which successors get that deeper pricing. Degrades the model on its own — see Results. |
| `--td-move-eps EPS` | also emit a training row for every legal successor within EPS (MWC) of the best 0-ply value, each targeted at its own 1-ply value. Does not change which move is played, so the state distribution is untouched. Used in the champion's lineage; never validated head-to-head. |
| `--met-scope cube,terminal,boundary` | which consumers read the real MET. `cube`: the value of being dropped, in offer/take decisions. `terminal`: what a win/gammon/backgammon is worth at this score, when pricing game-ending moves. `boundary`: the TD target when a game ends. Any scope left out falls back to the network's own estimate. |
| `--score-curriculum P` | fraction of episodes starting at a random away-away score rather than 0-0. |
| `--vectorized-collect --vectorized-batch B` | lockstep batched self-play. Keep `--episodes-per-round >= workers x B` or the fused batches stay under-filled. |
| `--gpu-eval-server --gpu-eval-route-moves` | one CUDA context shared by all workers (~2-3x; see `benchmarks/matchplay_gpu_eval_server_ab.md` for the noise floor). Not bitwise reproducible. |
| `--lr` / `--end-lr` | set equal for a flat rate. The champion's legs used flat 2.5e-6. |

**Oracle round, then pure TD** — the two legs that produced the current
champion, from an existing net. The oracle leg regresses toward p5jan
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
    --oneply --met auto --met-scope all --score-curriculum 0.5 \
    --gpu-eval-server --gpu-eval-route-moves --workers 16 \
    --num-episodes 600000 --episodes-per-round 2000 \
    --lr 2.5e-6 --end-lr 2.5e-6 --save models/mp_t.pt --save-every 100000
```

## Match equity table

`--met auto` resolves to the copy bundled at
`met_tables/Kazaross-XG2.xml`, so no gnubg install is required; it falls
back to gnubg's system table and raises if neither is present. Train and
evaluate against the same table — scoring a model against a MET it was
not trained on silently changes every cube decision.
