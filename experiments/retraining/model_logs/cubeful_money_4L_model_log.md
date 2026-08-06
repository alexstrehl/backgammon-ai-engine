# Cubeful money 4L — model log

**Produces** `cf4L_e7.pt` (step 17) — a cubeful-money equity net,
`[512,512,256,256]`, encoder `cubeful_perspective196`, 200 inputs.
**Starts from** `cm4L_c2_ep4000000.pt`
([`cubeless_money_4L_model_log.md`](cubeless_money_4L_model_log.md)).

Step 1 widens the input layer 196 → 200 — three cube-ownership indicators plus
`is_cube_action`, initialised fresh — while the hidden layers and the equity
head are copied unchanged.

| # | Ply | Ep. (M) | Start LR | End LR | |
|---|---|---|---|---|---|
| 1 | 0 | 2.0 | 1e-4 | const | cubeless → cubeful, 196 → 200 inputs (wu20) |
| 2 | 0 | 15.0 | 6e-5 | **1.53e-5 effective** | see note |
| 3 | 0 | 8.0 | 5e-5 | 5e-7 | |
| 4 | 1 | 2.5 | 1.8e-5 | const | |
| 5 | 0 | 9.0 | 2e-6 | const | polish |
| 6 | 1 | 8.0 | 1.234e-5 | 1e-6 | |
| 7 | 0 | 9.0 | 2e-6 | const | polish |
| 8 | 1 | 7.0 | 1.11e-5 | 9e-7 | |
| 9 | 0 | 9.0 | 2e-6 | const | polish — **last one; see note** |
| 10 | 1 | 21.0 | 1e-5 | 2e-7 | (wu30) |
| 11 | 1 | 4.0 | 9e-6 | const | (wu20) |
| 12 | 1 | 4.0 | 9e-6 | const | (wu20) |
| 13 | 1 | 12.5 | 1.8e-5 | const | (wu20) |
| 14 | 1 | 13.0 | 1.69e-5 | const | (wu20) |
| 15 | 1 | 11.5 | 1.69e-5 | 1.4e-6 | (wu20) — first GPU leg |
| 16 | 1 | 19.0 | 1.606e-5 | 1.33e-6 | (wu20) |
| 17 | 1 | 22.5 | 1.4454e-5 | 1.197e-6 | (wu20) — **final** |

## Where it got to

Capped ±128 mEq/game vs gnubg-ext, `--plies 0`:

| model | vs gnubg-ext | games |
|---|---|---|
| step 9 | +59.9 [57.0, 62.7] | 4M |
| step 10 | +66.5 [63.6, 69.3] | 4M |
| step 13 | +73.2 [70.9, 75.6] | 6M |
| step 14 | +75.3 [73.0, 77.7] | 6M |
| production (reference) | +78.7 [76.9, 80.5] | 10M |

**Step 15 vs production, direct: −1.1 mEq [−2.9, +0.7], p=0.235 at 10M games** —
the interval contains zero. Chained arithmetic (−0.9) and direct measurement
(−1.1) agree to 0.2 mEq.

## Commands

```bash
# 1  warm-start cubeful off the cubeless equity net
$PY train_batch.py --game-mode cubeful-money \
  --warm-start-cubeful $MCM/cm4L_c2_ep4000000.pt \
  --num-episodes 2000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-4 --warmup-cycles 20 \
  --workers 32 --device cuda --bf16-collect --pipeline-collect \
  --save-every 1000000 --save $M/cf4L_s0a.pt

# 2  START AS A 20M LEG, STOP AT 15M (see note). Use cf4L_s0bc_ep15000000.pt.
$PY train_batch.py --game-mode cubeful-money --resume $M/cf4L_s0a.pt \
  --num-episodes 20000000 --episodes-per-round 1000 --optimizer adam \
  --lr 6e-5 --end-lr 4e-7 \
  --workers 32 --device cuda --bf16-collect --pipeline-collect \
  --save-every 1000000 --save $M/cf4L_s0bc.pt

# 3  anneal restart from step 2's 15M checkpoint
$PY train_batch.py --game-mode cubeful-money \
  --resume $M/cf4L_s0bc_ep15000000.pt \
  --num-episodes 8000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-7 \
  --workers 32 --device cuda --bf16-collect --pipeline-collect \
  --save-every 1000000 --save $M/cf4L_s0e.pt

# 4  first 1-ply leg. START AS 8M, CUT AT 2.5M (plateau). LR is const, so the
#    cut does not distort the schedule. Use cf4L_c1_1ply_ep2500000.pt.
env CUDA_VISIBLE_DEVICES= $PY train_batch.py --game-mode cubeful-money \
  --resume $M/cf4L_s0e_ep8000000.pt --oneply --device cpu \
  --num-episodes 8000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1.8e-5 --end-lr 1.8e-5 \
  --workers 40 --pipeline-collect --c-engine \
  --save-every 500000 --save $M/cf4L_c1_1ply.pt

# 5  0-ply polish
$PY train_batch.py --game-mode cubeful-money --resume $M/cf4L_c1_1ply.pt \
  --num-episodes 9000000 --episodes-per-round 1000 --optimizer adam \
  --lr 2e-6 --end-lr 2e-6 \
  --workers 32 --device cuda --pipeline-collect \
  --save-every 3000000 --save $M/cf4L_c1.pt

# 6  1-ply
env CUDA_VISIBLE_DEVICES= $PY train_batch.py --game-mode cubeful-money \
  --resume $M/cf4L_c1.pt --oneply --device cpu \
  --num-episodes 8000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1.234e-5 --end-lr 1e-6 \
  --workers 40 --pipeline-collect --c-engine \
  --save-every 500000 --save $M/cf4L_c2_1ply.pt

# 7  0-ply polish
$PY train_batch.py --game-mode cubeful-money --resume $M/cf4L_c2_1ply.pt \
  --num-episodes 9000000 --episodes-per-round 1000 --optimizer adam \
  --lr 2e-6 --end-lr 2e-6 \
  --workers 32 --device cuda --pipeline-collect \
  --save-every 3000000 --save $M/cf4L_c2.pt

# 8  1-ply
env CUDA_VISIBLE_DEVICES= $PY train_batch.py --game-mode cubeful-money \
  --resume $M/cf4L_c2.pt --oneply --device cpu \
  --num-episodes 7000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1.11e-5 --end-lr 9e-7 \
  --workers 40 --pipeline-collect --c-engine \
  --save-every 500000 --save $M/cf4L_c3_1ply.pt

# 9  0-ply polish -- the last one. Do not add polish after this step.
$PY train_batch.py --game-mode cubeful-money --resume $M/cf4L_c3_1ply.pt \
  --num-episodes 9000000 --episodes-per-round 1000 --optimizer adam \
  --lr 2e-6 --end-lr 2e-6 \
  --workers 32 --device cuda --pipeline-collect \
  --save-every 3000000 --save $M/cf4L_c3.pt

# 10-14  1-ply legs, CPU c-engine path. Same shape each time:
#   10: --resume cf4L_c3.pt        -N 21000000 --lr 1e-5    --end-lr 2e-7  --warmup-cycles 30
#   11: --resume cf4L_e1_1ply.pt   -N  4000000 --lr 9e-6    --end-lr 9e-6
#   12: --resume cf4L_e2a.pt       -N  4000000 --lr 9e-6    --end-lr 9e-6
#   13: --resume cf4L_e2b.pt       -N 12500000 --lr 1.8e-5  --end-lr 1.8e-5
#   14: --resume cf4L_e3.pt        -N 13000000 --lr 1.69e-5 --end-lr 1.69e-5
env CUDA_VISIBLE_DEVICES= $PY train_batch.py --game-mode cubeful-money \
  --resume $M/<parent> --oneply --device cpu \
  --num-episodes <N> --episodes-per-round 1000 --optimizer adam \
  --lr <LR> --end-lr <END_LR> --warmup-cycles 20 \
  --workers 40 --pipeline-collect --c-engine \
  --max-episode-plies 1500 --save-every 500000 --save $M/<out>.pt

# 15-17  same legs on the GPU vectorized collector (~1.9x throughput, half the
#        cores). Identical results; only the collector differs.
#   15: --resume cf4L_e4.pt -N 11500000 --lr 1.69e-5   --end-lr 1.4e-6
#   16: --resume cf4L_e5.pt -N 19000000 --lr 1.606e-5  --end-lr 1.33e-6
#   17: --resume cf4L_e6.pt -N 22500000 --lr 1.4454e-5 --end-lr 1.197e-6
LD_PRELOAD=$PWD/c_engine/libfakeintel.so CUDA_VISIBLE_DEVICES=0 \
$PY -u train_batch.py --game-mode cubeful-money \
  --resume $M/<parent> --oneply --device cuda \
  --num-episodes <N> --episodes-per-round 1000 --optimizer adam \
  --lr <LR> --end-lr <END_LR> --warmup-cycles 20 \
  --workers 24 --vectorized-collect --vectorized-batch 32 \
  --max-episode-plies 1500 --save-every 500000 --save $M/<out>.pt
```

Per-leg validation — head-to-head against the immediately preceding leg:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LD_PRELOAD= $PY \
  experiments/head_to_head/run.py --side1 $M/<new>.pt --side2 $M/<parent>.pt \
  --game-mode cubeful-money --games 8000000 --workers 48
```

## Notes for retraining

- **Step 2's end LR is not 4e-7.** The anneal is linear over the *declared*
  `--num-episodes` (`trainer.py:822`), so a 20M schedule stopped at 15M only
  traverses 75% of the ramp and lands at **1.53e-5**. Declaring
  `--num-episodes 15000000 --end-lr 4e-7` runs the entire leg at a
  systematically lower LR and is a different run. Exact equivalent:
  `--num-episodes 15000000 --lr 6e-5 --end-lr 1.53e-5`. Step 4 is also cut
  short but is const-LR, so it is unaffected.
- **Do not add 0-ply polish after step 9.** Polish *is* 0-ply training, and
  0-ply saturates. Measured twice, on two legs at two LRs: direct 4M H-H
  **−2.3 mEq [−5.0, +0.5]**, corpus oracle **−1.9 [−2.6, −1.2]**. It was
  dropped from step 10 onward. Steps 5/7/9 keep it only because they are part
  of the recorded recipe.
- **Validate at ≥8M games.** σ ≈ 2837 mEq/game, so MDE at 80% power is 2.81 at
  8M but 3.97 at 4M — *larger than a typical leg's gain*. Step 11 first read as
  a null at 4M and only showed +2.2 when steps 11+12 were read cumulatively at
  8M. Measure cumulatively, not per-round.
- **Don't extend past step 17 on this schedule.** Steps 16 and 17 both measured
  null against their parents (+1.2 [−0.7, +3.2] and +1.6 [−0.4, +3.5] at 8M
  games) — 41.5M episodes for a pooled ~+2.8, against the +9.7 that steps
  13–15 bought over 37M. Step 15 already reached parity with the production
  model. If you want more, change a variable (a genuinely different LR, or
  2-ply targets); more episodes at this LR have stopped paying.
- **LR barely matters on a converged trunk; episodes are the lever** (through
  step 15 — see the note above for where this stops holding). const
  9e-6, const 1.8e-5, const 1.69e-5, anneal 1e-5→2e-7 and anneal 1.69e-5→1.4e-6
  all yield **~2.2–2.9 mEq per 10M 1-ply episodes**. Rate tracks how converged
  the model is, not the learning rate. Nothing in the schedule beat simply
  running more 1-ply episodes.
- **`--warmup-cycles` counts ROUNDS.** At `--episodes-per-round 1000`, `20` =
  20k episodes. Keep EPR at 1000 for the GPU path: the 555 ep/s figure was
  measured with it, the tail-drain penalty is already priced in, and raising it
  doubles target staleness *and* silently rescales warmup.
- **`--max-episode-plies 1500`** discards non-terminating self-play games. It
  never fired on a healthy leg here (0 discards across steps 15–17), but a
  badly-seeded net can reach a mutual-hitting fixed point with 3000+ ply games
  and exhaust RAM. When it fires it appends
  `| discarded N overlong episodes (>1500 plies)` to the round line.
- **GPU leg tuning:** `--workers 24` is the optimum, not a ceiling — an
  idle-box sweep gave W=16 472.8, W=20 527.8, W=24 555.7, W=26/28 flat, and
  VRAM was not the constraint. Equivalence to the CPU path was validated: a
  500k GPU arm scored +0.69 [+0.14, +1.25] against its CPU-trained twin.
- Step 14 actually ran as 10.5M + a 2.5M resume after an interruption; both
  const 1.69e-5, so a single 13.0M leg reproduces it. Only the first 10.5M
  carried warmup.
