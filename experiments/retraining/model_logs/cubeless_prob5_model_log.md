# Cubeless prob5 — model log

**Produces** `p5b_5g.pt` — a 5-output probability net (win / gammon-win /
bg-win / gammon-loss / bg-loss), `[512,512,256,128]`, encoder
`perspective196`, trained with `train_prob5.py`.
**Starts from** `probe_1e-5.pt` — the **step-2 intermediate** of
[`cubeless_money_4L_model_log.md`](cubeless_money_4L_model_log.md), *not*
that log's final output.

Twenty-two steps. Step 1 copies the equity base's first three hidden layers
(494,848 of prob5's 528,389 parameters, 93.7%); the final hidden layer
(narrowed 256 → 128) and the five-output head are initialised fresh. From
step 2 on, every odd step is a 1-ply leg and every even step is a 2M 0-ply
polish at constant 1e-6.

| # | Change | Ply | Ep. (M) | Start LR | End LR |
|---|---|---|---|---|---|
| 1 | equity `[512,512,256,256]` → prob5 `[512,512,256,128]` **(wu20)** | 0 | 5.0 | 1e-4 | 1e-5 |
| 2 | — (probe) | 1 | 1.0 | 1.6e-4 | const |
| 3 | — | 1 | 3.0 | 1.8e-4 | 6e-5 |
| 4 | — | 0 | 2.0 | 1e-6 | const |
| 5 | — | 1 | 3.0 | 1.8e-4 | 6e-5 |
| 6 | — | 0 | 2.0 | 1e-6 | const |
| 7 | — | 1 | 3.0 | 1.8e-4 | 6e-5 |
| 8 | — | 0 | 2.0 | 1e-6 | const |
| 9 | — | 1 | 3.0 | 1.8e-4 | 6e-5 |
| 10 | — | 0 | 2.0 | 1e-6 | const |
| 11 | — | 1 | 3.0 | 9e-5 | 3e-5 |
| 12 | — | 0 | 2.0 | 1e-6 | const |
| 13 | — | 1 | 3.0 | 9e-5 | 3e-5 |
| 14 | — | 0 | 2.0 | 1e-6 | const |
| 15 | — | 1 | 3.0 | 4.5e-5 | 1.5e-5 |
| 16 | — | 0 | 2.0 | 1e-6 | const |
| 17 | — (long anneal) | 1 | 15.3 | 1.5e-4 | 1.5e-5 |
| 18 | — | 0 | 2.0 | 1e-6 | const |
| 19 | — (anneal restart) | 1 | 10.8 | 1.0e-4 | 1.0e-5 |
| 20 | — | 0 | 2.0 | 1e-6 | const |
| 21 | — (anneal restart) | 1 | 7.8 | 9e-5 | 9e-6 |
| 22 | — | 0 | 2.0 | 1e-6 | const |

## Commands

`$MCM` is the cubeless-money model directory (for the step-1 input).

```bash
# 1  trunk warm-start off the cubeless equity net  (verify: the startup
#    block must print three "COPIED" layer lines and
#    "=> 494848/528389 params warm-started (93.7%)")
$PY train_prob5.py --warm-start-equity-trunk $MCM/probe_1e-5.pt \
  --hidden 512,512,256,128 \
  --num-episodes 5000000 --episodes-per-round 2000 \
  --lr 1e-4 --end-lr 1e-5 --warmup-cycles 20 \
  --workers 48 --device cuda --pipeline-collect --save $M/p5b_s1.pt

# 2  1-ply probe (constant LR)
CUDA_VISIBLE_DEVICES= $PY train_prob5.py --resume $M/p5b_s1.pt --oneply \
  --num-episodes 1000000 --episodes-per-round 1000 \
  --lr 1.6e-4 \
  --workers 64 --device cpu --pipeline-collect --save $M/p5b_probe.pt

# 3  1-ply
CUDA_VISIBLE_DEVICES= $PY train_prob5.py --resume $M/p5b_probe.pt --oneply \
  --num-episodes 3000000 --episodes-per-round 1000 \
  --lr 1.8e-4 --end-lr 6e-5 \
  --workers 64 --device cpu --pipeline-collect --save $M/p5b_c1_1ply.pt

# 4  0-ply polish
$PY train_prob5.py --resume $M/p5b_c1_1ply.pt \
  --num-episodes 2000000 --episodes-per-round 2000 \
  --lr 1e-6 --workers 48 --device cuda --pipeline-collect --save $M/p5b_c1.pt

# 5/6, 7/8, 9/10  three more cycles of the SAME pair of commands,
#    resuming each leg from the previous cycle's polish output:
#    c2: --resume p5b_c1.pt  -> p5b_c2_1ply.pt -> p5b_c2.pt   (1.8e-4 -> 6e-5)
#    c3: --resume p5b_c2.pt  -> p5b_c3_1ply.pt -> p5b_c3.pt   (1.8e-4 -> 6e-5)
#    c4: --resume p5b_c3.pt  -> p5b_c4_1ply.pt -> p5b_c4.pt   (1.8e-4 -> 6e-5)

# 11/12, 13/14  two cycles at the halved peak:
#    c5: --resume p5b_c4.pt  -> ... --lr 9e-5 --end-lr 3e-5 ... -> p5b_c5.pt
#    c6: --resume p5b_c5.pt  -> ... --lr 9e-5 --end-lr 3e-5 ... -> p5b_c6.pt

# 15/16  one cycle at the quartered peak:
#    c7: --resume p5b_c6.pt  -> ... --lr 4.5e-5 --end-lr 1.5e-5 ... -> p5b_c7.pt

# 17  long anneal — one monotone decade over 15.3M
CUDA_VISIBLE_DEVICES= $PY train_prob5.py --resume $M/p5b_c7.pt --oneply \
  --num-episodes 15300000 --episodes-per-round 1000 \
  --lr 1.5e-4 --end-lr 1.5e-5 \
  --workers 64 --device cpu --pipeline-collect \
  --save-every 1000000 --save $M/p5b_la_1ply.pt

# 18  0-ply polish
$PY train_prob5.py --resume $M/p5b_la_1ply.pt \
  --num-episodes 2000000 --episodes-per-round 2000 \
  --lr 1e-6 --workers 48 --device cuda --pipeline-collect --save $M/p5b_la.pt

# 19  anneal restart
CUDA_VISIBLE_DEVICES= $PY train_prob5.py --resume $M/p5b_la.pt --oneply \
  --num-episodes 10800000 --episodes-per-round 1000 \
  --lr 1.0e-4 --end-lr 1.0e-5 \
  --workers 64 --device cpu --pipeline-collect \
  --save-every 500000 --save $M/p5b_la2_1ply.pt

# 20  0-ply polish
$PY train_prob5.py --resume $M/p5b_la2_1ply.pt \
  --num-episodes 2000000 --episodes-per-round 2000 \
  --lr 1e-6 --workers 48 --device cuda --pipeline-collect --save $M/p5b_la2.pt

# 21  anneal restart
CUDA_VISIBLE_DEVICES= $PY train_prob5.py --resume $M/p5b_la2.pt --oneply \
  --num-episodes 7800000 --episodes-per-round 1000 \
  --lr 9e-5 --end-lr 9e-6 \
  --workers 64 --device cpu --pipeline-collect \
  --save-every 400000 --save $M/p5b_5g_1ply.pt

# 22  final 0-ply polish
$PY train_prob5.py --resume $M/p5b_5g_1ply.pt \
  --num-episodes 2000000 --episodes-per-round 2000 \
  --lr 1e-6 --workers 48 --device cuda --pipeline-collect --save $M/p5b_5g.pt
```

## Notes for retraining

- **Step 1 must actually warm-start.** The trunk copy is what makes the
  chain this episode-efficient; confirm the `COPIED` lines and the 93.7%
  figure in the startup block before spending the 1-ply budget.
- **`--warmup-cycles 20` appears in step 1 only.** It scales the scheduled
  LR linearly from 10% to 100% over the first 20 rounds (40k episodes at
  `--episodes-per-round 2000`). No other step uses warm-up.
- **1-ply legs are CPU-only** (`--device cpu`, `CUDA_VISIBLE_DEVICES=`).
  prob5 1-ply GPU collection was built, validated, and measured slower than
  the CPU pipeline (`experiments/benchmarks/prob5_collect_levers_ab.md` on
  the release branch); without the env guard, torch also pins ~500 MB of
  GPU per worker for nothing. 0-ply steps use `--device cuda` (+82% on the
  train step).
- **After step 1, 0-ply is polish only.** Extending 0-ply past the 5M seed
  at higher LR made the model measurably worse; every 0-ply step from 4 on
  is 2M at constant 1e-6, unchanged across the whole chain.
- **Episodes-per-round is 1000 at 1-ply and 2000 at 0-ply** throughout.
  Worker counts (64 CPU / 48 cuda) reflect the reference box, not a recipe
  choice.
- The LR anneal is linear over the *declared* `--num-episodes` (same
  semantics as `train_batch.py`); every leg here runs to its declared
  length, so end LRs are as listed.
- **Steps 19 and 21 each buy well under 1.5 mEq.** Verifying them takes a
  10,000,000-game head-to-head (±0.85 mEq) against a fixed reference; a
  4M-game gate (±1.35) reads both legs as null even when the gain is real.
