# Small models

Four small nets (one and two hidden layers) for cheap inference, distilled from larger nets and
then trained with 1-ply TD, a three-layer prob5 net trained by TD alone, and
a three-layer cubeful net distilled from the 6L cubeful net and then trained with 1-ply TD.
They are weaker than the production 4-layer nets and trade strength for size. All are plain
`[in, hidden..., out]` MLPs loadable the same way as the production files.

| file | mode | hidden | params | md5 |
|---|---|---|---|---|
| `cubeless_prob5_512_512.pt` | cubeless prob5 | [512, 512] | 366k | `e8c0c760` |
| `cubeless_prob5_512.pt` | cubeless prob5 | [512] | 103k | `fa2c6935` |
| `cubeful_money_512_512.pt` | cubeful-money | [512, 512] | 366k | `1ddbb775` |
| `cubeful_money_512.pt` | cubeful-money | [512] | 103k | `39d75783` |
| `cubeless_prob5_512_512_256.pt` | cubeless prob5 | [512, 512, 256] | 496k | `3da8e498` |
| `cubeful_money_512_512_256.pt` | cubeful-money | [512, 512, 256] | 497k | `27a728cf` |

## Results (mEq/game, 95% bootstrap CIs; cubeful reads are capped means, ±128)

| model | vs gnubg 0-ply (1M) | vs production 4L (10M) | vs its teacher |
|---|---|---|---|
| `cubeless_prob5_512_512` | **+32.5** [+29.8, +35.3] | -12.1 [-13.0, -11.3] | same as prod column (teacher = prod 4L) |
| `cubeless_prob5_512` | -35.7 [-38.3, -33.0] | -82.6 [-83.5, -81.8] | same as prod column |
| `cubeful_money_512_512` | **+57.6** [+51.9, +63.2] | -25.1 [-26.9, -23.3] | -34.2 [-36.9, -31.4] vs the 6L `cubeful_money_768_512_256_256_256_256` (4M) |
| `cubeful_money_512` | -79.1 [-85.0, -73.3] | -157.5 [-159.6, -155.3] | -159.2 [-162.3, -156.2] (4M) |
| `cubeless_prob5_512_512_256` | — | **-3.7** [-4.5, -2.8] | not distilled (see below) |
| `cubeful_money_512_512_256` | — | **-4.5** [-6.3, -2.7] | distilled from the 6L cubeful net (see below) |

The production column is in [h2h_experimental_vs_prod.csv](../../../experiments/head_to_head/results/h2h_experimental_vs_prod.csv). Per-game raws are not committed.

Reference points on the gnubg scale: the production prob5 4L reads +45.8, the 6L cubeful teacher about +91.
The two-layer nets beat gnubg 0-ply. The one-layer nets do not. The production comparisons are against
`cubeless_prob5_512_512_256_256.pt` (`ac95a983`) and `cubeful_money_512_512_256_256.pt` (`6afda0f8`),
10,000,000 games each.

## The three-layer prob5 net

`cubeless_prob5_512_512_256.pt` was not distilled. Its trunk was seeded from the recipe's 4-layer
cubeless-money net (`train_prob5.py --warm-start-equity-trunk`, three-layer prefix copied, fresh head;
1.5M 0-ply episodes at 1e-4 -> 1e-5 with `--warmup-cycles 20`), then trained 1-ply for 52M episodes in
four legs (10.5M at 1.8e-4 -> 2e-6, then restarts of 11.5M at 9e-5, 13.5M at 4.5e-5 and 16.2M at 9e-5,
each annealed to 2e-6).

## The three-layer cubeful net

`cubeful_money_512_512_256.pt` started from `best_models/cubeful_money_512_512_256.pt` (`799f8c75`,
the same shape). It was distilled for 150 minutes from a 6L cubeful net `[768,512,256,256,256,256]`
(an uncommitted checkpoint): the teacher's equity as the regression target on
positions from teacher self-play, one quarter of the rounds from student self-play, 4,000 episodes per
round, batch 4096, lr 3e-4 -> 1e-6. Then two 1-ply TD legs (the `train_batch.py` command of the recipe
below): 12M episodes at 1.8e-5 -> 3e-7 with `--warmup-cycles 20`, then 30M at 1.62e-5 -> 2.7e-7.

## Recipe

Tools: `tools/collect_positions.py` (self-play position banks) and `tools/distill.py` (student <- teacher
0-ply labels; BCE for prob5, MSE for equity; `--init-from` copies the leading shape-matching hidden
layers). The TD legs are plain `train_prob5.py` / `train_batch.py` runs. Everything runs from the repo
root with the C engine built.

1. **Distillation** (10 DAgger rounds). Teachers: prob5 = production `cubeless_prob5_512_512_256_256.pt`;
   cubeful = the 6L `best_models/experimental/cubeful_money_768_512_256_256_256_256.pt`. Round 0: student initialised
   from the leading layers of the 4L production net of its kind, fit to a teacher-acting bank. Round K:
   collect a bank with the round K-1 student acting (50-60k games, a fresh seed) and re-distil from that
   student on the round-0 bank plus the last three student banks; ~10M positions in all.
   Example commands (illustrative values):

       python tools/collect_positions.py --model $TEACHER --games 60000 --seed 1 --out rows/r0.npz
       python tools/distill.py --teacher $TEACHER --init-from $PROD_4L --hidden 512,512 \
           --rows rows/r0.npz --epochs 20 --lr 1e-3 --end-lr 1e-5 --out models/d0.pt
       python tools/collect_positions.py --model models/d0.pt --games 60000 --seed 1001 --out rows/r1.npz
       python tools/distill.py --teacher $TEACHER --resume models/d0.pt --rows rows/r0.npz rows/r1.npz \
           --epochs 10 --lr 3e-4 --end-lr 3e-6 --out models/d1.pt

2. **Descent**: 8M 1-ply episodes from a start LR of 1e-4 (prob5 [512,512]), 5e-5 (prob5 [512]) or
   1e-5 (cubeful), linearly annealed to 1/50 of the start. The start LR is the better of two
   1M-episode constant-LR probes by head-to-head.
3. **Restart**: 2M 1-ply episodes from 0.5x the descent start to 1/10 of that.

   A 1-ply leg, run with `LD_PRELOAD=c_engine/libfakeintel.so`,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `OMP_NUM_THREADS=1`:

       python train_prob5.py --resume IN.pt --oneply --num-episodes EPS --episodes-per-round 1000 \
           --lr LR --end-lr END_LR --workers 24 --device cuda --vectorized-collect 32 \
           --save-every 1000000 --save OUT.pt
       python train_batch.py --game-mode cubeful-money --resume IN.pt --oneply --num-episodes EPS \
           --episodes-per-round 1000 --optimizer adam --lr LR --end-lr END_LR --warmup-cycles 0 \
           --workers 24 --vectorized-workers 24 --device cuda --vectorized-collect \
           --vectorized-shared-context --vectorized-batch 32 --max-episode-plies 1500 \
           --save-every 1000000 --save OUT.pt

   The trainers' checkpoints are weights-only, so an interrupted leg is resumed by restating its tail
   as a new leg on the same LR slope: after N of `eps` episodes, `lr0' = lr0 + (lr1-lr0)*N/eps` and
   `eps' = eps-N`.

Each leg is gated against its parent by head-to-head (2M games cubeless, 4M cubeful). The best
checkpoint is kept, not the last. The restart improved the one-layer prob5 net (+2.2 [+0.3, +4.1]) and
tied on the two-layer nets, whose files are the descent checkpoints. Gain of the whole TD phase over
the distilled start, by direct head-to-head (per-game raws not committed): prob5 [512,512]
+2.6 [+0.7, +4.5]; prob5 [512] +9.0 [+7.1, +10.9]; cubeful [512] +12.2 [+9.1, +15.4]; cubeful [512,512]
+1.0 [-1.7, +3.7].
