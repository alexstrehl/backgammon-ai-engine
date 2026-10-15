# A/B twin gate (parity level 4)

Two trainings per ply from `best_models/cubeful_money_512_512_256_256.pt` with the
same hyperparameters, seed, worker count (48) and environment; only the collector
differs: C (`simd`) against the Python CPU collector it reproduces episode for
episode. 1M episodes each, epr 1000, `--pipeline-collect`, Adam 1.62e-5 -> 2.7e-7
(the `cf_restart` schedule compressed to 1M), `BG_POOL_ON_DEVICE=1`, eager train step
and pool pipe. Threadripper 9980X, RTX 5090.

```bash
COMMON="--game-mode cubeful-money --resume best_models/cubeful_money_512_512_256_256.pt \
  --oneply --num-episodes 1000000 --optimizer adam --lr 1.62e-5 --end-lr 2.7e-7 \
  --max-episode-plies 1500 --workers 48 --episodes-per-round 1000 --pipeline-collect \
  --seed 20261010 --torch-seed 20261010 --device cuda --log-every 50 --save-every 250000"
ENV="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BG_POOL_ON_DEVICE=1 BG_TRAIN_CUDA_GRAPH=0 \
  BG_COLLECT_SHM=0 LD_PRELOAD=$PWD/c_engine/libfakeintel.so"
C0="${COMMON/--oneply /}"
$ENV .venv/bin/python -u train_batch.py $COMMON --c-collect --save $OUT/c/cf_ab.pt
$ENV .venv/bin/python -u train_batch.py $COMMON --no-c-collect --no-vectorized-collect \
  --save $OUT/py/cf_ab.pt
$ENV .venv/bin/python -u train_batch.py $C0 --c-collect --save $OUT/c0/cf_ab.pt
$ENV .venv/bin/python -u train_batch.py $C0 --no-c-collect --save $OUT/py0/cf_ab.pt
```

| arm | collector | wall | final md5 |
|---|---|---:|---|
| c | C, 1-ply targets | 10 min (1650 ep/s) | 24b7cdd4 |
| py | Python CPU, 1-ply targets | 85 min (195 ep/s) | 18a0964f |
| c0 | C, 0-ply | 3 min | 4d578fea |
| py0 | Python CPU, 0-ply | 6 min | 185e0e20 |

Each log: 20 round lines, no DISCARDED or WARN lines; `C collection: ON` in the C
arms.

## Head-to-head: C-trained model (Model1) vs Python-trained model

```bash
.venv/bin/python -u experiments/head_to_head/run.py --side1 $OUT/c/cf_ab.pt \
  --side2 $OUT/py/cf_ab.pt --game-mode cubeful-money --jacoby --games 1000000 \
  --workers 48 --paired-dice --save $OUT/hh_1ply
```

0-ply pair: `--side1 $OUT/c0/cf_ab.pt --side2 $OUT/py0/cf_ab.pt --save $OUT/hh_0ply`.
0-ply play on both sides.

| pair | equity, Model1, mEq/game | 95% CI (pair bootstrap) | p | win rate |
|---|---:|---|---:|---|
| 1-ply (c vs py) | **-1.0** | [-3.6, +1.6] | 0.44 | 49.9% / 50.1% |
| 0-ply (c0 vs py0) | **+2.8** | [-1.6, +7.1] | 0.20 | 50.6% / 49.4% |

Both estimates are inside the 1M noise band (+-5.6 mEq unpaired; the paired CIs are
+-2.6 and +-4.4), neither is significant on equity, and the signs differ across the
plies: no systematic collector effect. PASS.

After 1M episodes, arms that differ by <= 4e-6 in a target and part on a checker
near-tie about once per 4,500 games diverge like two seeds, so the twin measures
seed-level noise. The pairing makes the win-rate CI (+-0.02 pp) far narrower than the
equity CI, which is why `run.py` flags the 1-ply win rate while the equity is flat.
