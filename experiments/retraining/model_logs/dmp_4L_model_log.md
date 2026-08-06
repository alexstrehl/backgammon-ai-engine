# DMP 4L — model log

**Produces** `dmp_512_512_256_256_repro.pt` — a double-match-point net,
`[512,512,256,256]`, encoder `perspective196`.
**Starts from** nothing (random init at step 1).
**Consumed by** [`cubeless_money_4L_model_log.md`](cubeless_money_4L_model_log.md).

Steps 1–15 grow the net through the progressive expansion.

Run from the repo root with `PY=.venv/bin/python`; `$M` is your model dir.
Reference box: Threadripper 9980X (64 physical cores) + RTX 5090.

| # | Change | Ply | Ep. (M) | Start LR | End LR |
|---|---|---|---|---|---|
| 1 | → [80] | 0 | 1.0 | 1e-3 | const |
| 2 | width → [150] | 0 | 0.5 | 5e-5 | const |
| 3 | depth → [150,150] | 0 | 0.5 | 5e-5 | const |
| 4 | width → [256,256] | 0 | 1.0 | 5e-5 | const |
| 5 | width → [512,512] | 0 | 1.0 | 5e-5 | const |
| 6 | depth → [512,512,256] | 0 | 3.0 | 5e-5 | 5e-6 |
| 7 | — | 0 | 3.0 | 2.5e-5 | 2.5e-6 |
| 8a | — | 1 | 0.5 | 5e-5 | 2.5e-6 |
| 8b | — | 0 | 0.5 | 2.5e-5 | 2.5e-6 |
| 9 | depth → [512,512,256,256] **(wu20)** | 0 | 3.0 | 5e-5 | 5e-6 |
| 10 | — | 0 | 5.0 | 1e-4 | 1e-5 |
| 11 | — | 1 | 0.5 | 1e-4 | 1e-5 |
| 12 | — | 0 | 0.5 | 1e-5 | const |
| 13 | — | 0 | 5.0 | 5e-5 | 5e-6 |
| 14 | — | 1 | 0.5 | 5e-5 | 5e-6 |
| 15 | — | 0 | 0.5 | 5e-6 | const |

## Commands

```bash
# 1  random init -> [80]
$PY train_batch.py --game-mode dmp --hidden 80 \
  --num-episodes 1000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-3 --workers 48 --save $M/b80.pt

# 2  width -> [150]
$PY train_batch.py --game-mode dmp --expand $M/b80.pt --hidden 150 \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --workers 48 --save $M/b150.pt

# 3  depth -> [150,150]        (--expand-depth-size defaults to the last layer)
$PY train_batch.py --game-mode dmp --expand-depth $M/b150.pt \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --workers 48 --save $M/b150_150.pt

# 4  width -> [256,256]
$PY train_batch.py --game-mode dmp --expand $M/b150_150.pt --hidden 256,256 \
  --num-episodes 1000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --workers 48 --save $M/b256_256.pt

# 5  width -> [512,512]
$PY train_batch.py --game-mode dmp --expand $M/b256_256.pt --hidden 512,512 \
  --num-episodes 1000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --workers 48 --save $M/b512_512.pt

# 6  depth -> [512,512,256]
$PY train_batch.py --game-mode dmp --expand-depth $M/b512_512.pt \
  --expand-depth-size 256 \
  --num-episodes 3000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --workers 48 --save $M/b512_512_256.pt

# 7  continue 0-ply
$PY train_batch.py --game-mode dmp --resume $M/b512_512_256.pt \
  --num-episodes 3000000 --episodes-per-round 1000 --optimizer adam \
  --lr 2.5e-5 --end-lr 2.5e-6 --workers 48 --save $M/b512_512_256_c.pt

# 8a 1-ply
$PY train_batch.py --game-mode dmp --resume $M/b512_512_256_c.pt --oneply \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 2.5e-6 --workers 48 --save $M/b512_512_256_1ply.pt

# 8b 0-ply polish  -> end of the 3L stage
$PY train_batch.py --game-mode dmp --resume $M/b512_512_256_1ply.pt \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 2.5e-5 --end-lr 2.5e-6 --workers 48 --save $M/dmp_512_512_256_repro.pt

# 9  depth -> [512,512,256,256]   *** the one expansion that warms up ***
$PY train_batch.py --game-mode dmp --expand-depth $M/dmp_512_512_256_repro.pt \
  --expand-depth-size 256 \
  --num-episodes 3000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --workers 48 --device cuda --save $M/d4L_s9.pt

# 10 0-ply
$PY train_batch.py --game-mode dmp --resume $M/d4L_s9.pt \
  --num-episodes 5000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-4 --end-lr 1e-5 --workers 48 --device cuda --save $M/c1_0ply.pt

# 11 1-ply
$PY train_batch.py --game-mode dmp --resume $M/c1_0ply.pt --oneply \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-4 --end-lr 1e-5 --workers 48 --device cuda --save $M/c1_1ply.pt

# 12 0-ply polish
$PY train_batch.py --game-mode dmp --resume $M/c1_1ply.pt \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-5 --end-lr 1e-5 --workers 48 --device cuda --save $M/c1_done.pt

# 13 0-ply
$PY train_batch.py --game-mode dmp --resume $M/c1_done.pt \
  --num-episodes 5000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --workers 48 --device cuda --save $M/c2_0ply.pt

# 14 1-ply
$PY train_batch.py --game-mode dmp --resume $M/c2_0ply.pt --oneply \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --workers 48 --device cuda --save $M/c2_1ply.pt

# 15 0-ply polish -> FINAL
$PY train_batch.py --game-mode dmp --resume $M/c2_1ply.pt \
  --num-episodes 500000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-6 --end-lr 5e-6 --workers 48 --device cuda \
  --save $M/dmp_512_512_256_256_repro.pt
```

Gate:

```bash
$PY experiments/head_to_head/run.py \
  --side1 $M/dmp_512_512_256_256_repro.pt --side2 gnubg \
  --plies 0 --game-mode dmp --games 1000000 --workers 48
```

## Notes for retraining

- **Warmup is used at step 9 only.** Steps 3 and 6 are also depth expansions
  and ran without it. That inconsistency is in the original run, not a typo
  here — reproduce it as written, or add `--warmup-cycles 20` to 3 and 6
  uniformly, but don't assume the recorded lineage had it.
- **`--warmup-cycles` counts ROUNDS, not episodes.** With
  `--episodes-per-round 1000`, `20` means 20k episodes. Changing
  `--episodes-per-round` silently rescales it.
- **0-ply polish after a 1-ply leg (steps 8b, 12, 15) belongs here.** It is
  part of the accepted DMP recipe. It is *not* used in the later cubeful
  extension legs, where it measured harmful — see the cubeful log.
- Steps 1–8b were run on CPU, 9–15 with `--device cuda`. The trainer is
  `set_num_threads(1)`-bound on CPU, so 0-ply training is ~3–4× faster on GPU.
