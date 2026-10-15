# Retraining recipe

The chain that trains the cubeful-money and prob5 models from scratch to
production strength. It is the recipe behind
[best_models/cubeful_money_512_512_256_256.pt](../../best_models/cubeful_money_512_512_256_256.pt)
(md5 `6afda0f8`) and
[best_models/cubeless_prob5_512_512_256_256.pt](../../best_models/cubeless_prob5_512_512_256_256.pt)
(md5 `ac95a983`), with schedule adjustments, and was run end to end twice
at these settings ([Results](#results)). Each architecture or task change
runs a subsequence of: 0-ply at a constant LR, 0-ply annealed, 1-ply at a
constant LR, 1-ply annealed, and for cubeful a second 1-ply anneal. 0-ply
training uses sampled Bellman targets, 1-ply training exact Bellman targets.

1. DMP (double match point) 4L: progressive expansion from scratch (the ladder below).
2. Cubeless equity 4L: `--warm-start-equity` from the DMP final.
3. Cubeful 4L: `--warm-start-cubeful` from the cubeless final (input widens
   196 to 200).
4. prob5 4L: `train_prob5.py --warm-start-equity-trunk` from the cubeless
   final (hidden layers copied, five-output head fresh). Steps 3 and 4 fork
   from the same checkpoint and run independently.

Legs at each task's final architecture (episodes, LR schedule). s1-s6 of the
DMP ladder, a further 8.3M episodes, precede the DMP row. `wu20` =
`--warmup-cycles 20`, on the three task transfers only.

| task | 0-ply const | 0-ply anneal (seed) | 1-ply const | 1-ply anneal (descent) | 1-ply 2nd anneal |
|---|---|---|---|---|---|
| DMP (final shape) | — | 9M, 5e-5 → 5e-6 | — | 1M, 1e-4 → 1e-5 | — |
| cubeless equity | — | 16M, 5e-5 → 5e-6, wu20 | 2.5M @ 1e-5 | 12M, 1e-5 → 9e-7 | — |
| cubeful | 2M @ 1e-4, wu20 | 25M, 6e-5 → 5e-7 | 12M @ 1.8e-5 | 115M, 1.8e-5 → 3e-7 | 44M, 1.62e-5 → 2.7e-7 |
| prob5 | — | 6M, 1e-4 → 1e-5, wu20 | 8M @ 1.8e-4 | 49M, 1.8e-4 → 2e-6 | 8M, 9e-5 → 9e-6 |

Every leg runs `train_batch.py` (`train_prob5.py` for the prob5 fork) from
the repo root at `--episodes-per-round 1000` (the default).

Collectors. Self-play is collected in C by default (`bgc_collect`,
`c_api/libpuretd_simd.so`, built on demand by the first run;
[../c_collect/SPEC.md](../c_collect/SPEC.md)). Its samples match the
Python collector's for the same seed (SPEC, gate section). Configurations
it does not cover (`--boltzmann-temp`, `--oneply-acting`, `--bf16-collect`,
`--c-engine`) print a NOTE and fall back to the vectorized GPU collector
for `--oneply --device cuda` with `--boltzmann-temp`, to the Python
collector otherwise. On a CUDA host every leg, 1-ply included, takes
`--device cuda` for the train step and keeps `--pipeline-collect`. The
train step then replays a CUDA graph with the round's pool on the GPU
(`BG_TRAIN_CUDA_GRAPH`), and the workers hand their rows to the master
through shared memory (`BG_COLLECT_SHM`). Both are bitwise equal to the
eager step and the pool pipe. For cubeful 1-ply at 48 workers the C
collector runs at 1.87x the GPU collector's ep/s
([../c_collect/results/p4_throughput.md](../c_collect/results/p4_throughput.md)).
The GPU-collector form of the 1-ply legs is `--device cuda
--vectorized-collect --vectorized-batch 32` (`train_prob5.py`: `--device
cuda --vectorized-collect 32`) in place of `--pipeline-collect`, plus
`BG_VEC_CUBELESS_FAST=1` on the cubeless 1-ply legs. An explicit
`--vectorized-collect` replaces the C collector, with a NOTE.
Opt-outs, each on its own:
- `--no-c-collect`: the Python collector (with `--oneply --device cuda`:
  the vectorized GPU collector);
- `BG_TRAIN_CUDA_GRAPH=0`: the eager train step, with the pool on the host;
- `BG_COLLECT_SHM=0`: rows pickled through the pool pipe.

Memory: the graphed step holds the round's pool in VRAM (~50 MB
per 1000-episode round, 4x that at 4000). The shared-memory arenas are
host memory, pinned when the pool is on the GPU: one per round in flight,
two under `--pipeline-collect`, ~100 MB each at 8 workers and 1000
episodes per round, ~390 MB at 48 workers and 4000.

## Step 1: DMP ladder

Alternate widening (`--expand`, function-preserving) and deepening
(`--expand-depth`, a square near-identity layer appended after the last
hidden layer; `--expand-depth-eps` is the symmetry-breaking noise on the
identity init), training briefly after each step. At `eps=1e-4` the
appended layer deviates from the parent by ~0.6 mEq (256-wide, 4,000
positions; linear in eps). No `--warmup-cycles` on expansions.

```bash
# Build the C engine first (optional, ~20x faster) and the C collector
# (the default collector; built on demand by the first run otherwise)
cd c_engine && bash build_unix.sh && cd ..
bash c_api/build.sh simd

# Working directory for checkpoints (created automatically on first save)
mkdir -p models

# s1. Train [80] from scratch
python train_batch.py --game-mode dmp --hidden 80 \
  --num-episodes 1200000 --optimizer adam --lr 1e-3 \
  --workers 8 --save models/dmp_b80.pt

# s2. Width-expand [80] -> [150]
python train_batch.py --game-mode dmp \
  --expand models/dmp_b80.pt --hidden 150 \
  --num-episodes 600000 --optimizer adam --lr 5e-5 \
  --workers 8 --save models/dmp_b150.pt

# s3. Depth-expand [150] -> [150,150]
python train_batch.py --game-mode dmp \
  --expand-depth models/dmp_b150.pt --expand-depth-eps 1e-4 \
  --num-episodes 600000 --optimizer adam --lr 5e-5 \
  --workers 8 --save models/dmp_b150_150.pt

# s4. Width-expand [150,150] -> [256,256]
python train_batch.py --game-mode dmp \
  --expand models/dmp_b150_150.pt --hidden 256,256 \
  --num-episodes 1200000 --optimizer adam --lr 5e-5 \
  --workers 8 --save models/dmp_b256_256.pt

# s5. Depth-expand [256,256] -> [256,256,256]
python train_batch.py --game-mode dmp \
  --expand-depth models/dmp_b256_256.pt --expand-depth-size 256 --expand-depth-eps 1e-4 \
  --num-episodes 1200000 --optimizer adam --lr 5e-5 \
  --workers 8 --save models/dmp_b256_256_256.pt

# s6. Width-expand [256,256,256] -> [512,512,256] (GPU)
python train_batch.py --game-mode dmp \
  --expand models/dmp_b256_256_256.pt --hidden 512,512,256 \
  --num-episodes 3500000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --device cuda --save models/dmp_b512_512_256.pt

# s7. Depth-expand [512,512,256] -> [512,512,256,256]  <- DMP row, 0-ply anneal
python train_batch.py --game-mode dmp \
  --expand-depth models/dmp_b512_512_256.pt --expand-depth-size 256 --expand-depth-eps 1e-4 \
  --num-episodes 9000000 --optimizer adam --lr 5e-5 --end-lr 5e-6 \
  --workers 8 --device cuda --save models/dmp_4L_0ply.pt

# s8. 1-ply refinement  <- DMP row, 1-ply anneal
# (1-ply legs on a CUDA host: add --device cuda; see Collectors above)
python train_batch.py --game-mode dmp \
  --resume models/dmp_4L_0ply.pt --oneply \
  --num-episodes 1000000 --optimizer adam --lr 1e-4 --end-lr 1e-5 \
  --workers 8 --save models/dmp_final.pt
```

## Step 2: cubeless equity

`--warm-start-equity` keeps the trunk and re-initialises the output layer
for an equity output; `cm_seed` takes `--warmup-cycles 20`.

```bash
# cm_seed: task transfer (wu20), 0-ply
python train_batch.py --game-mode cubeless-money --output-mode equity \
  --warm-start-equity models/dmp_final.pt \
  --num-episodes 16000000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --workers 8 --device cuda --pipeline-collect \
  --save-every 4000000 --save models/cm_seed.pt

# cm_const: 1-ply at the band top
# (1-ply legs on a CUDA host: add --device cuda; see Collectors above)
python train_batch.py --game-mode cubeless-money --resume models/cm_seed.pt --oneply \
  --num-episodes 2500000 --optimizer adam --lr 1e-5 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 500000 --save models/cm_const.pt

# cm_descent: 1-ply descent from that same LR
python train_batch.py --game-mode cubeless-money --resume models/cm_const.pt --oneply \
  --num-episodes 12000000 --optimizer adam --lr 1e-5 --end-lr 9e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cm_descent.pt
```

## Step 3: cubeful

`--warm-start-cubeful` widens the input 196 → 200 for the four cube features
and copies the hidden layers and the equity head; `cf_adapt` absorbs the widening at a
constant LR and takes `--warmup-cycles 20`.

```bash
# cf_adapt: task transfer (wu20), absorbs the 196 -> 200 input widening
python train_batch.py --game-mode cubeful-money \
  --warm-start-cubeful models/cm_descent.pt \
  --num-episodes 2000000 --optimizer adam --lr 1e-4 --warmup-cycles 20 \
  --workers 8 --device cuda --pipeline-collect --save models/cf_adapt.pt

# cf_seed: 0-ply
python train_batch.py --game-mode cubeful-money --resume models/cf_adapt.pt \
  --num-episodes 25000000 --optimizer adam --lr 6e-5 --end-lr 5e-7 \
  --workers 8 --device cuda --pipeline-collect \
  --save-every 4000000 --save models/cf_seed.pt

# cf_const: 1-ply at the band top
# (1-ply legs on a CUDA host: add --device cuda; see Collectors above)
python train_batch.py --game-mode cubeful-money --resume models/cf_seed.pt --oneply \
  --num-episodes 12000000 --optimizer adam --lr 1.8e-5 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_const.pt

# cf_descent: 1-ply descent from that same LR (the long leg)
python train_batch.py --game-mode cubeful-money --resume models/cf_const.pt --oneply \
  --num-episodes 115000000 --optimizer adam --lr 1.8e-5 --end-lr 3e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_descent.pt

# cf_restart: 2nd anneal; this is the finished model
python train_batch.py --game-mode cubeful-money --resume models/cf_descent.pt --oneply \
  --num-episodes 44000000 --optimizer adam --lr 1.62e-5 --end-lr 2.7e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_restart.pt
```

## Step 4: prob5

`--warm-start-equity-trunk` copies the hidden layers of the cubeless final
into a fresh prob5 net (`--hidden` names the shape) and starts the
five-output head fresh; `p5_seed` takes `--warmup-cycles 20`.

```bash
# p5_seed: task transfer (wu20), 0-ply
python train_prob5.py --warm-start-equity-trunk models/cm_descent.pt --hidden 512,512,256,256 \
  --num-episodes 6000000 --lr 1e-4 --end-lr 1e-5 --warmup-cycles 20 \
  --workers 8 --device cuda --pipeline-collect \
  --save-every 1000000 --save models/p5_seed.pt

# p5_const: 1-ply at the band top
# (1-ply legs on a CUDA host: add --device cuda; see Collectors above)
python train_prob5.py --resume models/p5_seed.pt --oneply \
  --num-episodes 8000000 --lr 1.8e-4 \
  --workers 8 --pipeline-collect \
  --save-every 1000000 --save models/p5_const.pt

# p5_descent: 1-ply descent from that same LR (the long leg)
python train_prob5.py --resume models/p5_const.pt --oneply \
  --num-episodes 49000000 --lr 1.8e-4 --end-lr 2e-6 \
  --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/p5_descent.pt

# p5_restart: 2nd anneal; this is the finished model
python train_prob5.py --resume models/p5_descent.pt --oneply \
  --num-episodes 8000000 --lr 9e-5 --end-lr 9e-6 \
  --workers 8 --pipeline-collect \
  --save-every 1000000 --save models/p5_restart.pt
```

## Results

Final checkpoint of each of the two end-to-end runs against production,
both sides 0-ply: cubeful money with Jacoby, capped
+/-128, or cubeless money. Negative = production ahead.

```bash
python experiments/head_to_head/run.py --side1 models/cf_restart.pt \
  --side2 best_models/cubeful_money_512_512_256_256.pt \
  --game-mode cubeful-money --jacoby --games 500000 --workers 48 --save DIR_CF
python experiments/head_to_head/run.py --side1 models/p5_restart.pt \
  --side2 best_models/cubeless_prob5_512_512_256_256.pt \
  --game-mode cubeless-money --games 500000 --workers 48 --save DIR_P5
python experiments/head_to_head/run.py --combine DIR_CF
python experiments/head_to_head/run.py --combine DIR_P5
```

| run | final checkpoint (md5) | games | mEq/game vs production |
|---|---|---|---|
| cubeful, run 1 | [../../best_models/experimental/recipe_runs/cubeful_money_512_512_256_256_run1.pt](../../best_models/experimental/recipe_runs/cubeful_money_512_512_256_256_run1.pt) `e9f7ddc8` | 20M | +0.7 [-0.5, +2.0] |
| cubeful, run 2 | [../../best_models/experimental/recipe_runs/cubeful_money_512_512_256_256_run2.pt](../../best_models/experimental/recipe_runs/cubeful_money_512_512_256_256_run2.pt) `60bd300a` | 20M | -0.0 [-1.3, +1.2] |
| prob5, run 1 | [../../best_models/experimental/recipe_runs/cubeless_prob5_512_512_256_256_run1.pt](../../best_models/experimental/recipe_runs/cubeless_prob5_512_512_256_256_run1.pt) `64860333` | 20M | -1.0 [-1.6, -0.4] |
| prob5, run 2 | [../../best_models/experimental/recipe_runs/cubeless_prob5_512_512_256_256_run2.pt](../../best_models/experimental/recipe_runs/cubeless_prob5_512_512_256_256_run2.pt) `5d130631` | 20M | -0.0 [-0.6, +0.6] |

Rows `recipe_*` of
[h2h_cubeful_summary.csv](../head_to_head/results/h2h_cubeful_summary.csv)
and
[h2h_cubeless_summary.csv](../head_to_head/results/h2h_cubeless_summary.csv);
pooled `run.py --combine` outputs in [results/](results/).
