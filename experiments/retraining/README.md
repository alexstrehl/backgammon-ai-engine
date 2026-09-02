# Retraining recipe

This is the recipe we use for training a cubeful-money model from scratch
to production strength. It follows the chain shape that trained
`best_models/cubeful_money_512_512_256_256.pt`, with some schedule
adjustments. An end-to-end replication at exactly these settings is in
progress, and the recipe will be amended if it disagrees.

Each time the architecture changes (e.g. one hidden layer to two) or the task
changes (e.g. DMP to cubeless money), we run a subsequence of the following:
0-ply training at a constant learning rate, 0-ply with an annealed learning
rate, 1-ply at a constant learning rate, and 1-ply with an annealed learning
rate. Additionally, a second 1-ply annealed run is the final step, which we
found necessary to reproduce production strength.

In the descriptions below, **0-ply** training means training on *Sampled
Bellman Targets*, while **1-ply** training refers to training on *Exact
Bellman Targets*.

## Chain order

1. **DMP 4L** — progressive expansion from scratch (ladder below).
2. **Cubeless equity 4L** — `--warm-start-equity` from the DMP final.
3. **Cubeful 4L** — `--warm-start-cubeful` from the cubeless final
   (input widens 196 → 200).

## Legs (episodes, LR schedule)

| task | 0-ply const | 0-ply anneal (seed) | 1-ply const-LR | 1-ply anneal (descent) | 1-ply 2nd-anneal |
|---|---|---|---|---|---|
| DMP (final shape) | — | 9M, 5e-5 → 5e-6 | — | 1M, 1e-4 → 1e-5 | — |
| cubeless equity | — | 16M, 5e-5 → 5e-6, **wu20** | 2.5M @ 1e-5 | 12M, 1e-5 → 9e-7 | — |
| cubeful | **2M @ 1e-4, wu20** | 25M, 6e-5 → 5e-7 | 12M @ 1.8e-5 | 115M, 1.8e-5 → 3e-7 | **44M, 1.62e-5 → 2.7e-7** |

A dash means that step is skipped. Only cubeful needs a 0-ply constant leg,
to absorb the 196 → 200 input widening; only cubeful gets a 2nd-anneal.

**This table covers only the legs run at each task's final architecture**;
for the entire recipe, see below.

`wu20` = `--warmup-cycles 20`, and it appears **only on task transfers**
(DMP → equity, equity → cubeful). A task transfer is exactly a
freshly-initialised head on a trained trunk.

Run flags: `--episodes-per-round 1000` everywhere;
`LD_PRELOAD=c_engine/libfakeintel.so` on AMD hosts. Every leg runs
`train_batch.py`; 1-ply legs take `--max-episode-plies 1500` and a collector,
either `--pipeline-collect` or `--vectorized-collect`. Collector choice and
worker counts are host-dependent — benchmark first.

## Shape rules

- **The constant leg and the descent share one LR.** The constant leg runs at
  the top of the band; the descent then starts at that same LR and anneals
  down from it, so the LR never rises between the two.
- **2nd-anneal leg**: resume the finished descent at ~half to 0.9× the band
  peak, anneal one decade, over ~12–35% of the descent's length. Why it helps
  is not established. Two hypotheses: `--resume` resets Adam's moments
  (checkpoints carry no optimizer state), and two shorter back-to-back
  anneals may be more robust than one long linear one.

## Step 1 — DMP ladder

Start small, then alternate widening and deepening, training briefly after
each step so the new units differentiate. Every expansion is
near-function-preserving at initialization (exactly function-preserving
for width; for depth, up to the `eps` symmetry-breaking noise), so no step
starts from a materially worse model than it inherited. Only the last two steps (s7,
s8) appear in the Legs table; s1–s6 build the architecture and total a
further 8.3M episodes.

```bash
# Build C engine first (optional but ~20x faster)
cd c_engine && bash build_unix.sh && cd ..

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
python train_batch.py --game-mode dmp \
  --resume models/dmp_4L_0ply.pt --oneply \
  --num-episodes 1000000 --optimizer adam --lr 1e-4 --end-lr 1e-5 \
  --workers 8 --save models/dmp_final.pt

```

Depth expansion appends a square near-identity layer, so it only deepens by
the last hidden size; widen first with `--expand` if you want a different
width. `--expand-depth-eps` is required — it is the symmetry-breaking noise on
the identity init. The append is exactly function-preserving at `eps=0`; `1e-4`
keeps the symmetry-breaking at ~0.6 mEq of deviation from the parent (measured,
256-wide, 4k real positions; deviation is linear in eps). **No
`--warmup-cycles` on expansions** — warmup is for task transfers only, per the
rule above.

## Step 2 — cubeless equity

`--warm-start-equity` converts the DMP model's win-probability output to an
equity output, re-initialising the output layer and keeping the trunk. That
fresh head on a trained trunk is the task transfer, so `cm_seed` takes
`--warmup-cycles 20`; the two legs after it do not.

```bash
# cm_seed — TASK TRANSFER (wu20), 0-ply
python train_batch.py --game-mode cubeless-money --output-mode equity \
  --warm-start-equity models/dmp_final.pt \
  --num-episodes 16000000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --workers 8 --device cuda --pipeline-collect \
  --save-every 4000000 --save models/cm_seed.pt

# cm_const — 1-ply at the band top
python train_batch.py --game-mode cubeless-money --resume models/cm_seed.pt --oneply \
  --num-episodes 2500000 --optimizer adam --lr 1e-5 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 500000 --save models/cm_const.pt

# cm_final — 1-ply descent from that same LR
python train_batch.py --game-mode cubeless-money --resume models/cm_const.pt --oneply \
  --num-episodes 12000000 --optimizer adam --lr 1e-5 --end-lr 9e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cm_final.pt
```

## Step 3 — cubeful

`--warm-start-cubeful` widens the input 196 → 200 for the four cube features
and re-initialises the output layer. `cf_adapt` absorbs that widening at a
constant LR before any annealing starts, and is the second task transfer, so
it takes `--warmup-cycles 20`.

```bash
# cf_adapt — TASK TRANSFER (wu20), absorbs the 196 -> 200 input widening
python train_batch.py --game-mode cubeful-money \
  --warm-start-cubeful models/cm_final.pt \
  --num-episodes 2000000 --optimizer adam --lr 1e-4 --warmup-cycles 20 \
  --workers 8 --device cuda --pipeline-collect --save models/cf_adapt.pt

# cf_seed — 0-ply
python train_batch.py --game-mode cubeful-money --resume models/cf_adapt.pt \
  --num-episodes 25000000 --optimizer adam --lr 6e-5 --end-lr 5e-7 \
  --workers 8 --device cuda --pipeline-collect \
  --save-every 4000000 --save models/cf_seed.pt

# cf_plateau — 1-ply at the band top
python train_batch.py --game-mode cubeful-money --resume models/cf_seed.pt --oneply \
  --num-episodes 12000000 --optimizer adam --lr 1.8e-5 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_plateau.pt

# cf_final — 1-ply descent from that same LR (the long leg)
python train_batch.py --game-mode cubeful-money --resume models/cf_plateau.pt --oneply \
  --num-episodes 115000000 --optimizer adam --lr 1.8e-5 --end-lr 3e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_final.pt

# cf_restart — 2nd anneal; this is the finished model
python train_batch.py --game-mode cubeful-money --resume models/cf_final.pt --oneply \
  --num-episodes 44000000 --optimizer adam --lr 1.62e-5 --end-lr 2.7e-7 \
  --max-episode-plies 1500 --workers 8 --pipeline-collect \
  --save-every 2000000 --save models/cf_restart.pt
```

## Notes

On a CUDA host the 1-ply legs run substantially faster with
`--device cuda --vectorized-collect --vectorized-batch 32` in place of
`--pipeline-collect`.

For the **cubeless** 1-ply legs (`cm_const`, `cm_final`) also set
**`BG_VEC_CUBELESS_FAST=1`**. It selects a batched target generator that is
~5x faster under multi-worker GPU collection — measured 522 vs 109 ep/s at
`--workers 24`, which over those two legs' 14.5M episodes is about 8 hours
instead of 37. The default generator is bit-exact against the CPU reference;
the fast one differs in the last ulp. The current 4-layer cubeless model,
and the cubeless stage in the production 4-layer cubeful model's lineage,
used this batched path. The cubeful legs are unaffected either way.
