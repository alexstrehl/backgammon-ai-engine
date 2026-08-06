# prob5 collection levers: bf16 / pipeline / vectorized A/B

`train_prob5.py` never imports `trainer.py`, so the three collection
optimisations that `train_batch.py` enjoys (`--bf16-collect`,
`--pipeline-collect`, `--vectorized-collect`) were unavailable to prob5.
All three were implemented, verified for correctness, and benchmarked.
**Only `--pipeline-collect` survived**; the other two are measurably
slower on this host and were removed.

## Host

- CPU: AMD Ryzen Threadripper PRO 9995WX — 96 physical / 192 logical
- GPU: NVIDIA GeForce GTX 1070 Ti, 8 GB (Pascal `sm_61`)
- torch 2.13.0+cu126. `sm_61` is absent from the arch list but the
  `sm_60` cubin runs correctly on the device (verified numerically).

## Method

- Model `best_models/cubeless_prob5_512_512_256_128.pt`
  (md5 `c8b2b265470e79e84320d8dadcf08253`).
- `--lr 1e-8` (net effectively frozen, so arms cannot diverge through
  training), `--seed 12345 --torch-seed 12345`, `BG_MAX_PLIES=200`.
- **`LD_PRELOAD=c_engine/libfakeintel.so` on every arm.** Without the
  MKL shim the fp32 CPU baseline is ~24% slow, which silently flatters
  every alternative.
- 2 interleaved reps per cell (config inner, rep outer).
- 0-ply: W=48, 24000 eps, `--device cuda`. 1-ply: W=64, 12000 eps,
  `--device cpu` (both are the arms that win their ply on this host —
  see the GPU A/B note below).

## Results

### 0-ply (W=48, `--device cuda`)

| config | ep/s (r1 / r2) | vs base |
|--------|---------------:|--------:|
| baseline | 1296 / 1328 | — |
| **+ pipeline** | **2133 / 2335** | **+70%** |
| + bf16 | 1194 / 1247 | −7% |
| + bf16 + pipeline | 2099 / 2126 | +61% |

### 1-ply (W=64, `--device cpu`)

| config | ep/s (r1 / r2) | vs base |
|--------|---------------:|--------:|
| baseline | 170.8 / 172.5 | — |
| **+ pipeline** | **235.5 / 231.7** | **+36%** |
| + bf16 | 166.1 / 162.8 | −4% |
| + bf16 + pipeline | 216.2 / 215.8 | +26% |
| + vectorized (W=16, B=16) | 136.6 / 136.5 | −20% |

Confirmation run after removing bf16/vectorized (same protocol):
0-ply 1308 → 2229 ep/s (**+70%**), 1-ply 175 → 234 ep/s (**+33%**).

## Why pipeline wins

Collection and the master train step are serialised in the base loop.
Pipelining runs round N+1's workers during round N's train step, so the
whole train step disappears behind collection. Per-round medians show it
directly — 0-ply `collect 1.00s train 0.40s` becomes `collect 0.00-0.20s
train 0.40-0.60s`: the collect wait is absorbed almost entirely.

**Cost:** the next round's workers are dispatched with the *current*
weights, before this round trains, so their targets are **one round
stale**. This is the same trade `train_batch.py` makes in its production
cubeful config. It is a training-semantics change, not just a speed knob
— worth a short convergence check before adopting it for a long run.

## Why bf16 loses (despite a faster forward)

bf16 is *not* slow here in isolation — measured on this CPU
(which does have `avx512_bf16`), single-thread, including the cast cost:

| rows | fp32 ms | bf16 ms | speedup |
|-----:|--------:|--------:|--------:|
| 20 | 0.190 | 0.136 | 1.39x |
| 147 | 0.727 | 0.428 | 1.70x |
| 4096 | 21.958 | 10.461 | 2.10x |

The problem is that the forward is not the bottleneck, and the batches
are small: the median collection block is **6 rows at 0-ply** and **147
rows at 1-ply** (p90 16 / 297) — far from the 4096-row shape where bf16
approaches 2x. Speeding up a minority of the round while adding cast and
allocation traffic across 48-64 concurrent workers is a net loss — the
same "move generation is the wall" conclusion the cubeful GPU A/B
reached.

Beware the measurement trap: the same microbenchmark *without* the MKL
shim reports bf16 at **5.6x** (fp32 26.6ms vs bf16 4.74ms) rather than
2.0x, because the unshimmed fp32 path falls back to SSE on AMD.

## Why vectorized loses

The design is sound — B games per worker driven as coroutines in
lockstep, all pending forwards fused into one batched call, no IPC — but
it puts a **CUDA context in every worker**. On an 8 GB card that caps it
at ~16 workers (W=32 OOMs outright), so only 16 of 96 cores generate
moves while the CPU baseline uses 64. Sweep at 4000 eps:

| W | B | ep/s |
|---|---|-----:|
| 8 | 32 | 87.3 |
| 16 | 8 | 117.5 |
| 16 | 16 | 130.6 |
| 16 | 32 | 127.5 |
| 20 | 16 | 124.7 |
| 24 | 12 | 118.7 |
| 32 | 32 | **OOM** |

Best case 130 ep/s vs 175 ep/s for the plain CPU baseline. On a
large-VRAM, strong-GPU host this conclusion would likely flip.

## Correctness

Before benchmarking, the vectorized coroutine rewrite was checked against
the original collection path, both driven with identical fp32 CPU
forwards, over 12 games / 742 positions at each ply:

```
0-ply: max|d enc| = 0   max|d tgt| = 0
1-ply: max|d enc| = 0   max|d tgt| = 0
```

i.e. **bit-identical**, which is the bar for a pure restructuring. Two
aliasing hazards had to be handled: `bg_fast.get_legal_plays_encoded`
returns `features` as a copy (safe to retain) but `next_states` as a
*lazy view over a shared C buffer*, invalidated by the next call — and in
lockstep mode sibling games call it while a coroutine is suspended at a
yield. Successors are therefore materialised before yielding, and yielded
blocks are `np.stack`ed (copied) first.

bf16 is a numeric change by construction, so it was measured rather than
asserted equal: mean |Δ| 3.4e-4 / max 4.8e-3 on identical positions, and
it **flips the greedy move choice on 3.9% of decisions**. That is a
data-quality consideration independent of speed, and another reason not
to adopt it here.

## Bottom line

- **Keep `--pipeline-collect`**: +70% at 0-ply, +33-36% at 1-ply. Mind
  the one-round-stale targets.
- **Dropped bf16 and vectorized**: both measurably slower here. The
  implementations were removed from the tree after this A/B; this doc
  preserves the evidence should the hardware change.
