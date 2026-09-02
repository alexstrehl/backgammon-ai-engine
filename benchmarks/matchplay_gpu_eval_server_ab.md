# Matchplay 1-ply collection: CPU vs `--vectorized-collect` vs `--gpu-eval-server`

Three ways to collect matchplay 1-ply self-play data, same host, same
contention (co-tenant training job, load 65-90 on 128 threads; arms
interleaved, so the ranking is sound but absolute rates are depressed).

    CPU serial          N workers, every forward local, single-threaded
    --vectorized-collect N lockstep drivers, each with its OWN CUDA
                        context (~475 MB); no IPC
    --gpu-eval-server   ONE CUDA context; N pure-CPU workers ship
                        feature matrices over per-worker shared memory

Setup: EPYC 7713P, RTX 3080 Ti (12 GB), 204-input 3pt net, MET on,
600 episodes, workers = 8, two reps interleaved.

## NOISE FLOOR — do not read the ranges as intervals

Four runs of an identical CPU-serial workload inside 15 minutes spanned
**1.71x** (3.70-6.33 collect ep/s). The two-rep min-max ranges below are
two samples from a distribution that wide. What survives: the ORDERING
(vectorized > server+moves > server > cpu, reproduced independently)
and order-of-magnitude ratios. No two-digit gain figure does.

## Results (W=8)

    arm             collect ep/s     wall ep/s   peak VRAM (MB)
    cpu serial       3.14 - 3.38    3.00 - 3.25         0
    vectorized      21.73 - 24.51   16.02 - 19.62    3856
    gpu-eval-server  7.20 - 8.82    6.20 - 7.76       412
      + route-moves  9.22 - 10.90   8.08 - 8.82       400

**Vectorized wins at W=8, by ~2.3x over the server.** Expected shape:
the server pays a queue round-trip per forward; vectorized's only cost
(VRAM) is not yet binding at 8 workers.

**What the server buys: VRAM.** Vectorized costs ~482 MB per worker and
is capped near W=20 on a 12 GB card. The server holds one CUDA context,
but its VRAM is NOT constant in the slot count — the coalesced batch's
activations grow with it. Saturating-client sweep of the real server:

    slots      1     2     4     8    16    32    48
    MB       328   350   350   412   640   976  1262

Real collection coalesces far less than a saturating driver (mean 1.67
requests/GPU call at W=8), so a real W=48 run lands between 640 and
1262 MB — still ~18x cheaper than vectorized's 23 GB at the same W.
`GPU_EVAL_MAX_BATCH_ROWS` caps the fused batch; without it 48 slots x
8192 rows fuses a 393216-row forward, which CUDA-OOMs the card. Host
cost is /dev/shm instead: 6.5 MB per worker at max_rows=8192 with the
207-feature encoder.

**`--gpu-eval-route-moves` helped in 5/5 measurements**, but the
magnitude spanned +14% to +121% — the sign is supported, no number is.
Re-measure at high W, where the sign could flip.

## Reproducibility

Coalescing fuses requests by wall-clock arrival order and fp32
reductions are batch-composition sensitive, so a routed run is NOT
bitwise reproducible by default: a last-ulp difference can flip a move
tie, changing the whole trajectory (measured: 3/4 identically-seeded
runs matched at W=4, 0/4 at W=2). Values are correct; seeded reruns are
not regenerable. `--gpu-eval-deterministic` runs one request per GPU
call: bit-exact against the unrouted collector at any worker count,
reproducible run to run (3/3), and measured to cost nothing at W=4
(fusion is nearly dormant at low W; expect that to reverse at high W).
`--vectorized-collect` is not bit-exact against serial either
(atol=1e-6) but reproduces run to run — its batch composition is fixed.

## What this does not settle

W=8 is near the server's worst case (IPC paid, VRAM advantage unused)
and vectorized's best. The production question — vectorized at its
VRAM-capped W (~20) vs the server at W=48 — needs a quiet box and was
not measured. A saturation estimate suggests the server reaches ~17
ep/s at W=48, which does not obviously beat vectorized's 21.7-24.5 at
W=8; treat as a lower bound (LLM co-tenant on the card).
