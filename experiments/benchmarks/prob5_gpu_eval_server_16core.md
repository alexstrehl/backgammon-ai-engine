# prob5 1-ply GPU eval server on a 16-core / large-VRAM host

`experiments/benchmarks/prob5_collect_levers_ab.md` measured the GPU 1-ply
collectors as **losers** and `reject_cuda_oneply` in `train_prob5.py`
encodes that verdict. This re-runs the question on the host archetype that
doc flags as the exception:

> "On a large-VRAM, strong-GPU host this conclusion would likely flip."

It flips. The GPU eval server is **2.3x the best CPU config** here.

## Host

- CPU: AMD Ryzen 9 9950X — 16 physical / 32 logical, cgroup quota 30.7 CPUs
- GPU: NVIDIA GeForce RTX 4080, 16 GB (Ada `sm_89`)
- torch 2.5.1+cu124, driver 595.84
- **Thermally saturated**: `Tctl` pinned 96.7-97.2 C under load, loaded cores
  hold 4.5-4.8 GHz (base spec 4.3). Stable, not degrading — see below.

This inverts the ratio of both hosts in the prior A/Bs:

| host | cores | GPU | cores per fp32 TFLOP | 1-ply verdict |
|---|---|---|---|---|
| earlier A/B (not included here) | 64 (9985WX) | RTX 5090 32GB | ~0.6 | eval server **+66-86%** |
| `prob5_collect_levers_ab.md` | 96 (9995WX) | GTX 1070 Ti 8GB | ~12 | eval server **-60%**, vectorized **-25%** |
| **this host** | **16 (9950X)** | **RTX 4080 16GB** | **~0.46** | **eval server +143%** |

## Why it was expected to flip here

Two independent reasons, both measured on this box before benchmarking:

1. **prob5 1-ply is forward-bound, not move-gen bound.** `cProfile` on
   `_collect_one_episode(..., oneply=True)`, 25 episodes, shimmed:

   | | share |
   |---|---|
   | net forward (`torch._C._nn.linear` + relu + dispatch) | **76%** |
   | `expand_21` (C move generation) | 14% |
   | ctypes glue / postprocess | 10% |

   Amdahl ceiling if forwards were free: **4.2x**. This is *not* the
   "move-gen is the wall" regime that capped the cubeful GPU paths at
   ~300 ep/s.

2. **The mechanism that killed the vectorized collector cannot occur here.**
   That loss was VRAM starving cores — "on an 8 GB card that caps it at ~16
   workers, so only 16 of 96 cores generate moves." The eval server uses ONE
   CUDA context total (server process; workers stay on CPU), and this box has
   16 cores against 16 GB anyway. No cores are forfeited.

## Method

Protocol copied from `prob5_collect_levers_ab.md` so the numbers are comparable:

- Model: the then-prod prob5 `[512,512,256,128]`
  (md5 `c8b2b265470e79e84320d8dadcf08253` — same file as that doc; since
  superseded and no longer shipped in `best_models/`).
- `--lr 1e-8` (net frozen, arms cannot diverge through training),
  `--seed 12345 --torch-seed 12345`.
- **`LD_PRELOAD=c_engine/libfakeintel.so` on every arm.** Unshimmed, this
  AMD host runs MKL at SSE: measured **2.23x** on the 1-ply forward shape
  (147x512 @ 512x512: 0.564 -> 0.253 ms). Benchmarking without it would
  flatter the GPU by that factor.
- 3000 episodes, 500/round -> 6 rounds. 2 interleaved reps.

```bash
LD_PRELOAD=c_engine/libfakeintel.so python3 train_prob5.py \
  --resume best_models/cubeless_prob5_512_512_256_256.pt --oneply \
  --num-episodes 3000 --episodes-per-round 500 --lr 1e-8 \
  --seed 12345 --torch-seed 12345 --workers 16 --device cpu --gpu-eval-server
```

## Results — 1-ply (ep/s, mean of 2 reps)

| W | cpu base | cpu +pipeline | **GPU eval server** | server vs best CPU |
|---|---:|---:|---:|---:|
| 8 | 40.5 | 42.5 | **106.2** | **2.50x** |
| 12 | 53.6 | 57.2 | **120.6** | **2.11x** |
| 16 | 52.2 | 53.5 | **129.9** | **2.43x** |

Worst rep-to-rep spread 1.6%.

**Best CPU (W=12 pipeline) 57.2 -> best GPU (W=16 server) 129.9 = +127%.**

Two structural observations:

- **CPU collection plateaus at W=12** (53.6 -> 52.2 going to W=16) while the
  **GPU arm is still climbing at W=16** (106 -> 121 -> 130). The server is not
  saturated; it is core-limited. On a host with more cores it would go further.
- **`--pipeline-collect` is nearly worthless at 1-ply here (+2-6%,** vs +33-36%
  on the 96-core host). Per-round split is `collect 9.0s / train 0.4s` — the
  train step is 4% of a round, so there is almost nothing to hide. This is an
  independent reason `reject_cuda_oneply`'s "cuda buys ~0% under pipeline"
  argument does not carry over to this box.
- The server also **avoids pipeline's staleness trade**: weights are pushed
  between rounds, so targets are FRESH, where `--pipeline-collect` dispatches
  round N+1 on round N's weights.

## 0-ply, same protocol (for completeness)

12000 episodes, 1000/round, mean of 2 reps. `--device cuda` is supported and
recommended at 0-ply; the eval server is 1-ply only.

| W | cpu base | cpu +pipeline | cuda base | **cuda +pipeline** |
|---|---:|---:|---:|---:|
| 8 | 399.0 | 562.0 | 525.5 | 580.2 |
| 12 | 485.9 | 741.5 | 698.5 | 817.1 |
| 16 | 536.0 | 727.5 | 816.2 | **974.8** |

Best GPU 974.8 vs best CPU-only 741.5 = **+31%**. Note this is the
decision-relevant figure; against the *unoptimized* W=16 CPU baseline it reads
+82% — the classic trap of quoting a win against a baseline nobody would run.

## Correctness of the port

The server (`gpu_eval_server.py`) is wired
into `train_prob5.py` via `_net_raw`, which routes the raw forward and leaves
`sigmoid`/`prob5_postprocess` on the caller. Routing changes only WHERE the
matmul runs.

- **Target parity vs the CPU reference**: `tests/test_gpu_eval_server_parity.py`
  drives 120 real self-play states through `_oneply_target_vec` with and
  without the server: **max|d| = 1.19e-07, mean 9.56e-09**.
  Bit-exactness is NOT the bar here and is not achievable — fp32 GPU and CPU
  GEMMs use different reduction orders. (It IS the correct bar for the
  CPU-to-CPU generator parity in `tests/test_vectorized_cubeless.py`.)
- **Trajectory identity**: per-round pool sizes are byte-identical between the
  CPU and server arms at W=8 and W=16 across all 6 rounds. At W=12 round 1
  differs (27995 vs 28037, 0.15%) and rounds 2-6 match — a near-tie argmin
  flipped on a last-ulp difference. **The arms are therefore NOT guaranteed
  trajectory-identical**; expect rare forks. This does not affect the
  throughput comparison but should be stated in any strength claim.
- **`activation` is carried** into `build_spec` and consumed by the server.
  Verified load-bearing: identical weights under tanh vs relu diverge by
  0.04, and `activation` is not a parameter tensor, so `load_state_dict`
  cannot catch a mismatch. This is the bug class `858ce0d` fixed on the
  `trainer.py` worker path.
- **`--gpu-eval-server` + `--pipeline-collect` now exits with an error.** The
  between-round `update_weights()` is only safe when no worker is mid-request;
  pipelining keeps round N+1's workers running across that push.

## Thermal stability of the measurements

The box is thermally saturated but does not degrade: across 18 configs
measured twice over ~25 minutes of continuous load, worst rep-to-rep spread
was **1.6%**, median 0.7%, with rep 2 faster in 7 of 18 cells. It reaches a
thermal floor within seconds and holds it. `amd-pstate-epp` with
EPP=`performance` is already optimal; nothing to tune.

## Recommended config on this host

```bash
# 0-ply legs
--workers 16 --device cuda --pipeline-collect          # 975 ep/s

# 1-ply legs
--workers 16 --device cpu --gpu-eval-server            # 130 ep/s
```

Relative to the 72-worker production host (2646 ep/s 0-ply, 310 ep/s 1-ply),
this box now reaches ~37% at 0-ply and ~42% at 1-ply — before the port, 1-ply
was ~19%.

## Not established

Throughput only. No strength/convergence run has been done with the server, and
the rare trajectory fork above means a strength twin is the right gate before
adopting it for a long leg.
