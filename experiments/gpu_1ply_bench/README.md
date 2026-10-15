# GPU 1-ply collection

The vectorized GPU collector collects 1-ply self-play on the GPU in
`train_batch.py` and `train_prob5.py`. The C collector is the default;
the GPU collector is opt-in: `--vectorized-collect`, or `--no-c-collect`
with `--oneply --device cuda`. It runs under one of two topologies.
Shared context, the default: one eval-server process
(`gpu_eval_server.py`) holds the only CUDA context, the workers run move
generation and the lockstep game generators (`vectorized_collect.py`) on
the CPU and ship each fused batch through pinned shared memory.
Per-worker context: the net on the GPU in every worker, one CUDA context
each. Both collect bitwise-identical data. This directory measures their
throughput and VRAM over a worker ladder, and the training levers around
them.

Hosts:

* A: Ryzen 9 9950X (16 cores) + RTX 4080 16 GB, driver 595.84, torch 2.13.0+cu132
* B: Threadripper 9980X (64 cores) + RTX 5090, torch 2.13.0+cu129
* C: Ryzen 9 9950X (16 cores) + RTX 5080, torch 2.8.0+cu128

Models:
[best_models/cubeful_money_512_512_256_256.pt](../../best_models/cubeful_money_512_512_256_256.pt)
(cubeful, `train_batch.py`) and
[best_models/cubeless_prob5_512_512_256_256.pt](../../best_models/cubeless_prob5_512_512_256_256.pt)
(prob5, `train_prob5.py`).

## Run

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python train_prob5.py --oneply --device cuda --no-c-collect --workers 32 \
  --resume best_models/cubeless_prob5_512_512_256_256.pt \
  --num-episodes 500000 --episodes-per-round 4000 --lr 1.5e-4 \
  --save models/prob5_1ply.pt

python train_batch.py --game-mode cubeful-money --oneply --device cuda --no-c-collect --workers 32 \
  --resume best_models/cubeful_money_512_512_256_256.pt \
  --num-episodes 500000 --episodes-per-round 4000 --optimizer adam --lr 1e-5 \
  --save models/cubeful_1ply.pt
```

The startup log names the collector (`AUTO: --oneply --device cuda ->
vectorized GPU collector ...`), the worker count and any VRAM cap. Rates
below are inter-round means of collect + train over rounds 2..N, not the
`Done in ... ep/s` line, which includes worker startup.

## Topology benchmark, host A

```bash
python experiments/gpu_1ply_bench/oneply_gpu_context_bench.py \
  --workers 4,8,12,16,24,32 --rounds 8 --episodes-per-round 500 --repeats 2 \
  --extra-args=--no-c-collect
python experiments/gpu_1ply_bench/oneply_gpu_context_bench.py --validate-only --no-out \
  --extra-args=--no-c-collect   # bitwise check through the real CLIs
```

lr 1e-8 (net frozen), 2 repeats interleaved. ep/s = steady state from the
`Round` line timestamps over rounds 2..8; VRAM = peak nvidia-smi used at
4 Hz minus idle. Per-worker rows above W=21 (cubeful) / W=28 (prob5) were
capped by the VRAM preflight; the tables show the W used. Raw output:
[results/oneply_gpu_context_9950x_rtx4080_20260904.txt](results/oneply_gpu_context_9950x_rtx4080_20260904.txt).

cubeful (`train_batch.py`, B=32):

| W requested | per-worker: W used | ep/s | peak MiB | shared: ep/s | peak MiB | shared vs per-worker |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 168.4 | 2,203 | 174.8 | 1,376 | +4% |
| 8 | 8 | 239.8 | 4,092 | 262.5 | 1,416 | +9% |
| 12 | 12 | 268.7 | 5,942 | 286.8 | 1,416 | +7% |
| 16 | 16 | 267.9 | 7,751 | 307.4 | 1,398 | +15% |
| 24 | 21 (capped) | 257.8 | 9,498 | 289.6 | 1,258 | +12% |
| 32 | 21 (capped) | 257.9 | 9,518 | 272.9 | 1,098 | +6% |

prob5 (`train_prob5.py`, B = ceil(500 / W)):

| W requested | per-worker: W used | ep/s | peak MiB | shared: ep/s | peak MiB | shared vs per-worker |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 122.5 | 2,961 | 127.1 | 2,496 | +4% |
| 8 | 8 | 193.4 | 4,322 | 209.0 | 1,696 | +8% |
| 12 | 12 | 228.1 | 5,604 | 257.7 | 1,436 | +13% |
| 16 | 16 | 234.6 | 6,945 | 274.0 | 1,296 | +17% |
| 24 | 24 | 225.3 | 9,448 | 267.8 | 1,136 | +19% |
| 32 | 28 (capped) | 218.4 | 10,888 | 245.6 | 1,056 | +12% |

## Settings, hosts B and C

No run logs are committed for this section.

* `--episodes-per-round 4000`: +16-24% over 1000/2000 (B). 4000 is the
  cap: 8M episodes per arm against the recipe's round size tied, -0.1 mEq
  [-2.1, +1.9]. A published chain keeps the recipe's value.
* `BG_EVAL_SLOT_DTYPE=fp16` at 4000 episodes per round: +1.6-8.7%. At
  1000 it loses (-1.5% cubeful, -3.1% prob5, B).
* `BG_POOL_ON_DEVICE=1` (0-ply): only where the train step is a large
  share of the round (+10-12% on B at 87%, -0.8% on C at 37%).
* Not used: `BG_EVAL_SERVER_DTYPE=fp16` (+2%, inside noise, flips 0.4% of
  1-ply moves); bf16 collection (-4..-7%, flips 3.9% of greedy moves); MPS
  (`nvidia-cuda-mps-control`, host A, W=16: within 0.7%).
* `--vectorized-workers`, from a sweep over ~0.5x..1.5x physical cores
  (`python -c 'import gpu_context; print(gpu_context.physical_cores())'`):

| workload | B (64 cores) | C (16 cores) |
|---|---|---|
| play_models | 40 | 12 |
| cubeful 0-ply | ~30 | 16 |
| cubeful 1-ply | 48 | 16 |
| prob5 1-ply | 48 | 24 |

## Parity

* Shared vs per-worker context: bitwise, every round, both workloads
  (`tests/test_shared_context_parity.py`; `oneply_gpu_context_bench.py
  --validate`, collected pools and final weights max|diff| = 0).
* Lockstep collector vs the stock per-game path, identical CPU forwards:
  bitwise at B=1, at B>1 targets move by at most 1 fp32 ulp
  (`tests/test_prob5_vectorized_parity.py`).
* GPU vs CPU forward: ~1e-7 (`tools/gpu_eval_server_validity.py`
  checks 1-ply targets at 1e-5).
* End to end: an 8M-episode leg collected on the GPU tied its CPU-collected
  twin, +0.4 mEq over 4M games, p=0.12.
