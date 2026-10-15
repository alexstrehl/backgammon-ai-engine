# Profile: where the master's train phase goes (eager step)

A replay of the per-round loop of
`Trainer.train` (the `BG_POOL_ON_DEVICE=1` branch and the host-pool branch) on one
dumped round. Threadripper 9980X, RTX 5090, torch 2.13.0+cu129,
`LD_PRELOAD=c_engine/libfakeintel.so`, `OMP_NUM_THREADS=1`, 2-4 cores of foreign
single-core load.

Workload: one cubeful-money 0-ply round of 4000 episodes collected by
`train_batch.py` with `BG_COLLECT_DUMP` (W=8, seed 1): 256,062 rows x 200 float32,
1001 Adam steps of batch 256, net `best_models/cubeful_money_512_512_256_256.pt`
(562,945 parameters, 10 tensors), stock `torch.optim.Adam(lr=1e-5)` (foreach, not
capturable), fresh optimizer each repeat, after one warm-up round.

## The round

| pool | wall s/round (repeats) | ms/step |
|---|---|---:|
| on device (`BG_POOL_ON_DEVICE=1`) | 0.517 0.518 0.522 0.518 0.610; other runs 0.41-0.55 | 0.52 |
| host (numpy fancy index, H2D per batch, `.item()` per batch) | 0.631 0.632 0.635 0.630 0.627 | 0.63 |

## Host time per section

Host time of each call, `perf_counter`, no syncs inside the loop; median of 5
repeats. A sync after every section moves the total only from 0.489 to 0.574 s: the GPU is not the
limit.

| section | pool on device: us/step | share | host pool: us/step | share |
|---|---:|---:|---:|---:|
| index (`pool_t[idx_t]` / numpy `pool[idx]`) | 20.7 | 4% | 27.5 | 5% |
| H2D of the batch (`torch.from_numpy(..).to`, `stack_targets`) | - | - | 36.5 | 7% |
| forward (`network(x)`) | 86.8 | 18% | 82.5 | 16% |
| loss (`0.5 * mse_loss`) | 28.7 | 6% | 27.5 | 5% |
| `optimizer.zero_grad()` (set_to_none) | 17.0 | 3% | 16.5 | 3% |
| `loss.backward()` | 195.3 | 40% | 176.6 | 34% |
| `optimizer.step()` (Adam, foreach) | 115.6 | 24% | 111.9 | 21% |
| `loss.detach()` / `loss.item()` | 1.4 | 0.3% | 48.3 (waits for the GPU) | 9% |
| per round: pool H2D 7.8 ms, shuffle 1.8 ms, loss readback 0.4 ms | 10.0 | 2% | 1.6 | 0.3% |
| total | **489** | | **528** | |

Per step: 52 kernel launches plus 3 memsets, and 20 `aten::item` calls on Adam's CPU
step counters (no sync). The host-pool path adds 3 `cudaStreamSynchronize` per step.

## GPU time per step (torch.profiler)

GPU kernel time 219 us/step (pool on device) and 213 us/step (host pool), against
489-528 us of host time: the step is launch-bound by ~2.3x.

| kernels | us/step |
|---|---:|
| GEMMs: 9 cutlass `simt_sgemm` (fwd 5 addmm, bwd 9 mm) + 7 cublasLt split-K reduces | ~79 |
| Adam foreach (`multi_tensor_apply`): addcdiv 22.4, div 20.4, sqrt 14.4, addcmul 8.2, lerp 8.2, mul 7.2, add 7.2 | ~88 |
| bias-gradient reductions (`reduce_kernel`, 4-5 per step) | ~19 |
| elementwise: relu, threshold_backward, mse, mul, mean, index | ~15 |

Adam's 7 foreach kernels are 40% of the GPU time: each `multi_tensor_apply` launch
covers only ~15 chunks of 65,536 elements for these 10 tensors (~15 blocks on a
170-SM GPU).

## Under collection load

The same profile beside 48 spawn processes running the cubeful 0-ply Python collector
in a loop (no pool in the profiled process):

| condition | wall s/round (plain, 5 repeats) | host-timers total | backward / opt_step / forward us/step |
|---|---|---:|---|
| idle (2-4 foreign cores) | 0.52 (0.41-0.55 over runs) | 0.489 | 195 / 116 / 87 |
| 48 collectors, unpinned (Tctl 80 °C) | 0.54-0.66, median 0.60 | 0.782 | 328 / 204 / 127 |
| 48 collectors on cores != {2, 66}, master pinned to core 2 (Tctl 69 °C) | 0.59-0.60 | 0.632 | 264 / 165 / 104 |

Every section inflates by about the same factor (1.5-1.7x) under load: the slowdown
is per instruction on the master's core, not the GIL. A whole physical core for the
master recovers part of it.

## Bitwise checks

* Two fresh runs of the same round give `torch.equal` weights and identical
  per-batch losses.
* `BG_POOL_ON_DEVICE=1` and `=0` are bitwise identical on this round (weights
  `torch.equal`, all 1001 losses equal).
