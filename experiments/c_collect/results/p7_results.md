# CUDA-graph train step (`BG_TRAIN_CUDA_GRAPH`), results

What the graphed train step buys on the 0-ply train phase, bitwise identical to the
eager step. Profile it starts from: [p7_profile.md](p7_profile.md) (0.49 ms host time
per Adam step against 0.22 ms of GPU work). Threadripper 9980X, RTX 5090.

## The lever

The graph is captured once per `Trainer.train` call, at the second full batch, and
replayed for every full batch after that; the first step and each round's ragged last
batch run eagerly. Mode `adam` (`=1` with default `torch.optim.Adam` and no
`grad_clip`): the graph holds forward, loss, backward and the step-independent half of
Adam (`lerp_`, `mul_`, `addcmul_`, `sqrt`); the bias-corrected tail runs eagerly with
the stock scalars. Mode `fwdbwd` (any optimizer, `grad_clip`): the graph holds forward,
loss and backward. Adam's foreach lists run on views of at most 16384 elements
(`BG_TRAIN_GRAPH_ADAM_CHUNK`, 0 = one entry per tensor): 62 -> 22.5 us/step, same
kernels. The first replay is checked against the eager step (`torch.equal` on
weights, Adam state, loss); a mismatch keeps the eager result with a WARN.

Bitwise: `tests/test_train_step_bitwise.py` (whole `Trainer.train` runs against the
host-pool loop with a `.item()` per batch: weights `torch.equal`, loss lists equal),
and 10-round `train_batch.py` CLI runs (cubeful and dmp, pipelined and not, every
graph mode): checkpoints `torch.equal`, all per-batch losses and log lines equal.

## Throughput

`bench.sh`, W=48, epr=4000, batch 256, 25 rounds (rounds 2..25 measured), `--seed
1`, `BG_POOL_ON_DEVICE=1`, Python collector, arms interleaved A B A B per row; box
otherwise idle (foreign load 0.0-0.1 cores).

```bash
WORKLOAD=cubeful-0ply ROUNDS=25 [PIPELINE=0] [BG_TRAIN_CUDA_GRAPH=1] bash experiments/c_collect/bench.sh
```

| workload | pipeline | arm | ep/s (2 runs) | mean ep/s | train s/round | mean train s |
|---|---|---|---|---:|---|---:|
| cubeful 0-ply | on | stock | 3723, 3617 | 3670 | 0.817, 0.892 | 0.855 |
| cubeful 0-ply | on | graph | 4493, 4325 | **4409 (+20%)** | 0.192, 0.193 | **0.193 (4.4x)** |
| cubeful 0-ply | off | stock | 2174, 2165 | 2170 | 0.510, 0.534 | 0.522 |
| cubeful 0-ply | off | graph | 2539, 2495 | **2517 (+16%)** | 0.189, 0.187 | **0.188 (2.8x)** |
| dmp 0-ply | on | stock | 3517, 3535 | 3526 | 0.819, 0.835 | 0.827 |
| dmp 0-ply | on | graph | 4143, 4218 | **4181 (+19%)** | 0.183, 0.182 | **0.183 (4.5x)** |
| dmp 0-ply | off | stock | 2146, 2151 | 2148 | 0.488, 0.499 | 0.494 |
| dmp 0-ply | off | graph | 2541, 2502 | **2522 (+17%)** | 0.175, 0.175 | **0.175 (2.8x)** |

Passes with 1-3 cores of foreign load, same protocol:

| workload | pipeline | stock ep/s | graph ep/s | stock train s | graph train s |
|---|---|---|---|---|---|
| cubeful 0-ply | on | 3533, 3491, 3630, 2729 (mean 3346) | 4151, 3745, 4367, 3987, 3618 (mean 3974, +19%) | 0.79-0.98 | 0.192-0.195 |
| cubeful 0-ply | off | 2136, 2311 (2224) | 2694, 2743 (2718, +22%) | 0.464, 0.476 | 0.187, 0.187 |
| dmp 0-ply | on | 3587, 3306 (3447) | 3937, 3812 (3874, +12%) | 0.761, 0.797 | 0.186, 0.184 |
| dmp 0-ply | off | 2277, 2192 (2234) | 2557, 2715 (2636, +18%) | 0.456, 0.450 | 0.175, 0.176 |

With the graph the pipelined rounds are collect-bound (the 48 Python workers take
~0.9-1.0 s per round); the train phase itself is 4.4-4.5x shorter.

Host-pool loss readback once per pass (the default eager path), cubeful 0-ply,
`BG_POOL_ON_DEVICE` unset, stock = `.item()` per batch:

| pipeline | pass | stock ep/s | readback-once ep/s | per-pair | stock train s | readback-once train s |
|---|---|---|---|---|---|---|
| off | 2 | 2206, 2188 | 2210, 2244 | +0.2%, +2.6% | 0.561, 0.584 | 0.508, 0.520 |
| on | 2, 4 | 3221, 3225, 2906 | 3026, 3111, 2871 | -6.0%, -3.5%, -1.2% | 1.124, 1.160, 1.118 | 1.014, 0.973, 0.970 |
| on | exclusive | 3289, 3407, 3386 | 3535, 3380, 3342 | +7.5%, -0.8%, -1.3% | 1.137, 0.979, 1.060 | 0.938, 0.998, 1.010 |

The train phase is 7-13% shorter; the ep/s effect is inside the noise.

## Where the step goes now

One cubeful round (1001 steps) replayed with the graphed Adam step:
0.175-0.190 s (eager 0.53 s). GPU busy 160 us/step (eager 219):

| kernels | us/step |
|---|---:|
| GEMMs (cutlass simt sgemm + split-K reduce, the stock cuBLAS choices) | ~78 |
| bias-gradient reductions | ~18.5 |
| Adam (7 foreach ops on 41 pieces) | ~28 |
| elementwise (relu, threshold_backward, mse), 2 gathers, loss copy | ~15 |

The step is GPU-bound; collection load beside it no longer matters (0.19 s pipelined
and idle).

## Not bitwise

One round of 1001 steps from the prod net at lr 1e-5 against the eager step (load1
20-29, times rough):

| variant | s/round | bitwise | max abs weight diff after 1 round |
|---|---:|---|---:|
| stock eager | 0.485 | yes | 0 |
| P7 graph, mode adam, pieces of 16384 | **0.181** | yes | 0 |
| eager Adam `capturable=True` | 0.541 | **no** | 2.5e-4 |
| eager Adam `fused=True` | 0.357 | no | 2.7e-4 |
| whole step in one graph (torch-docs style), `capturable=True` Adam | 0.244 | no | 2.5e-4 |
| whole step in one graph, `fused=True, capturable=True` Adam | 0.191 | no | 2.7e-4 |
| P7 graph + TF32 matmuls | 0.151 | no | 7.6e-4 |
| whole step, fused Adam + TF32 | 0.163 | no | 7.7e-4 |
| P7 graph with the Adam tail captured too (baked scalars) | 0.170 (vs 0.175) | only once the scalars stop changing | - |

`capturable=True` computes the bias corrections in fp32 device ops; `fused=True` is a
different kernel; TF32 (~17% faster) is a recipe change. Each needs its own A/B gate.
