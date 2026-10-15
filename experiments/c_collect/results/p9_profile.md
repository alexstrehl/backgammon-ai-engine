# Profile: where train_prob5.py's train phase goes (eager step)

A replay of the eager round of
`train_prob5.train` (net to the device, pool H2D, `torch.randperm` on the device, per
batch a gather, the five per-output losses, Adam, a `.item()`, net back to the CPU) on
one dumped round. Threadripper 9980X, RTX 5090, torch 2.13.0+cu129,
`LD_PRELOAD=c_engine/libfakeintel.so`, `OMP_NUM_THREADS=1`, process pinned to one
core, beside a 48-worker training (load1 48): the contended case of a pipelined W=48
run.

Workload: one prob5 0-ply round of 4000 episodes collected in C as a W=48
`train_prob5.py --seed 1` run collects round 1: 226,554 rows x 196 float32, targets
(n, 5), 885 Adam steps of batch 256. Net `best_models/cubeless_prob5_512_512_256_256.pt`
(561,925 parameters, sigmoid outputs, `F.binary_cross_entropy` per output), stock
`torch.optim.Adam(lr=1e-5)`, `pw_weight` 1.0, fresh optimizer and
`torch.manual_seed(1)` each repeat, after one warm-up round.

## The round

| | wall s/round (5 repeats) | ms/step |
|---|---|---:|
| stock train phase | 1.124 1.068 1.768 1.191 1.094, median 1.124 | 1.27 |

## Host time per section

`perf_counter` around each call, no syncs inside the loop except the per-batch
`.item()`; median of 5 repeats. "synced" column: a sync after every section.

| section | us/step | share | P7 cubeful, 48 collectors (us/step) | synced us/step |
|---|---:|---:|---:|---:|
| index (`perm` slice, `x_all[idx]`, `y_all[idx]`) | 32.3 | 3% | - | 37.8 |
| forward (`net(xb)`, 4 Linear + ReLU, head, sigmoid) | 131.8 | 12% | 127 | 140.9 |
| **loss** (5 x `F.binary_cross_entropy(pred[:, j], yb[:, j])`, `torch.stack`, `* pow_dev`, `.mean()`) | **153.1** | **13%** | ~46 (one `0.5 * mse_loss`; 28.7 idle x P7's 1.6 load factor) | 158.4 |
| `optimizer.zero_grad()` | 30.5 | 3% | - | 33.8 |
| **`loss.backward()`** | **456.8** | **40%** | 328 | 474.0 |
| `optimizer.step()` (Adam, foreach) | 195.5 | 17% | 204 | 230.4 |
| `float(loss.item())` (waits for the GPU) | 38.7 | 3% | - | 16.9 |
| `per_output_sum += per_out.detach()` | 10.0 | 1% | - | 13.8 |
| per round: `net.to(device)` 1.5 ms, pool H2D 84.8 ms, `randperm` 0.5 ms, `net.to("cpu")` + per-output readback 2.8 ms | 101 | 9% | | 93 |
| total | **1139** (1.008 s) | | 781 (0.782 s / 1001 steps) | 1200 (1.062 s) |

The difference to cubeful is the loss: five per-output BCE terms on strided columns
are 18 device events forward and ~28 backward, against 3 and ~7 for `0.5 *
mse_loss`: +124 us in the loss and +129 us in backward per step.

## GPU time per step (torch.profiler)

GPU busy 264.5 us/step against 1139 us of host time: launch-bound by ~4.3x. 100
device events per step (90 kernel launches, 6 memcpys, 4 memsets) and one
`cudaStreamSynchronize` (the `.item()`).

| kernels | events/step | us/step |
|---|---:|---:|
| GEMMs (cutlass `simt_sgemm` fwd 5 addmm, bwd 9 mm, cublasLt split-K reduces) | 24 | 94.8 |
| Adam foreach (7 `multi_tensor_apply`, one entry per tensor) | 7 | 87.8 |
| elementwise (relu / threshold_backward, sigmoid, 5 BCE + 5 BCE backward, div_, fill_, select-backward copies, gathers, accumulate) | 48 | 48.9 |
| reductions (5 bias grads, 6 means) | 11 | 27.6 |
| memcpy / memset | 10 | 5.4 |

Plus 76.6 ms per round of pageable H2D copies (the 178 MB pool and the weights).

## What the graph needs in train_prob5

1. The net leaves the device every round: the round's values go back into the device
   storage the graph was captured on.
2. `Module.to` moves the grads too: drop `p.grad` before the move to the CPU.
3. Targets are (B, 5) and the per-output loss vector comes out beside the loss.
4. The loss weights need one device tensor for the run.
5. Capture and replay of a graph without RNG ops leave the CUDA generator, and the
   next `randperm`, bitwise unchanged.
6. `.item()` per batch: read the round's losses back once.
