# C collector in the trainer: cost, round parity, throughput

`train_batch.py --c-collect` (`bgc_collect`, `simd` backend) through the trainer's
own worker pool, eager train step, pool pipe (`BG_TRAIN_CUDA_GRAPH=0
BG_COLLECT_SHM=0`). Threadripper 9980X (64 cores / 128 threads, AVX-512), RTX 5090,
Python 3.12, torch 2.13.0+cu129, gcc 15.2, `bash c_api/build.sh ref simd` (backend
`simd avx512 VL16 mr<=6 nv4 (mr1 nv8) sparse1=1 sparse_mr=12 delta0=1`). "pre-P5" =
the collector before the levers of [p5_results.md](p5_results.md).

## Export and open cost

One core, 4L cubeful prod net unless noted. The export/open rows vary 2-5x between
runs with the page cache and the load; read them as "about a millisecond".

| step | pre-P5 | P5 build |
|---|---:|---:|
| `export_live` in the master, CPU net (CUDA net) | 2.3 ms (3.7 ms) | 0.4 ms |
| `Engine(path)` open, file just written | 1.8 ms (dmp 0.5, 5L-wide 7.4) | 0.7 ms (dmp 0.5, 5L-wide 0.8) |
| fresh `Engine` + `collect(84 episodes)` minus a warm engine's `collect` of the same seed, 0-ply | +0.24 ms of 97 ms | +0.86 ms of 94 ms |
| the same, dmp 0-ply | +0.41 ms of 160 ms | +0.74 ms of 156 ms |
| the same, 1-ply (2 episodes) | +0.13 ms of 60 ms | +0.61 ms of 50 ms |
| master worker args per round, W=48: old (state_dict clone + 48 pickles) | 24.8 ms, 108 MB pickled (5L-wide 41.8 ms, 206 MB) | same |
| master worker args per round, W=48: `--c-collect` (export + 48 path pickles) | 2.65 ms (5L-wide 6.3 ms) | about 1 ms |

A fresh engine per task costs under 1% of a 0-ply worker task (84 episodes at W=48)
and about 0.03% of a 1-ply one, so workers open the engine per task.

## Parity level 3

Both arms are `train_batch.py` on the mode's 4L prod net with `--lr 1e-9`, `--device
cpu`, 8 workers, `--seed 11`, 2000 episodes per round, `BG_COLLECT_DUMP` on. Bound:
1e-5 at 0-ply, 1e-4 at 1-ply (x10 relative above 1).

```bash
python experiments/c_collect/round_parity.py --game-mode <m> [--oneply] [--no-jacoby] \
  --episodes 2000 --rounds <r> --workers 8 [--pipeline] --check-ties --out $OUT/<cell>
```

| cell | rounds | workers agreeing to the end | Python rows aligned (bit-exact encodings) | aligned targets max \|d\| | out of bound | target mean, Python / C | per-feature means max \|d\| |
|---|---:|---|---:|---:|---:|---|---:|
| cubeful 0-ply, pipelined | 2 | 7/8, 8/8 | 247,920 / 255,305 (97.11%) | 2.4e-6 | 0 | -0.113084 / -0.113084 (r1) | 1.6e-5 (r1), 0 (r2) |
| cubeful no-Jacoby 0-ply | 1 | 8/8 | 145,368 / 145,368 (100%) | 3.6e-6 | 0 | equal | 0 |
| cubeless 0-ply, pipelined | 2 | 8/8, 6/8 | 213,263 / 226,685 (94.08%) | 2.2e-6 | 0 | +0.011624 / +0.011624 (r2) | 0 (r1), 8.9e-6 (r2) |
| dmp 0-ply, pipelined | 2 | 8/8, 8/8 | 237,524 / 237,524 (100%) | 7.2e-7 | 0 | equal | 0 |
| cubeful 1-ply | 1 | 7/8 | 121,217 / 128,602 (94.26%) | 2.0e-6 | 0 | -0.112368 / -0.112368 | 1.6e-5 |
| cubeless 1-ply | 1 | 7/8 | 111,234 / 113,612 (97.91%) | 2.0e-6 | 0 | +0.012101 / +0.012102 | 2.2e-5 |
| dmp 1-ply | 1 | 8/8 | 117,683 / 117,683 (100%) | 6.6e-7 | 0 | equal | 0 |

Per-worker row counts and worker seeds are equal in every round. Every worker whose
streams part was replayed by `--check-ties`; the first decision C takes differently
is a near-tie:

| cell | round, worker (seed) | parts in episode (of the worker's 250) | decision | C score margin |
|---|---|---:|---|---:|
| cubeful 0-ply and 1-ply | r1 w3 (815623033) | 134 | checker | 1.2e-7 |
| cubeless 0-ply | r2 w0 (1918102982) | 131 | checker | 3.6e-7 |
| cubeless 0-ply | r2 w5 (1701494130) | 126 | checker | 0 (exact C tie, first candidate taken) |
| cubeless 1-ply | r1 w0 (1942955373) | 209 | checker | 2.4e-7 |

5 partings in 20,000 episodes (4 distinct in 18,000 distinct games: the cubeful 1-ply
cell replays the cubeful 0-ply games), all checker plays, margins <= 3.6e-7. Up to the
parting every row is bit-exact; the cells without a parting agree to 0.

## Throughput

Every cell is two `bench.sh` runs: 30 rounds x 4000 episodes, batch 256, `--lr 1e-9`,
`--seed 1 --torch-seed 1`, `--device cuda`, `BG_POOL_ON_DEVICE=1`, MKL shim,
`OMP_NUM_THREADS=1`. ep/s is rounds 2..30 over the wall time between round lines
(rounds 3..30 in brackets); collect / train are mean seconds per round; train share
= train / (collect + train). Pipelined, collect 0.000 means train-bound. Arms ran
interleaved with a reversed second pass, each start at load1 < 4 and < 3 busy cores,
foreign load <= 0.14 cores (1.05-1.08 for the first 7 runs).

```bash
python experiments/c_collect/p4_sweep.py --plan {zeroply,oneply,other} --out $OUT
WORKLOAD=<name> WORKERS=<W> [PIPELINE=0] bash experiments/c_collect/bench.sh   # one cell
```

### cubeful-money 0-ply

| collector | W | pipelined | ep/s, two runs (3..30) | **mean ep/s** | vs stock, same pipelining | collect s | train s | train share | rows/round |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| stock (Python CPU) | 48 | yes | 3423 (3415), 3534 (3518) | **3479** | 1.00 | 0.197 | 0.928 | 82% | 253,500 |
| C | 48 | yes | 5774 (5822), 5577 (5602) | **5675** | **1.63x** | 0.000 | 0.676 | 100% | 253,444 |
| C | 32 | yes | 5771 (5776), 5682 (5701) | **5726** | 1.65x | 0.000 | 0.671 | 100% | 254,258 |
| C | 16 | yes | 5468 (5481), 5834 (5840) | **5651** | 1.62x | 0.000 | 0.687 | 100% | 253,745 |
| stock (Python CPU) | 48 | no | 2269 (2266), 2162 (2167) | **2215** | 1.00 | 1.292 | 0.499 | 28% | 253,467 |
| C | 48 | no | 5106 (5130), 4879 (4905) | **4992** | **2.25x** | 0.250 | 0.536 | 68% | 253,453 |
| C | 32 | no | 5090 (5066), 4587 (4579) | **4838** | 2.18x | 0.286 | 0.527 | 65% | 254,277 |
| C | 16 | no | 4060 (4052), 3850 (3855) | **3955** | 1.79x | 0.449 | 0.547 | 55% | 253,664 |

### cubeful-money 1-ply

| collector | W | pipelined | ep/s, two runs (3..30) | **mean ep/s** | vs GPU recipe | per worker | collect s | train s | train share | rows/round |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPU recipe (vectorized, shared context, fp16 slots, `--vectorized-workers 48`) | 48 | no (the vectorized collector cannot pipeline) | 931 (931), 938 (938) | **934** | 1.00 | 19.5 | 3.711 | 0.554 | 13% | 254,034 |
| C | 32 | yes | 1325 (1326), 1324 (1324) | **1325** | 1.42x | 41.4 | 2.439 | 0.562 | 19% | 254,281 |
| C | 48 | yes | 1737 (1736), 1749 (1751) | **1743** | **1.87x** | 36.3 | 1.637 | 0.639 | 28% | 253,482 |
| C | 60 | yes | 1940 (1943), 1950 (1960) | **1945** | 2.08x | 32.4 | 1.230 | 0.804 | 40% | 253,693 |
| C | 64 | yes | 1974 (1998), 1985 (1979) | **1979** | **2.12x** | 30.9 | 1.148 | 0.849 | 42% | 253,676 |

### cubeless-money 1-ply and dmp 0-ply, W=48

| workload | collector | pipelined | ep/s, two runs (3..30) | **mean ep/s** | vs stock | collect s | train s | train share | rows/round |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| cubeless-money 1-ply | GPU recipe (vectorized, `BG_VEC_CUBELESS_FAST=1`, `--vectorized-workers 48`) | no | 895 (895), 890 (890) | **892** | 1.00 | 3.984 | 0.486 | 11% | 226,673 |
| cubeless-money 1-ply | C | yes | 1206 (1206), 1207 (1207) | **1206** | **1.35x** | 2.681 | 0.619 | 19% | 226,648 |
| dmp 0-ply | stock (Python CPU) | yes | 3578 (3598), 3500 (3468) | **3539** | 1.00 | 0.307 | 0.799 | 72% | 236,446 |
| dmp 0-ply | C | yes | 5571 (5611), 5475 (5497) | **5523** | **1.56x** | 0.000 | 0.697 | 100% | 236,455 |

Rows per round agree between the arms within 0.3%.

* 0-ply is train-bound with the C collector at any W >= 16, at ~5.7k ep/s: the train
  phase is 0.67-0.69 s pipelined and 0.53-0.55 s unpipelined.
* Unpipelined, the C collector needs 0.25 s per 4000-episode round at W=48 (Python:
  1.29 s), 0.45 s at W=16.
* 1-ply is collect-bound and scales to W=64: per worker 41.4 ep/s at W=32, 30.9 at
  W=64 (the all-core AVX-512 clock drop, p5_results.md). It beats the GPU recipe at
  every W.
* cubeless 1-ply gains less (1.35x): 25 ep/s per worker at W=48 against cubeful's 36.
