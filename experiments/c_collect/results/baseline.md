# Baseline: training throughput of the Python and GPU collectors

Threadripper 9980X (64 cores / 128 threads, AVX-512), RTX 5090 32 GB, Python 3.12,
torch 2.13.0+cu129, gcc 15.2. Every row is one [`../bench.sh`](../bench.sh) run with
the stock workloads (no C collector): 30 rounds of 4000 episodes, batch 256, `--lr
1e-9` (frozen net), `--seed 1 --torch-seed 1`, `BG_ROUND_TIMING=1`,
`OMP_NUM_THREADS=1`, MKL shim (`LD_PRELOAD=c_engine/libfakeintel.so`),
`BG_POOL_ON_DEVICE=1`. ep/s = episodes of rounds 2..30 over the wall time between the
round-1 and round-30 log lines (rounds 3..30 in brackets). collect / train = mean
seconds per round over rounds 2..30; train share = train / (collect + train).
Pipelined, collect is only the wait for the round submitted before the previous train
step.

```bash
WORKLOAD=<workload> OUT_DIR=$OUT bash experiments/c_collect/bench.sh
```

## The five workloads (W=48, epr=4000, 30 rounds)

| workload | collector (from the log) | ep/s, rounds 2..30 (3..30) | collect s | train s | train share | rows/round | load1 at start |
|---|---|---:|---:|---:|---:|---:|---:|
| cubeful-money 0-ply | Python CPU, pipelined | **3603** (3591) | 0.256 | 0.831 | 76% | 253,500 | 1.8 |
| cubeful-money 1-ply, GPU | vectorized, shared context, 48 procs x B=32, 3 eval servers | **858** (859) | 4.130 | 0.514 | 11% | 254,034 | 16.9 |
| cubeful-money 1-ply, CPU | Python CPU, pipelined | **187** (185) | 20.44 | 0.808 | 4% | 253,535 | 26.3 |
| cubeless-money 1-ply | vectorized, shared context, 48 procs x B=32, 3 eval servers | **838** (837) | 4.322 | 0.438 | 9% | 226,673 | 20.9 |
| dmp 0-ply | Python CPU, pipelined | **3221** (3205) | 0.361 | 0.853 | 70% | 236,446 | 14.6 |

| workload | bench.sh | resolved flags and env beyond the common ones above |
|---|---|---|
| cubeful-money 0-ply | `WORKLOAD=cubeful-0ply bash experiments/c_collect/bench.sh` | `--game-mode cubeful-money --pipeline-collect --resume best_models/cubeful_money_512_512_256_256.pt --device cuda --workers 48` |
| cubeful-money 1-ply, GPU | `WORKLOAD=cubeful-1ply-gpu ...` | `--game-mode cubeful-money --oneply --vectorized-workers 48 --resume best_models/cubeful_money_512_512_256_256.pt --device cuda --workers 48`, `BG_EVAL_SLOT_DTYPE=fp16` |
| cubeful-money 1-ply, CPU | `WORKLOAD=cubeful-1ply-cpu ...` | `--game-mode cubeful-money --oneply --no-vectorized-collect --pipeline-collect --resume best_models/cubeful_money_512_512_256_256.pt --device cuda --workers 48` |
| cubeless-money 1-ply | `WORKLOAD=cubeless-1ply ...` | `--game-mode cubeless-money --oneply --vectorized-workers 48 --resume best_models/cubeless_money_512_512_256_256.pt --device cuda --workers 48`, `BG_VEC_CUBELESS_FAST=1` |
| dmp 0-ply | `WORKLOAD=dmp-0ply ...` | `--game-mode dmp --pipeline-collect --resume best_models/dmp_512_512_256_256.pt --device cuda --workers 48` |

The GPU 1-ply rows pass `--vectorized-workers 48`: the default worker count is capped
at 32, which reads 692 ep/s (collect 5.29 s, train 0.47 s) against 858. They run
unpipelined (the vectorized collector disables `--pipeline-collect`). dmp 0-ply
without `BG_POOL_ON_DEVICE=1`: 3003 ep/s (3018), train 0.969 s.

## 0-ply: worker count, pipelining, noise

`WORKLOAD=cubeful-0ply` (or `dmp-0ply`), 30 rounds, epr 4000, seed 1. Each cell is
ep/s over rounds 2..30 / train-phase seconds per round. "f" = foreign CPU load in
cores at the start; Tctl at the start.

| config | first pass, idle box | 1 f, 85 °C | ~8 f | interleaved pass 1 | interleaved pass 2 |
|---|---|---|---|---|---|
| W=32 pipelined | 3410 / 0.58 | | 2242 / 0.84 | 2340 / 0.82 (9 f, 80 °C) | 2286 / 0.76 (5 f, 85 °C) |
| **W=48 pipelined** | **3603 / 0.83** | 3172 / 0.87 | 2522 / 0.91 | 3120 / 0.89 (1 f, 72 °C) | 2724 / 0.81 (7.5 f, 87 °C) |
| W=60 pipelined | 2933 / 1.15 | | | 2753 / 0.93 (8.6 f, 82 °C) | 2875 / 1.04 (5 f, 71 °C) |
| W=48 unpipelined | 2239 / **0.46** | | | 1774 / 0.52 (8.4 f, 78 °C) | 2078 / 0.44 (5 f, 72 °C) |
| dmp W=48 pipelined | 3221 / 0.85 | | | 2530 / 0.92 (8.3 f, 79 °C) | 3289 / 0.73 (5 f, 70 °C) |
| dmp W=48 unpipelined | 2127 / 0.51 | | | | |

Unpipelined collect (4000 episodes on 48 workers): 1.31 s idle, 1.71 / 1.47 s in the
two interleaved passes.

* The train phase is 0.44-0.52 s per round unpipelined and 0.81-0.91 s at W=48
  pipelined (~0.47 ms per 256-row Adam step alone: the master's Python/launch loop).
* Pipelining pays +31% to +76% at W=48. W=48 is best or tied.
* W=48 pipelined read 2522-3603 over five runs: a single 30-round read cannot resolve
  0-ply differences below ~15-20%. 5-9 cores of foreign load cost up to 30%. The GPU
  1-ply rows are steadier (every round interval within 8% of the median).

## C forward, `c_api/bench_forward.py --backends simd`

`bash c_api/build.sh ref simd` (gcc 15.2, `-O3 -march=native`, no `-ffast-math`);
backend `simd avx512 VL16 mr<=6 nv4 (mr1 nv8) sparse1=1 sparse_mr=12`, one thread.
Rows are random-playout encodings.

```bash
BENCH_CORE=62 CAPI_BINS=$OUT/bins \
taskset -c 62 env LD_PRELOAD=c_engine/libfakeintel.so OMP_NUM_THREADS=1 \
  python c_api/bench_forward.py --backends simd --model best_models/<net>.pt --json $OUT/<net>.json
```

us per row (us/call / n). torch-py = single-thread torch forward on the same rows:

| net | params | n=16 | n=64 | n=512 | n=2048 | torch-py n=16 / 64 / 512 / 2048 | max abs diff vs torch | load1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `cubeful_money_512_512_256_256` (4L prod) | 562,945 | **1.60** | **1.68** | **1.67** | **1.75** | 8.0 / 4.8 / 3.9 / 4.6 | 1.9e-6 | 2.9 |
| `cubeful_money_768_768_256_256_256` (5L-wide) | 1,073,665 | **3.72** | **3.55** | **3.63** | **3.69** | 12.7 / 8.4 / 7.3 / 7.2 | 3.3e-6 | 2.8 |
| `cubeful_money_768_512_256_256_256_256` (6L) | 877,057 | **2.98** | **2.88** | **2.97** | **3.05** | 11.3 / 7.2 / 6.1 / 6.1 | 3.1e-6 | 2.6 |
| `cubeless_prob5_512_512_256_256` | 561,925 | **1.82** | **1.63** | **1.73** | **1.81** | 8.3 / 5.0 / 3.9 / 3.8 | 1.2e-6 | 4.8 |

simd is flat from n=16 to n=2048 (590-700 GFLOPS on the 4L nets, 575-610 on 5L/6L)
and 3.4-5.0x faster than single-thread torch at n=16, 2.0-2.6x at n >= 512. At n=1 a
call costs 2.1 us (4L), 8.8 us (5L-wide), 5.2 us (6L), 2.5 us (prob5).

## CPU-side microbench

All 8000 positions of the fixed pool, stock `c_engine/libbg_engine.so` (`bash
c_engine/build_unix.sh`, `-O2`), `taskset -c 62`:

| step | us per position |
|---|---:|
| `expand_21` C call (21 move-gens + encodes) | 324.7 |
| 21x `get_legal_plays` (move-gen only) | 306.8 |
|   6 doubles only | 244.5 |
|   15 non-doubles only | 62.7 |
| `get_legal_plays_encoded(state, actual roll)` | 14.5 |
|   + materialise every successor (Python) | 37.8 |

Rows per target forward: mean 439, median 361, p90 939, max 3362. Duplicate boards
across rolls: 10.2% of rows. Doubles are 80% of the 1-ply move generation.

## Reading

* 0-ply is train-bound: at 0.46-0.51 s of training per round an infinitely fast
  collector caps cubeful 0-ply at ~8,300 ep/s and dmp at ~7,700 unpipelined; at the
  pipelined train phase (0.81-0.91 s) the cap is ~4,300-4,800 ep/s.
* No 1-ply path is train-bound: train is 11% (cubeful GPU), 9% (cubeless GPU) and 4%
  (cubeful CPU) of the round.
