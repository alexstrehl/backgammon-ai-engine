# Shared-memory row transport (`BG_COLLECT_SHM`), results

What the shared-memory transport (`row_transport.py`) buys on C-collector 0-ply
training against the pool pipe. Both arms use the graphed train step. Training is
bitwise unchanged (`tests/test_row_transport_bitwise.py`). Profile it starts from:
[p8_profile.md](p8_profile.md). Threadripper 9980X, RTX 5090, `bash c_api/build.sh ref
simd`, MKL shim, `OMP_NUM_THREADS=1`.

## Throughput

8 cells: {cubeful-0ply-c, dmp-0ply-c} x W in {48, 16} x {pipelined, `PIPELINE=0`},
each run pipe, shm, pipe, shm. Each run is `bench.sh` with 40 rounds x 4000 episodes,
batch 256, `--lr 1e-9`, `--seed 1`, `BG_POOL_ON_DEVICE=1`, `BG_TRAIN_CUDA_GRAPH=1`,
`BG_ROUND_PHASES=1`; the arms differ only in `BG_COLLECT_SHM=0/1`. ep/s is rounds
2..40 over the wall time between round lines (3..40 in brackets). Each start at load1
< 4 and < 3 busy cores; foreign load 0.04-0.08 cores. No slab overflow.

```bash
python experiments/c_collect/p8_table.py $OUT   # $OUT: sweep.tsv and run dirs in p4_sweep.py's layout
```

| cell | transport | ep/s, two runs (3..40) | **mean ep/s** | **shm / pipe** | round s | collect s | train s |
|---|---|---|---:|---:|---:|---:|---:|
| cubeful W=48 pipelined | pipe | 14346 (14296), 14872 (14859) | 14609 | | 0.274 | 0.015 | 0.227 |
| | shm | 20191 (20170), 21008 (20990) | **20600** | **1.41** | 0.194 | 0.004 | 0.190 |
| cubeful W=48 unpipelined | pipe | 8787 (8772), 8869 (8857) | 8828 | | 0.453 | 0.248 | 0.191 |
| | shm | 10972 (10962), 10999 (11009) | **10985** | **1.24** | 0.364 | 0.179 | 0.183 |
| cubeful W=16 pipelined | pipe | 11303 (11188), 11324 (11208) | 11314 | | 0.353 | 0.142 | 0.194 |
| | shm | 12340 (12204), 12346 (12210) | **12343** | **1.09** | 0.324 | 0.138 | 0.185 |
| cubeful W=16 unpipelined | pipe | 6362 (6367), 6246 (6248) | 6304 | | 0.635 | 0.427 | 0.192 |
| | shm | 7403 (7414), 7463 (7473) | **7433** | **1.18** | 0.538 | 0.354 | 0.182 |
| dmp W=48 pipelined | pipe | 14612 (14527), 14666 (14580) | 14639 | | 0.274 | 0.057 | 0.194 |
| | shm | 17408 (17297), 17562 (17453) | **17485** | **1.19** | 0.229 | 0.052 | 0.176 |
| dmp W=48 unpipelined | pipe | 7701 (7683), 7739 (7721) | 7720 | | 0.518 | 0.325 | 0.180 |
| | shm | 9024 (8998), 9094 (9094) | **9059** | **1.17** | 0.442 | 0.267 | 0.172 |
| dmp W=16 pipelined | pipe | 7685 (7641), 7672 (7623) | 7678 | | 0.521 | 0.322 | 0.183 |
| | shm | 8116 (7991), 8094 (7999) | **8105** | **1.06** | 0.493 | 0.318 | 0.174 |
| dmp W=16 unpipelined | pipe | 5033 (5038), 4992 (4991) | 5012 | | 0.798 | 0.604 | 0.181 |
| | shm | 5665 (5676), 5698 (5710) | **5681** | **1.13** | 0.704 | 0.530 | 0.172 |

"round s" is the mean interval between round lines; collect and train are the round
line's fields. The two runs of an arm agree within 1-4%; the shm gain is 6-41%, with
the same sign in all 16 pairs.

Cubeful 0-ply at W=48 pipelined:

| collector | ep/s |
|---|---:|
| Python CPU (P1) | 3603 |
| C (P4) | 5675 |
| C + graphed step (this sweep, pipe) | 14609 |
| **C + graphed step + shm** | **20600** |

## The master's phases (`BG_ROUND_PHASES`, s/round, means over both runs)

| cell | transport | collect | submit | assemble (concat) | h2d | perm | steps |
|---|---|---:|---:|---:|---:|---:|---:|
| cubeful 48p | pipe | 0.0168 | 0.0013 | 0.0299 | 0.0143 | 0.0056 | **0.2071** |
| | shm | 0.0050 | 0.0011 | **0.0000** | **0.0049** | 0.0053 | **0.1789** |
| cubeful 48u | pipe | 0.2477 | - | 0.0162 | 0.0091 | 0.0023 | 0.1777 |
| | shm | 0.1797 | - | 0.0000 | 0.0045 | 0.0022 | 0.1776 |
| cubeful 16p | pipe | 0.1419 | 0.0016 | 0.0181 | 0.0105 | 0.0028 | 0.1803 |
| | shm | 0.1374 | 0.0011 | 0.0000 | 0.0042 | 0.0036 | 0.1787 |
| cubeful 16u | pipe | 0.4277 | - | 0.0155 | 0.0112 | 0.0023 | 0.1778 |
| | shm | 0.3541 | - | 0.0000 | 0.0043 | 0.0021 | 0.1776 |
| dmp 48p | pipe | 0.0576 | 0.0017 | 0.0221 | 0.0123 | 0.0046 | 0.1767 |
| | shm | 0.0520 | 0.0011 | 0.0000 | 0.0043 | 0.0037 | 0.1686 |
| dmp 48u | pipe | 0.3251 | - | 0.0147 | 0.0082 | 0.0023 | 0.1678 |
| | shm | 0.2673 | - | 0.0000 | 0.0044 | 0.0022 | 0.1676 |
| dmp 16p | pipe | 0.3221 | 0.0012 | 0.0158 | 0.0127 | 0.0025 | 0.1679 |
| | shm | 0.3176 | 0.0011 | 0.0000 | 0.0040 | 0.0032 | 0.1687 |
| dmp 16u | pipe | 0.6047 | - | 0.0142 | 0.0094 | 0.0021 | 0.1676 |
| | shm | 0.5302 | - | 0.0000 | 0.0041 | 0.0021 | 0.1676 |

`h2d` includes a cuda sync under `BG_ROUND_PHASES`. Round 1 pins both arenas once:
0.12-0.17 s (cubeful W=48: 2 x 484,096 rows, 389 MB each, `cudaHostRegister`).

## Transport breakdown at W=48 (in-process timers, as in [p8_profile.md](p8_profile.md))

Cubeful 0-ply, W=48, epr 4000, graph, 12 rounds (2..12 averaged), two runs per arm
(pipe, shm, pipe, shm); each value is the mean of the two runs (runs in brackets).

| | pipelined, pipe | pipelined, shm | unpipelined, pipe | unpipelined, shm |
|---|---:|---:|---:|---:|
| round interval | 0.266 s (0.250, 0.282) | **0.197 s** (0.199, 0.196) | 0.446 s (0.442, 0.451) | **0.374 s** (0.379, 0.369) |
| main thread: assemble + h2d | 38.7 ms | 4.9 ms | 25.7 ms | 4.6 ms |
| main thread: steps | 214 ms (206, 222) | **185 ms** (185, 184) | 183 ms | 183 ms |
| worker `_c_collect` mean / max | 145 / 196 ms | 131 / 172 ms | 157 / 213 ms | 134 / 178 ms |
| worker handoff (collect end -> rows sent) mean / max | 30 / 54 ms | 0.1 / 0.1 ms | 19 / 34 ms | 0.4 / 1.8 ms |
| master result thread: read + unpickle per round | 102 + 17 ms | 0.6 + 0.5 ms | 89 + 14 ms | 0.7 + 0.5 ms |
| ... of it during the previous round's steps | 115 ms | 1 ms | 0 | 0 |
| last worker's collect end -> its rows loaded | 34 ms | 5 ms | 18 ms | 1 ms |
| last result -> the main thread asks for it (slack) | 48 ms | 45 ms | 0.3 ms | 0.4 ms |

* Pipelined W=48 is train-bound at 0.19 s per round with shm and main-thread-bound
  with the pipe: 39 ms of concatenate + pageable H2D, and steps 14% slower while the
  result thread read and unpickled 115 ms of rows during them. The next cap is the
  graphed step itself, ~4000 / 0.19 = 21k ep/s.
* Unpipelined, collection is 55-75 ms shorter in every cell.
* W=16 pipelined gains least (1.06-1.09x): its collection (0.32-0.35 s) is longer than
  the train step.
* With shm, W=48 0-ply pipelined reads 20.6k against W=16's 12.3k ep/s.
* Cost: 2 pinned arenas of ~1.3x-1.9x the round's rows, 2 x 389 MB at W=48 epr 4000.
