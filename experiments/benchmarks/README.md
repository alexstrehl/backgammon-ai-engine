# Per-decision compute cost

Single-core cost per decision of PureTD, gnubg 1.08.003, Open Sage and
HedgeHog, each through its own production code path (move generation,
evaluation, move filter, cube logic), on a fixed pool of decision points
that every engine replays: checker decisions (board and dice, plus the cube
state when cubeful) and pre-roll cube decisions (board and cube state).

PureTD runs through its C API ([c_api/](../../c_api/README.md)), one row
per forward backend: `torch` (libtorch/ATen `addmm` on MKL; the paper's
main tables) and `simd` (hand-written AVX2/AVX-512 kernels; the paper's
optimized-kernel appendix).

## Protocol

| | Zen 3 | Zen 5 |
|---|---|---|
| CPU | AMD Ryzen 9 5950X | AMD Ryzen 7 9700X |
| kernel | 6.8.0-142-generic | 6.8.0-137-generic |
| cpufreq | acpi-cpufreq, schedutil, boost on | amd-pstate-epp, EPP performance |
| results | [results/zen3_5950x_20261014/](results/zen3_5950x_20261014/) | [results/zen5_9700x_20261014/](results/zen5_9700x_20261014/) |

Both: Ubuntu 22.04, gcc 11.4, Python 3.12, torch 2.5.1+cpu (MKL), one
pinned core (core 1, its SMT sibling idle) for every engine, box otherwise
idle (`suite.sh` checks before each group). `env.txt` in each results
directory has the full environment and binary md5s.

- Two groups (`suite.sh fast|slow`), each run as eight sweeps; a sweep runs
  every job of its group once, each benchmark in a fresh process, one at a
  time, with 90 s idle between sweeps (`SWEEP_GAP`).
- `fast`: every 0-ply row, gnubg 1-ply, Open Sage 1-ply cubeless,
  HedgeHog 0-ply and the network sizes.
- `slow`: the other 1-ply rows, gnubg 2-ply and HedgeHog 1-ply.
- Every row times the full pools: 8,000 cubeless checker items; 10,000
  checker and 6,000 cube items cubeful.
- Every timed region follows a 5 s busy-loop clock ramp (`BENCH_WARMUP_S`,
  set by `suite.sh`; the scripts' own default is 3 s) and 20 untimed
  warm-up items.
- A row is the mean of its eight sweeps with a 95% t-interval over them
  (t(7) = 2.365, `aggregate_sweeps.py`); each results directory has the
  aggregate of every job in `aggregate.txt`.
- PureTD 1-ply rows (cubeful and prob5, both backends) use the C API root
  move filter at 0.08 (`P1_ROOT_FILTER`; see Fairness).
- gnubg: threads-disabled build (`--disable-threads`), timed through an
  additive 185-line patch to its Python module (`gnubgmodule.c`, the
  `benchpool` function) that calls `FindnSaveBestMoves` /
  `GeneralCubeDecisionE` per item in C; no engine code is modified.
- Open Sage: fixed pool only, timed by the loops in
  [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](../../third_party_opponents/opensage/opensage_bench/sage_bench.cpp), which
  call its decision functions directly (see Fairness).
- HedgeHog: fixed pool only, timed by the `hh_bench_*` loops in the shim
  (engine `e8847f4`).
- Earlier versions of these tables were measured with the MKL shim
  `c_engine/libfakeintel.so` preloaded; only the torch rows differ
  materially. The shim is off by default. It overrides MKL's CPU-vendor
  check, which affects only the torch rows.

Pools are fixed, so every sweep of a row times the same items, on both
hosts. The t-interval is repeatability: the between-sweep sd has a
median of 0.4-0.7% of the mean per job for PureTD, gnubg and Open Sage and
0.7-2.7% for HedgeHog (up to 5%, Aureus), and reaches 3-4% for some Zen 5
`torch` rows.
The per-sweep bootstrap CI on each raw line is the sampling uncertainty of
the fixed sample of items, which more sweeps do not reduce.

On Zen 5 the `torch` 0-ply rows spread more between sweeps: cubeless 4,946
[4,798, 5,094] decisions/s (±3.0%), cubeful checker (±2.7%) and cube
(±1.4%), against at most ±0.9% for every Zen 3 PureTD row in the engine tables
below. Open Sage cubeless 1-ply on Zen 3 is ±5.4%: one sweep at 667, the
other seven at 799-818.

## Fairness

Every engine runs its production defaults:

- gnubg: its pruning nets, the Normal move filter and its evaluation cache.
- Open Sage: cubeless 1-ply uses its default move filter (the top 5 moves
  within 0.08 at 0-ply). Cubeful 1-ply has no pruning setting: checker play
  scores every move and every reply; cube decisions pre-rank each roll's
  moves with PubEval and score the top 15 with the net. Open Sage clears its
  position cache after each 1-ply cube decision, as shipped; the clear runs
  outside the timer. Each 1-ply cubeful checker item starts on cleared
  caches (untimed), as after the cube decision that precedes a roll.
  Cubeless rows attach its bearoff database, as its analyzer does.
- HedgeHog: full width; at 0-ply the successors of a decision are
  evaluated in one batch with `eval_ensemble_successors`, which HedgeHog
  ships but its own search does not call. Cubeful rows run with its Jacoby
  rule on, as the pool's games and the other engines.
- PureTD: full width at 0-ply; at 1-ply the benchmark's root filter, 0.08
  (`P1_ROOT_FILTER`; the C API's default is full width; rows say
  `root-filter 0.08`): every legal move is
  screened at 0-ply, those within 0.08 of the best (and game-ending moves)
  are expanded with every reply scored, and cube decisions are full 1-ply.
  On every item of both committed pools the filtered choice
  equals the full-width choice on 100% of the prob5 and 99.91% of the
  cubeful decisions (7 of 8,073; 0.010 mEq/decision), expanding 5.7 and
  4.7 of about 20.5 candidates ([rootf_validate.py](rootf_validate.py),
  [results/rootf_validate_20261011.txt](results/rootf_validate_20261011.txt)).

Single thread everywhere (gnubg `--disable-threads`, MKL/OpenMP/ATen 1
thread, Open Sage `parallel_evaluate=False`). No engine batches across
decisions. Fixed-pool items are converted to each engine's native format
before timing; every engine's items are then timed inside native code
(`CLOCK_MONOTONIC`) around one full decision, with no Python call in the
timed region. The pools include dances and forced moves, timed as each
engine handles them. Open Sage (bgsage 2.0.20260907): cubeful checker and cube
decisions are the functions its C API (`bgsage_mobile_analyze`) runs,
without the C API's JSON reply and without the second 1-ply evaluation it
makes after a cube decision for that reply; cubeless decisions are its move
generation and `best_move_index` (the call behind its Python package's
`best_move_index`). Cubeful picks equal the C API's on every item it
answers within its 1 s budget (2 of the 10,000 1-ply checker items exceed
it); cubeless picks equal `best_move_index`'s with the bearoff database
attached.

| engine | timed per item: checker / cube |
|---|---|
| PureTD | `bgc_best_move` (move generation, encoding, forward, selection) / `bgc_offer_double` (both evaluations + decision) |
| gnubg | `FindnSaveBestMoves` + `RefreshMoveList` / `GeneralCubeDecisionE` + `FindCubeDecision` (`gnubg.benchpool`, [third_party_opponents/gnubg/gnubg_bench/](../../third_party_opponents/gnubg/gnubg_bench/)) |
| Open Sage | `possible_boards` + candidate scoring (cubeful), `possible_boards` + `best_move_index` (cubeless) / `cube_decision_1ply`, `cube_decision_nply` ([third_party_opponents/opensage/opensage_bench/](../../third_party_opponents/opensage/opensage_bench/sage_bench.cpp)) |
| HedgeHog | `hh_best_move` (cubeful), `hh_best_move_cubeless` / `hh_double_action` (`hh_bench_*` loops in the shim) |

Caches: gnubg and Open Sage keep their evaluation caches across items
(PureTD has none; Open Sage's serve its 1-ply searches and are cleared,
untimed, after each 1-ply cube decision). Build flags: PureTD and the
HedgeHog shim `-O3 -march=native`, Open Sage its wheel's `-O3 -ffast-math
-funroll-loops -mavx2 -mfma`, gnubg its configure flags with `--enable-simd=fma`.

## Inputs

- PureTD cubeful:
  [best_models/cubeful_money_512_512_256_256.pt](../../best_models/cubeful_money_512_512_256_256.pt)
  (md5 `6afda0f8`).
- PureTD cubeless (prob5):
  [best_models/cubeless_prob5_512_512_256_256.pt](../../best_models/cubeless_prob5_512_512_256_256.pt)
  (md5 `ac95a983`).
- PureTD C API: `VENV=/path/to/venv/bin/python bash c_api/build.sh torch simd`
  (`torch` links that interpreter's libtorch).
- gnubg 1.08.003 built `--disable-threads --enable-simd=fma` with the
  `gnubg.benchpool` patch: `bash third_party_opponents/gnubg/gnubg_bench/build.sh
  $HOME/gnubg-bench` (needs flex, bison, glib, python3 headers). A threaded
  build polls at 10 ms and reads ~99 moves/s at every ply; an unpatched
  build times the fixed pool per Python call and says so.
- [Open Sage](https://github.com/markbgsage/bgsage) `v2.0.20260907` exactly
  (bgsage 1.x gives different rows; commit
  `4b10c69318c3b80e2a7618232540b10a76217d20`), production model `stage11`;
  the fixed-pool loops built against its engine library with the wheel's
  flags: `bash third_party_opponents/opensage/opensage_bench/build.sh` (needs
  cmake).
- HedgeHog engine `e8847f4`: shim (with the `hh_bench_*` loops) and `.ogxf`
  models per [third_party_opponents/hedgehog/README.md](../../third_party_opponents/hedgehog/README.md).
- Pools: [fixedpool_positions.json](fixedpool_positions.json), 8,000
  cubeless checker points (board and dice) from seeded gnubg 0-ply self-play;
  [fixedpool_cubeful_positions.json](fixedpool_cubeful_positions.json), the
  first 10,000 checker points (board, dice, cube state) and 6,000 cube points
  of seeded PureTD cubeful self-play (seed 1, Jacoby on), generated by an
  earlier 4L cubeful checkpoint (md5 `77a3205c`). `fixedpool_bench.py --mode
  cubeful --regen-pool` overwrites it with a pool from the current cubeful
  model.

```bash
pip install bgsage==2.0.20260907
bash third_party_opponents/opensage/opensage_bench/build.sh
```

## Commands

The whole suite, from the repo root. `suite.sh` runs one group as eight
sweeps, one job at a time, and resumes a group from its finished sweep
files. `VENV` is a python with torch and bgsage; the HedgeHog shim and
models are found as in [third_party_opponents/hedgehog/README.md](../../third_party_opponents/hedgehog/README.md).

```bash
export BENCH_CORE=<core> VENV=/path/to/venv/bin/python GNUBG=$HOME/gnubg-bench/bin/gnubg
bash experiments/benchmarks/suite.sh ~/bench_results/<run> fast   # bench.sh fast tier, HedgeHog 0-ply
bash experiments/benchmarks/suite.sh ~/bench_results/<run> slow   # bench.sh slow tier, HedgeHog 1-ply
python experiments/benchmarks/aggregate_sweeps.py ~/bench_results/<run>/fast/bench_sweep*.txt   # one job at a time
```

One sweep of all rows, gaps included, takes about 8 min (`fast`) and
50 min (`slow`) on the 5950X, 8 and 34 min on the 9700X. Eight sweeps of
both groups take about 7.8 h and 5.5 h, plus about 6 min per `fast` sweep for the `cap` job (network sizes) on either host. `ROWS` selects rows (tokens in the `bench.sh` and
`suite.sh` headers). `SMOKE=1` runs every job with tiny counts. `bench.sh`
alone runs both tiers (`TIER=all`) as one sweep (`SWEEPS=1`); the paper rows
are eight sweeps (`suite.sh`).

The jobs of one sweep (`bench.sh`, then the extra `suite.sh` jobs). The
gnubg script runs inside the gnubg binary's own interpreter. PureTD rows are
labelled `prob5c[<b>]` (cubeless) and `PureTD-C[<b>]` (cubeful) for backend
`<b>` (`CAPI_BACKENDS`, default `torch simd`).

```bash
export BENCH_CORE=<core>
G=$HOME/gnubg-bench/bin/gnubg

# gnubg: cubeless and cubeful checker, cube; 0/1-ply (fast), 2-ply (slow)
GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 $G -t -q -p third_party_opponents/gnubg/fixedpool_bench_gnubg.py
GNUBG_FP_MODE=both GNUBG_FP_PLIES=2 $G -t -q -p third_party_opponents/gnubg/fixedpool_bench_gnubg.py

# fast: full pools
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-0ply,sage-0ply,sage-1ply --backends torch,simd
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,sage0 --backends torch,simd
# slow
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-1ply --backends torch,simd --root-filter 0.08
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,sage --plies 1 --backends torch,simd --root-filter 0.08

# HedgeHog 0-ply (fast; suite.sh job hh0) and 1-ply (slow; job hh1)
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs prob5c-0ply,hh-fox-0ply,hh-aureus-0ply --backends torch,simd
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines puretdc,hh-fox0,hh-aureus0 --backends torch,simd
python experiments/benchmarks/fixedpool_bench.py --mode cubeless --configs hh-fox-1ply,hh-aureus-1ply
python experiments/benchmarks/fixedpool_bench.py --mode cubeful --engines hh-fox1,hh-aureus1

# network sizes (fast; job cap)
python experiments/benchmarks/capacity_costbench.py --models all --backends torch,simd
```

## Results

Decisions/s, mean of eight sweeps [95% t-interval]; in parentheses the
per-decision cost relative to PureTD `torch` 0-ply on the same benchmark
and host (its rate over the row's). Zen 3 = Ryzen 9 5950X, Zen 5 = Ryzen 7
9700X. Raw sweeps: `results/<host>/{fast,slow}/<job>_sweep<i>.txt`; aggregates: `results/<host>/aggregate.txt`.
PureTD 1-ply = root filter 0.08.

### Cubeless money, fixed pool (checker)

| engine | depth | Zen 5 | Zen 3 |
|---|---|---|---|
| PureTD torch | 0-ply | 4,946 [4,798, 5,094] (1.00×) | 4,689 [4,679, 4,700] (1.00×) |
| PureTD simd | 0-ply | 37,308 [37,137, 37,478] (0.13×) | 15,838 [15,734, 15,941] (0.30×) |
| gnubg | 0-ply | 41,268 [41,173, 41,362] (0.12×) | 32,816 [32,629, 33,004] (0.14×) |
| Open Sage | 0-ply | 19,498 [19,452, 19,543] (0.25×) | 16,889 [16,851, 16,927] (0.28×) |
| PureTD torch | 1-ply | 80.9 [80.7, 81.1] (61×) | 79.2 [79.2, 79.3] (59×) |
| PureTD simd | 1-ply | 567 [565, 568] (8.7×) | 226 [225, 226] (21×) |
| gnubg | 1-ply | 1,394 [1,386, 1,402] (3.5×) | 1,027 [1,025, 1,030] (4.6×) |
| Open Sage | 1-ply | 1,055 [1,052, 1,058] (4.7×) | 793 [750, 836] (5.9×) |
| gnubg | 2-ply | 75.9 [75.7, 76.0] (65×) | 55.1 [54.9, 55.3] (85×) |

### Cubeful money, fixed pool

| engine | depth | checker Zen 5 | checker Zen 3 | cube Zen 5 | cube Zen 3 |
|---|---|---|---|---|---|
| PureTD torch | 0-ply | 5,596 [5,447, 5,746] (1.00×) | 5,200 [5,190, 5,210] (1.00×) | 23,835 [23,498, 24,172] (1.00×) | 13,419 [13,347, 13,490] (1.00×) |
| PureTD simd | 0-ply | 44,093 [43,886, 44,300] (0.13×) | 18,477 [18,422, 18,533] (0.28×) | 255,731 [252,934, 258,528] (0.09×) | 114,110 [113,059, 115,161] (0.12×) |
| gnubg | 0-ply | 40,341 [40,089, 40,592] (0.14×) | 32,562 [32,370, 32,755] (0.16×) | 736,019 [733,958, 738,080] (0.03×) | 605,593 [601,011, 610,174] (0.02×) |
| Open Sage | 0-ply | 19,696 [19,452, 19,941] (0.28×) | 16,713 [16,529, 16,897] (0.31×) | 339,400 [335,724, 343,075] (0.07×) | 277,828 [275,278, 280,377] (0.05×) |
| PureTD torch | 1-ply | 64.3 [64.1, 64.5] (87×) | 63.3 [63.2, 63.3] (82×) | 138 [138, 138] (172×) | 137 [137, 137] (98×) |
| PureTD simd | 1-ply | 477 [477, 478] (11.7×) | 185 [184, 186] (28×) | 1,067 [1,064, 1,070] (22×) | 426 [424, 429] (31×) |
| gnubg | 1-ply | 1,483 [1,477, 1,490] (3.8×) | 1,113 [1,109, 1,118] (4.7×) | 4,263 [4,231, 4,295] (5.6×) | 3,235 [3,217, 3,253] (4.1×) |
| Open Sage | 1-ply | 51.1 [50.8, 51.4] (109×) | 40.1 [39.8, 40.4] (130×) | 1,835 [1,820, 1,850] (13.0×) | 1,432 [1,423, 1,442] (9.4×) |
| gnubg | 2-ply | 77.8 [77.6, 78.1] (72×) | 58.0 [57.9, 58.2] (90×) | 256 [256, 257] (93×) | 192 [191, 193] (70×) |

### HedgeHog, fixed pool

Engine `e8847f4`, same hosts and protocol:
[results/zen5_9700x_20261014_hh/](results/zen5_9700x_20261014_hh/),
[results/zen3_5950x_20261014_hh/](results/zen3_5950x_20261014_hh/). Factors
against the PureTD `torch` 0-ply rows of the tables above (the HedgeHog 0-ply
job re-times them within 3.4%).

Cubeless checker:

| model | depth | Zen 5 | Zen 3 |
|---|---|---|---|
| Fox | 0-ply | 23,007 [22,850, 23,164] (0.21×) | 20,830 [20,691, 20,969] (0.23×) |
| Aureus | 0-ply | 14,713 [14,068, 15,357] (0.34×) | 7,733 [7,524, 7,943] (0.61×) |
| Fox | 1-ply | 54.3 [53.1, 55.4] (91×) | 45.3 [44.5, 46.2] (103×) |
| Aureus | 1-ply | 30.6 [29.9, 31.3] (161×) | 16.7 [16.0, 17.3] (281×) |

Cubeful checker:

| model | depth | Zen 5 | Zen 3 |
|---|---|---|---|
| Fox | 0-ply | 24,647 [24,435, 24,860] (0.23×) | 21,840 [21,755, 21,924] (0.24×) |
| Aureus | 0-ply | 15,473 [14,883, 16,063] (0.36×) | 8,325 [8,097, 8,553] (0.62×) |
| Fox | 1-ply | 50.8 [50.0, 51.5] (110×) | 42.7 [42.2, 43.2] (122×) |
| Aureus | 1-ply | 29.7 [28.5, 30.8] (189×) | 16.5 [16.0, 17.0] (314×) |

Cube:

| model | depth | Zen 5 | Zen 3 |
|---|---|---|---|
| Fox | 0-ply | 316,091 [313,390, 318,793] (0.08×) | 290,153 [286,845, 293,462] (0.05×) |
| Aureus | 0-ply | 220,980 [211,346, 230,615] (0.11×) | 124,181 [120,583, 127,779] (0.11×) |
| Fox | 1-ply | 869 [856, 882] (27×) | 734 [723, 746] (18.3×) |
| Aureus | 1-ply | 487 [468, 507] (49×) | 274 [267, 281] (49×) |

### Network sizes, fixed pool

The one- to six-layer cubeful and prob5 nets of
[../capacity/README.md](../capacity/README.md), timed by
[capacity_costbench.py](capacity_costbench.py) (job `cap`; standalone:
[capacity_sweeps.sh](capacity_sweeps.sh)). 0-ply checker decisions/s;
factors against the production net of the same family, backend and
host. Sweeps: `results/<host>_20261014/fast/capacity_sweep*.txt`; cube rows
in `aggregate.txt`. Between-sweep t-intervals: Zen 5 `torch` checker
±0.9-4.0% and cube up to ±6.0%, every other row within ±1.0%. Table:
`python experiments/capacity/network_size_table.py --bench`.

| model | hidden | params | Zen 5 torch | Zen 3 torch | Zen 5 simd | Zen 3 simd |
|---|---|---|---|---|---|---|
| cubeful 1-layer | [512] | 103,425 | 28,513 [28,094, 28,932] (0.20×) | 25,268 [25,173, 25,362] (0.20×) | 200,640 [200,307, 200,974] (0.22×) | 117,225 [116,700, 117,749] (0.16×) |
| cubeful 2-layer | [512,512] | 366,081 | 8,098 [7,789, 8,407] (0.70×) | 7,828 [7,817, 7,839] (0.66×) | 66,295 [66,078, 66,513] (0.67×) | 28,518 [28,457, 28,580] (0.65×) |
| cubeful 3-layer | [512,512,256] | 497,153 | 6,103 [5,871, 6,336] (0.92×) | 5,856 [5,845, 5,867] (0.88×) | 47,415 [47,268, 47,562] (0.93×) | 19,981 [19,841, 20,122] (0.92×) |
| cubeful 4-layer (production) | [512,512,256,256] | 562,945 | 5,638 [5,496, 5,779] (1.00×) | 5,169 [5,146, 5,191] (1.00×) | 44,138 [43,928, 44,348] (1.00×) | 18,458 [18,413, 18,504] (1.00×) |
| cubeful 5-layer | [512,512,256,256,256] | 628,737 | 5,025 [4,885, 5,165] (1.12×) | 4,648 [4,637, 4,660] (1.11×) | 38,157 [38,001, 38,313] (1.16×) | 15,921 [15,874, 15,967] (1.16×) |
| cubeful 5-layer wide | [768,512,256,256,256] | 811,265 | 3,937 [3,818, 4,056] (1.43×) | 3,650 [3,645, 3,654] (1.42×) | 28,046 [27,923, 28,168] (1.57×) | 11,352 [11,320, 11,384] (1.63×) |
| cubeful 6-layer wide | [768,512,256,256,256,256] | 877,057 | 3,651 [3,557, 3,745] (1.54×) | 3,379 [3,375, 3,382] (1.53×) | 24,624 [24,516, 24,732] (1.79×) | 9,967 [9,943, 9,991] (1.85×) |
| cubeful 5-layer 768x768 | [768,768,256,256,256] | 1,073,665 | 3,011 [2,919, 3,102] (1.87×) | 2,823 [2,820, 2,827] (1.83×) | 20,022 [19,916, 20,127] (2.20×) | 7,966 [7,932, 8,001] (2.32×) |
| prob5 1-layer | [512] | 103,429 | 22,837 [22,637, 23,037] (0.22×) | 19,272 [19,249, 19,296] (0.24×) | 171,536 [171,102, 171,971] (0.22×) | 100,228 [99,520, 100,936] (0.16×) |
| prob5 2-layer | [512,512] | 366,085 | 7,333 [7,043, 7,622] (0.69×) | 6,727 [6,706, 6,747] (0.69×) | 53,720 [53,531, 53,908] (0.69×) | 23,445 [23,337, 23,552] (0.68×) |
| prob5 3-layer | [512,512,256] | 496,133 | 5,726 [5,545, 5,907] (0.88×) | 5,237 [5,229, 5,246] (0.89×) | 42,485 [42,317, 42,653] (0.88×) | 18,115 [18,073, 18,157] (0.88×) |
| prob5 4-layer (production) | [512,512,256,256] | 561,925 | 5,032 [4,859, 5,205] (1.00×) | 4,644 [4,634, 4,654] (1.00×) | 37,288 [37,095, 37,482] (1.00×) | 15,866 [15,830, 15,903] (1.00×) |
| prob5 5-layer | [512,512,256,256,256] | 627,717 | 4,571 [4,445, 4,697] (1.10×) | 4,180 [4,175, 4,184] (1.11×) | 33,949 [33,883, 34,015] (1.10×) | 14,363 [14,310, 14,415] (1.10×) |
| prob5 5-layer wide | [768,512,256,256,256] | 809,221 | 3,619 [3,529, 3,708] (1.39×) | 3,297 [3,292, 3,301] (1.41×) | 27,182 [27,065, 27,298] (1.37×) | 11,169 [11,082, 11,256] (1.42×) |
| prob5 6-layer wide | [768,512,256,256,256,256] | 875,013 | 3,332 [3,239, 3,426] (1.51×) | 3,056 [3,052, 3,060] (1.52×) | 25,668 [25,564, 25,773] (1.45×) | 10,563 [10,531, 10,595] (1.50×) |
| prob5 5-layer 768x768 | [768,768,256,256,256] | 1,071,621 | 2,748 [2,661, 2,835] (1.83×) | 2,564 [2,561, 2,566] (1.81×) | 20,474 [20,346, 20,602] (1.82×) | 8,305 [8,275, 8,335] (1.91×) |
