# Self-play compute benchmark (single-core, per-decision)

Cross-engine computational-cost comparison — **PureTD vs gnubg 1.08.003 vs
Open Sage** — measured from **self-play games**: each engine plays itself on
one pinned core, and cost is quoted **per decision**: total engine wall-time
divided by all decisions made. Move generation, evaluation, filtering, and
cube logic all run through each engine's own production code path. This
measures compute cost, **not** playing strength.

Design rules (identical for every engine):

- **Same game count**: N=300 self-play games per engine per box.
- **One pinned core** (`sched_setaffinity`; `BENCH_CORE` to choose,
  `BENCH_NO_PIN` to disable): wall-time = true single-core cost even when an
  engine's internals spawn helper threads.
- **Same counting convention**: one checker decision per board+dice turn; one
  cube decision per pre-roll turn where the on-roll player may double, plus
  each take/drop response.
- Cubeful headline = **combined** cost over all decisions from one cube-on
  run; cubeless = checker-only cost (no cube decisions exist).

Scripts — **four**. Native engines (PureTD/prob5 + Open Sage) get one
self-play file and one fixed-pool file, each dispatching on `--mode`/`--engine`;
gnubg is the one unavoidable split because it runs inside the gnubg binary's
own embedded Python (which can't import torch/numpy/bgsage — or even `random`):

- `selfplay_bench.py` — all NATIVE self-play,
  `--mode {cubeless,cubeful} --engine {puretd,sage}`. prob5 uses the
  record-free fused chooser (move-gen + encode in one C call + one batched
  forward, selection parity-verified vs the standard path); Sage uses its own
  `BgBotAnalyzer` (cubeful) / native cubeless move selection.
- `selfplay_bench_gnubg.py` — gnubg self-play; runs **inside** the gnubg
  binary (`gnubg -t -q -p`). Its cube-off runs provide gnubg's cubeless rows.
- `fixedpool_bench.py` — all NATIVE fixed-pool per-decision rows,
  `--mode {cubeless,cubeful}` (see *Fixed-position pool benchmark* below).
- `fixedpool_bench_gnubg.py` — gnubg fixed-pool; runs inside the gnubg binary,
  `GNUBG_FP_MODE={cubeless,cubeful,both}`.

Every printed rate carries a bootstrap 95% CI (`bench_common`). `reproduce.sh`
runs the whole suite on one box (below).

Ply numbering: `0-ply` = raw NN; `1-ply` = one level of lookahead. PureTD and
gnubg use gnubg numbering; Open Sage's own API is XG-numbered (its `2P` = this
1-ply).

## Running

**Whole suite, one command** — the single entry point to regenerate every
benchmark row (writes labeled result blocks to `results/reproduce_<host>.txt`):

    BENCH_CORE=8 VENV=/path/to/venv/bin/python bash experiments/benchmarks/reproduce.sh

`reproduce.sh` pins one core, runs all four scripts (gnubg + native, fixed-pool
+ self-play), and skips the multi-hour deep-search self-play rows unless
`RUN_DEEP=1`. `VENV` must be a python with torch **and** bgsage. The individual
scripts, if you want one at a time:

    # gnubg self-play — MUST use a threads-disabled build (recipe below)
    GNUBG_BENCH_PLIES=0,1 GNUBG_BENCH_G0=300 GNUBG_BENCH_G1=300 \
      $HOME/gnubg-st/bin/gnubg -t -q -p experiments/benchmarks/selfplay_bench_gnubg.py

    # native self-play (on AMD preload the MKL CPUID shim). --plies is OUR
    # numbering (0/1-ply); Sage maps internally to sage_plies = plies+1.
    LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeful  --engine puretd --plies 0 --games 300
    LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeful  --engine sage   --plies 1 --sage-unpruned --games 300
    LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeless --engine puretd --plies 0 --games 300

`libfakeintel` (`c_engine/build_fakeintel.sh`) reports an Intel vendor ID so
MKL takes its fast code paths on AMD; it affects only the torch/MKL-backed
PureTD rows (~5–7%). Preload it (or don't) consistently when comparing boxes.

## Building the threads-disabled gnubg (required)

    curl -O https://ftp.gnu.org/gnu/gnubg/gnubg-release-1.08.003-sources.tar.gz
    tar xzf gnubg-release-1.08.003-sources.tar.gz && cd gnubg-1.08.003
    # needs flex, bison, glib-2.0 dev headers, python3 dev headers
    ./configure --disable-threads --with-python=python3 --without-gtk --prefix=$HOME/gnubg-st
    make -j && make install
    md5sum $HOME/gnubg-st/share/gnubg/gnubg.wd   # must be 14184acc9c60ef67be0fad88548fd51a

No source is modified — `--disable-threads` is an upstream configure option.
The build auto-selects gnubg's AVX SIMD tier (`-O3 -ffast-math -mavx`, the
same level as the official Windows "AVX" installer). gnubg's only higher
tier, `--enable-simd=fma`, measured performance-neutral in a direct A/B on
this suite (every row within the run-to-run drift band); gnubg has no
AVX-512 evaluation path.
It is required because the stock threaded build routes every in-play
evaluation through its multi-thread task queue, whose completion wait is a
~10 ms sleep-poll (`WaitForAllTasks`, multithread.c): self-play measures
~101 decisions/s at ANY depth — pure dispatch latency, not compute (~350×
the true 0-ply cost, ~12× at 1-ply). Affinity pinning cannot remove a timed
sleep; compiling the queue out can. The single-thread build's costs
cross-check against gnubg's own `calibrate` rate.

## Why one combined number is fair: the decision-mix ratio

The combined metric folds cube and checker decisions into one denominator,
which is comparable across engines only if the decision *mix* is similar. It
is — measured on the SAME N=300 runs as the results (Zen 5 column shown):

| engine | checker/game | cube/game | cube:checker ratio |
|---|---|---|---|
| gnubg 0-ply | 34.5 | 25.8 | 0.747 |
| gnubg 1-ply | 35.6 | 26.6 | 0.747 |
| PureTD 0-ply | 38.5 | 27.8 | 0.723 |
| PureTD 1-ply | 35.0 | 25.9 | 0.741 |
| Open Sage 0-ply | 40.2 | 27.5 | 0.685 |
| Open Sage 1-ply | 42.4 | 28.5 | 0.673 |

(gnubg checker counts exclude dance turns, matching the Python adapters.)
Ratio 0.71 ± 0.04 for every engine and depth, and game lengths are
near-identical — a combined per-decision cost misprices no engine by more
than a few percent of denominator mix.

## Results — cubeful money (combined ms/decision; dec/s), N=300, one pinned core

| engine | Zen 3 | Zen 5 |
|---|---|---|
| gnubg 0-ply | 0.030 (32,862/s) | 0.018 (54,564/s) |
| PureTD 0-ply (4L) | 0.332 (3,013/s) | 0.114 (8,737/s) |
| gnubg 1-ply, default pruning | 0.876 (1,141/s) | 0.530 (1,885/s) |
| Open Sage 0-ply (analyzer) | 10.2 (97.9/s) | 4.87 (205.5/s) |
| Open Sage 1-ply production | 34.2 (29.2/s) | 18.5 (54.1/s) |
| Open Sage 1-ply unpruned (paper config) | 42.2 (23.7/s) | 23.8 (42.0/s) |
| PureTD 1-ply | 172 (5.8/s) | 56.8 (17.6/s) |

## Results — cubeless money (checker ms/decision; dec/s), N=300, one pinned core

| engine | Zen 3 | Zen 5 |
|---|---|---|
| gnubg 0-ply (cube-off) | 0.064 (15,547/s) | 0.034 (29,445/s) |
| Open Sage 0-ply (`best_move_index`) | 0.379 (2,640/s) | 0.183 (5,460/s) |
| PureTD prob5 0-ply | 0.462 (2,165/s) | 0.202 (4,959/s) |
| gnubg 1-ply, default pruning (cube-off) | 1.45 (692/s) | 0.885 (1,130/s) |
| Open Sage 1-ply production | 1.85 (542/s) | 1.02 (977/s) |
| PureTD prob5 1-ply (full-width) | 86.0 (11.6/s) | 31.0 (32.3/s) |

Every engine's 0-ply row is its in-play per-call cost (gnubg
`findbestmove` cube-off, Sage `best_move_index`, PureTD fused
`choose_next_state`), so Sage and PureTD are near-parity at 0-ply and
both ~5-7x behind gnubg's tiny nets — a model-size gap (Sage/gnubg run
tiny hand-tuned nets; prob5 is ~16x more MACs/eval through general MKL),
not a harness gap: PureTD's loop-free kernel is only ~1.1x its in-play
number (`results/fixedpool_*`), so a fully-C++ loop would not close it.

Footnote: Sage's fully native C++ self-play loop
(`play_games_pair_vs_self`) clocks 0.075 / 0.045 ms (13.3k / 22.2k
dec/s), ~4x faster than the `best_move_index` row (its tiny eval means
the per-call pybind glue dominates). It is NOT tabulated because it
measures an un-anchorable config: it runs a reduced 17-net strategy
(the 19-net `create_multipy_stage9` API rejects 17 nets, so this config
exists only inside the native loop — it cannot be built as a strategy
object, hence cannot be played head-to-head or given a strength number),
and its `GameStats` returns only game outcomes, so its decision count
can only be estimated, not counted. The tabulated `best_move_index`
number is the 19-net production config that all strength results use. Also measured, not tabulated (raw in `results/`):
Sage 1-ply unpruned 14.8 / 8.19 ms (the root filter is an ~8x cubeless
cost lever).

## Headline ratios (equal-strength framing, cubeful combined)

- PureTD 0-ply **beats** gnubg 1-ply head-to-head (+11.9 mEq cubeless /
  +22.4 cubeful) at **2.6× (Zen 3) to 4.6× (Zen 5) less compute** per
  decision.
- gnubg 0-ply is ~6–11× cheaper than PureTD 0-ply per decision but loses to
  it by 44.6 / 78.7 mEq.

## Measurement notes and caveats

- **Dance turns** (no legal move) are excluded from the gnubg harness's
  decision counts (empty-move records are skipped), matching the Python
  adapters; the session-start auto-played game is also excluded. Deep-search
  rows (Sage unpruned, PureTD 1-ply, prob5 1-ply) are single-run point
  estimates - CI not computed (hours per run).
- **gnubg 1-ply uses default production settings** (pruning neural nets +
  Normal move filter), verified applied in-session.
- **Open Sage's analyzer spawns ~2 internal threads** even at
  `parallel_threads=1`; pinning showed the second thread is mostly
  unproductive spin (pinned 1-ply cost within 10% of unpinned; 0-ply
  *faster* pinned). Pinned numbers are quoted — rigorously single-core.
- **Sage's cube machinery dominates its cubeful cost** (~30 ms/call at
  1-ply, ~65× gnubg's native 1-ply cube primitive) — its cubeful combined
  rates are cube-bound, matching its weak cube skill in the strength tables.
- **Sage's root move filter matters enormously cubeless**: unpruned
  (keep-all) 1-ply is ~8× the production-filter cost.
- Run-to-run reproducibility on an idle box is ~2% (spot-checked across the
  pilot and final runs; e.g. Sage 1-ply unpruned 41.9 vs 42.0 dec/s).

## Strength (mEMG), not speed

- `../offline_analysis/Sage_mEMG_calculation.md` — Open Sage money-game mEMG
  via gnubg N-ply.
- `../offline_analysis/run_cubeless_memg.py` / `run_cubeful_memg.py` —
  cubeless / cubeful mEMG drivers.

## Dependencies

- This repo (parent); `c_engine/` buildable by `gcc`; PyTorch, NumPy.
- gnubg 1.08.003 built `--disable-threads` (recipe above) — for the gnubg rows.
- [Open Sage](https://github.com/markbgsage/bgsage) `v0.8.20260611`
  (`pip install bgsage`) — optional, only for the Sage rows.

## Fixed-position pool benchmark

A second, independent methodology over a shared pool of 8,000 realistic
decision points (board + dice), committed once as
`fixedpool_positions.json` (seeded gnubg 0-ply cubeless-money
self-play; full provenance in the artifact's meta block).
`fixedpool_bench_gnubg.py` runs INSIDE the threads-disabled official build
and times gnubg's own `findbestmove` (default pruning + Normal filter) and
`cfevaluate` (cubeful evaluation), pinned to one core;
`fixedpool_bench.py --mode {cubeless,cubeful}` does the native engines
(PureTD/prob5 + Open Sage) on the same pools:

    # gnubg  (GNUBG_FP_MODE = cubeless | cubeful | both)
    GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1,2 \
      $HOME/gnubg-st/bin/gnubg -t -q -p experiments/benchmarks/fixedpool_bench_gnubg.py
    # native engines
    python experiments/benchmarks/fixedpool_bench.py --mode cubeless
    python experiments/benchmarks/fixedpool_bench.py --mode cubeful --cap 10000  # full pool, matches the committed results

The CUBEFUL rows replay a second committed artifact,
`fixedpool_cubeful_positions.json` — the first 10,000 checker points
(board + dice + cube state) and first 6,000 cube points (board + cube
state) of one seeded prod-cubeful self-play pass (253 games; regenerate
deterministically with `fixedpool_bench.py --mode cubeful --regen-pool`).
ALL engines — gnubg included — replay identical positions with identical
per-position cube states; 49% of the cube points are recube situations
(cube ≥ 2, owned), matching real cubeful play. Per-position cube states
matter: centered `cfevaluate` measures ~6–18% dearer than owned-cube at
1-ply, so a constant-centered convention would overstate gnubg's cube cost.

| `findbestmove` cubeless (ms/move) | Zen 3 | Zen 5 |
|---|---|---|
| 0-ply | 0.053 | 0.029 |
| 1-ply | 1.341 | 0.778 |

| `cfevaluate` (ms/eval, shared cubeful pool) | Zen 3 | Zen 5 |
|---|---|---|
| 0-ply | 0.0053 | 0.0027 |
| 1-ply | 0.443 | 0.252 |

Fixed-pool and self-play checker costs agree at the ~10–20% level at 0-ply
and ~8–14% at 1-ply, with self-play always the dearer: its wall-clock
carries game bookkeeping and dance-turn handling that the denominator
(decisions) excludes, a fixed overhead that weighs most on the tiniest
evals — two independent methodologies agreeing to within that understood
overhead is strong validation of both. Full tables with CIs:
`results/fixedpool_cubeless_20260802.txt`,
`results/fixedpool_cubeful_20260805.txt`.
