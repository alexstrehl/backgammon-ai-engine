# Self-play compute benchmark (single-core, per-decision)

Cross-engine computational-cost comparison — **PureTD vs gnubg 1.08.003 vs
Open Sage** — measured from **self-play games**: each engine plays itself on
one pinned core, and cost is quoted **per decision**: total engine wall-time
divided by all decisions made. Move generation, evaluation, filtering, and
cube logic all run through each engine's own production code path. This
measures compute cost, **not** playing strength.

Method:

- **Same game count per row across boxes.** Native rows are seeded, so a row
  replays an identical workload on every box and in every sweep
  (`PureTD 0-ply cube-on`: the same 22,129 checker decisions each time), which
  makes a cross-box ratio a pure hardware comparison. gnubg draws its own dice
  internally, so its workload varies (19,829–21,206 checker decisions across
  the five Zen 5 sweeps) and its ratios hold only statistically. Counts vary
  by row (200–600).
- **A single pinned core, the same one for every engine** (`sched_setaffinity`;
  `BENCH_CORE` selects it, `BENCH_NO_PIN` disables pinning). Wall-time is then
  a true single-core cost even where an engine spawns helper threads. Rows or
  sweeps timed on different cores are not comparable. `bench.sh` exports one
  `BENCH_CORE` to all four scripts.
- **Counting convention.** One checker decision per board-and-dice turn; one
  cube decision per pre-roll turn in which the player on roll may double, plus
  each take/drop response.
- **The core is warmed before every timed region** (`BENCH_WARMUP_S`, default
  3 s; a cache-neutral scalar spin in `bench_common._spin_ramp`). An idle core
  sits at base clock and requires ~2 s of sustained load to reach boost
  (3720 → 5005 MHz measured on Zen 5); timed cold, prob5 0-ply reads
  0.197 ms/dec against 0.170 warm. The penalty is approximately fixed at
  ~0.4 s, so it distorts cross-engine ratios, affecting the cheapest engines
  most. All four scripts ramp.
- **Repeat each benchmark to sample variance, rather than lengthen it.**
  `SWEEPS=5` runs the whole suite five times as five fresh processes (90 s idle
  between them, `SWEEP_GAP`), writing `$OUTDIR/bench_<host>_sweep<N>.txt`.
  `aggregate_sweeps.py` then reports each row as the mean of the five with a
  95% t-interval over them.
- **Game counts are tiered by measured cost**: `G_FAST`=600 for rows under
  ~100 ms/game, `G_SLOW`=200 for the two Open Sage analyzer rows. Search depth
  is not a reliable proxy — Sage's analyzer at 0-ply costs 0.70 s/game against
  gnubg's 0.055 s/game at 1-ply. The counts place each row's within-sweep
  sampling error below the between-sweep standard deviation rather than
  equalising precision, which is unattainable given a ~1100× span in per-game
  cost; the gnubg rows consequently carry the widest intervals (~1.5–2%).
  Every native timed region runs at least ~2.5 s, most between 15 and 85 s.
- **gnubg's session record is reset every 250 games** (`GNUBG_RESET_EVERY`).
  gnubg's per-game bookkeeping scales with the in-memory record's length:
  3.5 → 6.1 ms/game across 2000 games in one session, flat at ~3.0 when reset.
  This affects only the benchmark, which drives gnubg's own game loop; the
  head-to-head runs use the external-player interface, where gnubg only
  evaluates positions. Per-game decision counts are harvested from the record
  *before* each reset, so the per-game (time, decisions) pairs are identical to
  an unreset run. `new session` does not clear gnubg's evaluation cache.
- **The per-sweep output also carries within-sweep bootstrap CIs**
  (`bench_common`, resampling calls or games). These are not the intervals the
  tables quote. Resampling games estimates variation across different games;
  repeating the sweep estimates reproducibility of the measurement, which is
  the claim being made.

Scripts — **four**. Native engines (PureTD/prob5 + Open Sage) get one
self-play file and one fixed-pool file, each dispatching on `--mode`/`--engine`;
gnubg is split off because it runs inside the gnubg binary's own embedded
Python (which can't import torch/numpy/bgsage — or even `random`):

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

Every tabulated rate is a five-sweep mean with a t-interval; the ATTRIBUTION
lines in the raw output are single-run point estimates. `RUN_DEEP=1` adds
multi-hour 1-ply self-play rows, which are not tabulated here.

Ply numbering: `0-ply` = raw NN; `1-ply` = one level of lookahead. PureTD and
gnubg use gnubg numbering; Open Sage's own API is XG-numbered (its `2P` = this
1-ply).

## Running

**Whole suite, one command** (writes labeled result blocks to
`$OUTDIR/bench_<host>.txt` at the default `SWEEPS=1`, or
`$OUTDIR/bench_<host>_sweep<N>.txt` per sweep when `SWEEPS>1`. `OUTDIR`
defaults to `$HOME/bench_results` — outside the repo, so a run never dirties
the tracked tree; the committed `results/` directory holds only curated,
published files):

    BENCH_CORE=8 VENV=/path/to/venv/bin/python bash experiments/benchmarks/bench.sh

`bench.sh` pins one core, runs all four scripts (gnubg + native,
fixed-pool + self-play), and skips the multi-hour deep-search rows unless
`RUN_DEEP=1`. `VENV` must be a python with torch **and** bgsage. The individual
scripts, if you want one at a time:

    # gnubg self-play — MUST use a threads-disabled build (recipe below).
    # BENCH_CORE must match the native runs below; rows timed on different
    # cores are not comparable.
    BENCH_CORE=8 GNUBG_BENCH_PLIES=0,1 GNUBG_BENCH_G0=600 GNUBG_BENCH_G1=600 \
      $HOME/gnubg-st/bin/gnubg -t -q -p experiments/benchmarks/selfplay_bench_gnubg.py

    # native self-play (on AMD preload the MKL CPUID shim). --plies is OUR
    # numbering (0/1-ply); Sage maps internally to sage_plies = plies+1.
    # --games matches the per-row counts bench.sh uses (G_FAST=600, G_SLOW=200);
    # the committed results were produced at these counts, not at the
    # scripts' default of 300.
    BENCH_CORE=8 LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeful  --engine puretd --plies 0 --games 600
    BENCH_CORE=8 LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeful  --engine sage   --plies 1 --games 200
    BENCH_CORE=8 LD_PRELOAD=c_engine/libfakeintel.so \
      python experiments/benchmarks/selfplay_bench.py --mode cubeless --engine puretd --plies 0 --games 600

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
The build auto-selects gnubg's AVX SIMD tier (`-O3 -ffast-math -mavx`, the same
level as the official Windows "AVX" installer); its only higher tier,
`--enable-simd=fma`, measured performance-neutral in a direct A/B on this suite
(no row moved beyond the ~1% between-run sd; that A/B's raw output is not
committed). gnubg has no AVX-512 evaluation path.

Threads-disabled is required because the stock threaded build routes every
in-play evaluation through its multi-thread task queue, whose completion wait is
a ~10 ms sleep-poll (`WaitForAllTasks`, multithread.c): self-play measures ~101
decisions/s at ANY depth — pure dispatch latency, not compute (~350× the true
0-ply cost, ~12× at 1-ply). Affinity pinning cannot remove a timed sleep;
compiling the queue out can. The single-thread build's costs cross-check against
gnubg's own `calibrate` rate.

## Why one combined number is fair: the decision-mix ratio

The combined metric folds cube and checker decisions into one denominator,
which is comparable across engines only if the decision *mix* is similar. It
is — measured on the same 2026-08-11 five-sweep runs as the results (Zen 5
column, means over the five sweeps; Sage analyzer rows N=200, the rest N=600):

| engine | checker/game | cube/game | cube:checker ratio |
|---|---|---|---|
| gnubg 0-ply | 34.1 | 25.7 | 0.754 |
| gnubg 1-ply | 35.7 | 26.6 | 0.744 |
| PureTD 0-ply | 36.9 | 27.1 | 0.735 |
| Open Sage 0-ply | 40.8 | 27.7 | 0.679 |
| Open Sage 1-ply | 42.0 | 28.4 | 0.677 |

The ratio spans 0.68–0.75 across every engine and depth, and game lengths are
near-identical — a combined per-decision cost misprices no engine by more
than a few percent of denominator mix.

## Results — cubeful money (combined ms/decision; dec/s [95% CI]), one pinned core

| engine | Zen 3 | Zen 5 |
|---|---|---|
| gnubg 0-ply | 0.029 (35,074 [34,532, 35,616]) | 0.018 (54,534 [53,902, 55,165]) |
| PureTD 0-ply (4L) | 0.335 (2,988 [2,954, 3,022]) | 0.119 (8,391 [8,301, 8,481]) |
| gnubg 1-ply, default pruning | 0.869 (1,151 [1,133, 1,168]) | 0.544 (1,839 [1,802, 1,876]) |
| Open Sage 0-ply (analyzer) | 10.050 (100 [99, 100]) | 5.003 (200 [199, 200]) |
| Open Sage 1-ply production | 34.014 (29 [29, 30]) | 19.342 (52 [51, 52]) |

## Results — cubeless money (checker ms/decision; dec/s [95% CI]), one pinned core

| engine | Zen 3 | Zen 5 |
|---|---|---|
| gnubg 0-ply (cube-off) | 0.053 (18,973 [18,530, 19,417]) | 0.033 (30,371 [29,827, 30,915]) |
| Open Sage 0-ply (`best_move_index`) | 0.365 (2,742 [2,727, 2,758]) | 0.165 (6,061 [6,008, 6,114]) |
| PureTD prob5 0-ply | 0.447 (2,237 [2,230, 2,244]) | 0.169 (5,911 [5,892, 5,929]) |
| gnubg 1-ply, default pruning (cube-off) | 1.430 (699 [690, 709]) | 0.890 (1,124 [1,101, 1,146]) |
| Open Sage 1-ply production | 1.780 (562 [560, 564]) | 1.024 (976 [962, 991]) |

Both tables are the mean of **five independent sweeps** (2026-08-11); brackets
are 95% t-intervals over those five, not within-run bootstrap CIs. Per-sweep
values are in `results/selfplay_zen3_epyc7713p_20260811.txt` and
`results/selfplay_zen5_9995wx_20260811.txt`.

Every engine's 0-ply row is its in-play per-call cost (gnubg `findbestmove`
cube-off, Sage `best_move_index`, PureTD fused `choose_next_state`), so Sage and
PureTD are near-parity at 0-ply and both ~5-8x behind gnubg. The gap is not
harness overhead: PureTD's loop-free fixed-pool kernel is only ~1.1x its in-play
number (`results/fixedpool_*`).

Not benchmarked: Sage's fully native C++ self-play loop
(`bgbot_cpp.play_games_pair_vs_self`). It cannot run the production net —
`GamePlanPairStrategy requires exactly 17 weight paths and hidden sizes,
got 19/19` — so it is restricted to the 17-net stage7/stage8 models, an
older generation than the 19-net stage9 config every Open Sage number here uses.
The tabulated `best_move_index` rows are the production config throughout.

## Measurement notes and caveats

- **gnubg's session-start auto-played game is excluded** from its counts.
- **gnubg and Open Sage run their default production settings** — gnubg's
  pruning neural nets plus Normal move filter, Open Sage's production root
  filter — verified applied in-session.
- **Open Sage does not currently implement efficient APIs for cubeful money
  play or for self-play**, so we use the fastest paths it provides. Only its
  cubeless fixed-position measurement is truly representative of its
  evaluation cost.
- **Open Sage's analyzer spawns ~2 internal threads** even at
  `parallel_threads=1`. Pinned 1-ply cost is within 10% of unpinned, and 0-ply
  is faster pinned. Pinned numbers are quoted — rigorously single-core.
- **Quoted CIs are t-intervals over five independent sweeps** (t(4)=2.776), not
  within-sweep bootstrap CIs. Each sweep contributes one number per row — itself
  a ratio-of-sums over that sweep — and the five are the sample. The self-play
  games are seeded, so every sweep replays identical work and the spread is pure
  machine variation (boost residency, thermal state, cache and allocator
  layout). Across both hosts: between-sweep sd median 0.53%, max 3.29%, with the
  fixed-pool rows tighter since their timed region excludes per-game
  bookkeeping. Do NOT pool raw per-game points across sweeps — games in a sweep
  share its state, so pooling treats clustered data as independent and gives an
  interval several times too narrow.

## Strength (mEMG), not speed

- `../offline_analysis/Sage_mEMG_calculation.md` — Open Sage money-game mEMG
  via gnubg N-ply.
- `../offline_analysis/run_cubeless_memg.py` / `run_cubeful_memg.py` —
  cubeless / cubeful mEMG drivers.

## Dependencies

- This repo (parent); `c_engine/` buildable by `gcc`; PyTorch, NumPy.
- gnubg 1.08.003 built `--disable-threads` (recipe above) — for the gnubg rows.
- [Open Sage](https://github.com/markbgsage/bgsage) `v1.3.20260723` (commit `a9576e8ab300e36969329e842962232883f7d90d`)
  (`pip install bgsage`) — optional, only for the Sage rows.

## Fixed-position pool benchmark

A second, independent methodology over a shared pool of 8,000 realistic
decision points (board + dice), committed once as `fixedpool_positions.json`
(seeded gnubg 0-ply cubeless-money self-play; full provenance in the artifact's
meta block). `fixedpool_bench_gnubg.py` runs INSIDE the threads-disabled
official build and times gnubg's own `findbestmove` (default pruning + Normal
filter) and `cfevaluate` (cubeful evaluation), pinned to one core;
`fixedpool_bench.py --mode {cubeless,cubeful}` does the native engines
(PureTD/prob5 + Open Sage) on the same pools:

    # gnubg  (GNUBG_FP_MODE = cubeless | cubeful | both)
    GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 \
      $HOME/gnubg-st/bin/gnubg -t -q -p experiments/benchmarks/fixedpool_bench_gnubg.py
    # native engines
    python experiments/benchmarks/fixedpool_bench.py --mode cubeless
    python experiments/benchmarks/fixedpool_bench.py --mode cubeful --cap 10000  # full pool, matches the committed results

The CUBEFUL rows replay a second committed artifact,
`fixedpool_cubeful_positions.json` — the first 10,000 checker points
(board + dice + cube state) and first 6,000 cube points (board + cube state) of
one seeded prod-cubeful self-play pass (253 games; regenerate deterministically
with `fixedpool_bench.py --mode cubeful --regen-pool`). ALL engines — gnubg
included — replay identical positions with identical per-position cube states;
49% of the cube points are recube situations (cube ≥ 2, owned), matching real
cubeful play. Per-position cube states matter: centered `cfevaluate` measures 6%
costlier than owned-at-2 and 21% costlier than owned-at-4 at 1-ply (0.409 vs 0.385
vs 0.337 ms), so a constant-centered convention would overstate gnubg's cube
cost.

| `findbestmove` cubeless (ms/move [95% CI]) | Zen 3 | Zen 5 |
|---|---|---|
| 0-ply | 0.0505 [0.0500, 0.0511] | 0.0301 [0.0299, 0.0303] |
| 1-ply | 1.304 [1.292, 1.316] | 0.798 [0.794, 0.802] |

| `cfevaluate` (ms/eval [95% CI], shared cubeful pool) | Zen 3 | Zen 5 |
|---|---|---|
| 0-ply | 0.00484 [0.00482, 0.00486] | 0.00285 [0.00282, 0.00289] |
| 1-ply | 0.4130 [0.4080, 0.4181] | 0.2590 [0.2582, 0.2598] |

Fixed-pool and self-play checker costs agree within ~15% for gnubg (+4.7 to
+15.5%), prob5 0-ply (+2.6 to +5.2%) and Sage 1-ply (+5.0 to +15.2%), with
self-play the costlier: its wall-clock carries game bookkeeping and dance-turn
handling that the denominator (decisions) excludes. One row sits outside the
band:

- **Sage 0-ply**: self-play costs 2.7× (Zen 5) to 4.0× (Zen 3) the fixed-pool
  figure — 0.165/0.365 ms against 0.062/0.091 ms fixed-pool, an absolute gap of
  0.103 ms (Zen 5) and 0.273 ms (Zen 3). prob5 0-ply, measured the same way, is
  within ~5% of its fixed-pool cost on both hosts (0.169/0.447 ms self-play
  against 0.167/0.424 ms fixed-pool, a gap of 0.002 and 0.023 ms).

Full tables with CIs (five-sweep t-intervals): self-play rows in
`results/selfplay_zen3_epyc7713p_20260811.txt` /
`results/selfplay_zen5_9995wx_20260811.txt`; fixed-position rows in
`results/fixedpool_zen3_epyc7713p_20260811.txt` /
`results/fixedpool_zen5_9995wx_20260811.txt` (one bench.sh run produced
both; the fixed-position files carry the split rows plus the cube-state
sensitivity and timing-overhead measurement notes).
