# PureTD paper: tables to code

The paper's tables mapped to code, commands and committed results. Links are repo-relative; line anchors are pinned to this release. Tables are numbered in the order of their `table` environments, appendix included.

## Prerequisites

In build order.

1. Python 3.10–3.13 with torch: [experiments/head_to_head/README.md#L11-L32](experiments/head_to_head/README.md?plain=1#L11-L32), which also builds the C engine (`bash c_engine/build_unix.sh`).
2. C API (Tables 4, 5, 11, 12, 13, 22, 23): `VENV=/path/to/venv/bin/python bash c_api/build.sh torch simd`, [c_api/README.md](c_api/README.md).
3. gnubg 1.08.003: the packaged or source-built binary for head-to-head ([experiments/head_to_head/README.md#L18-L22](experiments/head_to_head/README.md?plain=1#L18-L22)); for the benchmarks a `--disable-threads` build with the `gnubg.benchpool` patch, `bash third_party_opponents/gnubg/gnubg_bench/build.sh $HOME/gnubg-bench` (needs flex, bison, glib, python3 headers).
4. Open Sage: `pip install bgsage==2.0.20260907`, exactly this version (bgsage 1.x gives different rows); native C API `bash third_party_opponents/opensage/opensage_mobile_build.sh`, benchmark loops `bash third_party_opponents/opensage/opensage_bench/build.sh` ([experiments/head_to_head/OPENSAGE.md](experiments/head_to_head/OPENSAGE.md)).
5. HedgeHog (Tables 22, 23): engine, shim and models per [third_party_opponents/hedgehog/README.md](third_party_opponents/hedgehog/README.md).
6. Bear-off tables (Table 7): `python -c 'import bearoff; bearoff.build_table(); bearoff.build_ts_table()'`.

Host-specific paths and defaults in the commands (the repo `.venv`, `$HOME/gnubg-bench`, `~/xg_kits`, `--device cuda`, `--workers 48` to `64`) are placeholders: set `--workers` to the local core count; `--device cpu` works.

`play_models.py` and `run.py` play checkpoint and `janowski:` sides through the C API by default (built on demand). The committed rows of Tables 2 and 3 were played on the Python path; `--combine` refuses to pool across paths, so add `--python-api` to extend them.

Abbreviations: mEq, thousandths of a point (equity) per game; PR, XG++ Performance Rating, 500 × equity lost per decision (lower is better); DMP, double match point, the last game of a match, where only winning counts.

## Table 2: Cubeless-money strength

- Code: [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/head_to_head/README.md#L69-L87](experiments/head_to_head/README.md?plain=1#L69-L87)
- Results: [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv), [experiments/head_to_head/results/prob5_vs_opensage0_cbl_10M.txt](experiments/head_to_head/results/prob5_vs_opensage0_cbl_10M.txt), [experiments/head_to_head/results/prob5_vs_opensage1_cbl_10M.txt](experiments/head_to_head/results/prob5_vs_opensage1_cbl_10M.txt) (pooled Open Sage outputs)

## Table 3: Cubeful-money strength

XG++ PR column: no exports committed.

- Code: [experiments/head_to_head/run.py](experiments/head_to_head/run.py), [tools/analyze_xg_results.py](tools/analyze_xg_results.py)
- Command: [experiments/head_to_head/README.md#L89-L116](experiments/head_to_head/README.md?plain=1#L89-L116)
- Results: [experiments/head_to_head/results/h2h_cubeful_summary.csv](experiments/head_to_head/results/h2h_cubeful_summary.csv), XG++ per-game export not committed

## Table 4: Cubeless-money cost versus strength

Benchmark results (Tables 4, 5, 11, 12, 13 and 23): earlier versions of these tables were measured with the MKL shim preloaded; only the torch rows differ materially.

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](third_party_opponents/opensage/opensage_bench/sage_bench.cpp), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job), [experiments/head_to_head/README.md#L69-L87](experiments/head_to_head/README.md?plain=1#L69-L87) (equity cells)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt), [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv) (equity cells)

## Table 5: Cubeful-money cost versus strength

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/opensage/opensage_bench/sage_bench.cpp](third_party_opponents/opensage/opensage_bench/sage_bench.cpp), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job), [experiments/head_to_head/README.md#L89-L116](experiments/head_to_head/README.md?plain=1#L89-L116) (equity cells)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt), [experiments/head_to_head/results/h2h_cubeful_summary.csv](experiments/head_to_head/results/h2h_cubeful_summary.csv) (equity cells)

## Table 7: Learned versus heuristic cube handling

Both sides play with the exact bearoff tables, built locally ([experiments/janowski/README.md#L41-L50](experiments/janowski/README.md?plain=1#L41-L50)) by [bearoff.py#L80-L145](bearoff.py#L80-L145) (one-sided, up to 15 checkers) and [bearoff.py#L429-L512](bearoff.py#L429-L512) (two-sided cubeful, up to 8 checkers per side). XG++ PR: no exports committed. Prob5Jan plays a fixed cube-life index x = 0.76 (`JANOWSKI_X` overrides).

- Code: [janowski.py](janowski.py), [janowski_agent.py](janowski_agent.py), [bearoff.py](bearoff.py), [bearoff_td.py](bearoff_td.py), [play_models.py#L498-L536](play_models.py#L498-L536) (the `bo:` and `janowski:` sides), [experiments/head_to_head/run.py](experiments/head_to_head/run.py), [tools/analyze_xg_results.py](tools/analyze_xg_results.py)
- Command: [experiments/janowski/README.md#L63-L69](experiments/janowski/README.md?plain=1#L63-L69) (head-to-head), [experiments/janowski/README.md#L86-L88](experiments/janowski/README.md?plain=1#L86-L88) (XG++ PR)
- Results: [experiments/janowski/results/janowski_h2h.csv](experiments/janowski/results/janowski_h2h.csv), [experiments/janowski/README.md#L73-L78](experiments/janowski/README.md?plain=1#L73-L78) (head-to-head table), XG++ per-game exports not committed; PR values in [experiments/janowski/README.md#L90-L93](experiments/janowski/README.md?plain=1#L90-L93) (XG++ PR table)

## Table 9 and Figure 3: Network size

Strength of the one- to six-layer cubeful and prob5 nets head-to-head against the production net of the family, and their 0-ply cost per decision on the fixed pools; the figure plots strength against Zen 5 `torch` checker decisions/s.

- Code: [experiments/benchmarks/capacity_costbench.py](experiments/benchmarks/capacity_costbench.py), [experiments/benchmarks/capacity_sweeps.sh](experiments/benchmarks/capacity_sweeps.sh), [experiments/capacity/network_size_table.py](experiments/capacity/network_size_table.py), [experiments/head_to_head/run.py](experiments/head_to_head/run.py)
- Command: [experiments/capacity/README.md#L15-L25](experiments/capacity/README.md?plain=1#L15-L25) (head-to-head), [experiments/capacity/README.md#L38-L43](experiments/capacity/README.md?plain=1#L38-L43) (decisions/s and table), [experiments/capacity/README.md#L81-L83](experiments/capacity/README.md?plain=1#L81-L83) (figure data)
- Results: [experiments/head_to_head/results/h2h_experimental_vs_prod.csv](experiments/head_to_head/results/h2h_experimental_vs_prod.csv), [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt) (`fast/capacity` sections), [experiments/capacity/README.md#L54-L71](experiments/capacity/README.md?plain=1#L54-L71) (table), [experiments/capacity/results/capacity_cubeful.dat](experiments/capacity/results/capacity_cubeful.dat), [experiments/capacity/results/capacity_prob5.dat](experiments/capacity/results/capacity_prob5.dat) (figure data)

## Table 11: Cubeless-money throughput

PureTD: C API `torch` rows; the `simd` rows are Table 13's.

- Code: [experiments/benchmarks/bench.sh](experiments/benchmarks/bench.sh), [experiments/benchmarks/aggregate_sweeps.py](experiments/benchmarks/aggregate_sweeps.py)
- Command: [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt)

## Table 12: Cubeful-money throughput

PureTD: C API `torch` rows; the `simd` rows are Table 13's.

- Code: [experiments/benchmarks/bench.sh](experiments/benchmarks/bench.sh), [experiments/benchmarks/aggregate_sweeps.py](experiments/benchmarks/aggregate_sweeps.py)
- Command: [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt)

## Table 13: PyTorch versus the optimized kernel

PureTD C API, `torch` and `simd` backends, in the same sweeps as Tables 11 and 12. 1-ply rows: root filter 0.08.

- Code: [c_api/backend_torch.cpp](c_api/backend_torch.cpp), [c_api/backend_simd.c](c_api/backend_simd.c), [c_api/README.md#L161-L173](c_api/README.md?plain=1#L161-L173) (root filter), [experiments/benchmarks/rootf_validate.py](experiments/benchmarks/rootf_validate.py)
- Command: [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt), [experiments/benchmarks/README.md#L225-L251](experiments/benchmarks/README.md?plain=1#L225-L251) (`PureTD torch` and `PureTD simd` rows, with intervals), [experiments/benchmarks/results/rootf_validate_20261011.txt](experiments/benchmarks/results/rootf_validate_20261011.txt) (root filter agreement)

## Table 22: Head-to-head strength against Hedgehog

HedgeHog engine, shim binary and models are third-party and not committed: [third_party_opponents/hedgehog/README.md#L21-L45](third_party_opponents/hedgehog/README.md?plain=1#L21-L45). HedgeHog engine `e8847f4`. Cubeful cells: the cube-capped (64, both sides) rows. The paper gives HedgeHog's equity: signs flipped against the CSVs.

- Code: [third_party_opponents/hedgehog/hedgehog_agent.py](third_party_opponents/hedgehog/hedgehog_agent.py), [third_party_opponents/hedgehog/shim/hh_shim.cpp](third_party_opponents/hedgehog/shim/hh_shim.cpp), [play_models.py#L486-L496](play_models.py#L486-L496), [third_party_opponents/gnubg/gnubg_eval.py#L283-L286](third_party_opponents/gnubg/gnubg_eval.py#L283-L286) (cube cap)
- Command: [experiments/head_to_head/HEDGEHOG.md#L14-L37](experiments/head_to_head/HEDGEHOG.md?plain=1#L14-L37)
- Results: [experiments/head_to_head/results/h2h_cubeless_summary.csv](experiments/head_to_head/results/h2h_cubeless_summary.csv), [experiments/head_to_head/results/h2h_cubeful_cap64_summary.csv](experiments/head_to_head/results/h2h_cubeful_cap64_summary.csv), [experiments/head_to_head/HEDGEHOG.md#L39-L63](experiments/head_to_head/HEDGEHOG.md?plain=1#L39-L63)

## Table 23: Fixed-position throughput against Hedgehog

HedgeHog engine `e8847f4`, as Table 22. 0-ply rows: the `fast` group, 1-ply rows (Fox and Aureus): the `slow` group, both eight sweeps per host. Factors: per-decision cost over the PureTD `torch` 0-ply rows of the same host; gnubg rows as Tables 11 and 12.

- Code: [experiments/benchmarks/fixedpool_bench.py](experiments/benchmarks/fixedpool_bench.py), [third_party_opponents/gnubg/fixedpool_bench_gnubg.py](third_party_opponents/gnubg/fixedpool_bench_gnubg.py), [third_party_opponents/hedgehog/shim/hh_shim.cpp](third_party_opponents/hedgehog/shim/hh_shim.cpp), [third_party_opponents/hedgehog/validate_costpath.py](third_party_opponents/hedgehog/validate_costpath.py)
- Command: [experiments/benchmarks/README.md#L207-L211](experiments/benchmarks/README.md?plain=1#L207-L211) (HedgeHog), [experiments/benchmarks/README.md#L173-L178](experiments/benchmarks/README.md?plain=1#L173-L178) (suite), [experiments/benchmarks/README.md#L192-L215](experiments/benchmarks/README.md?plain=1#L192-L215) (per job: gnubg, PureTD)
- Results: [experiments/benchmarks/results/zen5_9700x_20261014_hh/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014_hh/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014_hh/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014_hh/aggregate.txt); PureTD reference rows: [experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt](experiments/benchmarks/results/zen5_9700x_20261014/aggregate.txt), [experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt](experiments/benchmarks/results/zen3_5950x_20261014/aggregate.txt)
