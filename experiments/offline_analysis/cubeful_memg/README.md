# Cubeful-money mEMG — gnubg 1.08.003 judge

Judge: **gnubg 1.08.003** CLI at **4-ply** (`gnubg.wd` md5
`14184acc9c60ef67be0fad88548fd51a`). All scoring with `LC_ALL=C` (en_US
locale silently breaks the stats parser), `OMP_NUM_THREADS=1`. n=4000
legal-opening self-play games per row (Jacoby on).

**PureTD (`ours-*`) model**: `best_models/cubeful_money_512_512_256_256.pt`
md5 `77a3205cc7582dbe7da4ae4e6c89fef1` (prod).

## Layout

- `cubeful_memg.csv` — summary table (chequer / cube / overall means +
  game-level bootstrap 95% CIs via `--combine`, B=10,000 seed 42), regenerated from the
  per-game data below.
- `pergame/<player>.jsonl` — one line per game (`chequer`, `cube`, `overall`
  mEMG), one file per main-table row; names match
  `run_cubeful_memg.py --players` keys. Every row, including sage 0-ply, is a
  plain 4,000-game jsonl (the old pooled-shard format was retired in the
  2026-08-13 re-measurement).

Provenance: ALL rows re-measured 2026-08-13. The previous values were
generated with the defective move generator (false forced pass, fixed in
2149c48) and judged by hosts whose bearoff configuration was not recorded --
a provisioning script had been deleting `gnubg_ts0.bd`, so gnubg fell back to
a net estimate for cubeful bearoff equities and fabricated cube errors. Every
row here was regenerated on the fixed engine and judged by a gnubg asserted to
carry BOTH bearoff databases (`gnubg_ts0.bd` md5 `44b6040b49b46cb9dd2ce8caa947044d`,
`gnubg.wd` md5 `14184acc9c60ef67be0fad88548fd51a`) with zero
"will not use the two-sided bearoff" warnings; `gnubg_eval.assert_gnubg_bearoff`
now enforces this before any judging runs.

The correction is cube-side, as expected: every cubeful row's cube component
fell (e.g. ours-0ply 1.873 -> 1.613, sage-analyzer-0ply 7.048 -> 6.179) while
chequer play barely moved. Engine orderings are unchanged. The cubeless table
moved by at most 0.05. Pre-correction values are in git history.

## Rows (overall mEMG, lower is better)

| player | config | chequer | cube | overall |
|---|---|---|---|---|
| ours-1ply | 4L cubeful | 0.975 | 1.030 | 0.967 [0.934, 1.001] |
| ours-0ply | 4L cubeful | 1.902 | 1.613 | 1.848 [1.795, 1.903] |
| gnubg-ext-1ply | true gnubg 1.08 binary, full/unpruned | 2.093 | 1.845 | 2.027 [1.976, 2.077] |
| sage-analyzer-1ply | SAGE_UNPRUNED (paper convention) | 1.902 | 5.656 | 2.204 [2.147, 2.263] |
| gnubg-ext-0ply | true gnubg 1.08 binary | 3.983 | 3.307 | 3.806 [3.716, 3.899] |
| sage-analyzer-0ply | analyzer 0-ply | 3.795 | 6.179 | 3.929 [3.838, 4.021] |

Notes:

- **gnubg rows are gnubg-ext** (real 1.08.003 binary over the external-player
  socket, native money cube) — the same opponent as the H-H tables.
- **sage-analyzer rows** use `OpenSageAnalyzerAgent` (Sage's own
  `BgBotAnalyzer`: cube-aware checker play + native `cube_action`), identical
  to the H-H runs. The paper row is a config-verified `SAGE_UNPRUNED=1` run
  (2026-07-27). The pruned-vs-unpruned contrast is carried by the 10M-game
  head-to-head matrix
  (`../../head_to_head/results/h2h_native_pruning_matrix.csv`), not by this
  4,000-game table.
- Sage's cube component (~5.8) dominates its overall gap to PureTD — the same
  cube-skill signature as the XG++ analysis.
- Reproduction wrapper: `../cubeful_money_memg.sh`.
