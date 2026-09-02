# Cubeful-money mEMG — gnubg 1.08.003 judge

Judge: **gnubg 1.08.003** CLI at **4-ply** (`gnubg.wd` md5
`14184acc9c60ef67be0fad88548fd51a`). All scoring with `LC_ALL=C` (en_US
locale silently breaks the stats parser), `OMP_NUM_THREADS=1`. n=4000
legal-opening self-play games per row (Jacoby on).

**PureTD (`ours-*`) model**: `best_models/cubeful_money_512_512_256_256.pt`
md5 `6afda0f85b039a186f018753120ed74c` (prod; `ours-*` rows re-measured
2026-08-23/24 on this model — the runner prints the md5 of the model it
loaded, so check it against this pin before trusting a run).

## Layout

- `cubeful_memg.csv` — summary table (chequer / cube / overall means +
  game-level bootstrap 95% CIs via `--combine`, B=100,000, seed 42), regenerated
  from the per-game data below.
- `pergame/<player>.jsonl` — one line per game (`chequer`, `cube`, `overall`
  mEMG), one file per main-table row; names match
  `run_cubeful_memg.py --players` keys. Every row, including sage 0-ply, is a
  plain 4,000-game jsonl.

Provenance: every row is generated on the corrected move generator and judged
by a gnubg asserted to carry BOTH bearoff databases (`gnubg_ts0.bd` md5
`44b6040b49b46cb9dd2ce8caa947044d`, `gnubg.wd` md5
`14184acc9c60ef67be0fad88548fd51a`) with zero "will not use the two-sided
bearoff" warnings. `gnubg_eval.assert_gnubg_bearoff` enforces this before any
judging run, and `experiments/head_to_head/run.py` checks it up front before a
head-to-head starts.

## Rows (overall mEMG, lower is better)

| player | config | chequer | cube | overall |
|---|---|---|---|---|
| ours-1ply | 4L cubeful | 0.94 | 0.90 | 0.93 [0.90, 0.97] |
| ours-0ply | 4L cubeful | 1.78 | 1.70 | 1.76 [1.70, 1.82] |
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
