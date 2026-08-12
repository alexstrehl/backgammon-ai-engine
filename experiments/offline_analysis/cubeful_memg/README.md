# Cubeful-money mEMG — gnubg 1.08.003 judge

Judge: **gnubg 1.08.003** CLI at **4-ply** (`gnubg.wd` md5
`14184acc9c60ef67be0fad88548fd51a`). All scoring with `LC_ALL=C` (en_US
locale silently breaks the stats parser), `OMP_NUM_THREADS=1`. n=4000
legal-opening self-play games per row (Jacoby on) — except the
sage-analyzer-0ply row, which pools its three raw shards (n=6,664 per-side
samples; see Layout).

**PureTD (`ours-*`) model**: `best_models/cubeful_money_512_512_256_256.pt`
md5 `77a3205cc7582dbe7da4ae4e6c89fef1` (prod).

## Layout

- `cubeful_memg.csv` — summary table (chequer / cube / overall means +
  game-level bootstrap 95% CIs via `--combine`, B=10,000 seed 42), regenerated from the
  per-game data below.
- `pergame/<player>.jsonl` — one line per game (`chequer`, `cube`, `overall`
  mEMG), one file per main-table row; names match
  `run_cubeful_memg.py --players` keys. The sage 0-ply row is instead stored
  as its three original raw shards (`sage_1P_raw_{local,epyc,topup}.json`,
  per-side sample arrays pooled for the published value).

Provenance of the `ours-*` rows: generated 2026-07-10/11 on a rented
Threadripper PRO 9975WX box; judge weights bit-identical to the reference
(md5 gate `14184acc...`), so the samples pool with all other rows here.

## Rows (overall mEMG, lower is better)

| player | config | chequer | cube | overall |
|---|---|---|---|---|
| ours-1ply | 4L cubeful | 0.987 | 1.411 | 1.027 [0.971, 1.093] |
| ours-0ply | 4L cubeful | 1.968 | 1.873 | 1.936 [1.871, 2.007] |
| gnubg-ext-1ply | true gnubg 1.08 binary, full/unpruned | 2.022 | 2.145 | 1.997 [1.936, 2.067] |
| sage-analyzer-1ply | SAGE_UNPRUNED (paper convention) | 1.974 | 5.758 | 2.258 [2.189, 2.330] |
| gnubg-ext-0ply | true gnubg 1.08 binary | 4.011 | 3.819 | 3.898 [3.779, 4.027] |
| sage-analyzer-0ply | analyzer, pooled raw shards | 3.795 | 7.048 | 4.020 [3.926, 4.114] |

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
