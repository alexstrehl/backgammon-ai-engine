# Cubeless-money mEMG — gnubg 1.08.003 judge

Judge: **gnubg 1.08.003** CLI at **4-ply** (`gnubg.wd` md5
`14184acc9c60ef67be0fad88548fd51a`; gnubg ≥1.08 ships retrained nets vs
1.06/1.07 — see `../cubeful_memg/README.md` for the same-judge cubeful table).
All scoring with `LC_ALL=C`, `OMP_NUM_THREADS=1`, `GNUBG_CUBELESS_EVAL=1`
(cube analysis off; move selection AND judging use cubeless money equity —
gammons/backgammons count, no cube leverage). n=4000 self-play games per row,
legal openings by construction.

## Layout

- `cubeless_memg.csv` — summary table (means + game-level bootstrap 95% CIs
  via `--combine`, B=10,000 seed 42), regenerated from the per-game data below.
- `pergame/<player>.jsonl` — one line per game (`chequer` mEMG), one file per
  main-table row. Player names match `run_cubeless_memg.py --players` keys, so
  the pooled CI is reproducible directly:

      python experiments/offline_analysis/run_cubeless_memg.py \
        --combine experiments/offline_analysis/cubeless_memg/pergame --players sage-1ply

- `robustness/` — same format, non-paper configs kept for the filter-contrast
  discussion (currently: `sage-1ply.jsonl` = Sage production filter 5/0.08).

## Rows (chequer mEMG, lower is better)

| player | config | mEMG |
|---|---|---|
| prob5-1ply | full-width 1-ply | 0.694 [0.675, 0.713] |
| prob5-0ply | prod prob5 | 1.343 [1.312, 1.375] |
| gnubg-ext-1ply | true gnubg 1.08 binary, full/unpruned | 1.648 [1.611, 1.686] |
| sage-1ply | SAGE_UNPRUNED (paper convention) | 1.669 [1.613, 1.731] |
| sage-0ply | native, production filter | 3.043 [2.981, 3.106] |
| gnubg-ext-0ply | true gnubg 1.08 binary | 2.932 [2.875, 2.989] |
| sage-1ply (robustness) | production filter 5/0.08 | 1.768 [1.697, 1.850] |

Notes:

- **gnubg rows are gnubg-ext** (the real 1.08.003 binary driven over the
  external-player socket) — the same opponent as the H-H tables.
- **sage rows** use the native `OpenSageAgent` adapter (`best_move_index`),
  identical to the H-H runs. The paper row disables the root move filter
  (`SAGE_UNPRUNED=1` → 1000/10.0), Sage's stronger config; the production
  filter row is kept under `robustness/`.
- Cubeless rows have no cube/overall split: with cube analysis off,
  overall == chequer (the per-game `overall` field includes gnubg's
  never-double artifact and is ignored).

## Reproduction

```bash
# one row (repeat per player: prob5-0ply, prob5-1ply, gnubg-ext-0ply,
# gnubg-ext-1ply, sage-0ply, sage-1ply; sage paper row: prefix SAGE_UNPRUNED=1)
LC_ALL=C OMP_NUM_THREADS=1 GNUBG_CMD=/usr/games/gnubg \
python3 experiments/offline_analysis/run_cubeless_memg.py \
    --players prob5-0ply --games 4000 --plies 4 \
    --play-workers 64 --gnubg-workers 64 \
    --save-pergame experiments/offline_analysis/cubeless_memg/pergame

# pooled table + game-level bootstrap CIs
python3 experiments/offline_analysis/run_cubeless_memg.py \
    --combine experiments/offline_analysis/cubeless_memg/pergame
```

`../cubeless_money_memg.sh` wraps the same invocations. Runs may be sharded
across machines (disjoint entropy-seeded RNG streams) and pooled by
concatenating the per-game jsonl; a single-box run is statistically
equivalent.
