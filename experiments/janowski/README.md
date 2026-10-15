# Learned versus heuristic cube handling

PureTD's cubeful money net against Prob5Jan: the prob5 cubeless net with
an analytic Janowski cube layer. `janowski.py` converts the five
probabilities into cubeful equity, `(1 - x) * dead + x * live`, with the
piecewise-linear live-cube curve (knots at the live take and cash points)
and a fixed cube-life index x = 0.76 (`janowski.DEFAULT_X`; env
`JANOWSKI_X=<x>` overrides); `janowski_agent.py` plays cubeful money with
it. Checker play ranks candidates by the cubeful equity of the resulting
position with the opponent's cube decision folded in: the opponent
doubles when double/take and double/pass both beat no double, then takes
or passes, whichever is worse for it; exact ties go to the better
cubeless equity. Cube actions follow the repo's money rule
(`janowski.money_cube_action`: PureTD's `TDAgent._money_cube_decision`,
take iff DT <= 1). 1-ply has the structure of PureTD's cubeful 1-ply:
every candidate is expanded, and per roll each cube branch takes its own
best reply (the paper plays Prob5Jan at 0-ply only). Both sides play
0-ply and route bearoff-class positions through the same exact bearoff
tables (`bearoff.py`; `bearoff_td.py` gives them to PureTD); with gammons
still possible Prob5Jan converts the exact probabilities. Prob5Jan zeroes
the gammon and backgammon probabilities the board rules out
(`prob_agent.prob5_sanity`; env `JANOWSKI_BOARD_SANITY=0` turns it off).
Plain prob5 does not (`PROB5_BOARD_SANITY=1` turns it on). The code is
`JANOWSKI_REV` 2026-10-13. Saved head-to-heads record the sanity settings
and the revision; `run.py --combine` refuses runs that differ in either.

Bearoff-class positions against gnubg's databases: within 3.0e-4 (gnubg's
database rounding) in the two-sided class and with gammons possible;
beyond the two-sided table without gammons the hazard DP is approximate
(up to 0.30 in ND).

## Inputs

- PureTD, side `bo:`:
  [best_models/cubeful_money_512_512_256_256.pt](../../best_models/cubeful_money_512_512_256_256.pt)
  (md5 `6afda0f85b039a186f018753120ed74c`).
- Prob5Jan, side `janowski:`:
  [best_models/cubeless_prob5_512_512_256_256.pt](../../best_models/cubeless_prob5_512_512_256_256.pt)
  (md5 `ac95a983a114c269257225e0b84a5d9f`). `JANOWSKI_X=<x>` sets the
  cube-life index; unset is 0.76.
- Bearoff tables, built once into `~/.cache`
  (`bg_bearoff_os6g.npz` md5 `0439a7bf`, `bg_bearoff_ts8.npz` md5 `8dbf76bc`;
  `BG_BEAROFF_TABLE` and `BG_BEAROFF_TS_TABLE` override the paths), ~30 min
  on one core. The first `janowski:` run exports them to the 155 MB
  `~/.cache/bg_bearoff_c.bin` the C API reads (`BG_BEAROFF_C_TABLE`
  overrides):

```bash
python -c 'import bearoff; bearoff.build_table(); bearoff.build_ts_table()'
```

- XG++ for the PR.

## Head-to-head

Cubeful money, Jacoby, 0-ply both sides, chunks pooled with
`run.py --combine`. Positive = PureTD ahead. All rows at
`JANOWSKI_REV` 2026-10-06c (recorded in every npz); new runs (2026-10-13)
do not pool with their raws. The command plays
x = 0.76; prefix `JANOWSKI_X=<x>` for the other rows. By default
`janowski:` plays the C Prob5Jan.

```bash
OMP_NUM_THREADS=1 python experiments/head_to_head/run.py \
    --side1 bo:best_models/cubeful_money_512_512_256_256.pt \
    --side2 janowski:best_models/cubeless_prob5_512_512_256_256.pt \
    --game-mode cubeful-money --games 1000000 --workers 48 --log-runaway 256 --save DIR
python experiments/head_to_head/run.py --combine DIR
```

[results/janowski_h2h.csv](results/janowski_h2h.csv).

| x | games | PureTD ahead, mEq/game | Prob5Jan win % |
|---|---|---|---|
| 0.76 | 10M | +15.5 [+13.8, +17.2] | 50.3 |
| 0.70 | 10M | +17.0 [+15.1, +18.9] | 48.7 |
| 0.68 | 10M | +17.4 [+15.6, +19.3] | 48.2 |
| 0.60 | 10M | +29.2 [+27.2, +31.2] | 46.6 |

## XG++ PR

1,000 self-play cubeful money games per side, analysed in XG++; the PR is
decision-weighted over both players of each game, with a game-level
bootstrap 95% interval. The XG++ exports are not committed.

```bash
python tools/analyze_xg_results.py <export.txt> --rows-per-game 2
```

| | PR | checker | cube |
|---|---|---|---|
| PureTD 0-ply | 0.80 [0.76, 0.84] | 0.85 [0.80, 0.89] | 0.56 [0.46, 0.67] |
| Prob5Jan 0-ply | 1.13 [1.06, 1.22] | 1.05 [0.97, 1.14] | 1.58 [1.36, 1.83] |
