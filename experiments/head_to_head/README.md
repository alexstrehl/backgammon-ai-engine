# Head-to-head strength

PureTD at 0-ply against gnubg 1.08.003 (the real binary, driven over its
external-player socket) and Open Sage, each at 0-ply and 1-ply, in cubeless
and cubeful money. `run.py` plays the two sides and reports side1's win rate
and mEq/game (thousandths of a point per game) with bootstrap 95% CIs. Game generation is unseeded: a rerun lands
within the reported CI, it does not replay the games.

## Requirements

- Python 3.10–3.13 and
  `pip install torch==2.13.0 numpy==2.2.6 bgsage==2.0.20260907`. `bgsage`
  is needed for Open Sage sides only: release
  [v2.0.20260907](https://github.com/markbgsage/bgsage/releases/tag/v2.0.20260907),
  commit `4b10c69318c3b80e2a7618232540b10a76217d20`, production model
  `stage11`. `opensage-native` also needs the native library:
  `bash third_party_opponents/opensage/opensage_mobile_build.sh`.
- gnubg 1.08.003 for the `gnubg-ext*` sides: `apt install gnubg` on a release
  that ships 1.08.003 (Ubuntu 26.04), or a build from GNU source. `GNUBG_CMD`
  names the binary when it is not `/usr/games/gnubg`. 1.06 and 1.07 carry
  different nets: `md5sum /usr/share/gnubg/gnubg.wd` must be
  `14184acc9c60ef67be0fad88548fd51a`.
- PureTD:
  [best_models/cubeless_prob5_512_512_256_256.pt](../../best_models/cubeless_prob5_512_512_256_256.pt)
  (md5 `ac95a983`) and
  [best_models/cubeful_money_512_512_256_256.pt](../../best_models/cubeful_money_512_512_256_256.pt)
  (md5 `6afda0f8`). Every `.npz` and CSV row carries the side's md5, and
  `--combine` refuses to pool across md5s.

```bash
bash c_engine/build_unix.sh                       # c_engine/libbg_engine.so
```

## Sides

`--side1` / `--side2` is a checkpoint path or an engine keyword:
`gnubg-ext` (gnubg as shipped: pruning nets, Normal move filter),
`gnubg-ext-full` (pruning off, keep-all filter),
`opensage` (cubeless money: Sage's cubeless `BgBotAnalyzer` with its
bearoff database; DMP: `MultiPlyStrategy` win probability), `opensage-native` (cubeful money, 0-
and 1-ply: Sage's native C API), `opensage-analyzer` (cubeful money:
Sage's `BgBotAnalyzer`, same decisions, needed for 2-ply;
[OPENSAGE.md](OPENSAGE.md)), `hedgehog:<model.ogxf>`
([HEDGEHOG.md](HEDGEHOG.md)), `janowski:<prob5.pt>` and `bo:<cubeful.pt>`
(cubeful money; [../janowski/README.md](../janowski/README.md)), `random`.
`--plies` is gnubg's depth.
`--sage-plies` is XG-numbered: 1 = 0-ply, 2 = 1-ply. `SAGE_UNPRUNED=1`
disables Sage's root move filter. Cubeless
only: Open Sage's cubeful 1-ply has no unpruned mode
([OPENSAGE.md](OPENSAGE.md)). Cubeful
money plays Jacoby by default (`--no-jacoby` to disable) with gnubg's native
money cube; `--log-runaway 256` archives the record of any game that reaches
cube 128.

Checkpoint and `janowski:` sides play through the C API (`c_api/`, simd,
built on demand; needs gcc) with the Python agents' decisions up to float
near-ties. A run it cannot serve (2-ply, `--bf16` prob5) plays on the
Python path after one NOTE; `--c-api` makes that an error, `--python-api`
selects the Python path, `--c-api-backend` picks the backend. The npz
records the path and `--combine` refuses to pool across paths: the rows
below were played on the Python path, so add `--python-api` to extend or
pool with their npz.

## Commands

1-ply baselines run at their default move pruning; the unpruned runs are
for the move-filter table.

### Cubeless money

```bash
# PureTD 0-ply vs gnubg 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 gnubg-ext --plies 0 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 1-ply
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 gnubg-ext --plies 1 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 opensage --sage-plies 1 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 1-ply
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 48
```

### Cubeful money, Jacoby on

```bash
# PureTD 0-ply vs gnubg 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext --plies 0 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 1-ply
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext --plies 1 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 2-ply
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext --plies 2 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 opensage-native --sage-plies 1 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 1-ply (no pruning setting)
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 opensage-native --sage-plies 2 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48
```

### Unpruned 1-ply opponents

```bash
# PureTD 0-ply vs gnubg 1-ply unpruned, cubeless
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 gnubg-ext-full --plies 1 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 1-ply unpruned, cubeless
SAGE_UNPRUNED=1 python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_256.pt \
  --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 1-ply unpruned, cubeful
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext-full --plies 1 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48
```

A 10M-game row is either `--games 10000000` or batches run with
`--save DIR` and pooled with `--combine DIR`:

```bash
python experiments/head_to_head/run.py --combine DIR
```

## Results

mEq/game from PureTD's perspective (positive = PureTD ahead), with bootstrap
95% CIs. 1-ply baselines use their default move pruning. Cubeful rows are the capped (±128) mean, Jacoby on. Summaries:
[results/h2h_cubeless_summary.csv](results/h2h_cubeless_summary.csv),
[results/h2h_cubeful_summary.csv](results/h2h_cubeful_summary.csv); the
Best 3L cubeful, Best 4L cubeless, Best 3L cubeless and Best DMP rows:
[results/h2h_model_table_rows_20260810.csv](results/h2h_model_table_rows_20260810.csv).
HedgeHog rows: [HEDGEHOG.md](HEDGEHOG.md). Open Sage cubeless pooled outputs:
`results/prob5_vs_opensage*_cbl_*.txt`. Experimental and small models vs
production ([../../best_models/experimental/README.md](../../best_models/experimental/README.md)):
[results/h2h_experimental_vs_prod.csv](results/h2h_experimental_vs_prod.csv).

### Cubeless money, 10M games per row

| Matchup | PureTD edge | 95% CI |
|---|---|---|
| PureTD 0-ply vs gnubg 0-ply | +45.8 | [+45.0, +46.7] |
| PureTD 0-ply vs gnubg 1-ply | +12.1 | [+11.2, +12.9] |
| PureTD 0-ply vs Open Sage 0-ply | +36.8 | [+35.9, +37.7] |
| PureTD 0-ply vs Open Sage 1-ply | +5.9 | [+5.1, +6.8] |

### Cubeful money, capped ±128, Jacoby on

| Matchup | PureTD edge | 95% CI | games |
|---|---|---|---|
| PureTD 0-ply vs gnubg 0-ply | +85.2 | [+83.5, +87.0] | 10M |
| PureTD 0-ply vs gnubg 1-ply | +27.2 | [+25.3, +29.0] | 10M |
| PureTD 0-ply vs Open Sage 0-ply | +79.5 | [+75.2, +83.8] | 2M |
| PureTD 0-ply vs Open Sage 1-ply¹ | +23.8 | [+19.5, +28.2] | 2M |
| PureTD 0-ply vs gnubg 2-ply | −20.7 (gnubg has the edge) | [−24.8, −16.6] | 2M |

### Opponent move filter

PureTD 0-ply vs each 1-ply baseline with its move filter on (pruned) and off
(unpruned). Cubeless rows 10M games (Open Sage unpruned 2M); cubeful 10M
(gnubg) and 2M (Open Sage).
Rows are in [results/h2h_cubeless_summary.csv](results/h2h_cubeless_summary.csv)
and [results/h2h_cubeful_summary.csv](results/h2h_cubeful_summary.csv).

| Matchup | pruned | unpruned | difference |
|---|---|---|---|
| vs gnubg 1-ply, cubeless | +12.1 [+11.2, +12.9] | +12.3 [+11.5, +13.2] | −0.2 [−1.4, +1.0] |
| vs Open Sage 1-ply, cubeless | +5.9 [+5.1, +6.8] | +6.7 [+4.8, +8.6] | −0.8 [−2.9, +1.3] |
| vs gnubg 1-ply, cubeful | +27.2 [+25.3, +29.0] | +27.8 [+25.9, +29.6] | −0.6 [−3.2, +2.0] |
| vs Open Sage 1-ply, cubeful | +23.8 [+19.5, +28.2]¹ | n/a | |

¹ Open Sage's cubeful 1-ply is not configurable: checker play scores every
legal move and every opponent reply with the full net; cube decisions pre-rank
replies with PubEval and keep the top 15 when more than 16 are legal.
