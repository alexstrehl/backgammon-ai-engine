# Experimental models

Deeper/wider variants of the 4L cubeful and prob5 nets. mEq/game vs the
former production 4L nets, both sides 0-ply, with 95% CIs:

- cubeful: `best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt` (md5 `6afda0f8`), capped ±128
- prob5: `best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt` (md5 `ac95a983`)

The current 4L nets, `4e25e4f7` (cubeful) and `88b03c1c` (prob5), are
+7.2 [+6.4, +8.0] and +2.9 [+2.5, +3.3] over these references (luck-adjusted, 200k games).

| file | type | md5 | mEq [95% CI] |
|---|---|---|---|
| `cubeful_money_512_512_256_256_256.pt` | cubeful-money | `37e41475` | +5.0 [+3.3, +6.8] |
| `cubeful_money_768_512_256_256_256.pt` | cubeful-money | `f65c49fe` | +8.4 [+6.6, +10.2] |
| `cubeful_money_768_512_256_256_256_256.pt` | cubeful-money | `4f86345a` | +11.1 [+9.3, +12.9] |
| `cubeful_money_768_768_256_256_256.pt` | cubeful-money | `16039ea7` | +19.8 [+18.9, +20.7]¹ |
| `cubeless_prob5_512_512_256_256_256.pt` | cubeless prob5 | `bade7d51` | +1.9 [+1.1, +2.8] |
| `cubeless_prob5_768_512_256_256_256_256.pt` | cubeless prob5 | `a9df2ffc` | +6.7 [+5.8, +7.6] |
| `cubeless_prob5_768_768_256_256_256.pt` | cubeless prob5 | `b8d4306c` | +10.1 [+9.7, +10.5]¹ |

Rows without ¹ are in [h2h_experimental_vs_prod.csv](../../experiments/head_to_head/results/h2h_experimental_vs_prod.csv). Cost per decision: [../../experiments/capacity/README.md](../../experiments/capacity/README.md).

¹ Luck-adjusted (the [variance.py](../../experiments/head_to_head/variance.py) estimator, beta 1, 4L reference), paired dice, 200k games.

`small_models/` holds one- to three-layer nets; training in [small_models/README.md](small_models/README.md).
`recipe_runs/` holds the final models of the two end-to-end runs of the retraining recipe ([../../experiments/retraining/README.md](../../experiments/retraining/README.md)).

## Head-to-head

The variant as side 1, so `run.py` reports the table's sign. One row:

```bash
python experiments/head_to_head/run.py \
  --side1 best_models/experimental/cubeful_money_768_512_256_256_256.pt \
  --side2 best_models/history/cubeful_money_512_512_256_256_6afda0f8.pt \
  --game-mode cubeful-money --jacoby --games 10000000 --workers 48

python experiments/head_to_head/run.py \
  --side1 best_models/experimental/cubeless_prob5_768_512_256_256_256_256.pt \
  --side2 best_models/history/cubeless_prob5_512_512_256_256_ac95a983.pt \
  --game-mode cubeless-money --games 10000000 --workers 48
```
