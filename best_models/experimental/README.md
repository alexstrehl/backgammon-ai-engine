# Experimental models

Deeper/wider variants of the production cubeful and prob5 net. mEq/game
vs the matching production net, with bootstrap 95% CIs:

- cubeful: `best_models/cubeful_money_512_512_256_256.pt` (md5 `6afda0f8`), capped ±128
- prob5: `best_models/cubeless_prob5_512_512_256_256.pt` (md5 `ac95a983`)

| file | type | md5 | mEq [95% CI] |
|---|---|---|---|
| `cubeful_money_512_512_256_256_256.pt` | cubeful-money | `37e41475` | +5.0 [+3.3, +6.8] |
| `cubeful_money_768_512_256_256_256.pt` | cubeful-money | `f65c49fe` | +8.4 [+6.6, +10.2] |
| `cubeful_money_768_512_256_256_256_256.pt` | cubeful-money | `4f86345a` | +11.1 [+9.3, +12.9] |
| `cubeful_money_768_768_256_256_256.pt` | cubeful-money | `de9e6e84` | +16.4 [+14.7, +18.2]¹ |
| `cubeless_prob5_512_512_256_256_256.pt` | cubeless prob5 | `bade7d51` | +1.9 [+1.1, +2.8] |
| `cubeless_prob5_768_512_256_256_256_256.pt` | cubeless prob5 | `a9df2ffc` | +6.7 [+5.8, +7.6] |
| `cubeless_prob5_768_768_256_256_256.pt` | cubeless prob5 | `b3fad4eb` | +9.7 [+8.9, +10.6] |

Rows are in [h2h_experimental_vs_prod.csv](../../experiments/head_to_head/results/h2h_experimental_vs_prod.csv). Cost per decision: [../../experiments/capacity/README.md](../../experiments/capacity/README.md).

¹ Converged: a further 33M-episode leg tied it, +0.2 [-0.5, +0.9] (12M paired games).

`small_models/` holds one- to three-layer nets; training in [small_models/README.md](small_models/README.md).
`recipe_runs/` holds the final models of the two end-to-end runs of the retraining recipe ([../../experiments/retraining/README.md](../../experiments/retraining/README.md)).

## Head-to-head

The variant as side 1, so `run.py` reports the table's sign. One row:

```bash
python experiments/head_to_head/run.py \
  --side1 best_models/experimental/cubeful_money_768_512_256_256_256.pt \
  --side2 best_models/cubeful_money_512_512_256_256.pt \
  --game-mode cubeful-money --jacoby --games 10000000 --workers 48

python experiments/head_to_head/run.py \
  --side1 best_models/experimental/cubeless_prob5_768_512_256_256_256_256.pt \
  --side2 best_models/cubeless_prob5_512_512_256_256.pt \
  --game-mode cubeless-money --games 10000000 --workers 48
```
