# HedgeHog head-to-head rows

HedgeHog as the opponent: side spec `hedgehog:<model.ogxf>`, `--oneply2`
for HedgeHog depth 1. The shim `libhhagent.so` and the `.ogxf` models are
not in the repo: [third_party_opponents/hedgehog/README.md](../../third_party_opponents/hedgehog/README.md). PureTD
plays through the C API, `simd` backend (`--c-api --c-api-backend simd`).
Neither side uses a bearoff database.

| model | md5 |
|---|---|
| `fox_v0.34_2026-09-26.ogxf` | `240a6a1605d58c44decf4fc3db413559` |
| `aureus_v0.12_2026-09-26.ogxf` | `379f8f20dd2b0ce05f9e24f05098fb87` |

## Commands

```bash
P5=best_models/cubeless_prob5_512_512_256_256.pt
C4=best_models/cubeful_money_512_512_256_256.pt
HH=third_party_opponents/hedgehog

# cubeless: fox 0-ply (aureus: the other file; 1-ply rows, cubeless and
# cubeful: --oneply2 --games 4000000)
python experiments/head_to_head/run.py \
  --side1 $P5 --side2 hedgehog:$HH/fox_v0.34_2026-09-26.ogxf \
  --c-api --c-api-backend simd \
  --game-mode cubeless-money --games 10000000 --workers 64 --save <dir>

# cubeful, Jacoby on, cube capped at 64 for both sides (the paper's rows)
python experiments/head_to_head/run.py \
  --side1 $C4 --side2 hedgehog:$HH/fox_v0.34_2026-09-26.ogxf \
  --c-api --c-api-backend simd \
  --game-mode cubeful-money --jacoby --cube-cap 64 \
  --games 10000000 --workers 64 --save <dir>

# pool the --save legs
python experiments/head_to_head/run.py --combine <dir>
```

## Results

Positive = PureTD ahead, mEq/game, bootstrap 95% CI. Engine `e8847f4`
([third_party_opponents/hedgehog/README.md](../../third_party_opponents/hedgehog/README.md)).

### Cubeless: prob5 `ac95a983` 0-ply vs HedgeHog

Rows in [results/h2h_cubeless_summary.csv](results/h2h_cubeless_summary.csv).

| opponent | games | mEq/game | 95% CI |
|---|---:|---:|---|
| fox 0-ply | 10M | +26.6 | [+25.7, +27.5] |
| fox 1-ply | 4M | −0.8 | [−2.1, +0.6] |
| aureus 0-ply | 10M | +13.0 | [+12.1, +13.9] |

### Cubeful, Jacoby on: PureTD 4L `6afda0f8` 0-ply vs HedgeHog

Cube capped at 64 for both sides (`--cube-cap 64`),
[results/h2h_cubeful_cap64_summary.csv](results/h2h_cubeful_cap64_summary.csv):

| opponent | games | mEq/game | 95% CI | win rate |
|---|---:|---:|---|---:|
| fox 0-ply | 10M | +68.5 | [+66.7, +70.4] | 52.6% |
| aureus 0-ply | 10M | +47.8 | [+46.0, +49.7] | 53.2% |
| fox 1-ply | 4M | +18.0 | [+15.2, +21.1] | 51.4% |
