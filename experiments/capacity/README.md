# Network size

Eight cubeful and eight prob5 nets of one to six hidden layers: strength
head-to-head at 0-ply against the production 4-layer net of the family,
and cost per decision on the fixed pools of
[../benchmarks/README.md](../benchmarks/README.md). One- to three-layer
nets: [../../best_models/experimental/small_models/README.md](../../best_models/experimental/small_models/README.md);
five- and six-layer nets: [../../best_models/experimental/README.md](../../best_models/experimental/README.md).

## Head-to-head against production

The variant as side 1. 10M games per row, chunks pooled with
`run.py --combine`. Cubeful rows: Jacoby on, capped mean (±128).

```bash
python experiments/head_to_head/run.py \
    --side1 best_models/experimental/small_models/cubeful_money_512_512_256.pt \
    --side2 best_models/cubeful_money_512_512_256_256.pt \
    --game-mode cubeful-money --jacoby --games 250000 --workers 48 --save DIR
python experiments/head_to_head/run.py \
    --side1 best_models/cubeless_prob5_768_512_256_256_256.pt \
    --side2 best_models/cubeless_prob5_512_512_256_256.pt \
    --game-mode cubeless-money --games 250000 --workers 48 --save DIR
python experiments/head_to_head/run.py --combine DIR
```

Cells: [../head_to_head/results/h2h_experimental_vs_prod.csv](../head_to_head/results/h2h_experimental_vs_prod.csv);
pooled outputs of four rows: `../head_to_head/results/*_vs_prod_10M.txt`.

## Decisions per second

The `cap` job of the benchmark suite's `fast` group
([../benchmarks/README.md](../benchmarks/README.md), network sizes):
[capacity_costbench.py](../benchmarks/capacity_costbench.py), eight
sweeps per host, checkpoints pinned by md5, every net through the C API
(`torch` and `simd` rows), 0-ply checker and cube decisions.

```bash
bash c_api/build.sh torch simd
BENCH_CORE=<core> bash experiments/benchmarks/capacity_sweeps.sh   # or ROWS=cap suite.sh <out> fast
python experiments/benchmarks/aggregate_sweeps.py $HOME/bench_results/capacity/capacity_sweep*.txt
python experiments/capacity/network_size_table.py   # the table below
```

## Results

Checker decisions/s at 0-ply, mean of eight sweeps; in parentheses the
per-decision cost relative to the production net on the same backend and
host. Zen 5 = Ryzen 7 9700X, Zen 3 = Ryzen 9 5950X. Sweeps and
aggregates, cube rows included: `../benchmarks/results/<host>_20261014/`
(`fast/capacity_sweep*.txt`, `aggregate.txt`); intervals in
[../benchmarks/README.md](../benchmarks/README.md) (network sizes).

| model | file | hidden | params | md5 | vs production, mEq [95% CI] | games | Zen 5 torch | Zen 3 torch | Zen 5 simd | Zen 3 simd |
|---|---|---|---|---|---|---|---|---|---|---|
| cubeful 1-layer | [`cubeful_money_512.pt`](../../best_models/experimental/small_models/cubeful_money_512.pt) | [512] | 103,425 | `39d75783` | -157.5 [-159.6, -155.3] | 10M | 28,513 (0.20×) | 25,268 (0.20×) | 200,640 (0.22×) | 117,225 (0.16×) |
| cubeful 2-layer | [`cubeful_money_512_512.pt`](../../best_models/experimental/small_models/cubeful_money_512_512.pt) | [512,512] | 366,081 | `1ddbb775` | -25.1 [-26.9, -23.3] | 10M | 8,098 (0.70×) | 7,828 (0.66×) | 66,295 (0.67×) | 28,518 (0.65×) |
| cubeful 3-layer | [`cubeful_money_512_512_256.pt`](../../best_models/experimental/small_models/cubeful_money_512_512_256.pt) | [512,512,256] | 497,153 | `27a728cf` | -4.5 [-6.3, -2.7] | 10M | 6,103 (0.92×) | 5,856 (0.88×) | 47,415 (0.93×) | 19,981 (0.92×) |
| cubeful 4-layer (production) | [`cubeful_money_512_512_256_256.pt`](../../best_models/cubeful_money_512_512_256_256.pt) | [512,512,256,256] | 562,945 | `6afda0f8` | --- |  | 5,638 (1.00×) | 5,169 (1.00×) | 44,138 (1.00×) | 18,458 (1.00×) |
| cubeful 5-layer | [`cubeful_money_512_512_256_256_256.pt`](../../best_models/experimental/cubeful_money_512_512_256_256_256.pt) | [512,512,256,256,256] | 628,737 | `37e41475` | +5.0 [+3.3, +6.8] | 10M | 5,025 (1.12×) | 4,648 (1.11×) | 38,157 (1.16×) | 15,921 (1.16×) |
| cubeful 5-layer wide | [`cubeful_money_768_512_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_512_256_256_256.pt) | [768,512,256,256,256] | 811,265 | `f65c49fe` | +8.4 [+6.6, +10.2] | 10M | 3,937 (1.43×) | 3,650 (1.42×) | 28,046 (1.57×) | 11,352 (1.63×) |
| cubeful 6-layer wide | [`cubeful_money_768_512_256_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_512_256_256_256_256.pt) | [768,512,256,256,256,256] | 877,057 | `4f86345a` | +11.1 [+9.3, +12.9] | 10M | 3,651 (1.54×) | 3,379 (1.53×) | 24,624 (1.79×) | 9,967 (1.85×) |
| cubeful 5-layer 768x768 | [`cubeful_money_768_768_256_256_256.pt`](../../best_models/experimental/cubeful_money_768_768_256_256_256.pt) | [768,768,256,256,256] | 1,073,665 | `de9e6e84` | +16.4 [+14.7, +18.2] | 10M | 3,011 (1.87×) | 2,823 (1.83×) | 20,022 (2.20×) | 7,966 (2.32×) |
| prob5 1-layer | [`cubeless_prob5_512.pt`](../../best_models/experimental/small_models/cubeless_prob5_512.pt) | [512] | 103,429 | `fa2c6935` | -82.6 [-83.5, -81.8] | 10M | 22,837 (0.22×) | 19,272 (0.24×) | 171,536 (0.22×) | 100,228 (0.16×) |
| prob5 2-layer | [`cubeless_prob5_512_512.pt`](../../best_models/experimental/small_models/cubeless_prob5_512_512.pt) | [512,512] | 366,085 | `e8c0c760` | -12.1 [-13.0, -11.3] | 10M | 7,333 (0.69×) | 6,727 (0.69×) | 53,720 (0.69×) | 23,445 (0.68×) |
| prob5 3-layer | [`cubeless_prob5_512_512_256.pt`](../../best_models/experimental/small_models/cubeless_prob5_512_512_256.pt) | [512,512,256] | 496,133 | `3da8e498` | -3.7 [-4.5, -2.8] | 10M | 5,726 (0.88×) | 5,237 (0.89×) | 42,485 (0.88×) | 18,115 (0.88×) |
| prob5 4-layer (production) | [`cubeless_prob5_512_512_256_256.pt`](../../best_models/cubeless_prob5_512_512_256_256.pt) | [512,512,256,256] | 561,925 | `ac95a983` | --- |  | 5,032 (1.00×) | 4,644 (1.00×) | 37,288 (1.00×) | 15,866 (1.00×) |
| prob5 5-layer | [`cubeless_prob5_512_512_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_512_512_256_256_256.pt) | [512,512,256,256,256] | 627,717 | `bade7d51` | +1.9 [+1.1, +2.8] | 10M | 4,571 (1.10×) | 4,180 (1.11×) | 33,949 (1.10×) | 14,363 (1.10×) |
| prob5 5-layer wide | [`cubeless_prob5_768_512_256_256_256.pt`](../../best_models/cubeless_prob5_768_512_256_256_256.pt) | [768,512,256,256,256] | 809,221 | `bd1a7931` | +5.2 [+4.3, +6.0] | 10M | 3,619 (1.39×) | 3,297 (1.41×) | 27,182 (1.37×) | 11,169 (1.42×) |
| prob5 6-layer wide | [`cubeless_prob5_768_512_256_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_768_512_256_256_256_256.pt) | [768,512,256,256,256,256] | 875,013 | `a9df2ffc` | +6.7 [+5.8, +7.6] | 10M | 3,332 (1.51×) | 3,056 (1.52×) | 25,668 (1.45×) | 10,563 (1.50×) |
| prob5 5-layer 768x768 | [`cubeless_prob5_768_768_256_256_256.pt`](../../best_models/experimental/cubeless_prob5_768_768_256_256_256.pt) | [768,768,256,256,256] | 1,071,621 | `b3fad4eb` | +9.7 [+8.9, +10.6] | 10M | 2,748 (1.83×) | 2,564 (1.81×) | 20,474 (1.82×) | 8,305 (1.91×) |

## Figure data

[results/capacity_cubeful.dat](results/capacity_cubeful.dat),
[results/capacity_prob5.dat](results/capacity_prob5.dat): one row per net,
`params` (thousands), `edge` with error-bar lengths `ehi`, `elo` (mEq),
`label`, `zen5` (Zen 5 `torch` checker decisions/s, the x-axis); `anc` and
`anc2` place the label.

```bash
python experiments/capacity/network_size_table.py --dat
```
