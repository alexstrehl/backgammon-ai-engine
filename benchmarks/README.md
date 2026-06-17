# Self-play compute benchmark (single-thread moves/s)

Self-contained reproduction of the moves/s figures: our 4L DMP network vs
gnubg-nn, measured on a single thread. Everything needed is vendored here or
built by `build.sh`; no hard-coded paths.

## Contents

| File | Purpose |
|------|---------|
| `bench_agent.py` | **AGENT path** — times the production decision call each engine uses |
| `bench_lean.py`  | **LEAN path** — same evaluators, Python orchestration stripped |
| `build.sh`       | Builds `libbg_engine.so` (our C move-gen) and installs `gnubg-nn==1.1.0a8` |
| `run.sh`         | Runs all configs across seeds, single-thread |
| `models/dmp_512_512_256_256.pt` | Pinned model — 4L DMP, cubeless (the exact weights benchmarked) |

The harness imports our engine modules (`td_agent`, `model`, `agents`,
`backgammon_engine`, `bg_fast`) from the parent repo via a path computed from
`__file__`, so it works from any checkout location.

## Dependencies

- This repository (parent of this folder), with `c_engine/` buildable by `gcc`.
- [`gnubg-nn`](https://pypi.org/project/gnubg-nn/) `== 1.1.0a8` — published wheel,
  GNU Backgammon's neural evaluator. Provides `gnubg_nn.best_move` (native C
  move-gen + eval) and is required for the `gnubg0` / `gnubg1` configs.
- PyTorch, NumPy.

## Usage

```bash
./build.sh                       # build libbg_engine.so + ensure gnubg-nn
./run.sh                         # full run: n=40000, seeds 1..5, AGENT + LEAN
./run.sh 40000 "1 2 3" agent     # custom n / seeds / path
python3 bench_agent.py --config ours0 --n 40000 --seed 1   # single config
```

Configs: `ours0` (our net, 0-ply), `gnubg0` (gnubg-nn 0-ply),
`gnubg1` (gnubg-nn 1-ply).

## What is measured

Move generation runs in C for **every** config (a shared, fair engine). Each
evaluator then uses its fastest native path:

- **ours** — `bg_fast` C move-gen + encode, then a single **batched PyTorch**
  forward (fp32) over all legal successors, then `argmin`.
- **gnubg-nn** — its `best_move` C API (native move-gen + eval).

Set `BF16=1` to evaluate our network in bfloat16 (faster, but ~6% move
disagreement vs fp32 — not used for the reported figures).

All runs pin `OMP_NUM_THREADS=MKL_NUM_THREADS=torch.num_threads=1`.

## Reference results (single thread, AMD Zen 5, n=40000 × 5 seeds)

| Config | AGENT (moves/s) | LEAN (moves/s) |
|--------|-----------------|----------------|
| ours 0-ply      | 4,087 ± 31  | 6,237 ± 84  |
| gnubg-nn 0-ply  | 21,313 ± 455 | 21,870 ± 424 |
| gnubg-nn 1-ply  | 249 ± 6     | 248 ± 7     |

gnubg-nn 0-ply is ~5.2× our 0-ply throughput; our 0-ply is ~16× faster than
gnubg-nn 1-ply.
