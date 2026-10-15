# C self-play collector

`train_batch.py` (dmp, cubeless-money, cubeful-money) and `train_prob5.py` collect
self-play in C by default: `bgc_collect` (`c_api/puretd_collect.c`) in
`libpuretd_<backend>.so` (`--c-collect-backend`, default `simd`). The master exports
each round's weights once (`export_weights.export_live`); each worker plays on one
`random.Random(seed)` stream and returns the samples the Python collector
(`trainer.collect_episode`, `train_prob5._collect_one_episode`) produces for the
same seed. Play is 0-ply argmin; targets are 0- or 1-ply.

## Parity

| level | compares | pass | where |
|---|---|---|---|
| 1 | C API primitives and every emitted sample vs the torch agent | 0-ply 1e-5, 1-ply 1e-4 (x10 relative above 1, `test_c_api.close`) | `tests/test_c_api.py`, `tests/test_c_collect.py` |
| 2 | one episode per seed, C vs Python; cells dmp, cubeless, cubeful x jacoby, prob5, each 0-/1-ply, `ref` and `simd` | row count equal, encodings bit-exact, targets within level 1 up to the first decision C takes differently, which must be a near-tie (C scores within 1e-5) | `tests/test_c_collect.py` |
| 3 | one training round per collector, same `--seed` and `--workers`, from `BG_COLLECT_DUMP` | per-worker seeds and row counts equal; aligned rows as level 2; every stream parting is a near-tie | `round_parity.py`, `tests/test_c_collect_trainer.py`, `tests/test_c_collect_prob5_trainer.py` |
| 4 | A/B twin: two 1M-episode trainings from one parent, only the collector differs | 1M-game head-to-head inside the noise band (+-5.6 mEq), no common sign over 0-/1-ply | [results/ab_twin.md](results/ab_twin.md) |

prob5 1-ply targets keep the whole vector of each roll's max-equity reply, so a
near-tie between replies can move a component by p(roll) x their difference. Such a
row passes when some choice among each roll's near-tied replies (equity within
1e-5) reproduces the C vector within 1e-4 (`prob5_tie_target`).

## Defaults and fallbacks

| condition | default | explicit `--c-collect` |
|---|---|---|
| `--boltzmann-temp` != 0, `--oneply-acting`, `--bf16-collect`, `--c-engine`, an encoder other than `perspective196` / `cubeful_perspective196`, dmp with an equity net, a mode/net pair with no C engine kind (prob5: a non-`perspective196` net) | NOTE per reason, then the `--no-c-collect` routing | NOTE per reason, Python collector |
| `libpuretd_<backend>.so` cannot be built or loaded (a missing one is built on demand) | NOTE, then the `--no-c-collect` routing | error |
| `--vectorized-collect` (prob5 also `--gpu-eval-server`) | that collector, NOTE | error |
| `--oneply --device cuda` | C collector, NOTE | C collector, NOTE |

`--no-c-collect`: the Python collector, or with `--oneply --device cuda` the
vectorized GPU collector. `--pipeline-collect`, `--save-every` and pool restarts work
with the C collector. At `--workers 1` the master collects one worker's share on a
seed drawn from the run RNG, so its episodes differ from the Python single-process
path; from 2 workers on both collectors play the same episodes up to near-tie partings.

## Environment

- `BG_TRAIN_CUDA_GRAPH`: unset or `1` = graphed train step on a cuda device (mode
  auto), `0` = eager, `adam` / `fwdbwd` = that mode; implies the on-device pool;
  bitwise equal to the eager step (`cuda_graph_step.py`).
- `BG_COLLECT_SHM`: unset or `1` = C workers write rows into shared-memory arenas,
  `0` = the pool pipe; bitwise equal (`row_transport.py`).
- `BG_COLLECT_DUMP=<dir>`: each round's pool to `round_NNNN.npz` (`enc`, `tgt`,
  `worker_rows`, `worker_seeds`), for level 3.
- `BG_ROUND_TIMING=1`, `BG_ROUND_PHASES=1`: per-round collect/train and master
  phase seconds (read by `bench.sh`).

The defaults give checkpoints bitwise equal to `BG_TRAIN_CUDA_GRAPH=0 BG_COLLECT_SHM=0`
(`tests/test_collector_defaults.py`). C build switches: `c_api/README.md`.

## Harness

Results: [results/](results/), one note per measurement.

| script | does |
|---|---|
| `bench.sh` | one training workload, N rounds, frozen weights: ep/s over rounds 2..N, collect/train split, `SUMMARY` line; `-c` workloads use the C collector |
| `p4_sweep.py` | bench.sh arms one at a time behind a load gate, interleaved, one `sweep.tsv` row per run |
| `p8_table.py` | per-cell tables of a sweep (transport arms) |
| `round_parity.py` | parity level 3 |
| `ab_timing.py` | paired single-core A/B of two builds (or two environments), `Engine.collect` ep/s per cell |
| `p5_profile.py` | `-DBGC_PROFILE` counters |
