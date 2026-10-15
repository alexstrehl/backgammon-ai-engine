# HedgeHog opponent

[HedgeHog](https://hedgehog-bg.com)
([hedgehog-public](https://gitlab.com/eranlambooij/hedgehog-public), engine
pinned at `e8847f4`) as a money-game side of the head-to-head runner: side
spec `hedgehog:<model.ogxf>`, `--oneply1` / `--oneply2` for HedgeHog depth 1.
Money modes only. Rows and commands:
[experiments/head_to_head/HEDGEHOG.md](../../experiments/head_to_head/HEDGEHOG.md); per-decision
cost: [experiments/benchmarks/README.md](../../experiments/benchmarks/README.md).

```bash
python experiments/head_to_head/run.py \
    --side1 best_models/cubeless_prob5_512_512_256_256.pt \
    --side2 hedgehog:third_party_opponents/hedgehog/fox_v0.34_2026-09-26.ogxf --c-api --c-api-backend simd \
    --game-mode cubeless-money --games 100000 --workers 32 --save <dir>
```

The saved npz records the `.ogxf` md5, so pooled results cannot mix model
versions.

## Build

The engine, the shim binary and the model weights are not tracked.

```bash
git clone https://gitlab.com/eranlambooij/hedgehog-public.git
(cd hedgehog-public && git checkout e8847f4 && make example -j && make run-tests)
bash third_party_opponents/hedgehog/shim/build_shim.sh ./hedgehog-public   # -> third_party_opponents/hedgehog/libhhagent.so
```

Full pin: `e8847f498f7be8798c0b71314e83c8b4075f2043`, used by the
HedgeHog head-to-head and benchmark rows.

The shim is built `-march=native`: build it on the machine that runs it.
`hh_lib.py` loads `$HH_LIB` when set, else
`third_party_opponents/hedgehog/libhhagent.so`.

Models (hedgehog-bg.com/models/), placed in `third_party_opponents/hedgehog/` or named
by `$HH_FOX` / `$HH_AUREUS` (benchmarks) and `$HH_MODEL` (tests):

| model | file | md5 |
|---|---|---|
| fox v0.34 | `fox_v0.34_2026-09-26.ogxf` | `240a6a1605d58c44decf4fc3db413559` |
| aureus v0.12 | `aureus_v0.12_2026-09-26.ogxf` | `379f8f20dd2b0ce05f9e24f05098fb87` |
| default-small v1.0 | `default-small_v1.0_2026-05-13.ogxf` | tests only |

## Engine configuration

- Cube: Janowski formulas, x = 0.68, Jacoby-aware; the agent sets the shim's
  Jacoby flag from `match_state.jacoby` every game. Cube cap 64
  (`CUBE_MAX_VALUE`); `--cube-cap 64` applies the same ceiling to both sides.
- No bearoff database, no beavers.
- Legality from our `get_legal_plays`; HedgeHog ranks the successors.
  Leaves are evaluated with the side to move and flipped to the root.

## Validation

- `python third_party_opponents/hedgehog/verify_conversion.py [games]`: board
  round-trip, successor sets (ours a subset of HedgeHog's) and terminal
  winner and points on random games. Engine `e8847f4`, 300 games: 29,663
  positions exact, no larger-die-laxity positions on HedgeHog's side.
- `HH_LIB=<shim.so> python third_party_opponents/hedgehog/validate_costpath.py`:
  the benchmark's native C path (0-ply successors evaluated in one batch,
  `eval_ensemble_successors`) picks the agent's move (each candidate
  evaluated alone) on every item of both committed pools, Fox and Aureus:
  0/8000 cubeless and 0/8994 cubeful mismatches each
  ([experiments/hedgehog/results/validate_costpath_e8847f4_20261010.txt](../../experiments/hedgehog/results/validate_costpath_e8847f4_20261010.txt)).
