# Open Sage as an H-H baseline (optional dependency)

Open Sage (`bgsage`) is an OPTIONAL opponent used only to reproduce the
head-to-head tables.

## Install (pinned for reproduction)

    pip install bgsage==1.3.20260723

## Which agent / spec keyword

| Spec keyword        | Class                   | Use |
|---------------------|-------------------------|-----|
| `opensage`          | `OpenSageAgent`         | cubeless-money / DMP (errors in cubeful mode) |
| `opensage-analyzer` | `OpenSageAnalyzerAgent` | cubeful-money |

`--sage-plies` is XG-numbered: 1 = 0-ply (1P), 2 = 1-ply (2P).

## Why two agents (cubeless vs cubeful)

The two agents exist because Sage itself exposes two different play paths,
and each of ours mirrors the corresponding native one:

- **Cubeless / DMP** (`OpenSageAgent`): Sage's lean `best_move_index` move
  selection on the stage9 `MultiPlyStrategy` — its native cubeless money
  objective, with no cube machinery in the loop.
- **Cubeful money** (`OpenSageAnalyzerAgent`): Sage's own `BgBotAnalyzer`,
  exactly as Sage's self-play generator drives it — cube-AWARE
  `checker_play` (the move choice receives cube value/owner) plus native
  `cube_action`.

Fidelity of the cubeful agent was verified decision-for-decision against
Sage's own self-play: 100% agreement on 1105 cube offers, 60 responses,
and 1494 real checker moves over 40 replayed seeded games (forced-pass
bookkeeping differs but outcomes are identical; the beaver flag flipped
0/1165 cube decisions). Cube-aware checker play matters: replacing it
with cube-blind (cubeless) move selection costs Sage ~7 mEq/game against
PureTD, so the analyzer path is the fair representation of Sage's
cubeful strength.

## Reproduce the cubeful H-H cell

    python experiments/head_to_head/run.py \
      --side1 best_models/cubeful_money_512_512_256_256.pt \
      --side2 opensage-analyzer --sage-plies 1 \
      --game-mode cubeful-money --games 1000000 --workers 30 \
      --save ~/hh_out/puretd_vs_opensage_analyzer_0ply   # raw .npz stays outside the repo; only summary CSVs are committed

Runs may be split across machines and pooled with `--combine <dir>`;
worker RNGs are entropy-seeded and the combiner rejects byte-identical
duplicate streams (see `tests/test_rng_independence.py`).
