# Head-to-Head Benchmarks — Reproduction Guide

This directory runs two backgammon engines against each other and reports the
win rate and (for money modes) per-game equity, each with bootstrap 95% CIs.
`run.py` contains **only library calls** — all logic lives in the repo
(`play_models.py`, `agents.py`, `gnubg_external.py`, `gnubg_eval.py`,
`modes.py`, `backgammon_engine.py`, `stats.py`).

The matchups below pit **our engine ("PureTD")** against two public engines:
**gnubg** (real 1.08 binary) and **Open Sage**. You can reproduce any cell from scratch with the
commands here. Results are **statistically reproducible, not bit-identical**:
game generation is unseeded, so a rerun lands within the reported CI rather than
reproducing the exact games.

---

## Requirements (pinned)

- **Python** 3.10 or 3.11 (both have wheels for the engines below). On modern
  distros that ship only newer Pythons and PEP-668-locked pip, create the venv
  with e.g. `uv venv --python 3.11 && uv pip install ...` (or `pyenv`) rather
  than the system interpreter.
- A C toolchain (`gcc`/`make`) to build the bundled fast engine
- Python packages (all on public PyPI):

  ```bash
  pip install torch numpy==1.26.3 scipy \
              bgsage==1.3.20260723
  ```

  - `bgsage` (bundles `bgbot_cpp`) — only needed for matchups with an
    `opensage` side. The pinned release is tagged upstream at
    <https://github.com/markbgsage/bgsage/releases/tag/v1.3.20260723>,
    commit `a9576e8ab300e36969329e842962232883f7d90d`; production model `stage9`.
  - The real **gnubg 1.08.003 binary IS required** for the paper's `gnubg-ext*`
    sides (external-player socket). Easiest install: `apt install gnubg` on a
    recent Debian/Ubuntu (Ubuntu 26.04 ships exactly 1.08.003; older releases
    ship 1.06/1.07, whose nets are **different** — always verify
    `md5sum /usr/share/gnubg/gnubg.wd` = `14184acc9c60ef67be0fad88548fd51a`).
    Otherwise build from GNU source — recipe in
    `experiments/benchmarks/README.md` (the `--disable-threads` flag there is
    only needed for the compute benchmarks; a stock build is fine here).
    If the binary isn't at `/usr/games/gnubg`, point the `GNUBG_CMD` env var
    at it.

Our model checkpoints ship in the repo under `best_models/` (~2 MB each).

---

## One-time setup

```bash
git clone https://github.com/alexstrehl/backgammon-ai-engine bg-h2h && cd bg-h2h
pip install torch numpy==1.26.3 scipy bgsage==1.3.20260723

# Build the bundled fast engine (required):
bash c_engine/build_unix.sh          # -> c_engine/libbg_engine.so

# Optional, AMD CPUs only: build the MKL "fakeintel" shim for a big speedup,
# then LD_PRELOAD it before every run. Skip entirely on Intel.
bash c_engine/build_fakeintel.sh     # -> c_engine/libfakeintel.so
export LD_PRELOAD="$PWD/c_engine/libfakeintel.so"
```

---

## Running a matchup (general form)

```bash
python experiments/head_to_head/run.py \
    --side1 <model-path | gnubg-ext | gnubg-ext-full | opensage | opensage-analyzer | random> \
    --side2 <model-path | gnubg-ext | gnubg-ext-full | opensage | opensage-analyzer | random> \
    --game-mode <dmp | cubeless-money | cubeful-money> \
    --games <N> --workers <N> [--plies P] [--sage-plies S] [--jacoby/--no-jacoby]
```

Key arguments:

| Arg | Meaning |
|-----|---------|
| `--side1 / --side2` | a model checkpoint path, or an engine keyword |
| `--game-mode` | `dmp` (1-pt match), `cubeless-money` (gammon-weighted), `cubeful-money` (doubling cube) |
| `--plies` | gnubg search depth for a `gnubg-ext*` side (0 or 1) |
| `--sage-plies` | Open Sage ply (XG numbering): **1 = 0-ply (1P), 2 = 1-ply (2P)** |
| `--jacoby` | Jacoby rule for `cubeful-money` (default **on**); `--no-jacoby` to disable |
| `--games` | games to play (default 1,000,000) |
| `--workers` | parallel worker processes (default 32; gnubg 1-ply is slow — use as many cores as you have) |
| `--save DIR` | save raw per-game results (one `.npz`) so runs can be pooled/extended |
| `--combine DIR` | pool all `.npz` in a `--save` dir and report the combined result (no new games) |

Notes:
- **Paper opponents (real gnubg 1.08 over the external socket):**
  `gnubg-ext` = gnubg as shipped (prune nets + Normal move filter);
  `gnubg-ext-full` = pruning disabled + keep-all filter — **all paper rows
  use `gnubg-ext-full`**. Both work in cubeless and cubeful modes and use
  gnubg's native money cube (Jacoby) in cubeful. NOTE: pruning is NOT a
  separate flag — it is encoded in the side name.
- **Open Sage pruning:** the paper rows disable Sage's root move filter
  with the env var `SAGE_UNPRUNED=1` (production filter 5/0.08 otherwise).
- Cubeful runs should pass `--log-runaway 256` (archives runaway
  redouble-war games; scoring uses the capped ±128 mean either way).
- Cubeful-money Open Sage sides must use `opensage-analyzer` (cube-AWARE
  checker play + native cube action via Sage's own `BgBotAnalyzer`); the
  plain `opensage` keyword is cubeless-only and hard-errors in cubeful mode.

---

## Exact commands for each reported cell

PureTD's checkpoints: prob5 (cubeless) = `best_models/cubeless_prob5_512_512_256_128.pt`;
cubeful 4L = `best_models/cubeful_money_512_512_256_256.pt`. Prepend the
`LD_PRELOAD` shim on AMD. `--workers` shown as 32; raise to match your cores.

### Cubeless money (gammon-weighted, no cube)

```bash
# PureTD 0-ply vs gnubg 0-ply  (real gnubg over socket)
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_128.pt \
  --side2 gnubg-ext --plies 0 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 1-ply UNPRUNED (paper row)
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_128.pt \
  --side2 gnubg-ext-full --plies 1 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_128.pt \
  --side2 opensage --sage-plies 1 --game-mode cubeless-money --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 1-ply UNPRUNED (paper row)
SAGE_UNPRUNED=1 python experiments/head_to_head/run.py --side1 best_models/cubeless_prob5_512_512_256_128.pt \
  --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 48

# gnubg 1-ply vs Open Sage 1-ply (engine baseline, both unpruned)
SAGE_UNPRUNED=1 python experiments/head_to_head/run.py --side1 gnubg-ext-full --plies 1 \
  --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 48
```

### Cubeful money (doubling cube, Jacoby on)

```bash
# PureTD 0-ply vs gnubg 0-ply  (real gnubg, native money cube)
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext --plies 0 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs gnubg 1-ply UNPRUNED (paper row)
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 gnubg-ext-full --plies 1 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 0-ply
python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 opensage-analyzer --sage-plies 1 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# PureTD 0-ply vs Open Sage 1-ply UNPRUNED (paper row)
SAGE_UNPRUNED=1 python experiments/head_to_head/run.py --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 opensage-analyzer --sage-plies 2 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48

# gnubg 1-ply vs Open Sage 1-ply (engine baseline, both unpruned)
SAGE_UNPRUNED=1 python experiments/head_to_head/run.py --side1 gnubg-ext-full --plies 1 \
  --side2 opensage-analyzer --sage-plies 2 --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 48
```

To reach 10M, either pass `--games 10000000`, or run in batches with
`--save DIR` and pool them with `--combine DIR` (point estimates are identical
in expectation; pooling just tightens the CI).

---

## Reading the output

For money modes `run.py` reports, from **side1's** perspective:
- **Win rate** with a Wilson 95% CI.
- **Per-game equity** in mEq/game: raw mean, median, trimmed (1%), and — for
  cubeful — the **capped mean (±128)**, each with a bootstrap 95% CI and p-value.
- The **cubeful headline is the capped (±128) mean** — capping individual game
  equities tames fat tails from large cube/gammon swings; raw and capped agree
  unless the tails are heavy.

---

## Reproducibility notes

- **Statistical, not bit-exact.** Game generation is unseeded (parallel workers
  seed independently), so a rerun reproduces the value *within
  its CI*, not the exact games. At 1M–10M games CIs are tight.
- **Jacoby is always on** for cubeful unless you pass `--no-jacoby`.
- **gnubg cube for money** is real gnubg's native money cube (via the
  external socket).
- **Open Sage ply numbering** is XG-style: `--sage-plies 1` = 0-ply, `2` = 1-ply.
- Runtime: **gnubg 1-ply is the bottleneck** (memory-bound, ~hundreds of
  games/sec even on many cores); the 0-ply and prob5-vs-prob5 cases are far
  faster.
- **Pin `bgsage`.** Verify with `pip show bgsage` before generating games.

---

## Results

**10M games for every row**, in the
machine-readable summaries `results/h2h_cubeless_summary.csv` and
`results/h2h_cubeful_summary.csv` (main-results convention: gnubg 1-ply =
full-width (no pruning, `gnubg-ext-full`), Open Sage 1-ply = native unpruned; config variants in
`results/h2h_native_pruning_matrix.csv`). Equity is mEq/game from PureTD's
perspective (+ = PureTD ahead); cubeful is the capped (±128) mean, Jacoby on.

### Cubeless money (all 10M games; gnubg = real 1.08 binary via `gnubg-ext` / `gnubg-ext-full`)
| Matchup | PureTD edge | 95% CI |
|---|---|---|
| PureTD 0-ply vs gnubg 0-ply | +46.3 | [+45.5, +47.2] |
| PureTD 0-ply vs gnubg 1-ply unpruned | +12.8 | [+11.9, +13.6] |
| PureTD 0-ply vs Open Sage 0-ply | +39.4 | [+38.6, +40.3] |
| PureTD 0-ply vs Open Sage 1-ply unpruned | +8.1 | [+7.2, +9.0] |
| gnubg 1-ply vs Open Sage 1-ply (both unpruned) | −3.4 (Open Sage has the edge) | [−4.2, −2.5] |

Robustness (default move filters instead of unpruned): 10M games per row, in
`results/h2h_native_pruning_matrix.csv` — directly comparable to the tables
here. **No filter effect is significant** in any of the four matchups:

| Matchup | pruned | unpruned | difference |
|---|---|---|---|
| vs gnubg 1-ply, cubeless | +13.3 | +12.8 | +0.5 [−0.7, +1.7] |
| vs Open Sage 1-ply, cubeless | +7.9 | +8.1 | −0.2 [−1.5, +1.1] |
| vs gnubg 1-ply, cubeful | +18.6 | +19.8 | −1.2 [−3.9, +1.5] |
| vs Open Sage 1-ply, cubeful | +16.2 | +17.4 | −1.2 [−4.0, +1.6] |

Neither contrast reaches significance, so no claim about the opponent's filter
is made.

### Cubeful money (capped ±128, Jacoby on; all 10M games; real gnubg cube)
| Matchup | PureTD edge | 95% CI |
|---|---|---|
| PureTD 0-ply vs gnubg 0-ply | +78.8 | [+77.0, +80.6] |
| PureTD 0-ply vs gnubg 1-ply unpruned | +19.8 | [+18.0, +21.7] |
| PureTD 0-ply vs Open Sage 0-ply | +81.9 | [+80.0, +83.8] |
| PureTD 0-ply vs Open Sage 1-ply unpruned | +17.4 | [+15.5, +19.4] |
| gnubg 1-ply vs Open Sage 1-ply (both unpruned) | −3.8 (Open Sage has the edge) | [−5.8, −1.8] |

Headline: **PureTD at 0-ply (no search) beats both gnubg and Open Sage at 1-ply
on money equity, in both cubeless and cubeful play.**
