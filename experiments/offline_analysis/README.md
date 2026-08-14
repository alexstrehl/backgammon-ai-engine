# offline_analysis

Reproduction scripts for the main offline strength results. These files are
**thin command wrappers** — all functionality lives in the codebase
(`gnubg_eval.py` play/export/judge helpers, `gnubg_external.py`,
`experiments/opensage_agents.py`, `stats.py`). Each script just records the exact invocations used and the
output-combining step, so the published numbers can be regenerated.

Files:

| file | role |
|------|------|
| `cubeless_money_memg.sh` | entry point: cubeless-money mEMG table (wraps `run_cubeless_memg.py`) |
| `cubeful_money_memg.sh` | entry point: cubeful-money mEMG table (wraps `run_cubeful_memg.py`) |
| `run_cubeless_memg.py` | cubeless runner: self-play → `.mat` export → gnubg judge → bootstrap CIs |
| `run_cubeful_memg.py` | cubeful counterpart (live-cube games; judges chequer AND cube decisions) |
| `sage_money_memg.py` | scores Open Sage's native money self-play transcripts with the same judge |
| `memg_common.py` | shared helpers (resilient gnubg worker, per-game persistence, `--combine` pooling) |

Published tables + per-game raws live in `cubeless_memg/` and `cubeful_memg/`
(see their READMEs).

## `cubeless_money_memg.sh` — cubeless-money mEMG (gnubg N-ply judge)

Mean error per move (milli-equity) for **PureTD-prob5 / gnubg (real 1.08
binary, `gnubg-ext`) / Open Sage**, each at
**0-ply and 1-ply**, judged by gnubg at `--plies` (default 4). Lower = stronger.

**Cubeless-money throughout:** every engine selects moves by cubeless money equity
(`2·P(win)+P(wg)+P(wbg)−P(lg)−P(lbg)−1`; gammons/backgammons count, **no cube**), and the
gnubg judge scores chequer play with cubeless equity too — gnubg's default is *cubeful*,
so the driver switches the judge to cubeless scoring: it sets `GNUBG_CUBELESS_EVAL=1`
itself, which makes `gnubg_eval._write_gnubg_script` issue two commands —
`set analysis chequerplay evaluation cubeful off` (errors measured in the same
cubeless-equity units the engines optimize) and `set analysis cube off` (gnubg scores
*missed* doubles even in a 0-point session, so without this the never-doubling cubeless
agents would accrue phantom cube errors that swamp "overall", making it ~85% artifact).
With cube analysis off, overall == chequer — so for cubeless we report **chequer only**.

**Prereqs:** gnubg 1.08.003 CLI (`/usr/games/gnubg` or `GNUBG_CMD`; it is both
the judge and the `gnubg-ext` players), `pip install bgsage`, torch + this
repo, and `best_models/cubeless_prob5_512_512_256_128.pt`.

**Run:**
```bash
GAMES=4000 PLIES=4 PLAY_WORKERS=56 GNUBG_WORKERS=90 bash experiments/offline_analysis/cubeless_money_memg.sh
```
Outputs a pooled, equal-n table (chequer + overall mEMG with bootstrap 95% CIs) for the six
players. Cost is dominated by the gnubg N-ply analysis (heavy tail at 4-ply: budget ~1 h per
1000 games on ~30 cores); `--save-pergame` makes it resumable/poolable across machines.

## `cubeful_money_memg.sh` — cubeful-money mEMG (gnubg N-ply judge)

Cubeful-money counterpart: engines play **live-cube** money games (Jacoby — real
double/take/drop decisions), and gnubg scores **both chequer and cube** decisions at
`--plies`. Reports chequer mEMG (move quality), cube, and overall (chequer+cube).

- **ours (4L cubeful) + gnubg-ext** via `run_cubeful_memg.py` — in-process live-cube self-play
  (`play_and_record_cubeful`, Jacoby), `export_mat(money_game=True)`, gnubg cubeful judge (its
  default; `GNUBG_CUBELESS_EVAL` is NOT set). Model `best_models/cubeful_money_512_512_256_256.pt`.
- **Open Sage 1-ply** via the same `run_cubeful_memg.py` (`--players sage-analyzer-1ply`:
  cube-aware `OpenSageAnalyzerAgent` self-play). `SAGE_UNPRUNED=1` reproduces the
  published paper-convention row; without it you are measuring Sage's production
  filter instead. The wrapper runs it after the main quartet.
- **Open Sage 0-ply** via the same `run_cubeful_memg.py`
  (`--players sage-analyzer-0ply`), at Sage's production filter — do not set
  `SAGE_UNPRUNED`. `sage_money_memg.py` is the separate transcript path, which
  scores Sage's own generated transcripts rather than self-play here.

**Prereqs:** gnubg 1.08.003 CLI, `pip install bgsage`, torch + this repo, and the cubeful model.

**Run:**
```bash
GAMES=4000 PLIES=4 PLAY_WORKERS=48 GNUBG_WORKERS=48 bash experiments/offline_analysis/cubeful_money_memg.sh
```
Set workers to **physical** cores (gnubg N-ply is compute-bound; SMT/oversubscription hurts).

**Caveats to disclose:** runs are not
seeded / no common-random-numbers across engines; Sage transcripts use Beaver on while
ours/gnubg in-process self-play has no beaver (different money variant if tabled together).
