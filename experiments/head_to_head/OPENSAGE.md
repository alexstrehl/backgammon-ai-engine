# Open Sage as a head-to-head side

`pip install bgsage==2.0.20260907` (release
[v2.0.20260907](https://github.com/markbgsage/bgsage/releases/tag/v2.0.20260907),
commit `4b10c69318c3b80e2a7618232540b10a76217d20`, production model `stage11`:
its 21 networks (24 slots), `backgame_pair_phased`, weights md5 over the 24 slot files in load order
`84f8621f3e274f6d25f83c07da657a4f`, bearoff database md5
`c1234d7b9a1c4cbce591bf4f93a3cb5f`). The rows need exactly this version;
bgsage 1.x gives different rows.

| spec keyword | class | game modes |
|---|---|---|
| `opensage` | `OpenSageAgent` | cubeless money: Sage's `BgBotAnalyzer(eval_level, cubeful=False)` with its bearoff database, `cube_owner=None`; DMP: win probability from the production `MultiPlyStrategy` |
| `opensage-native` | `OpenSageMobileAgent` | cubeful money, 0-ply and 1-ply: Sage's native C API (`bgsage_mobile_analyze`), nets loaded once |
| `opensage-analyzer` | `OpenSageAnalyzerAgent` | cubeful money: Sage's `BgBotAnalyzer`, cube-aware `checker_play` plus `cube_action`; reloads its nets on every cube call |

`opensage-native` and `opensage-analyzer` make the same decisions (0 of
2,000 0-ply and 0 of 500 1-ply checker decisions, 0 of 1,500 cube decisions
differ on the cubeful fixed pool); native calls that exceed the API's 1 s
budget are answered by the analyzer. `opensage-analyzer` is needed only for
2-ply (`--sage-plies 3`). The native library is built from the release
source: `bash third_party_opponents/opensage/opensage_mobile_build.sh` (or `BGSAGE_MOBILE_LIB`).

`--sage-plies` is XG-numbered: 1 = 0-ply, 2 = 1-ply, 3 = 2-ply.
`SAGE_UNPRUNED=1` disables the root move filter (production 5/0.08 becomes
1000/10.0). It changes cubeless 1-ply decisions only. Cubeful 1-ply has no
setting: checker play re-scores every candidate through the cube-aware tree,
which scores every opponent reply with the full net; cube decisions pre-rank
replies with PubEval and keep the top 15 when more than 16 are legal.

Move generation: `bgsage.possible_moves` offers a one-die play when a
two-dice play exists (e.g. bearing off with the 2 instead of hitting with
the 1 and bearing off with the 2); 178 of 194,508 position-roll pairs in the
differential against the repo's gnubg-validated generator. Every agent here
picks from the repo's legal plays only.

```bash
bash third_party_opponents/opensage/opensage_mobile_build.sh
python experiments/head_to_head/run.py \
  --side1 best_models/cubeful_money_512_512_256_256.pt \
  --side2 opensage-native --sage-plies 1 \
  --game-mode cubeful-money --jacoby --log-runaway 256 \
  --games 1000000 --workers 30 --save DIR
```

Cubeless money plays the analyzer's best-ranked legal move: on 2,000
fixed-pool positions and 3,000 self-play decisions, 100% of picks equal
the stock analyzer's at 0-ply and 1-ply. The analyzer runs one thread per
process: it forces `parallel_threads` to at least 2 at 1-ply, but Sage's
search uses threads only above 2 plies. The agent skips the analyzer's
per-decision flush of its 1-ply position cache (lookups check board and
depth, so values are unchanged; the flush is memory-bound with many
workers per host). `cube_owner=None` keeps the 1-ply
root filter on cubeless equity; a cube owner ranks it by Janowski equity.

The speed benchmarks time Open Sage on the fixed pools only, calling its
decision functions directly
([third_party_opponents/opensage/opensage_bench/](../../third_party_opponents/opensage/opensage_bench/sage_bench.cpp)).

`--save` legs pool with `--combine DIR`; `--combine` refuses to pool across
bgsage versions, SAGE_UNPRUNED settings or `sage_path` (cubeless money:
`analyzer`; absent = `best_move_index`).
