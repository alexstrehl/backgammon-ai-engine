# Open Sage money-game mEMG (gnubg N-ply reference)

How we measure Open Sage's playing strength in **cubeful money games** as
**mEMG** (mean error per move, in milli-equity), using **gnubg as the N-ply
reference analyzer**. Default reference depth is **4-ply**.

The pipeline is two stages: (1) Open Sage self-plays money games and writes
XG-import transcripts; (2) gnubg scores every decision at N-ply and we average
the per-decision errors into chequer / cube / overall mEMG with bootstrap CIs.

## Pinned versions (for reproducibility)

| Component | Version |
|-----------|---------|
| Open Sage (`bgsage`) | `v0.8.20260611`, commit `e09bf9c87a5644272765da53cff511f895c0270a` (see the version note below) |
| bgsage repo | https://github.com/markbgsage/bgsage |
| gnubg (reference analyzer) | GNU Backgammon **1.08.003** (`gnubg.wd` md5 `14184acc9c60ef67be0fad88548fd51a`; gnubg ≥1.08 ships retrained nets — 1.06/1.07 judges are NOT comparable) |

Open Sage 0-ply = its level **`1P`** (raw NN); 1-ply = **`2P`** (one level of
lookahead). XG numbering, so **Sage 1P = 0-ply**.

## Stage 1 — generate Open Sage money self-play games

Run from the **bgsage repo** at the pinned commit. Money games (Jacoby +
Beaver on), both sides played by Sage at the given level. Game `i` uses RNG
seed `initial_seed + i`, so the seed range fully determines the games.

```bash
cd /path/to/bgsage          # at commit e09bf9c
# 0-ply (1P): 1000 games, seeds 1000000..1000999, 30 parallel workers
python scripts/run_sage_vs_sage_games.py 1000000 1000 \
    --level 1P --workers 30 \
    --out-dir logs/sage_money_1P_fresh
```

Each game is written to `<out-dir>/seed_<N>.txt`. Use a **fresh, disjoint seed
range** to generate a new sample rather than reusing prior transcripts (the
original set used seeds `1..1000`; this run uses `1000000..1000999`).

## Stage 2 — score with gnubg at N-ply and compute mEMG

`sage_money_memg.py` (this directory) cleans the transcripts, runs gnubg at
`--plies` over all games in parallel, and reports chequer / cube / overall
mEMG with bootstrap 95% CIs. Both sides are scored, then averaged to one
per-game mean; the bootstrap resamples games (n = games).

```bash
cd /path/to/backgammon-ai/experiments/offline_analysis   # back in THIS repo (stage 1 left you in bgsage)
python sage_money_memg.py \
    --src /path/to/bgsage/logs/sage_money_1P_fresh \
    --plies 4 --workers 30 \
    --label "Open Sage 0-ply (1P)"
```

Output (and raw per-game means) are saved under `--work`
(default `/tmp/sage_money_ana`). Set the gnubg binary with the
`GNUBG_CMD` env var — a **1.08.003** build is required (recent Debian/Ubuntu
apt ships it; older releases ship 1.06/1.07, a different judge — the runtime
preflight verifies version + weights md5).

## Notes

- **Money game, not DMP.** These transcripts include cube actions, so the
  `cube` and `overall` components are meaningful (unlike the 1-point-match /
  DMP pipeline, which is win-only).
- gnubg's `show statistics match` emits three `Error rate ... mEMG` lines per
  game (chequer, cube, overall), each with a (player1, player2) pair; the
  scorer collects every pair.
- 4-ply scoring of ~1000 games is the slow stage (~1 h on 30 cores);
  generation at 0-ply is minutes.

### Open Sage version note

These numbers were produced with `bgsage` **v0.8.20260611**; the repo now pins
**v1.3.20260723** (commit `a9576e8ab300e36969329e842962232883f7d90d`). The two are interchangeable here:

* both ship the same production model — `PRODUCTION_MODEL = "stage9"`, the
  19-network backgame-aware model that determines play;
* they were verified to make **identical decisions** over ~26,000 compared
  values on 4,000 positions sampled across all game phases (3,327 opening /
  midgame, 469 bearing off, 204 deep bear-off, 1,557 with a checker on the
  bar), at 0-ply and 1-ply, under both the production and unpruned filters,
  covering the cubeless `best_move_index` path and the analyzer's
  `checker_play` and `cube_action` — zero differences, including equity
  values to six decimals;
* four configurations of **seeded self-play produced bit-identical game
  streams** (matching SHA-256 over full game outcomes).

So re-running these rows under the pinned release reproduces them unchanged.
