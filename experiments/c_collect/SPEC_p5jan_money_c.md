# Prob5Jan in C: the money Janowski agent on the PureTD C API

Prob5Jan = a cubeless prob5 net under the
Janowski cube layer at fixed x = 0.76 (`janowski.py`, `janowski_agent.py`,
`bearoff.py`). The Python agent is the reference; this port serves the same
decisions from C: engine kind `BGC_KIND_PROB5JAN`, opened by
`bgc_open_janowski` on a prob5 export, through the existing money entry
points (`bgc_best_move`, `bgc_value_1ply`, `bgc_offer_double`,
`bgc_respond_to_double`, `bgc_play_game`, the bench calls).

## From the match-play implementation

Carried over: the file split (conversion + evaluator in its own
`c_api/puretd_janowski.c`, a kind flag in `bgc_open`, dispatch on the kind
at the entry points); double numerics in Python's operation order with
`fp-contract=off`; row values as doubles beside the fp32 batch; the prob5
clamp exported to the Janowski file (`bgc_i_prob5_clamp`); the ctypes
pattern (`Engine(model, backend, janowski=True)`, a `JanowskiC` mirror of
the Python functions for parity tests); the test plan's shape (conversion
alone bitwise, then the full stack vs the Python agent on backend pairs).

Not carried over: everything match-specific. No MET, no `bgc_jan_ctx`, no
cash-point ladder, no dead-cube gate, no position-class x, no
match memos, no `puretd_match.c`. The money conversion
is a different function (below).

## The reference, in C terms

Conversion (`janowski.py`), on the nested 5 probabilities p = (P(w),
P(wg), P(wbg), P(lg), P(lbg)) of the player on roll, in double:

    W = 1 + (p1 + p2) / max(p0, 1e-7),  L = 1 + (p3 + p4) / max(1 - p0, 1e-7)
    degenerate (p0 <= 1e-7 or p0 >= 1 - 1e-7): W = L = 1, value = dead
    dead = cubeless money equity: 2p0 - 1 with Jacoby on a centered cube,
           else (2p0 + p1 + p2 - p3 - p4) - 1
    live = piecewise-linear in p0 through (0, lo), (TP, -1), (CP, +1), (1, hi)
           TP = (L - 0.5) / (W + L + 0.5), CP = (L + 1) / (W + L + 0.5)
           CENTERED: all four knots, (lo, hi) = (-1, 1) under Jacoby else (-L, W)
           MINE: (0, -L), (CP, 1), (1, W);  THEIRS: (0, -L), (TP, -1), (1, W)
           segment: the first with p0 < right knot, else the last;
           y0 + (y1 - y0) * (p0 - x0) / (x1 - x0), left to right
    value = (1 - x) * dead + x * live
    folded (the on-roll player's own cube decision, CENTERED / MINE only):
           nd = value(persp), dt = 2 * value(THEIRS), dp = 1:
           (dt >= nd and dp >= nd) ? min(dt, dp) : nd
    cube action (money_cube_action = TDAgent._money_cube_decision):
           dt > 1: double iff (jacoby and CENTERED) or 1 >= nd; else double iff dt > nd
           take iff dt <= 1

Agent (`janowski_agent.py`, defaults: x 0.76, cube_fold, fold_0ply,
bearoff_exact, board_sanity):

- Row probabilities: net sigmoids -> `prob5_impossible_mask` zeroing (the
  features' gnubg impossible-gammon rules, `c_engine/bg_engine.c`) -> the
  prob5 clamps -> double.
- Successor rows (a successor's player on roll = the leaf player), per
  branch (perspective of the leaf player, jacoby): terminal row (mover
  won, result r) value -(r if gammons count else 1), probs (0,0,0,r>=2,r>=3);
  bearoff-class row (both sides home, no bar): exact tier when it exists
  (two-sided table within 8 checkers a side; hazard DP beyond, gammon-free),
  value from the table (with the leaf's own double window when folding),
  probs (P_exact, 0, 0, 0, 0); bearoff-class with gammons possible: exact
  probabilities through the conversion; else the net's probs through the
  conversion. `sec` = cubeless equity (gammons counted) of the probs used.
- Pick: the lowest leaf-view value compared in float32; ties to the lowest
  float32 `sec`; then the first.
- 0-ply checker: successors at the opponent's perspective, folded.
- 1-ply checker: every candidate (n > 1); exact rows (terminal, table tier)
  keep their 0-ply (value, sec); a net candidate S (roller = the opponent)
  is valued by `_oneply(S, branches)`: branches = [cube] + [cube after the
  roller takes] when the roller may double; per roll (d1 = 1..6, d2 = d1..6,
  weight 1 or 2) and branch the leaf pick above (fold on), acc[b] += w *
  value; probs follow branch 0's pick; v[b] = -acc[b] / 36; score =
  fold(v[0], 2 * v[1]) (or v[0] alone), sec1 = cubeless equity of the
  averaged probs flipped to the roller. A forced pass is one row: the
  board with the turn switched. Then pick over (score, sec1).
- Cube: 0-ply (nd, dt) = (value(persp), 2 * value(THEIRS)) of the root's
  probs: the exact tables at a gammon-free bearoff root, exact probs at a
  gammonish one, else the net; 1-ply: a table root keeps the tables, else
  `_oneply(root, [cube, cube after take])` -> (v[0], 2 v[1]). Offer only at
  CENTERED / MINE. Respond: the offer's decision when hinted, else the same
  values' take.

Bearoff (`bearoff.py`): one-sided table (54,264 positions x 24 rolls, f and
g distributions, float32 in the file, widened to double), two-sided table
(3,003 positions a side: P win probability, V cubeful value per perspective
with the on-roll window folded, float32 widened). Exact probabilities by
cross-side convolution (`exact_probs5`); hazard DP `_race_dp` on hazards
rounded to 12 decimals (`rint(h * 1e12) / 1e12`); `_ts_wnd`: the window-free
values of (ia, ib) by one dice pass over V with the legal plays of the
parked one-sided board. Tiers as `_exact_tier`.

## Interface

C (`puretd.h`), all in `c_api/puretd_janowski.c`:

    enum { ..., BGC_KIND_PROB5JAN = 6 };      /* 4, 5 reserved */
    bgc_engine *bgc_open_janowski(const char *bgnn_path, char *err, size_t errlen);
    typedef struct { double x; int board_sanity; } bgc_jan_opts;   /* defaults 0.76, 1 */
    int bgc_set_jan(bgc_engine *, const bgc_jan_opts * /* NULL = defaults */);
    typedef struct bgc_bearoff bgc_bearoff;    /* read-only after open; shareable by engines */
    bgc_bearoff *bgc_bearoff_open(const char *path, char *err, size_t errlen);  /* mmap */
    void bgc_bearoff_close(bgc_bearoff *);
    int bgc_set_bearoff(bgc_engine *, const bgc_bearoff * /* NULL = bearoff_exact off */);
    /* the conversion alone */
    double bgc_jan_equity(const double p[5], int persp, double x, int jacoby);
    double bgc_jan_equity_folded(const double p[5], int persp, double x, int jacoby);
    double bgc_jan_cubeless_equity(const double p[5]);
    double bgc_jan_fold(double nd, double dt);
    int    bgc_jan_money_cube_action(double nd, double dt, int persp, int jacoby, int *take);
    /* the tables alone (1 found / 0 not applicable / -1 error) */
    int bgc_bo_exact_probs5(const bgc_bearoff *, const BoardState *, double p[5]);
    int bgc_bo_race_equity(const bgc_bearoff *, const BoardState *, int persp, int fold, double *v);
    int bgc_bo_race_cube_values(const bgc_bearoff *, const BoardState *, int persp, double *nd, double *dt);
    int bgc_bo_ts_win_prob(const bgc_bearoff *, const BoardState *, double *p);
    bgc_cube_eval gains `int take` at the END (the responder's decision in double).

Entry points on a PROB5JAN engine (cube required; NULL cube -> the error
value): `bgc_value` = 0-ply nd (`_static_cube`); `bgc_value_prob5` = the
sanitized clamped probs; `bgc_value_1ply` = the 1-ply value with the
roller's own cube decision folded (`_value_oneply(s, cube, fold=True)`);
`bgc_best_move` / `bgc_score_moves` (scores: mover view = -leaf value,
float; ties resolved by `sec`, not visible in the scores);
`bgc_value_1ply_play` = the two calls; `bgc_offer_double` (out: nd, dt
as float, take); `bgc_respond_to_double` (hint -> hint->take); the bench
calls and `bgc_play_game`. `bgc_collect` refuses (-1).

Seams in `puretd.c`: `struct bgc_jan *jan` at the END of
`bgc_engine`; `bgc_open_janowski` -> the open path with a flag; dispatch
lines in `bgc_value`, `bgc_value_prob5`, `value1k`, `choose`, `value1_play`
(no fast path), `bgc_offer_double`, `bgc_respond_to_double`, `bgc_collect`,
`bgc_close`; hidden helpers in `puretd_internal.h`: `bgc_i_expand21`,
`bgc_i_encode_base`, `bgc_i_forward_succ`, `bgc_i_ensure_rows`, a batch
view struct (x, y, ldx, plays, iplays, cdoff, cpass, rep_row, rep_gr,
cscore). The Janowski code reads a row's bearoff view (home counts, off)
from its 196 features, as the Python agent does: expand21 is untouched.

Bearoff file: `bearoff.export_c_tables(path)` writes a flat little-endian
file (magic `BGBO`, version 1, n_os, R_MAX, n_ts, ts_max, then os counts
uint8[n_os][6], f float32[n_os][24], g float32[n_os][24], ts counts
uint8[n_ts][6], P float32[n_ts][n_ts], V float32[n_ts][n_ts][3]); default
`~/.cache/bg_bearoff_c.bin` (`BG_BEAROFF_C_TABLE`), rewritten by
`bearoff.c_table_path()` when missing or older than either npz. C mmaps it
read-only and ranks (c1..c6) combinatorially into a position index.

Python (`c_api/puretd_py.py`): `Engine(model, backend, janowski=True,
x=None, board_sanity=True, bearoff=True)` (bearoff: True = the default
table, a path, a `BearoffTable`, or False); `BearoffTable(path=None)`;
`JanowskiC(backend)` mirroring `janowski.janowski_equity`,
`janowski_equity_folded`, `cubeless_equity`, `fold_cube_decision`,
`money_cube_action`, `bearoff.exact_probs5`, `race_equity`,
`race_cube_values`, `ts_win_prob`; `CAgent(model, plies, backend,
janowski=True, **jan_kw)` (is_cubeful). `play_models.py`: a `janowski:`
side with `--c-api`, or the side `janowski-c:<prob5.pt>` regardless, plays
through the C engine (JANOWSKI_X, JANOWSKI_NO_BEAROFF, JANOWSKI_BOARD_SANITY
honoured; JANOWSKI_NO_FOLD*, JANOWSKI_CUBELESS_CHECKER rejected; `--bf16`
rejected).

## Verification

Tests: `tests/test_janowski_c.py` (default tier simd; `BG_TESTS_SLOW=1`
ref, simd, torch; `CJAN_FULL=1` 1000 + 1000 pool points).

1. Conversion alone: 48,000 `janowski_equity` / `_folded` values (1000
   nested vectors incl. the 0 / 1e-7 / 1 - 1e-7 / 1 edges x 3 persp x
   jacoby x x in {0.76, 0, 1, 0.5}), `cubeless_equity`, `fold`: bitwise;
   `money_cube_action` equal on 2,640 (nd, dt) pairs x persp x jacoby.
2. Bearoff alone (460 boards, both tiers, gammonish, parked, edges,
   non-class): `race_equity`, `race_cube_values`, `ts_win_prob` bitwise;
   `exact_probs5` max |diff| 2.2e-16 (Python's `np.dot` sums in BLAS
   order; bound 1e-12).
3. Full stack vs `JanowskiAgent` (torch, `nn.Linear`) on the committed
   cubeful pool (seed 0 draw) + jacoby-off draw + constructed
   bearoff-class points. No backend's forward is bitwise the agent's
   (torch: `addmm` on a transposed weight copy, up to 5.4e-7 per
   probability); the live-cube slope amplifies that in v_dt. Test bound
   1e-5; measured:

   | depth, backend | checker / cube points | decisions equal (checker, offer, take, respond) | near-ties | max abs diff: scores / v_nd / v_dt / value |
   |---|---|---|---|---|
   | 0-ply torch | 12,000 / 8,000 | 100% | 0 | 1.8e-6 / 1.8e-6 / 3.5e-6 / 1.8e-6 |
   | 0-ply simd | 12,000 / 8,000 | 100% | 0 | 4.2e-6 / 1.5e-6 / 2.8e-6 / 1.5e-6 |
   | 1-ply torch | 1,400 / 1,400 | 100% | 0 | 1.5e-6 / 2.9e-7 / 3.9e-7 / 2.9e-7 |
   | 1-ply simd | 1,400 / 1,400 | 100% | 0 | 1.2e-6 / 7.7e-7 / 1.3e-6 / 7.7e-7 |

   0-ply: the full pool (10,000 checker + 6,000 cube) + 1,000 + 1,000
   jacoby-off + 1,000 + 1,000 bearoff-class; 1-ply: 1,000 + 1,000 pool +
   200 + 200 jacoby-off + 200 + 200 bearoff-class, every candidate
   scored at 1-ply. Mean abs diff ~5e-8 (scores) to ~3e-7 (v_dt).
   `bgc_value_prob5` max 1.2e-6. Bearoff-table values exact.
4. Speed, one pinned core (Zen 3 7713P, threads 1), per decision, mean
   (median) incl. the ctypes call, seeded pool draw (300 checker + 300
   cube; Python 1-ply checker on 100):

   | decision | Python agent | C simd | C torch | Python / C simd |
   |---|---|---|---|---|
   | 0-ply checker | 4.2 ms (0.75) | 0.12 ms (0.09) | 0.36 ms (0.29) | 34x |
   | 0-ply cube | 0.31 ms | 0.025 ms | 0.075 ms | 13x |
   | 1-ply checker | 247 ms (100) | 39 ms (10) | 108 ms (30) | 6.3x |
   | 1-ply cube | 14.6 ms | 2.4 ms | 6.8 ms | 6.2x |

   1-ply expands and forwards each candidate separately (no batching
   across candidates).
5. Paired-dice H-H `janowski-c:` (simd) vs `janowski:`, same net, Jacoby:
   0-ply 20,000 games: 10,000 / 10,000 pairs mirrored, +0.0 mEq [+0.0,
   +0.0]; 1-ply 2,000 games: 1,000 / 1,000 mirrored, +0.0 mEq.
