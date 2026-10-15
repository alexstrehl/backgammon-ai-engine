# PureTD C API (libpuretd)

C library for the PureTD agents: 0- and 1-ply checker play, money cube
decisions, 1-ply values, seeded self-play and the batch-training sample
collector, for the money net types, with no Python in the loop.
`puretd_py.py` is the ctypes binding. 2-ply, the optimizer step and GPU
are out of scope. Match play: the `matchplay1h64met_` / `matchplay1h64met6_`
nets open as `BGC_KIND_MATCH` (`puretd_match.c`, "Match play: the C core"
below); a prob5 net opens
as the p5jan match agent, `BGC_KIND_MATCH_JANOWSKI` ("p5jan / Janowski match
play" below); the other matchplay nets use the backends through
`fast_forward.py` (below).

## Build

    bash c_api/build.sh               # every backend whose dependencies are found
    bash c_api/build.sh ref simd      # named backends only
    OUT=/some/dir bash c_api/build.sh # outputs elsewhere (PURETD_LIBDIR=/some/dir to load them)

`torch` links the libtorch of `PURETD_TORCH_PYTHON`, else `VENV`, else the
repo's `.venv`, else `python3`; it is skipped when that interpreter has no
torch.

Per backend `<b>`: `libpuretd_<b>.so` (the API), `libbgt_<b>.so` (bare
forward, for `bench_forward.py`) and `puretd_selfplay_<b>` (the example),
each linked to a temp name and renamed into place (objects in a
per-process `obj_<b>.<pid>` directory, removed at exit). The trainers' C
collector builds the library it needs on demand (`puretd_py.ready_lib`,
one NOTE), as do the p5jan fast decisions (`ensure_lib`).

ABI: `BGC_ABI_VERSION` (`puretd.h`) equals `puretd_py.ABI_VERSION`; bump
both on any public struct or signature change. `ensure_lib` rebuilds a
library whose `puretd-abi:<v>` tag differs (`rebuild_stale=False` refuses
it instead, naming the rebuild); `load_lib` refuses one whose
`bgc_abi_version()` or `bgc_struct_sizes()` (every struct the ctypes
binding declares, the match-play ones included, in `puretd.h`'s order)
differ from the ctypes declarations, naming the rebuild
(`tests/test_c_api_abi.py`).

`PORTABLE=1` drops `-march=native`. Never add `-ffast-math`.
`CFLAGS_EXTRA=-DBGC_BF16` (simd, CPUs with `avx512_bf16`; off by default)
runs the 1-ply target rows through a bf16 forward (~1e-3 target error, a
systematic shift of a few 1e-4: outside the parity bounds, for A/B runs
only; `BGC_BF16=0` at open disables it in such a build). `-DBGC_PROFILE`
compiles in the profile counters (`bgc_prof_read` / `bgc_prof_reset`).

Nets load from the BGNN export: `python export_weights.py model.pt model.bin`
(the Python binding exports a `.pt` itself). The export names the
checkpoint's encoder (format 2, magic `BGNV`; see "Encoders" below). A
library without format-2 support fails `bgc_open` on such a file (bad
magic); `puretd_py.load_lib` refuses that library and names the rebuild
command.

## Backends

The forward pass is the only per-backend code (`puretd_backend.h`); move
generation, selection, cube logic and the game loop are shared (`puretd.c`).

| backend | forward | dependencies |
|---|---|---|
| `torch` | ATen `addmm`, one thread (default) | the venv's torch wheel (`libtorch_cpu`) |
| `simd` | AVX2/FMA and AVX-512 kernels, runtime dispatch (AVX-512 where the CPU has it, e.g. Zen 5; AVX2 on Zen 3), scalar fallback; skips exact zero inputs and activations (the result is unchanged); successor batches (0-ply candidates, 1-ply expansions) get layer 0 as the parent position's pre-activation plus the few changed inputs (`bgc_backend_forward_delta`, NNUE-style; within ~1e-6 of the plain forward; `BGC_SIMD_DELTA0=0` turns it off) | none |
| `cblas` | `cblas_sgemm`/`sgemv`, one thread | OpenBLAS (`OPENBLAS_DIR`, pkg-config or system) |
| `ref` | scalar reference | none |

## API (`puretd.h`)

| function | does |
|---|---|
| `bgc_open(path)` / `bgc_open_ex(path, err, len)` / `bgc_close` | load a `.bin`; the encoder from its header (`bgc_encoder_name`), the kind from the encoder and the output mode (`bgc_kind`); `_ex` returns the failure's message |
| `bgc_abi_version()` / `bgc_struct_sizes(out, n)` | `BGC_ABI_VERSION`; sizeof the public structs (order in `puretd.h`) |
| `bgc_open_janowski(path, err, len)` | a prob5 `.bin` as the p5jan match agent (`BGC_KIND_MATCH_JANOWSKI`), below |
| `bgc_encoder_lookup(name, &info)` | the encoder table's entry for a name: base features, cube columns, width, C collector; match block and MET width |
| `bgc_value(e, s, cube, ica)` | 0-ply net value, `s->turn`'s view (prob5: clamped equity) |
| `bgc_value_prob5(e, s, probs, eq)` | prob5: the 5 clamped probabilities and the equity |
| `bgc_value_1ply(e, s, cube)` | 1-ply checker value for the player on roll |
| `bgc_best_move(e, s, d1, d2, cube, plies, ...)` | checker play: index in `get_legal_plays` order, -1 forced pass |
| `bgc_score_moves(...)` | every candidate's selector score (mover's view) |
| `bgc_value_1ply_play(e, s, d1, d2, cube, &v1, ...)` | `bgc_value_1ply` + `bgc_best_move` at 0-ply from one expansion (the roll's candidates are expansion replies; bitwise the two calls' results on `ref`/`simd`, the two calls on the others) |
| `bgc_set_search(e, opts)` | 1-ply root move filter: expand only candidates within `root_thresh` of the best 0-ply score (game-ending kept; others score NAN); cube decisions stay full 1-ply; NULL or 0 = full width (default) |
| `bgc_offer_double` / `bgc_respond_to_double` | money cube at 0/1-ply (cubeful nets); `bgc_cube_eval.take` = the responder's decision |
| `bgc_rng_*` | `random.Random(seed)`-compatible MT19937 (`bgc_rng_random` = `random()`) |
| `bgc_set_met` / `bgc_match_*` | match play (`BGC_KIND_MATCH`, `BGC_KIND_MATCH_JANOWSKI`), below |
| `bgc_set_jan_x` / `bgc_jan_*` | p5jan: the x policy; the Janowski conversion alone (no net), below |
| `bgc_play_game(white, black, opts, rng, res, rec, ctx)` | one cubeful or cubeless game, optional per-decision callback |
| `bgc_collect_match(e, opts, seed, n, enc, tgt, cap, &n_discarded)` | `trainer._collect_episode_matchplay` samples of `n` matches (MATCH engines; below) |
| `bgc_collect(e, opts, seed, n, enc, tgt, cap, &n_discarded)` | `trainer.collect_episode` samples of `n` self-play episodes (dmp, cubeless money, cubeful money; 0-ply or 1-ply targets, 0-ply play, argmin), one `random.Random(seed)` stream; -2 when `cap` rows do not fit (replay the seed); -3 (`BGC_COLLECT_UNSUPPORTED`) for an encoder without a C collector |
| `bgc_bench_checker` / `bgc_bench_cube` / `bgc_bench_selfplay` | benchmark loops timed inside C |

| kind | net | NULL cube |
|---|---|---|
| `BGC_KIND_PROB` | cubeless encoder, sigmoid P(win) (DMP) | yes |
| `BGC_KIND_EQUITY` | cubeless encoder, cubeless money equity | yes |
| `BGC_KIND_CUBEFUL` | cubeful encoder, per-unit cubeful equity | checker play as `TDAgent.choose_checker_action` (centered, 0-ply); values NAN |
| `BGC_KIND_PROB5` | cubeless encoder, 5 sigmoids | yes |
| `BGC_KIND_MATCH` | match encoder, sigmoid P(win the match) | no cube (a `bgc_match`); the money entry points return their error value |
| `BGC_KIND_MATCH_JANOWSKI` | cubeless encoder, 5 sigmoids, opened by `bgc_open_janowski` | as `BGC_KIND_MATCH` |

## Encoders

`bgc_open` takes the encoder from the export's header (the checkpoint's
`encoder_name`) and resolves it through one table (`puretd.c`
`BGC_BASES` x `BGC_CUBE_BLOCKS`), as `encoding.get_encoder` does: a base
encoder's name, or `cubeful_<base>`. The net's
input size must equal the encoder's width; an unknown encoder or base
encoder, a width mismatch, or an output mode the encoder has no kind for
fails `bgc_open` with a message naming the encoder (`puretd_py.Engine`
raises it).

| encoder | width | row | `bgc_collect` |
|---|---|---|---|
| `perspective196` | 196 | the base | yes |
| `cubeful_perspective196` | 200 | `[196 base \| centered \| mine \| theirs \| is_cube_action]` | yes |
| `matchplay1h64met_perspective196` | 350 | `[196 base \| 154 match columns]` (`Matchplay1H64METEncoder`) | `bgc_collect_match` |
| `matchplay1h64met6_perspective196` | 354 | `[196 base \| 158 match columns]` (`Matchplay1H64MET6Encoder`) | `bgc_collect_match` |

Adding a base encoder is one `BGC_BASES` row (name, feature count, a C
encode function bit-exact with the Python encoder, `collect` 0 until the
collector is parity-tested on it) and that function; its `cubeful_` names
follow (the comment above the table).

Format-1 exports (magic `BGNN`, no encoder name) open by input width
(196, 200); `bgc_format_version` (`Engine.format_version`) is 1 for
them. `bench_forward.py` keys its export cache by the format version, so
it re-exports rather than reuse a format-1 cache entry.

A match encoder's name is a match-block prefix (`BGC_MATCH_BLOCKS`,
read before the cube blocks) + a base encoder; the other matchplay
encoders (`matchplay_`, `matchplay1h_`, `matchplay1h64_`, met4, met5,
`matchplaymet_`) are not in the table: `bgc_open` refuses them with
`unknown encoder '<name>'` and they run through the fast forward below.

## Root filter

`bgc_set_search(e, &(bgc_search){.root_thresh = T})` sets a 1-ply root
move filter on engine `e`: 1-ply checker decisions (`bgc_best_move`,
`bgc_score_moves`, `bgc_play_game`) score every candidate at 0-ply and
expand only those within `T` of the best 0-ply score; game-ending moves
are always expanded, the others score NAN, and a single survivor keeps its
0-ply score. Cube decisions stay full 1-ply. `T = 0` or NULL restores full
width (the default). The benchmark's 1-ply rows use 0.08: on every item of
both committed pools it picks the full-width move on 100% of prob5 and
99.91% of cubeful decisions (0.010 mEq/decision), expanding 5.7 of 20.5
(prob5) and 4.7 of 20.6 (cubeful) candidates.

## Conventions

- `s->turn` is the player to act; successors have the turn switched, so a
  successor's net value is the opponent's view. `out_value` and scores are
  the mover's view.
- Cube: `owner` 0 centered, 1 WHITE, 2 BLACK; `value`; `jacoby`. Gammons
  count unless Jacoby with a centered cube.
- `bgc_respond_to_double`: `s->turn` is the doubler; the result is the
  responder's take (1) or pass (0). The hint is used when its `plies`
  match.
- Ties: first candidate in `get_legal_plays` order.
- Terminal successors score their result (1/2/3; 1 under Jacoby with a
  centered cube; 1.0 for P(win) nets), never the net output.
- `bgc_play_game` with `white == black` passes the offer to the response as
  a hint (Python self-play does the same).
- Counting: `bgc_game_result.n_checker` includes forced passes in a
  cubeful game, not in a cubeless one; `n_dance` counts them in both. The
  `bgc_bench_selfplay` decision counts exclude forced passes in both modes
  (gnubg's count).
- Errors: `bgc_best_move` -2, cube functions -1, game/bench functions -1.

## Thread model

An engine (`bgc_engine`) holds its backend and scratch buffers and is used
by one thread at a time. Engines share nothing (no globals), so one engine
per thread runs in parallel. `get_legal_plays` keeps ~0.5 MB on the stack:
threads need stacks of at least 1 MB. Backends never start threads.

## Parity

Decisions are the Python agents' (`td_agent.TDAgent`,
`prob_agent.ProbAgent`, `gnubg_eval.play_and_record_cubeful`), checked by
`tests/test_c_api.py` against torch on the committed benchmark pools:

| check | bound |
|---|---|
| forward, 0-ply values and scores | 1e-5 (relative 1e-4 above 1) |
| 1-ply values | 1e-4 |
| decisions | equal, or C scores of both choices within the bound (near-tie) |
| self-play | every decision reproduced up to a near-tie; dice equal `random.Random(seed)` |
| collector (`tests/test_c_collect.py`) | per seed, rows equal `trainer.collect_episode`'s up to the first near-tie decision: encodings bit-exact, targets 1e-5 (1-ply 1e-4) |

Sums over the 21 rolls are in double in Python's order; prob5 equity uses
torch's fp32 summation order.

`pytest tests/test_c_api.py tests/test_c_collect.py` checks the simd backend
at reduced sizes; `--slow` adds every built backend and the 1-ply prod-net
games; `CAPI_FULL=1 CCOLLECT_FULL=1 pytest --slow ...` runs the full pools.

The tests run against this tree's nets and Python agents: the prob5
parity cases skip (no `prob_agent.py` here), and `tests/test_c_collect.py`
also checks `bgc_collect` bitwise against the same calls on a recorded
reference build (`tests/m0_collect_hashes.py`; the record is
`tests/data/m0_collect_hashes_frozen.json`).

## Python

    from c_api.puretd_py import Engine, CAgent, available_backends
    eng = Engine("best_models/cubeful_money_512_512_256_256.pt", backend="simd")
    idx, next_state, value = eng.best_move(state, (6, 5), match_state, plies=1)
    agent = CAgent("best_models/cubeless_prob5_512_512_256_128.pt", plies=1)
    enc, tgt, n_discarded = eng.collect({"oneply": True, "jacoby": True}, seed=7, n_episodes=80)

Backend: argument, else `PURETD_BACKEND`, else the first built of torch,
simd, cblas, ref; `play_backend()` is the play paths' choice (argument,
else `PURETD_BACKEND`, else simd). `CAgent` is an `agents.Agent` for the
Python game loops. Open one `Engine` per thread or process. Root filter:
`Engine(..., root_filter=0.08)` or `set_root_filter`; `CAgent` uses
`PURETD_ROOT_FILTER` when not given one.

## Example

    python export_weights.py best_models/cubeful_money_512_512_256_256.pt /path/cubeful.bin
    c_api/puretd_selfplay_simd -n 100 -s 1 -p 0 /path/cubeful.bin
    c_api/puretd_selfplay_simd -n 100 --cubeless /path/prob5.bin /path/cubeless.bin

Prints the result tally, points per game, decisions/s and games/s
(`example_selfplay.c`).

## Match play: the C core (`puretd_match.c`, `BGC_KIND_MATCH`)

Ports of the `TDAgent` matchplay methods with a MET grounding every
`_eval_new_game` scope, `boltzmann_temp` 0, `cube_fold` off, and of
`modes.MatchState`. A `bgc_match` is a
`MatchState` in WHITE's view (`away[0]` = WHITE's away score); values are
P(the named player wins the MATCH).

| function | Python |
|---|---|
| `bgc_set_met(e, pre, post, n)` | `MatchEquityTable._pre` (n x n) / `_post` (n), copied; required first |
| `bgc_match_encode(e, s, m, ica, row)` | `_encode_matchplay` (base + `match_extras`, bit-exact) |
| `bgc_match_new_game_mwc(e, after_game, player)` | `_eval_new_game` (start_game, then the MET) |
| `bgc_match_value(e, s, m, ica)` | `evaluate_matchplay` |
| `bgc_match_offer(e, s, m, &cube)` | `offer_double_matchplay` (v_nd, v_dt 0-ply; v_dp the MET); take = `v_dp > v_dt` |
| `bgc_match_best_move` / `bgc_match_score_moves` | `choose_checker_action_matchplay` at temp 0 (first argmin of the opp-view values, terminal successors priced through the MET) |
| `bgc_match_value_1ply(e, s, m)` | `value_oneply_checker_matchplay` (double; ungrouped dice sum, clamp, all-pass rule) |
| `bgc_match_value_1ply_play` | the two above from one expansion |

The batch machinery is the money kinds' (`puretd_internal.h`): expand21
with the cross-roll row dedup, the delta forward from the parent
position, the 0-ply play taken from the 1-ply expansion. A row's columns
after the base are the context's match block, the same for every row of
one expansion or candidate set. Parity (`tests/test_c_match.py`,
`matchplay_parity.py check-capi`): bitwise the `BG_FAST_FORWARD=<b>`
agent when C runs on `<b>` with the plain forward (simd:
`BGC_SIMD_DELTA0=0`; ref); within 1e-5 / 1e-4 of the torch agent.
Python: `Engine.set_met(met)`, `Engine.match_*`, `match_from_ms(ms)`.

The 1-ply memo: `bgc_match_value_1ply` / `_1ply_play` reuse the value an
engine computed earlier for the same (board, match state) -- a pure
function of the two for fixed weights and MET, so bitwise the value they
would compute (on a hit the 0-ply play is a separate select). Every match
of a round starts at the same score, so early positions recur: ~2% of an
11pt worker's checker turns. On by default; `BGC_MATCH_MEMO=0` at open
turns it off; `bgc_match_memo_stats` / `Engine.match_memo_stats()` count;
`bgc_set_met` clears it; at most 2^17 entries.

### The match-episode collector (`bgc_collect_match`, `puretd_match_collect.c`)

`bgc_collect_match(e, &opts, seed, n, enc, tgt, cap, &n_discarded)`: the
samples of `trainer._collect_episode_matchplay` (TDAgent at temp 0, 0-ply
play, the engine's MET on every scope) for `n` matches, rows and float32
targets in the Python order. `opts`
(`bgc_match_collect_opts`, Python `match_collect_opts(...)`):

| field | Python |
|---|---|
| `match_length`, `oneply` | `MatchplayMode.match_length`; `--oneply` (1-ply targets, play stays 0-ply) |
| `curriculum_p`, `force_my` / `force_opp` | `trainer._SCORE_CURRICULUM_P`; `TRAINER_CURRICULUM_FORCE` (0 = unset) |
| `cube_explore_eps`, `boundary_upweight` | `trainer._CUBE_EXPLORATION_EPS`, `_BOUNDARY_UPWEIGHT` |
| `outcome_mix` | `MatchplayMode.outcome_mix` |
| `max_plies` | `<= 0` unbounded (the serial collector); `> 0` the vectorized per-game bound: the match is discarded, the stream goes on |
| `rng_mode` | 0: one `random.Random(seed)` stream (`_collect_worker`); 1: per match `Random(base.randint(0, 2**31-1))` (the vectorized worker) |

The RNG draws are Python's, coins drawn at probability 0 included.
Returns the rows written, -1 on an error or bad option, -2 when `cap` is
too small (replay the seed), `BGC_COLLECT_UNSUPPORTED` for a collect-0
entry (`BGC_COLLECT_OFF=1` at open forces that, for tests). Python:
`Engine.collect_match(opts, seed, n, max_rows=None, out=None)` ->
`(enc, tgt, n_discarded)` (1024 rows per match to start, grow and replay
on -2). Parity: `tests/test_c_collect_match.py` (bitwise the serial /
bounded / vectorized-B=1 Python collector on `BG_FAST_FORWARD=<b>` with
the plain forward; the episode rule against torch).

## p5jan / Janowski match play (`puretd_janowski.c`, `BGC_KIND_MATCH_JANOWSKI`)

p5jan, the reference opponent of every matchplay promotion read
(`eval_matchplay.py --model2 p5jan:PATH`), is a cubeless money prob5 net
under gnubg's cubeless->cubeful match-winning-chance conversion
(`janowski_match.py`). Its
whole decision runs in C: `bgc_open_janowski` opens the prob5 export as a
match engine and the `bgc_match_*` entry points above serve it, through an
evaluator seam in `puretd_match.c` (the decision structure is the TD
match agent's: move generation, batches, delta forward, the one-expansion
play and the memos are shared).

| call | `janowski_match` |
|---|---|
| `bgc_open_janowski(path, err, len)`, `bgc_set_met` | `JanowskiMatchAgent(net, met)` |
| `bgc_set_jan_x(e, constant, x)` | `x_policy`: constant 0 = `janowski.cube_x` of each evaluated board (the default), 1 = `x` |
| `bgc_match_offer` | `offer_double` / `cube_decision` (one forward; v_nd, v_dt at the doubled cube THEIRS, v_dp = `met.mwc_after`) |
| `bgc_match_best_move` / `_score_moves` | `choose_checker_action_cubeful` (the first maximum of the mover-view doubles; the float outputs are float32(1 - them)) |
| `bgc_match_value_1ply` / `_1ply_play` | `value_oneply` (per roll the mover's best, summed in dice order; no clamp, no all-pass rule) |
| `bgc_match_value` | `cubeful_mwc` at the position for the player on roll |
| `bgc_match_encode` | the base row |

A row's value is `cubeful_mwc(prob5_postprocess(row), persp, ctx, x)` in
double: the context's MET lookups once per candidate set or expansion
(`bgc_i_jan_prep`), the arithmetic per row in Python's operation order
without FMA (`bgc_i_jan_eval`), `x` from the row's board (recorded where
the row is encoded). `bgc_collect_match` and the money calls refuse the
kind. The conversion alone, no net: `bgc_met`, `bgc_jan_ctx` (`MatchCtx`),
`bgc_jan_cubeful_mwc / _dead_mwc / _get_points / _do_cubeful /
_cube_decision / _cube_x` (Python: `puretd_py.JanowskiC`, the same names
and signatures as `janowski_match`'s).

Python: `Engine(model, backend, janowski=True)`, `set_met`,
`set_jan_x(x_policy)`, the `match_*` methods; the agent's opt-in is
`JanowskiMatchAgent.enable_fast_decisions(backend)` (offer, respond, the
checker choice and `value_oneply` through the engine; re-opened when the
net changes in place). `eval_matchplay.py --fast-forward B` and
`p5jan:ff-B:PATH` select it; `p5jan:ffnet-B:PATH` keeps the net-only
forward below (decisions in Python, for A/B); the `--raw-out` file records
each side's path (`decision_paths`).

Parity (`tests/test_janowski_c.py`): the conversion bitwise
`janowski_match`'s on every reachable state of 3/5/7/11-point matches x
300 prob5 vectors x both x policies and on the loose 11pt grid; the engine
bitwise the agent on the same backend's net-only plain forward (ref; simd
with `BGC_SIMD_DELTA0=0`) and, on `torch`, the torch agent's 0-ply choices
and cube triples; production simd (delta forward) vs the torch agent: no
flips on 1,037 match positions, values within 4e-7, 1-ply within 1e-7.
`tests/test_janowski_match*.py` run their conversion cases on both
implementations.

## Match play: fast forward (`fast_forward.py`)

The matchplay nets (350-356 inputs, 4-5 hidden layers, one sigmoid) and
the p5jan prob5 net run their forward on a backend through the bare
library (`libbgt_<b>.so`); encoding, move generation and every decision
rule stay in Python, so any input size or depth works. (p5jan's
decisions themselves run in C with `enable_fast_decisions`, above; in
eval_matchplay that is the default for a p5jan side and this net-only
path is `p5jan:ffnet-<b>:`.)

    BG_FAST_FORWARD=simd python eval_matchplay.py ...     # or --fast-forward simd
    python eval_matchplay.py --model1 ff-simd:A.pt --model2 model:ff-off:A.pt ...

Off by default. Set, it covers the matchplay `TDAgent` (0-ply move
selection, which also takes the C move generator, cube decisions, 1-ply
values), `JanowskiMatchAgent` (p5jan; the net forward, at construction),
and the CPU collectors (serial workers inherit the variable; the
vectorized CPU driver uses it).

| backend | vs torch Python | eval_matchplay speed-up |
|---|---|---|
| `torch` | bitwise (weights kept as nn.Linear's [out, in], one addmm over all rows) | 1.7x (1 worker), 1.1x (W=96) |
| `simd` | <= 2e-5 per output, flips only near-ties; AVX-512, AVX2 and scalar give identical bits | 3.7x (1 worker), 5.2x (W=96) |

`libbgt_torch.so` must be built for the interpreter that loads it
(`VENV=$(which python) OUT=dir bash c_api/build.sh torch`,
`PURETD_LIBDIR=dir`); `FastForward` refuses a torch version mismatch.
Weights are re-copied when the source net changes in place. Parity:
`tests/test_fast_forward.py`, `matchplay_parity.py` (real positions, all
heads, decisions), `movegen_fuzz.py` (C vs Python move generation).
