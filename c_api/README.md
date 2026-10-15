# PureTD C API (libpuretd)

C library for the PureTD agents: 0- and 1-ply checker play, money cube
decisions, 1-ply values, seeded self-play and the batch-training sample
collector, for every `best_models/*.pt` net type, with no Python in the
loop. `puretd_py.py` is the ctypes binding. 2-ply, match play, the
optimizer step and GPU are out of scope.

## Build

    bash c_api/build.sh               # every backend whose dependencies are found
    bash c_api/build.sh ref simd      # named backends only; exits 1 if one is skipped
    OUT=/some/dir bash c_api/build.sh # outputs elsewhere (PURETD_LIBDIR=/some/dir to load them)

`torch` links the libtorch of `PURETD_TORCH_PYTHON`, else `VENV`, else the
repo's `.venv`, else `python3`; it is skipped when that interpreter has no
torch.

Per backend `<b>`: `libpuretd_<b>.so` (the API), `libbgt_<b>.so` (bare
forward, for `bench_forward.py`) and `puretd_selfplay_<b>` (the example),
each linked to a temp name and renamed into place.
`play_models.py`, `experiments/head_to_head/run.py` and the trainers build
the library they need on demand (`puretd_py.ensure_lib`, one NOTE).

ABI: `BGC_ABI_VERSION` (`puretd.h`) equals `puretd_py.ABI_VERSION`; bump
both on any public struct or signature change. `ensure_lib` rebuilds a
library whose `puretd-abi:<v>` tag differs (`--c-api` refuses it instead,
naming the rebuild); `load_lib` refuses one whose `bgc_abi_version()` or
`bgc_struct_sizes()` differ from the ctypes declarations.
`PORTABLE=1` drops `-march=native`. Never add `-ffast-math`.
`CFLAGS_EXTRA=-DBGC_BF16` (simd, CPUs with `avx512_bf16`; off by default)
runs the 1-ply target rows through a bf16 forward (~1e-3 target error, a
systematic shift of a few 1e-4: outside the parity bounds, for A/B runs
only; `BGC_BF16=0` at open disables it in such a build). `-DBGC_PROFILE`
compiles in the profile counters (`experiments/c_collect/p5_profile.py`).

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
| `bgc_encoder_lookup(name, &info)` | the encoder table's entry for a name: base features, cube columns, `is_cube_action` column, width, C collector |
| `bgc_value(e, s, cube, ica)` | 0-ply net value, `s->turn`'s view (prob5: clamped equity) |
| `bgc_value_prob5(e, s, probs, eq)` | prob5: the 5 clamped probabilities and the equity |
| `bgc_value_1ply(e, s, cube)` | 1-ply checker value for the player on roll |
| `bgc_best_move(e, s, d1, d2, cube, plies, ...)` | checker play: index in `get_legal_plays` order, -1 forced pass |
| `bgc_score_moves(...)` | every candidate's selector score (mover's view) |
| `bgc_value_1ply_play(e, s, d1, d2, cube, &v1, ...)` | `bgc_value_1ply` + `bgc_best_move` at 0-ply from one expansion (the roll's candidates are expansion replies; bitwise the two calls' results on `ref`/`simd`, the two calls on the others) |
| `bgc_set_search(e, opts)` | 1-ply root move filter: expand only candidates within `root_thresh` of the best 0-ply score (game-ending kept; others score NAN); cube decisions stay full 1-ply; NULL or 0 = full width (default) |
| `bgc_offer_double` / `bgc_respond_to_double` | money cube at 0/1-ply (cubeful nets, Prob5Jan); `bgc_cube_eval.take` = the responder's decision |
| `bgc_open_janowski`, `bgc_set_jan`, `bgc_bearoff_open` / `_close`, `bgc_set_bearoff` | Prob5Jan (below) |
| `bgc_jan_*`, `bgc_bo_*` | the Janowski conversion and the bearoff tables alone (parity tests) |
| `bgc_rng_*` | `random.Random(seed)`-compatible MT19937 |
| `bgc_play_game(white, black, opts, rng, res, rec, ctx)` | one cubeful or cubeless game, optional per-decision callback |
| `bgc_collect(e, opts, seed, n, enc, tgt, cap, &n_discarded)` | `trainer.collect_episode` samples of `n` self-play episodes (dmp, cubeless money, cubeful money; 0-ply or 1-ply targets, 0-ply play, argmin), one `random.Random(seed)` stream; -2 when `cap` rows do not fit (replay the seed); -3 (`BGC_COLLECT_UNSUPPORTED`) for an encoder without a C collector |
| `bgc_bench_checker` / `bgc_bench_cube` / `bgc_bench_selfplay` | benchmark loops timed inside C |

| kind | net | NULL cube |
|---|---|---|
| `BGC_KIND_PROB` | cubeless encoder, sigmoid P(win) (DMP) | yes |
| `BGC_KIND_EQUITY` | cubeless encoder, cubeless money equity | yes |
| `BGC_KIND_CUBEFUL` | cubeful encoder, per-unit cubeful equity | checker play as `TDAgent.choose_checker_action` (centered, 0-ply); values NAN |
| `BGC_KIND_PROB5` | cubeless encoder, 5 sigmoids | yes |
| `BGC_KIND_PROB5JAN` (6) | a perspective196 prob5 net opened by `bgc_open_janowski` | no: the error value |

## Prob5Jan (money Janowski)

`janowski_agent.JanowskiAgent` at its defaults (x 0.76, every cube fold
on, `board_sanity` on, exact bearoff when a table is attached) in C,
`c_api/puretd_janowski.c`: the prob5 net's probabilities (gnubg's
impossible-gammon zeroing, the clamps) through `janowski.py`'s money
conversion, with `bearoff.py`'s exact tiers in bearoff-class positions.

    bgc_engine *e = bgc_open_janowski("prob5.bin", err, sizeof err);   /* BGC_KIND_PROB5JAN */
    bgc_bearoff *t = bgc_bearoff_open(path, err, sizeof err);         /* one per process */
    bgc_set_bearoff(e, t);                  /* NULL: no exact bearoff */
    bgc_jan_opts o = {0.76, 1};             /* x, board_sanity */
    bgc_set_jan(e, &o);

Then the money entry points (`bgc_best_move`, `bgc_score_moves`,
`bgc_value`, `bgc_value_1ply`, `bgc_offer_double`, `bgc_respond_to_double`,
`bgc_play_game` with cubeful opts, the bench calls); a cube is required.
No root filter (`bgc_set_search` accepts 0 only), no collector.

The bearoff file is `bearoff.export_c_tables()`'s flat copy of the two
npz tables (~155 MB, mmapped read-only, shareable by engines and
threads): `bearoff.c_table_path()` returns it, exporting it when it is
missing or older than either npz (`BG_BEAROFF_C_TABLE`, default
`~/.cache/bg_bearoff_c.bin`).

Python: `Engine(prob5_pt, janowski=True, x=None, board_sanity=True,
bearoff=True)`, `CAgent(prob5_pt, plies, janowski=True)`; `JanowskiC`
mirrors `janowski.py` and `bearoff.py`. `play_models.py`: the side
`janowski-c:<prob5.pt>`, or `janowski:<prob5.pt>` unless `--python-api`
(`JANOWSKI_X`, `JANOWSKI_BOARD_SANITY`, `JANOWSKI_NO_BEAROFF` honoured;
the `JANOWSKI_NO_FOLD*` / `JANOWSKI_CUBELESS_CHECKER` ablations and
`--bf16` play `janowski:` on the Python path).

Parity with `JanowskiAgent` (`tests/test_janowski_c.py`; full numbers in
`experiments/c_collect/SPEC_p5jan_money_c.md`):

| check | result |
|---|---|
| conversion (`bgc_jan_*`), random and edge vectors x persp x jacoby x folded | bitwise |
| bearoff tables (`bgc_bo_*`): race equity, cube values, two-sided P | bitwise |
| `exact_probs5` (Python's `np.dot` is BLAS order) | <= 2.2e-16 |
| decisions (checker, offer, take, respond) on the cubeful pool and bearoff positions: 0-ply 12k + 8k, 1-ply 1.4k + 1.4k, torch and simd | 100% equal, no near-ties |
| values and scores (fp32 forward differences amplified by the live-cube slope) | max 4.2e-6 (0-ply), 1.5e-6 (1-ply); bearoff-table values exact |
| paired-dice H-H vs the Python agent: 20k games 0-ply, 2k games 1-ply | every pair mirrored (0.0 mEq) |

Per decision on one Zen 3 core (simd): 0-ply checker 0.12 ms (Python
4.2 ms), 0-ply cube 0.025 ms (0.31), 1-ply checker 39 ms (247), 1-ply
cube 2.4 ms (14.6).

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

Adding a base encoder is one `BGC_BASES` row (name, feature count, a C
encode function bit-exact with the Python encoder, `collect` 0 until the
collector is parity-tested on it) and that function; its `cubeful_` names
follow (the comment above the table).

Format-1 exports (magic `BGNN`, no encoder name) open by input width
(196, 200); `bgc_format_version` (`Engine.format_version`) is 1 for
them. `bench_forward.py` keys its export cache by the format version, so
it re-exports rather than reuse a format-1 cache entry.

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
(prob5) and 4.7 of 20.6 (cubeful) candidates
([../experiments/benchmarks/rootf_validate.py](../experiments/benchmarks/rootf_validate.py)).

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

## Python

    from backgammon_engine import BoardState
    from modes import MatchState
    from c_api.puretd_py import Engine, CAgent, available_backends
    state, match_state = BoardState.initial(), MatchState()
    eng = Engine("best_models/cubeful_money_512_512_256_256.pt", backend="simd")
    idx, next_state, value = eng.best_move(state, (6, 5), match_state, plies=1)
    agent = CAgent("best_models/cubeless_prob5_512_512_256_256.pt", plies=1)
    enc, tgt, n_discarded = eng.collect({"oneply": True, "jacoby": True}, seed=7, n_episodes=80)

Backend: argument, else `PURETD_BACKEND`, else simd (if not built, the
first built of torch, cblas, ref); the play paths use `play_backend()`
(argument, else `PURETD_BACKEND`, else simd). `CAgent` is an `agents.Agent` for the
Python game loops. Open one `Engine` per thread or process.

Play path (`play_models.py`, `experiments/head_to_head/run.py`): checkpoint
and `janowski:` sides are `CAgent`s on simd by default, and model-vs-model
games without a record run in `bgc_play_game`. A run the C API cannot
serve (2-ply sides, `--bf16` prob5 sides, `PROB5_BOARD_SANITY=1`, the
Janowski ablations, an encoder it lacks, a library that does not build)
plays on the Python path after one NOTE naming the reason. `--c-api`
makes that an error; `--python-api` selects the Python path;
`--c-api-backend b` picks the backend. The npz records the path
(`c_api`, `c_api_backend`); runs on different paths refuse to pool.
Root filter: `Engine(..., root_filter=0.08)` or `set_root_filter`;
`CAgent` and `play_models.py --c-api-root-filter T` use
`PURETD_ROOT_FILTER` (> 0 requires the C API), and a saved H-H run records
it in the side depth (`1ply root-filter 0.08`), so pruned and full-width
runs never pool.

## Example

    python export_weights.py best_models/cubeful_money_512_512_256_256.pt /path/cubeful.bin
    c_api/puretd_selfplay_simd -n 100 -s 1 -p 0 /path/cubeful.bin
    python export_weights.py best_models/cubeless_prob5_512_512_256_256.pt /path/prob5.bin
    python export_weights.py best_models/cubeless_money_512_512_256_256.pt /path/cubeless.bin
    c_api/puretd_selfplay_simd -n 100 --cubeless /path/prob5.bin /path/cubeless.bin

Prints the result tally, points per game, decisions/s and games/s
(`example_selfplay.c`).
