# Third-party opponents

Adapters that put other backgammon engines behind the `Agent` interface, for
head-to-head play (`play_models.py`, `experiments/head_to_head/run.py`) and
the fixed-pool benchmarks (`experiments/benchmarks/`). No third-party source
code or weights are included: each engine is installed or built from its own
pinned release. An adapter is imported only when a side spec names its
engine. The adapters are MIT (root `LICENSE`) except the two files marked
below.

## gnubg

[GNU Backgammon](https://www.gnu.org/software/gnubg/), GPL-3.0-or-later.
Pinned: gnubg 1.08.003.

- `gnubg_external.py`: the gnubg binary over its external-player socket
  (gnubg's own checker play and money cube, Jacoby).
- `gnubg_nn_agent.py`: `gnubg-nn` (`pip install gnubg-nn`), a cube-blind
  approximation for sanity checks only.
- `gnubg_eval.py`: game recording, `.mat` export, binary preflight
  (version, weights md5, bearoff database).
- `fixedpool_bench_gnubg.py`: fixed-pool benchmark, run inside the gnubg
  binary.
- `gnubg_bench/benchpool.patch` (GPL-3.0-or-later, see its header) and
  `build.sh`: a threads-disabled gnubg with the `gnubg.benchpool` timing loop.

Install: `apt install gnubg` on a release that ships 1.08.003, or a build from
GNU source; `GNUBG_CMD` names the binary when it is not `/usr/games/gnubg`.
Benchmark build: `bash third_party_opponents/gnubg/gnubg_bench/build.sh
$HOME/gnubg-bench` (needs flex, bison, glib, python3 headers).

Side specs: `gnubg-ext` (production pruning nets and move filter),
`gnubg-ext-full` (no pruning, every legal move evaluated), `gnubg-ext-rootfull`
(keep-all root filter, default pruning below), `gnubg-nn` (or `--gnubg-nn`);
`--plies` sets gnubg's depth.

## HedgeHog

[HedgeHog](https://hedgehog-bg.com)
([hedgehog-public](https://gitlab.com/eranlambooij/hedgehog-public)). Its
`LICENSE` at `e8847f4`: MIT, except the position-ID and match-record format
specifications `docs/OGID.md`, `docs/OGXM_FORMAT_SPEC.md`,
`docs/OGXM_JSON_SPEC.md`, `docs/OGXM_V1_FORMAT_SPEC.md` and
`docs/OGXM_V1_JSON_SPEC.md`, which are CC BY 4.0. The vendored Highway
(`hwy/`) carries its own licence (Apache-2.0 or BSD-3-Clause). Pinned: engine
`e8847f4`; models fox v0.34 and aureus v0.12 (md5s in
[hedgehog/README.md](hedgehog/README.md)).
HedgeHog's models are available for non-commercial use per
[hedgehog-bg.com/community](https://hedgehog-bg.com/community) and are not
covered by this repository's MIT license.

- `hedgehog/hedgehog_agent.py`, `hedgehog/hh_lib.py`: the agent and its
  ctypes wrapper.
- `hedgehog/shim/`: `hh_shim.cpp`, a flat C API over the engine, and
  `build_shim.sh`.

Build: [hedgehog/README.md](hedgehog/README.md#build); the shim lands in
`third_party_opponents/hedgehog/libhhagent.so` (`$HH_LIB` overrides), models
go next to it.

Side spec: `hedgehog:<model.ogxf>`, money modes only; `--oneply1` /
`--oneply2` for HedgeHog depth 1. Rows and commands:
[experiments/head_to_head/HEDGEHOG.md](../experiments/head_to_head/HEDGEHOG.md).

## Open Sage

[Open Sage](https://github.com/markbgsage/bgsage) (bgsage), MPL-2.0 (tag `v2.0.20260907`).
Pinned: bgsage 2.0.20260907 (tag `v2.0.20260907`, model `stage11`); 1.x gives
different rows.

- `opensage/opensage_agents.py`: agents over the `bgsage` Python package.
- `opensage/opensage_mobile.py`, `opensage_mobile_build.sh`: agent over Open
  Sage's native C API (`libbgsage_mobile.so`, `$BGSAGE_MOBILE_LIB` overrides).
- `opensage/opensage_bench/`: `sage_bench.cpp` (links Open Sage), its CMake file and `build.sh`, the fixed-pool benchmark loops.

Install: `pip install bgsage==2.0.20260907`. Native C API:
`bash third_party_opponents/opensage/opensage_mobile_build.sh`. Benchmark
loops: `bash third_party_opponents/opensage/opensage_bench/build.sh` (needs
cmake). Both clone the tagged source into their build directory.

Side specs: `opensage` (cubeless money or DMP), `opensage-analyzer` (cubeful
money, Open Sage's self-play config), `opensage-native` (cubeful money, 0/1-ply,
native C API); `--sage-plies` is XG-numbered (1 = 0-ply, 2 = 1-ply, 3 = 2-ply).
Rows and commands: [experiments/head_to_head/OPENSAGE.md](../experiments/head_to_head/OPENSAGE.md).
