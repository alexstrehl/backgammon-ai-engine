"""Self-play computational benchmark: gnubg 1.08 native, cubeful money.

Run INSIDE the gnubg binary:  gnubg -t -q -p selfplay_bench_gnubg.py
(the game loop, evaluation, filters and cube logic are gnubg's own C
code; this script only issues commands and counts).

Per-decision-type counting so game length cannot influence conclusions:
  checker decisions = non-dance 'move' actions in the match record
                      (a dance is recorded with an empty move — no legal
                      play, so no decision; matches the Python adapters)
  cube decisions    = pre-roll turns where the mover holds cube access
                      (centered or own), replayed from the record; plus
                      take/drop responses. This matches the PureTD
                      counter definition (offer_double call sites).

Configs: plies from GNUBG_BENCH_PLIES (default "0,1"; DEFAULT pruning —
prune nets + Normal move filter, i.e. gnubg as shipped), cube off
(pure checker cost) then cube on. threads 1.
"""
import os
import time

import gnubg


class _LCG:
    """Math-free deterministic PRNG for bootstrap resampling. gnubg's
    embedded Python 3.14 can't import `random` (it pulls in `math`, a C
    extension absent from this interpreter), so use a tiny inline LCG.
    Seeded like random.Random(1): reproducible across runs."""

    def __init__(self, seed=1):
        self.s = (seed * 2862933555777941757 + 3037000493) & 0xFFFFFFFFFFFFFFFF

    def _n(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 11

    def randrange(self, n):
        return self._n() % n

# Pin to one core (kernel affinity): with multi-threaded internals (e.g.
# Sage's analyzer floors at ~2 threads) this makes wall-time equal true
# single-core cost. Override core with BENCH_CORE, disable with BENCH_NO_PIN.
if not os.environ.get("BENCH_NO_PIN"):
    _env = os.environ.get("BENCH_CORE", "").strip()
    if _env:
        # An explicitly requested core must be honored — fail loudly.
        _core = int(_env)
        if _core == 0:
            raise SystemExit("BENCH_CORE=0 is the IRQ core; pick another core")
        try:
            os.sched_setaffinity(0, {_core})
        except AttributeError:
            raise SystemExit("sched_setaffinity unavailable on this platform; "
                             "set BENCH_NO_PIN=1 to run unpinned")
        except OSError as e:
            raise SystemExit("cannot pin to core %d (%s); pick a core this host has"
                             % (_core, e))
    else:
        # No core requested: prefer 8 (core 0 handles IRQs), else the highest
        # core this process may use.
        try:
            _avail = os.sched_getaffinity(0)
            _core = 8 if 8 in _avail else max(_avail)
            os.sched_setaffinity(0, {_core})
            if _core != 8:
                print("[bench] BENCH_CORE unset; pinned to core %d" % _core)
        except (AttributeError, OSError):
            print("[bench] cannot pin on this platform; running unpinned")


PLIES = [int(x) for x in os.environ.get("GNUBG_BENCH_PLIES", "0,1").split(",")]
RESET_EVERY = int(os.environ.get("GNUBG_RESET_EVERY", "250"))
WARMUP_S = float(os.environ.get("BENCH_WARMUP_S", "3.0"))
NGAMES = {0: int(os.environ.get("GNUBG_BENCH_G0", "150")),
          1: int(os.environ.get("GNUBG_BENCH_G1", "40")),
          2: int(os.environ.get("GNUBG_BENCH_G2", "20"))}

# On a threads-disabled build this prints "Unknown keyword 'threads'" —
# harmless (that build has no thread pool to configure).
gnubg.command("set threads 1")
gnubg.command("set display off")
gnubg.command("set confirm new off")
gnubg.command("set player 0 gnubg")
gnubg.command("set player 1 gnubg")
gnubg.command("set jacoby on")
gnubg.command("set automatic roll on")
gnubg.command("set automatic game off")


def count_record():
    """Per-game [(checker_moves, cube_opportunities, cube_responses)]
    from the session record, replaying cube ownership."""
    m = gnubg.match(analysis=0)
    per_game = []
    for g in m["games"]:
        nch = ncu_opp = ncu_resp = 0
        owner = -1          # -1 centered, else player id
        first = True
        for r in g["game"]:
            a = r.get("action")
            if a == "move":
                if r.get("move"):  # dance = empty move, not a decision
                    nch += 1
                p = r.get("player")
                pid = 0 if p in (0, "O", "gnubg") else 1
                if not first and (owner == -1 or owner == pid):
                    ncu_opp += 1   # mover had a pre-roll cube decision
                first = False
            elif a == "double":
                pass               # counted as opportunity on the move
            elif a in ("take", "drop"):
                ncu_resp += 1
                p = r.get("player")
                owner = 0 if p in (0, "O", "gnubg") else 1
        per_game.append((nch, ncu_opp, ncu_resp))
    return per_game


def ratio_ci(dts, counts, B=10000):
    """Game-level percentile bootstrap 95% CI of sum(counts)/sum(dts)."""
    n = len(dts)
    rng = _LCG(1)
    out = []
    for _ in range(B):
        sd = sc = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sd += dts[j]
            sc += counts[j]
        out.append(sc / sd)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


for plies in PLIES:
    for p in (0, 1):
        gnubg.command("set player %d chequer evaluation plies %d" % (p, plies))
        gnubg.command("set player %d cube evaluation plies %d" % (p, plies))
        # DEFAULT pruning: do not touch prune nets / movefilter settings.
    for cube in ("off", "on"):
        gnubg.command("set cube use %s" % cube)
        ngames = NGAMES[plies]
        gnubg.command("new session")
        # 'new session' auto-plays one game (both players computer, auto
        # roll) — untimed, so exclude it from the counts as well.
        n0 = len(count_record())
        # Ramp the pinned core before timing (cannot import bench_common:
        # gnubg's embedded Python has no torch). These are the shortest timed
        # regions in the suite (0.34-0.80 s), so the most cold-exposed.
        _t_end = time.time() + WARMUP_S
        _x = 0.0
        while time.time() < _t_end:
            for _ in range(10000):
                _x = _x * 1.000001 + 1.0
        del _x  # accumulator exists only to keep the spin loop live
        gdts = []
        per_game = []
        # gnubg's per-game bookkeeping scales with the session record's
        # length (3.5 -> 6.1 ms/game over 2000 games); reset periodically to
        # remove the artifact. Counts are harvested BEFORE each reset, so the
        # per-game (time, decisions) pairs match an unreset run. 'new session'
        # does not clear the evaluation cache.
        for i in range(ngames):
            if RESET_EVERY and i and i % RESET_EVERY == 0:
                per_game.extend(count_record()[n0:])
                gnubg.command("new session")
                n0 = len(count_record())
            t0 = time.perf_counter()
            gnubg.command("new game")
            gdts.append(time.perf_counter() - t0)
        dt = sum(gdts)
        per_game.extend(count_record()[n0:])
        assert len(per_game) == ngames, (n0, len(per_game), ngames)
        nch = sum(g[0] for g in per_game)
        nopp = sum(g[1] for g in per_game)
        nresp = sum(g[2] for g in per_game)
        tot = nch + (nopp + nresp if cube == "on" else 0)
        gtots = [g[0] + (g[1] + g[2] if cube == "on" else 0) for g in per_game]
        lo, hi = ratio_ci(gdts, gtots)
        print("GNUBG %d-ply cube-%s: %d games %.2fs (%.1f g/s) | checker %d (%.1f dec/s)"
              " | cube-opps %d responses %d | combined %.1f dec/s 95%%CI [%.1f,%.1f]"
              % (plies, cube, ngames, dt, ngames / dt, nch, nch / dt,
                 nopp, nresp, tot / dt, lo, hi), flush=True)
        if cube == "off":
            c_ch_time = dt / nch
        else:
            c_cu_time = (dt - nch * c_ch_time) / max(nopp + nresp, 1)
            # negative residual = below run-to-run noise, not a real cost
            print("GNUBG %d-ply ATTRIBUTION: checker %.3f ms/dec (%.1f dec/s) | cube %.3f ms/dec (%.1f dec/s)"
                  % (plies, 1000 * c_ch_time, 1 / c_ch_time,
                     1000 * c_cu_time,
                     (1 / c_cu_time) if c_cu_time > 0 else float("nan")),
                  flush=True)
print("ALLDONE", flush=True)
