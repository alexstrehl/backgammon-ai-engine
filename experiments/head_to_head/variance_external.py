"""GNU Backgammon players under the native luck reference.

The recording loops' roll hook observes every roll, forced passes and
rolls after a take included. Ordinary and corrected executions use
separate GNU processes, so neither mode is served from the other's
evaluation cache. CPU accounting adds the live single-threaded GNU
subprocesses' time from /proc (Linux only) to the worker's own.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

GNUBG_SIDES = ("gnubg-ext", "gnubg-ext-full", "gnubg-ext-rootfull")
DICE = [(a, b) for a in range(1, 7) for b in range(a, 7)]
WEIGHTS = np.array([1 if a == b else 2 for a, b in DICE]) / 36


def preflight(plies):
    if not 0 <= plies <= 4:
        raise ValueError("GNU Backgammon --plies must be between 0 and 4")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("external variance runs require Linux /proc for subprocess CPU accounting")
    from third_party_opponents.gnubg import gnubg_eval as G
    G.assert_gnubg_bearoff()
    binary = Path(shutil.which(G.GNUBG_CMD) or G.GNUBG_CMD).resolve()
    data = binary.parent.parent / "share/gnubg"
    paths = [binary] + [data / n for n in ("gnubg.wd", "gnubg_os0.bd", "gnubg_ts0.bd")]
    return dict(command=str(binary), version=G._gnubg_version(str(binary)),
                weights_md5=G._gnubg_weights_md5(str(binary)),
                files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths if p.is_file()},
                allow_version_mismatch=os.environ.get("GNUBG_ALLOW_VERSION_MISMATCH", "0"),
                allow_onesided_bearoff=os.environ.get("GNUBG_ALLOW_ONESIDED_BEAROFF", "0"))


class RollObserver:
    """Accumulates the centered luck of the selected rolls and hashes the
    pre-move trajectory. Without a reference it only hashes."""

    def __init__(self, reference=None, every=1):
        self.reference, self.every = reference, every
        self.turns = self.corrected = 0
        self.luck = 0.0
        self.trace = hashlib.sha256()

    def __call__(self, state, dice, ms):
        self.turns += 1
        cube = None if ms is None else (int(ms.cube_owner), ms.cube_value, ms.jacoby)
        if ms is not None and not 0 < ms.cube_value <= 2**30:
            raise OverflowError("cube exceeds the native reference's representable money state")
        # Hash the whole pre-move board/dice/cube trajectory, not only the scores.
        self.trace.update(repr((state.points, state.bar, state.off, state.turn,
                                dice, cube)).encode("ascii"))
        if self.reference is None or self.turns == 1 or (self.turns-2) % self.every:
            return
        q = self.reference.dice_values(state, ms)
        correction = q[DICE.index(tuple(sorted(dice)))] - WEIGHTS @ q
        self.luck += (1 if state.turn == 0 else -1) * (ms.cube_value if ms else 1) * correction
        self.corrected += 1


def observed_game(white, black, rng, cubeful, jacoby, reference=None, every=1,
                  record_moves=False):
    from third_party_opponents.gnubg.gnubg_eval import play_and_record, play_and_record_cubeful
    observer = RollObserver(reference, every)
    if cubeful:
        rec = play_and_record_cubeful(white, black, jacoby=jacoby, rng=rng,
                                     record_moves=record_moves, on_roll=observer)
    else:
        rec = play_and_record(white, black, rng=rng, record_moves=record_moves,
                              on_roll=observer)
    return rec, observer


def _child_cpu(pid):
    # /proc/pid/stat field 2 is a parenthesized name that can contain spaces.
    # utime+stime have clock-tick (usually 10 ms) resolution. A GNU 2-ply
    # game takes hundreds of ms, and a run's totals sum thousands of games.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


class ExternalRunner:
    """One worker's players, reference engine and GNU processes."""

    def __init__(self, paths, backend, every, cubeful, jacoby, plies, native_plies=(0, 0),
                 root_filter=0.0):
        from c_api.puretd_py import Engine, CAgent
        from third_party_opponents.gnubg import gnubg_external as G
        from multiprocessing.util import Finalize
        self.agents = ([], [])  # per mode: ordinary, corrected
        self.reference = None
        self.backend, self.every, self.cubeful, self.jacoby = backend, every, cubeful, jacoby
        # This runner owns and closes its servers. Do not reap unrelated jobs.
        G._REAPER_RAN = True
        try:
            self.reference = Engine(paths[2], backend)
            for path, native in zip(paths[:2], native_plies):
                if path in GNUBG_SIDES:
                    # One GNU process per mode: a shared process would serve
                    # each game's second execution from its evaluation cache.
                    for agents in self.agents:
                        agent = G.GnubgExternalAgent(plies=plies, cubeless=not cubeful,
                                                    full=path == "gnubg-ext-full",
                                                    root_full=path == "gnubg-ext-rootfull")
                        agents.append(agent)
                        agent._srv()  # start now, outside the game timings
                else:
                    agent = CAgent(path, plies=native, backend=backend, root_filter=root_filter)
                    for agents in self.agents:
                        agents.append(agent)
            self.finalizer = Finalize(self, self.close, exitpriority=10)
        except BaseException:
            self.close()
            raise

    def close(self):
        for agent in {id(a): a for agents in self.agents for a in agents}.values():
            if getattr(agent, "_server", None) is not None:
                agent._server.close()
            if hasattr(agent, "engine"):
                agent.engine.close()
        if self.reference is not None:
            self.reference.close()

    def cpu_time(self, mode):
        return time.process_time() + sum(_child_cpu(a._server.proc.pid)
            for a in self.agents[mode] if getattr(a, "_server", None) is not None)

    def chunk(self, args, seed_for_pair):
        from c_api.puretd_py import Rng
        seed, start, count = args
        data = {k: np.zeros((count, 2), dtype=dtype) for k, dtype in (
            ("raw", float), ("luck", float), ("turns", np.int32), ("corrected", np.int32),
            ("cube_value", np.int32), ("ended_by_drop", bool), ("seconds", float),
            ("cpu_seconds", float), ("gnubg_cube_clamps", np.int32))}
        for i, pid in enumerate(range(start, start+count)):
            for leg in range(2):
                results, observers, states, clamps = {}, {}, {}, {}
                for mode in ((0,1) if (pid+leg) % 2 == 0 else (1,0)):
                    a, b = self.agents[mode]
                    white, black = (a, b) if leg == 0 else (b, a)
                    rng = Rng(seed_for_pair(seed, pid), self.backend)
                    c_before = sum(getattr(x, "cube_clamps", 0) for x in (a, b))
                    c0, t0 = self.cpu_time(mode), time.perf_counter()
                    rec, obs = observed_game(white, black, rng, self.cubeful, self.jacoby,
                                            self.reference if mode else None, self.every)
                    data["seconds"][i,mode] += time.perf_counter()-t0
                    data["cpu_seconds"][i,mode] += self.cpu_time(mode)-c0
                    clamps[mode] = sum(getattr(x, "cube_clamps", 0) for x in (a, b))-c_before
                    results[mode], observers[mode], states[mode] = rec, obs, rng.getstate()
                if (results[0] != results[1] or states[0] != states[1]
                        or observers[0].trace.digest() != observers[1].trace.digest()
                        or clamps[0] != clamps[1]):
                    raise RuntimeError(f"correction changed external play/dice at pair {pid}, leg {leg}")
                rec, obs = results[1], observers[1]
                sign = 1 if leg == 0 else -1
                data["raw"][i,leg] = sign * (1 if rec.winner == 0 else -1) * rec.result * rec.cube_value
                data["luck"][i,leg] = sign * obs.luck
                data["turns"][i,leg] = obs.turns
                data["corrected"][i,leg] = obs.corrected
                data["cube_value"][i,leg] = rec.cube_value
                data["ended_by_drop"][i,leg] = rec.ended_by_drop
                data["gnubg_cube_clamps"][i,leg] = clamps[0]
        return start, data
