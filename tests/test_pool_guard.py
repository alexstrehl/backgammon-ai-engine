"""A dead collection worker must not hang training (pool_guard).

These tests kill real workers and require (a) PoolWatch to raise
WorkerDied promptly and (b) the trainers to rebuild the pool and finish
the run.
"""

import os
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp

import pytest
import torch

from pool_guard import GuardedPool, PoolWatch, WorkerDied, shutdown_pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _die_on_zero(x):
    if x == 0:
        os._exit(3)
    time.sleep(0.2)
    return x


def _slow(x):
    time.sleep(0.2)
    return x


def test_death_mid_task_raises_instead_of_hanging():
    # Not `with Pool(...)`: its __exit__ is the plain terminate(), which
    # can deadlock after a worker death; shutdown_pool avoids that.
    pool = mp.get_context("spawn").Pool(2)
    try:
        watch = PoolWatch(pool, poll=0.2)
        t0 = time.monotonic()
        with pytest.raises(WorkerDied):
            watch.map(_die_on_zero, range(4))
        assert time.monotonic() - t0 < 30
    finally:
        shutdown_pool(pool, timeout=5)


def test_death_between_rounds_is_caught_before_dispatch():
    pool = mp.get_context("spawn").Pool(2)
    try:
        watch = PoolWatch(pool, poll=0.2)
        assert watch.map(_slow, range(4)) == [0, 1, 2, 3]
        os.kill(pool._pool[0].pid, signal.SIGKILL)
        time.sleep(1.0)   # the pool has reaped and replaced it by now
        with pytest.raises(WorkerDied):
            watch.map(_slow, range(4))
    finally:
        shutdown_pool(pool, timeout=5)


def test_healthy_pool_is_untouched():
    with mp.get_context("spawn").Pool(2) as pool:
        watch = PoolWatch(pool, poll=0.2)
        for _ in range(3):
            assert watch.map(_slow, range(4)) == [0, 1, 2, 3]


def test_idle_worker_death_then_shutdown_does_not_hang():
    # An idle worker waits for tasks INSIDE the task-queue lock. Killed
    # there, it leaves the lock held and a plain Pool.terminate() blocks
    # forever. The watch must still raise, and shutdown_pool must return.
    pool = mp.get_context("spawn").Pool(2)
    watch = PoolWatch(pool, poll=0.2)
    assert watch.map(_slow, range(4)) == [0, 1, 2, 3]
    time.sleep(0.5)
    for p in list(pool._pool):
        os.kill(p.pid, signal.SIGKILL)
    time.sleep(1.0)
    with pytest.raises(WorkerDied):
        watch.map(_slow, range(4))
    t0 = time.monotonic()
    shutdown_pool(pool, timeout=5)
    assert time.monotonic() - t0 < 15


def _segfault(x):
    import ctypes
    if x == 0:
        ctypes.string_at(0)   # SIGSEGV
    time.sleep(0.2)
    return x


def test_death_reports_the_signal():
    pool = GuardedPool(2, context=mp.get_context("spawn"))
    try:
        watch = PoolWatch(pool, poll=0.2)
        assert watch.map(_slow, range(2)) == [0, 1]
        os.kill(pool._pool[0].pid, signal.SIGKILL)
        time.sleep(1.0)
        with pytest.raises(WorkerDied, match="killed by SIGKILL"):
            watch.map(_slow, range(4))
    finally:
        shutdown_pool(pool, timeout=5)


def test_maxtasksperchild_recycling_is_not_a_death():
    pool = GuardedPool(2, maxtasksperchild=1, context=mp.get_context("spawn"))
    try:
        watch = PoolWatch(pool, poll=0.2)
        for _ in range(4):   # every round retires and replaces both workers
            assert watch.map(_slow, range(2)) == [0, 1]
    finally:
        shutdown_pool(pool, timeout=5)


def test_native_crash_prints_a_traceback(tmp_path):
    # faulthandler is enabled in every GuardedPool worker, so a SIGSEGV
    # leaves a stack on the worker's stderr (the training log).
    code = (
        "import multiprocessing as mp, sys\n"
        f"sys.path[:0] = [{REPO!r}, {os.path.dirname(__file__)!r}]\n"
        "from pool_guard import GuardedPool, PoolWatch, WorkerDied, shutdown_pool\n"
        "from test_pool_guard import _segfault\n"
        "if __name__ == '__main__':\n"
        "    pool = GuardedPool(2, context=mp.get_context('spawn'))\n"
        "    try:\n"
        "        PoolWatch(pool, poll=0.2).map(_segfault, range(4))\n"
        "    except WorkerDied as e:\n"
        "        print('DIED:', e, flush=True)\n"
        "    shutdown_pool(pool, timeout=5)\n")
    script = tmp_path / "crash.py"
    script.write_text(code)
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=120)
    assert "Fatal Python error: Segmentation fault" in out.stderr
    assert "_segfault" in out.stderr
    assert "killed by SIGSEGV" in out.stdout


# ── End to end: kill a worker inside a real training run ─────────────
#
# These cases run train_batch.py with 2 workers: a money mode on the
# Python and on the C collector, match play on the Python, the C match
# and (cuda) the lockstep GPU collector, the restart limit, and SIGTERM
# during a restart. (The money-game engine's prob5, shared-context and
# 4-worker cases need train_prob5.py and --vectorized-workers, which this
# tree does not have.) While a run is going this process is the child
# subreaper, so a worker the run leaves behind is re-parented HERE and "no
# orphan" is a check by parent pid, made once the run has exited (not at
# pipe EOF: a leftover worker holds the pipe open).

import ctypes
import queue
import threading
from typing import List, NamedTuple

_libc = ctypes.CDLL(None, use_errno=True)
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _proc_table():
    """pid -> (ppid, state, starttime, cmdline) of every process (/proc)."""
    table = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        table[int(d)] = (int(rest[1]), rest[0], int(rest[19]), cmd)
    return table


def _children(pid):
    """Live multiprocessing spawn children of `pid`, oldest first. Selected
    by parent pid; the cmdline only tells a spawn child from, e.g., the
    resource tracker."""
    kids = [(start, p) for p, (ppid, state, start, cmd) in _proc_table().items()
            if ppid == pid and state not in "ZX" and b"spawn_main" in cmd]
    return [p for _, p in sorted(kids)]


class _Subreaper:
    """PR_SET_CHILD_SUBREAPER on this process for the duration: whatever a
    run orphans is re-parented here, not to init."""

    def __enter__(self):
        was = ctypes.c_int(0)
        _libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(was),
                    ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))
        self._was = was.value
        if _libc.prctl(_PR_SET_CHILD_SUBREAPER, ctypes.c_ulong(1),
                       ctypes.c_ulong(0), ctypes.c_ulong(0),
                       ctypes.c_ulong(0)) != 0:
            pytest.skip(f"prctl(PR_SET_CHILD_SUBREAPER): errno "
                        f"{ctypes.get_errno()}")
        return self

    def __exit__(self, *exc):
        _libc.prctl(_PR_SET_CHILD_SUBREAPER, ctypes.c_ulong(self._was),
                    ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))


def _reap_leftovers(before, grace=15.0):
    """After a run has exited: the processes it left behind are children of
    this process (the subreaper) that were not children before. Reap the
    zombies; wait up to `grace` s for the rest to exit. A spawn child still
    alive then is an orphaned worker: SIGKILLed by pid, reaped, returned."""
    me = os.getpid()
    deadline = time.monotonic() + grace
    while True:
        new = {p: v for p, v in _proc_table().items()
               if v[0] == me and p not in before}
        for p, (_, state, _, _) in new.items():
            if state in "ZX":
                try:
                    os.waitpid(p, os.WNOHANG)
                except ChildProcessError:
                    pass
        alive = {p: v for p, v in new.items() if v[1] not in "ZX"}
        if not alive or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    orphans = sorted(p for p, v in alive.items() if b"spawn_main" in v[3])
    for p in orphans:
        try:
            os.kill(p, signal.SIGKILL)
            os.waitpid(p, 0)
        except (OSError, ChildProcessError):
            pass
    return orphans


class _Run(NamedTuple):
    rc: int
    out: str
    killed: List[int]    # the workers this harness SIGKILLed, in order
    orphans: List[int]   # spawn workers alive after the run (killed here)


def _run_killing_workers(cmd, n_kill=1, timeout=240, on_line=None):
    """Start `cmd`, SIGKILL its newest pool worker when `Round 2/` is
    printed (then at Round 5/, 8/, ... for `n_kill` kills; eval servers
    start first, so the newest child is a pool worker), wait for the end
    and return a _Run. `on_line(line, proc)` sees every output line."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", BGC_MATCH_SHM="0",
               BG_MAX_POOL_RESTARTS="2")
    with _Subreaper():
        before = {p for p, v in _proc_table().items() if v[0] == os.getpid()}
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        q = queue.Queue()
        threading.Thread(
            target=lambda: [q.put(l) for l in p.stdout] + [q.put(None)],
            daemon=True).start()
        lines, killed = [], []
        deadline = time.monotonic() + timeout
        exited = None
        try:
            while True:
                try:
                    line = q.get(timeout=0.5)
                except queue.Empty:
                    if p.poll() is not None:
                        # Exited. A leftover child holding the pipe must
                        # not keep us reading until it exits by itself.
                        exited = exited or time.monotonic()
                        if time.monotonic() - exited > 2.0:
                            break
                    elif time.monotonic() > deadline:
                        # A silent hang must fail the test, not block it.
                        pytest.fail("training hung after a worker was "
                                    "killed:\n" + "".join(lines[-20:]))
                    continue
                if line is None:
                    break
                lines.append(line)
                if len(killed) < n_kill and re.match(
                        r"Round\s+%d/" % (2 + 3 * len(killed)), line):
                    kids = _children(p.pid)
                    assert kids, "no pool workers found"
                    os.kill(kids[-1], signal.SIGKILL)
                    killed.append(kids[-1])
                if on_line is not None:
                    on_line(line, p)
            p.wait(timeout=60)
        finally:
            if p.poll() is None:
                for k in _children(p.pid):
                    try:
                        os.kill(k, signal.SIGKILL)
                    except OSError:
                        pass
                p.kill()
                p.wait(timeout=60)
            orphans = _reap_leftovers(before)
    return _Run(p.returncode, "".join(lines), killed, orphans)


def _run_and_kill_a_worker(cmd, n_kill=1, timeout=240):
    """The interface tests/test_c_collect_trainer.py uses:
    _run_killing_workers -> (returncode, output); a worker left behind
    fails the test."""
    run = _run_killing_workers(cmd, n_kill, timeout)
    assert run.orphans == [], (f"orphaned workers {run.orphans}:\n"
                               + run.out[-3000:])
    return run.rc, run.out


_RESTART = re.compile(
    r"^  WARN: (collection worker.*) in round (\d+) -- rebuilding the "
    r"collection pool and any eval servers, then redoing the round "
    r"\(restart (\d+)/(\d+), BG_MAX_POOL_RESTARTS\)$", re.M)
_ROUND = re.compile(r"^Round\s+(\d+)/(\d+) ", re.M)


def _rounds(out):
    return [(int(k), int(n)) for k, n in _ROUND.findall(out)]


def _assert_worker_died(err, pid):
    # PoolWatch.check's two messages: the worker seen dead, or its
    # replacement seen (the exit code recorded at reap time).
    assert err.startswith("collection worker"), err
    assert f"pid {pid} " in err and "killed by SIGKILL" in err, err


def _assert_recovered(run, save, n_rounds, c_collect):
    """One SIGKILLed worker: WorkerDied names it and its signal, the restart
    is announced, the round it hit is redone and completes, every round is
    logged once, the run saves and exits 0 with nothing left behind."""
    out = run.out
    assert run.rc == 0, out[-3000:]
    assert len(run.killed) == 1, out[-3000:]
    restarts = list(_RESTART.finditer(out))
    assert len(restarts) == 1, out[-3000:]
    m = restarts[0]
    _assert_worker_died(m.group(1), run.killed[0])
    k = int(m.group(2))
    assert (m.group(3), m.group(4)) == ("1", "2"), m.group(0)
    assert 3 <= k <= n_rounds, m.group(0)   # killed after Round 2
    # The round was redone: Round k-1 before the restart, Round k after.
    assert re.search(r"^Round\s+%d/%d " % (k - 1, n_rounds),
                     out[:m.start()], re.M), out[-3000:]
    assert re.search(r"^Round\s+%d/%d " % (k, n_rounds),
                     out[m.end():], re.M), out[-3000:]
    assert _rounds(out) == [(i, n_rounds) for i in range(1, n_rounds + 1)]
    assert f"Saved network to {save}" in out
    assert os.path.getsize(save) > 0
    assert run.orphans == [], f"orphaned workers {run.orphans}"
    _assert_c_dirs_removed(out, c_collect)


def _assert_c_dirs_removed(out, c_collect):
    # The C collectors' weight-export dir (c_collect_tmpdir), as logged.
    dirs = re.findall(r"exported once per round to (\S+/bg_ccollect_\w+)",
                      out)
    assert len(dirs) == (1 if c_collect else 0), out[-3000:]
    assert not any(os.path.exists(d) for d in dirs), dirs


_CPU2 = ["--workers", "2", "--device", "cpu"]
_MONEY = ["--game-mode", "cubeful-money", "--hidden", "32,32",
          "--torch-seed", "0"]
_MATCH = ["--game-mode", "matchplay", "--match-length", "3", "--oneply",
          "--met", "auto", "--met-scope", "all", "--hidden", "40",
          "--encoder", "matchplay1h64met_perspective196",
          "--output-mode", "probability", "--torch-seed", "0"]
_MATCH_C = ["--c-collect", "--c-collect-backend", "simd",
            "--c-collect-rng", "vectorized"]

# name -> (flags, episodes, episodes per round); 10 rounds each.
_RECOVERY_CASES = {
    # Pipelined Python: PoolWatch.map_async / get, the pre-roll.
    "money-python-pipelined": (
        _MONEY + ["--no-c-collect", "--pipeline-collect"], 200, 20),
    # bgc_collect, pipelined: two rounds' weight files in flight.
    "money-c-pipelined": (
        _MONEY + ["--c-collect", "--pipeline-collect"], 400, 40),
    # The serial Python match collector, synchronous: PoolWatch.map.
    "match-python": (_MATCH, 40, 4),
    # bgc_collect_match, pipelined (the round is re-exported on redo).
    "match-c-pipelined": (_MATCH + _MATCH_C + ["--pipeline-collect"], 160, 16),
    # The lockstep GPU match collector (one CUDA context per worker):
    # PoolWatch.map on vectorized_matchplay_worker.
    "match-vectorized-cuda": (
        _MATCH + ["--vectorized-collect", "--vectorized-device", "cuda",
                  "--vectorized-batch", "8"], 160, 16),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", [
    pytest.param(c, marks=[pytest.mark.cuda] if "cuda" in c else [])
    for c in _RECOVERY_CASES])
def test_dead_worker_is_replaced_and_the_round_redone(case, tmp_path):
    flags, n_eps, epr = _RECOVERY_CASES[case]
    save = str(tmp_path / "net.pt")
    run = _run_killing_workers(
        [sys.executable, "-u", "train_batch.py", *flags, *_CPU2,
         "--num-episodes", str(n_eps), "--episodes-per-round", str(epr),
         "--save", save])
    _assert_recovered(run, save, n_eps // epr, "--c-collect" in flags)


@pytest.mark.slow
def test_restart_limit_fails_loudly_instead_of_hanging(tmp_path):
    # BG_MAX_POOL_RESTARTS=2 in the helper: the first two deaths are
    # recovered, the third ends the run with an error, not a
    # hang. Match play on the synchronous C collector; the periodic
    # checkpoints the message points to are there, the final save is not.
    save = str(tmp_path / "net.pt")
    run = _run_killing_workers(
        [sys.executable, "-u", "train_batch.py", *_MATCH, *_MATCH_C, *_CPU2,
         "--num-episodes", "192", "--episodes-per-round", "16",
         "--save-every", "32", "--save", save], n_kill=3)
    out = run.out
    assert run.rc == 1, out[-3000:]
    assert len(run.killed) == 3, out[-3000:]
    restarts = list(_RESTART.finditer(out))
    assert [(r.group(3), r.group(4)) for r in restarts] == [
        ("1", "2"), ("2", "2")], out[-3000:]
    for r, pid in zip(restarts, run.killed):
        _assert_worker_died(r.group(1), pid)
    m = re.search(r"^ERROR: (collection worker.*) in round (\d+); giving up "
                  r"after 2 pool restart\(s\) \(BG_MAX_POOL_RESTARTS\)\. "
                  r"Resume from the last checkpoint\.$", out, re.M)
    assert m, out[-3000:]
    _assert_worker_died(m.group(1), run.killed[2])
    k = int(m.group(2))
    assert _rounds(out) == [(i, 12) for i in range(1, k)]
    assert "Saved network to" not in out and not os.path.exists(save)
    last = (k - 1) // 2 * 32    # --save-every 32 = every 2nd round
    assert os.path.exists(str(tmp_path / f"net_ep{last}.pt")), out[-3000:]
    assert run.orphans == [], f"orphaned workers {run.orphans}"
    _assert_c_dirs_removed(out, True)


# SIGTERM while a restart is under way: the handler must save the weights
# of the last whole round (`_ep{N}_sigterm.pt`, N = the episodes before the
# round being redone), tear down whichever pool is live and exit 143,
# leaving no worker and no C export dir behind. "warn": the signal is sent
# when the restart is announced (the old pool is being shut down);
# "rebuild": it is raised inside the construction of the new pool, whose
# workers are already started while `pool` still names the old one.
_INJECT_IN_REBUILD = """\
import os, runpy, signal, sys
sys.path.insert(0, {repo!r})
import trainer


class _SignalInRebuild(trainer.GuardedPool):
    built = 0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _SignalInRebuild.built += 1
        if _SignalInRebuild.built == 2:   # the restart's pool
            print("INJECT: SIGTERM inside the pool rebuild", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)


trainer.GuardedPool = _SignalInRebuild
sys.argv = ["train_batch.py"] + sys.argv[1:]
runpy.run_path("train_batch.py", run_name="__main__")
"""


@pytest.mark.slow
@pytest.mark.parametrize("when", ["warn", "rebuild"])
def test_sigterm_during_a_restart_saves_and_leaves_nothing_behind(
        when, tmp_path):
    flags, n_eps, epr = _RECOVERY_CASES["match-c-pipelined"]
    save = str(tmp_path / "net.pt")
    if when == "rebuild":
        main = tmp_path / "inject.py"
        main.write_text(_INJECT_IN_REBUILD.format(repo=REPO))
        hook = None
    else:
        main = "train_batch.py"
        sent = []

        def hook(line, proc):
            if not sent and "rebuilding the collection pool" in line:
                os.kill(proc.pid, signal.SIGTERM)
                sent.append(line)
    run = _run_killing_workers(
        [sys.executable, "-u", str(main), *flags, *_CPU2,
         "--num-episodes", str(n_eps), "--episodes-per-round", str(epr),
         "--save", save], on_line=hook)
    out = run.out
    assert run.rc == 128 + signal.SIGTERM, out[-3000:]
    restarts = list(_RESTART.finditer(out))
    assert len(restarts) == 1, out[-3000:]
    _assert_worker_died(restarts[0].group(1), run.killed[0])
    k = int(restarts[0].group(2))
    if when == "rebuild":
        assert out.index("INJECT: SIGTERM") > restarts[0].end()
    n = (k - 1) * epr
    sig_ckpt = str(tmp_path / f"net_ep{n}_sigterm.pt")
    m = re.search(r"^  -> signal 15: checkpoint saved: (\S+) \| episodes "
                  r"done (\d+) ", out, re.M)
    assert m and m.start() > restarts[0].end(), out[-3000:]
    assert (m.group(1), int(m.group(2))) == (sig_ckpt, n)
    assert os.path.getsize(sig_ckpt) > 0
    assert _rounds(out) == [(i, n_eps // epr) for i in range(1, k)]
    assert "Saved network to" not in out and not os.path.exists(save)
    assert run.orphans == [], f"orphaned workers {run.orphans}"
    _assert_c_dirs_removed(out, True)
