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

def _children(pid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        if ppid == pid and b"spawn_main" in cmd:
            out.append(int(d))
    return sorted(out)


def _run_and_kill_a_worker(cmd, n_kill=1, timeout=240):
    """Start `cmd`, wait for its 2nd round, SIGKILL the newest pool
    worker(s) (eval servers start first, so they have the lowest pids),
    and return (returncode, output)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", BG_MAX_POOL_RESTARTS="2")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    import queue
    import threading
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in p.stdout] + [q.put(None)],
                     daemon=True).start()
    lines, killed = [], 0
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                # A silent hang must fail the test, not block it.
                line = q.get(timeout=max(1.0, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail("training hung after a worker was killed:\n"
                            + "".join(lines[-20:]))
            if line is None:
                break
            lines.append(line)
            if killed < n_kill and re.match(r"Round\s+%d/" % (2 + 3 * killed), line):
                kids = _children(p.pid)
                assert kids, "no pool workers found"
                os.kill(kids[-1], signal.SIGKILL)
                killed += 1
        p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
            for k in _children(p.pid):
                try:
                    os.kill(k, signal.SIGKILL)
                except OSError:
                    pass
    return p.returncode, "".join(lines)


@pytest.mark.slow
def test_prob5_cpu_pipeline_recovers_from_a_dead_worker(tmp_path):
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_prob5.py", "--hidden", "32,32",
        "--num-episodes", "240", "--episodes-per-round", "40",
        "--workers", "4", "--pipeline-collect", "--device", "cpu",
        "--no-c-collect", "--save", str(tmp_path / "p.pt")])
    assert rc == 0, out[-3000:]
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "p.pt").exists()


@pytest.mark.slow
def test_cubeful_cpu_recovers_from_a_dead_worker(tmp_path):
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_batch.py", "--game-mode", "cubeful-money",
        "--hidden", "32,32", "--num-episodes", "240",
        "--episodes-per-round", "40", "--workers", "4", "--pipeline-collect",
        "--device", "cpu", "--no-c-collect", "--save", str(tmp_path / "c.pt")])
    assert rc == 0, out[-3000:]
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "c.pt").exists()


@pytest.mark.slow
def test_restart_limit_fails_loudly_instead_of_hanging(tmp_path):
    # BG_MAX_POOL_RESTARTS=2 in the helper; three deaths must end the run
    # with a clear error, not a hang.
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_prob5.py", "--hidden", "32,32",
        "--num-episodes", "2000", "--episodes-per-round", "40",
        "--workers", "4", "--device", "cpu", "--no-c-collect",
        "--save", str(tmp_path / "p.pt")], n_kill=3)
    assert rc != 0
    assert "giving up after 2 pool restart(s)" in out


@pytest.mark.cuda
@pytest.mark.slow
def test_prob5_shared_context_gpu_recovers_from_a_dead_worker(tmp_path):
    # 1-ply, vectorized collector, eval-server slots claimed in the pool
    # initializer.
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_prob5.py", "--hidden", "64,64",
        "--oneply", "--device", "cuda:0", "--no-c-collect",
        "--num-episodes", "600", "--episodes-per-round", "100",
        "--workers", "4", "--vectorized-workers", "4",
        "--save", str(tmp_path / "g.pt")])
    assert rc == 0, out[-3000:]
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "g.pt").exists()


@pytest.mark.cuda
@pytest.mark.slow
def test_cubeful_shared_context_gpu_recovers_from_a_dead_worker(tmp_path):
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_batch.py", "--game-mode", "cubeful-money",
        "--hidden", "64,64", "--oneply", "--device", "cuda", "--no-c-collect",
        "--num-episodes", "600", "--episodes-per-round", "100",
        "--workers", "4", "--vectorized-workers", "4",
        "--save", str(tmp_path / "g.pt")])
    assert rc == 0, out[-3000:]
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "g.pt").exists()
