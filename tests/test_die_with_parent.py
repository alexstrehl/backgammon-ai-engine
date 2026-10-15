"""proc_utils.die_with_parent: a BUSY spawn-pool worker is reaped when its
parent is SIGKILLed. The H-H scripts' pools install it as initializer;
without it an interrupted run leaves one `multiprocessing.spawn` worker
per core running to the end of its task (an idle worker exits on the
closed pipe, a busy one does not notice)."""
import glob
import os
import signal
import subprocess
import sys
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parent script (a file: spawn re-imports __main__): a 2-worker spawn
# pool, initializer die_with_parent iff argv[1] == "1"; each worker
# writes its pid into argv[2] and stays busy for 60 s.
_PARENT = '''
import multiprocessing as mp, os, sys, time
sys.path.insert(0, %r)
from proc_utils import die_with_parent

def _init(flag):
    if flag:
        die_with_parent()

def _busy(d):
    open(os.path.join(d, str(os.getpid())), "w").close()
    t0 = time.time()
    while time.time() - t0 < 60:
        pass
    return os.getpid()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(2, initializer=_init, initargs=(sys.argv[1] == "1",)) as pool:
        r = pool.map_async(_busy, [sys.argv[2]] * 2, chunksize=1)
        r.wait(120)
'''


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pids, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
@pytest.mark.parametrize("flag", ["1", "0"], ids=["pdeathsig", "control"])
def test_busy_workers_die_with_killed_parent(tmp_path, flag):
    script = tmp_path / "parent.py"
    script.write_text(_PARENT % _REPO)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    parent = subprocess.Popen([sys.executable, str(script), flag, str(pid_dir)],
                              cwd=str(tmp_path))
    pids = []
    try:
        t0 = time.time()
        while len(pids) < 2 and time.time() - t0 < 30:
            pids = [int(os.path.basename(p)) for p in glob.glob(str(pid_dir / "*"))]
            time.sleep(0.1)
        assert len(pids) == 2 and all(_alive(p) for p in pids), pids
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)
        gone = _wait_gone(pids, timeout=5.0 if flag == "1" else 2.0)
    finally:
        for p in pids:                  # never leave the control's orphans
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            parent.kill()
    if flag == "1":
        assert gone, f"workers {pids} survived their parent's SIGKILL"
    else:
        # The control is the problem itself: busy workers outlive a killed
        # parent (reaped by the finally above).
        assert not gone
