"""Process helpers shared by the multiprocessing pools of the play /
evaluation scripts (eval_matchplay.py, gnubg_eval_matchplay.py,
play_models.py).

`die_with_parent()` asks the kernel to SIGKILL the calling process when
its parent dies (PR_SET_PDEATHSIG). Pool workers use the 'spawn' start
method, so they land in their own process group and a `kill -- -PGID`
on the parent never reaches them; `Pool.terminate()` would, but it does
not run when the parent is killed, so an interrupted run leaves one
worker per core running forever (`multiprocessing.spawn` in its argv).
Same mechanism as the gnubg subprocess reaper in gnubg_external.py.
Linux only; a no-op elsewhere and on any failure.
"""
import signal
import sys


def die_with_parent() -> None:
    """Pool initializer (or its first call): SIGKILL this process when the
    parent dies. Best-effort; never raises."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass
