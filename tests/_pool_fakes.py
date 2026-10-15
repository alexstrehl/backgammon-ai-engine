"""Intercept pool dispatch in tests.

The trainers dispatch through pool_guard.PoolWatch, which calls
Pool.map_async (never the blocking Pool.map, which hangs on a dead
worker). A test that fakes `Pool.map` must fake `map_async` too.
"""

import multiprocessing.pool as mpp


class _Done:
    """A finished AsyncResult."""

    def __init__(self, value):
        self._value = value

    def ready(self):
        return True

    def wait(self, timeout=None):
        return None

    def get(self, timeout=None):
        return self._value


def patch_pool_map(monkeypatch, fake_map):
    """Route both Pool.map and Pool.map_async through `fake_map`."""
    monkeypatch.setattr(mpp.Pool, "map", fake_map)
    monkeypatch.setattr(
        mpp.Pool, "map_async",
        lambda self, fn, iterable, chunksize=None, *a, **kw:
            _Done(fake_map(self, fn, iterable, chunksize)))
