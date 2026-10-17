"""Chunked dispatch helper for GIL-free work callables.

The batch kernels themselves thread natively (``optics._native``); this
module keeps the historical chunking API for pure-Python callers.
"""

import os
import threading
from collections.abc import Callable


def run_in_chunks(work: Callable[[int, int], None], n: int) -> None:
    """Split ``range(n)`` across GIL-free chunk calls on threads."""
    if n <= 0:
        return
    nt = max(1, min(os.cpu_count() or 1, n))
    chunk = -(-n // nt)
    errors: list[BaseException] = []

    def run(lo: int, hi: int) -> None:
        try:
            work(lo, hi)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            errors.append(exc)

    threads = []
    for t in range(nt):
        lo = t * chunk
        if lo >= n:
            break
        hi = min(n, lo + chunk)
        th = threading.Thread(target=run, args=(lo, hi))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
