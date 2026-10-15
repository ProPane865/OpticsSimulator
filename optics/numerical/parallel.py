"""Chunked GIL-free dispatch for ``@njit(nogil=True)`` kernels."""

import threading
from collections.abc import Callable

import numba


def run_in_chunks(work: Callable[[int, int], None], n: int) -> None:
    """Split ``range(n)`` across GIL-free njit chunk calls on threads.

    ``prange`` cannot carry the element-stack tuple through its gufunc
    packing, so stack kernels are dispatched as disjoint slices instead;
    ``nogil=True`` lets the compiled chunks run truly in parallel.
    """
    if n <= 0:
        return
    nt = max(1, min(numba.get_num_threads(), n))
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
