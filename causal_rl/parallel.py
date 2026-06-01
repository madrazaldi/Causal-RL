from __future__ import annotations

from contextlib import contextmanager
import os
from collections.abc import Iterator


def resolve_n_jobs(n_jobs: int | str | None = None) -> int:
    """Resolve user-facing parallelism settings to a positive worker count."""
    cpu_count = os.cpu_count() or 1
    raw: int | str | None = n_jobs
    if raw is None:
        raw = os.environ.get("CAUSAL_RL_N_JOBS", "auto")
    if isinstance(raw, str):
        raw = raw.strip().lower()
        if raw in {"", "auto"}:
            return max(1, cpu_count - 1)
        try:
            raw = int(raw)
        except ValueError as exc:
            raise ValueError("n_jobs must be an integer, -1, or 'auto'.") from exc
    if raw == 0:
        raise ValueError("n_jobs=0 is invalid; use 'auto', -1, or a positive integer.")
    if raw < 0:
        return max(1, cpu_count + 1 + raw)
    return max(1, min(raw, cpu_count))


@contextmanager
def limit_inner_threads(n_jobs: int) -> Iterator[None]:
    """Prevent BLAS/OpenMP oversubscription while joblib runs outer tasks."""
    if n_jobs <= 1:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1):
        yield
