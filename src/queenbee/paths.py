"""Default filesystem locations."""

from __future__ import annotations

import os
from pathlib import Path

SILO_BENCH_DIR_ENV = "SILO_BENCH_DIR"


def default_benchmarks_dir() -> Path:
    """Silo-Bench ``benchmarks`` directory.

    ``$SILO_BENCH_DIR`` when set, else
    ``<repo root>/third_party/acl26-silo-bench/benchmarks``.
    """
    configured = os.environ.get(SILO_BENCH_DIR_ENV)
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "third_party" / "acl26-silo-bench" / "benchmarks"
