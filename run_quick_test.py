#!/usr/bin/env python3
"""
run_quick_test.py — THE fast validation command.

Exercises exactly the same pipeline as ``run_full_simulation.py`` (same sweep
loop, same per-run output, same plots) at a tiny scale, so you can confirm the
whole thing works before committing hours of CPU time.

    python run_quick_test.py

Expect it to finish in well under a minute.  It writes to ``quick_test_results/``
— deliberately *not* ``benchmark_results/`` — so a smoke test can never clobber
the production dataset.

Scale: 10×10 grid, 20 agents, 30 cycles, 3 actions/cycle, 3 difficulty levels,
2 seeds per cell, all 8 governments (48 runs total).  These numbers exist to
exercise code paths quickly; they are NOT a meaningful experiment and their
results should never be reported.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from benchmark_core import (
    BenchmarkConfig,
    ConfigError,
    DEFAULT_GOVERNMENTS,
    add_common_arguments,
    main_from_config,
)

# ---------------------------------------------------------------------------
# Smoke-test configuration — small enough to finish in seconds, wide enough to
# touch every government and both ends of the difficulty range.
# ---------------------------------------------------------------------------
DIFFICULTIES = (1, 50, 100)
K_RUNS      = 2
GRID_SIZE   = 10
N_AGENTS    = 20
MAX_CYCLES  = 30
MAX_STEPS   = 3
BASE_SEED   = 42

#: Log stem — ``sim_quick_<UTC>.log`` in the current directory.  Distinct from
#: the production stem so a smoke test's log is never mistaken for a sweep's.
LOG_PREFIX  = "sim_quick"
HEARTBEAT_EVERY = 10  # 30 cycles → 4 heartbeat lines per run

DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "quick_test_results")


def build_config() -> BenchmarkConfig:
    """Return the quick-test benchmark configuration."""
    return BenchmarkConfig(
        # The paper's eight primary regimes.  Still NOT
        # `tuple(GOVERNMENT_REGISTRY)` even though they currently match — see
        # the same comment in run_full_simulation.py for why the distinction is
        # kept.  The quick test must sweep whatever the full run sweeps, or it
        # stops being a smoke test of the production path.
        governments=DEFAULT_GOVERNMENTS,
        difficulties=DIFFICULTIES,
        k_runs=K_RUNS,
        grid_size=GRID_SIZE,
        n_agents=N_AGENTS,
        max_cycles=MAX_CYCLES,
        max_steps=MAX_STEPS,
        output_root=DEFAULT_OUTPUT_DIR,
        base_seed=BASE_SEED,
        label="Quick Test (NOT a valid experiment)",
        log_prefix=LOG_PREFIX,
        heartbeat_every=HEARTBEAT_EVERY,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_quick_test.py",
        description=(
            "Fast end-to-end validation of the benchmark pipeline at a tiny "
            "scale. Results are for smoke-testing only — never report them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_common_arguments(parser, DEFAULT_OUTPUT_DIR).parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return main_from_config(build_config(), args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
