#!/usr/bin/env python3
"""
run_full_simulation.py — THE production command.

Runs the complete benchmark that backs the paper: every government, at every
difficulty level, with ``--k-runs`` independent seeds per cell.

``--k-runs`` IS REQUIRED.  Without it, the bare command would silently start a
~47-hour job at the published ``K_RUNS`` (100) — the single most expensive
thing in this tree — with no statement of intent and nothing in the archive
recording that the scale had been chosen rather than defaulted.  Stating the
scale is a required part of invoking the sweep:

    python run_full_simulation.py --k-runs 10     # ~5 h validation pass
    python run_full_simulation.py --k-runs 100    # ~47 h, the published design

Neither is privileged; both are one flag.  ``--plots-only`` does not need it,
because a replot reads the scale out of the archive's own manifest.

To keep it running after you close the terminal:

    nohup python3 run_full_simulation.py --k-runs 100 > full_run_console.log 2>&1 &

Use ``run_quick_test.py`` first — it takes minutes, needs no flags, and
exercises the same pipeline.

Reported methodology this reproduces (at ``--k-runs 100``):
  * 21 difficulty levels — 1, then 5 through 100 in steps of 5.
  * 100 independent seeds per (government, difficulty) cell (``K_RUNS``).
  * 21 difficulties × 100 seeds = 2,100 runs per government; × 8 governments =
    16,800 runs in total.  Both counts are pinned by import-time guards below,
    so this docstring cannot silently drift from the numbers that actually run.
  * All eight primary regimes run; there are no excluded controls today.  See
    DEFAULT_GOVERNMENTS, which stays derived so a future exclusion is one edit.
  * 50×50 grid, 500 agents, 150 cycles, 10 agent actions per cycle.

This is a long job: expect hours, not minutes.  Use ``run_quick_test.py`` first
to confirm the pipeline works end to end.

All of the actual work lives in ``benchmark_core.py`` — this file only pins the
configuration, so the quick test and the full run can never diverge in logic.
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
    PAPER_K_RUNS,
    add_common_arguments,
    main_from_config,
)

# ---------------------------------------------------------------------------
# Production configuration — matches the paper exactly.  Do not tune casually:
# these values define the published dataset.
# ---------------------------------------------------------------------------

#: Difficulty sweep: {1} ∪ {5, 10, …, 100} → 21 levels.
DIFFICULTIES = (1,) + tuple(range(5, 101, 5))

#: Independent seeds per (government, difficulty) cell in the PUBLISHED design.
#:
#: NOT a second literal.  benchmark_core.PAPER_K_RUNS is the one definition of
#: this number and this name is an alias for it, so the two cannot disagree.
#:
#: This is the *design* constant, not what any given invocation runs.  The
#: sweep's actual k comes from the REQUIRED --k-runs flag (see main()), and the
#: value that ran is recorded in manifest.json alongside its provenance.
K_RUNS      = PAPER_K_RUNS
GRID_SIZE   = 50     # 50×50 grid
N_AGENTS    = 500
MAX_CYCLES  = 150
MAX_STEPS   = 10     # agent actions per cycle
BASE_SEED   = 42     # sweep root seed; every run's seed is derived via
                      # engine.scenario_plan.derive_seed(BASE_SEED, domain, ...)
                      # keyed by government, difficulty, and run index

#: Stem of the per-invocation log file, written to the current directory as
#: ``sim_full_<UTC>.log``.  A log is always produced; no flag is needed.
LOG_PREFIX  = "sim_full"
HEARTBEAT_EVERY = 25  # cycles between per-run progress lines

DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "benchmark_results")

# Fail loudly at import time if the sweep ever stops matching the paper's claim.
#
# NOT an `assert` — see the matching note in governments/__init__.py.  `python -O`
# strips asserts, and `python3 -O run_full_simulation.py` is a plausible way to
# launch a 40-60 hour job, which is precisely when an unnoticed difficulty-count
# drift is most expensive.
if len(DIFFICULTIES) != 21:
    raise RuntimeError(
        f"expected 21 difficulty levels (the paper's stated design), "
        f"got {len(DIFFICULTIES)}: {DIFFICULTIES}"
    )

# The same guard for the run count, and for the same reason: an unnoticed
# mismatch between the run count in the source and the one actually swept
# would silently invalidate the archive's claim to match the published
# design.
#
# Deliberately a literal `100` compared against the constant, exactly as the
# check above compares a literal `21` against len(DIFFICULTIES).  The point of
# an assertion is to restate the published claim independently of the value it
# checks; `K_RUNS != K_RUNS` would guard nothing.
#
# This constrains the SOURCE, not the command line.  `--k-runs 10` is a runtime
# override that is recorded in the manifest and does not trip this — source
# edits are guarded, deliberate overrides are documented.  Also NOT an `assert`:
# `python -O` strips asserts and is a plausible way to launch a 47-hour job.
if K_RUNS != 100:
    raise RuntimeError(
        f"expected 100 runs per cell (the paper's stated design), got "
        f"{K_RUNS!r}. K_RUNS aliases benchmark_core.PAPER_K_RUNS; if the "
        f"published design really has changed, change it there and update this "
        f"guard, the module docstring's run counts, and the paper together. To "
        f"run a sweep at a different scale WITHOUT changing the published "
        f"design, pass --k-runs instead — that is what it is for."
    )

if not isinstance(K_RUNS, int) or isinstance(K_RUNS, bool):
    raise RuntimeError(f"K_RUNS must be an int, got {type(K_RUNS).__name__}")


def build_config() -> BenchmarkConfig:
    """Return the production benchmark configuration."""
    return BenchmarkConfig(
        # The paper's eight primary regimes.  Still NOT
        # `tuple(GOVERNMENT_REGISTRY)`, even though the two currently produce
        # the same tuple: the registry is the validation set and may grow an
        # entry that should not sweep by default, and writing the registry here
        # is exactly how such an entry would silently change the published
        # run count.  Say what is meant.
        governments=DEFAULT_GOVERNMENTS,
        difficulties=DIFFICULTIES,
        k_runs=K_RUNS,
        grid_size=GRID_SIZE,
        n_agents=N_AGENTS,
        max_cycles=MAX_CYCLES,
        max_steps=MAX_STEPS,
        output_root=DEFAULT_OUTPUT_DIR,
        base_seed=BASE_SEED,
        label="Full Production Simulation",
        log_prefix=LOG_PREFIX,
        # 150 cycles / 25 → 7 heartbeat lines per run.
        heartbeat_every=HEARTBEAT_EVERY,
    )


def _paper_scale_description() -> str:
    """The one-line scale summary, derived so it cannot go stale."""
    runs = len(DEFAULT_GOVERNMENTS) * len(DIFFICULTIES) * K_RUNS
    return (
        f"Run the full production benchmark: {len(DEFAULT_GOVERNMENTS)} primary "
        f"governments x {len(DIFFICULTIES)} difficulty levels x N seeds "
        f"({GRID_SIZE}x{GRID_SIZE} grid, {N_AGENTS} agents, {MAX_CYCLES} "
        f"cycles). --k-runs is REQUIRED: the published design is "
        f"--k-runs {K_RUNS} ({runs:,} runs, ~47 hours). Use --k-runs 10 for a "
        f"validation pass first."
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_full_simulation.py",
        # Derived from the constants rather than written out in prose.  A
        # hard-coded "100 seeds = 16,800 runs" here would be a second place
        # the published scale is spelled out (the module docstring is the
        # first), and a second place it could go stale.
        description=_paper_scale_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_common_arguments(parser, DEFAULT_OUTPUT_DIR).parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # The scale must be stated, not defaulted.
    #
    # This entry point's most expensive failure mode is not a wrong answer, it
    # is a right answer that took 47 hours nobody meant to spend: a bare
    # command would launch the full k=100 sweep immediately, and an operator
    # who wanted a short validation pass would find out hours later.
    #
    # Mechanism chosen: REQUIRE the flag. Rejected alternatives --
    #   * an interactive confirmation prompt, because the documented way to run
    #     this is `nohup ... &` with no tty, where a prompt either hangs forever
    #     or is auto-answered by a stray EOF;
    #   * a `--yes`/`--force` gate on long runs, because any threshold that lets
    #     k=10 through un-gated makes the expensive path the one with fewer
    #     keystrokes, and a gate that catches both is just this check wearing a
    #     second flag.
    #
    # Requiring it keeps `--k-runs 10` and `--k-runs 100` exactly equally easy,
    # which was the explicit constraint: the safe path must not be the harder
    # one. It is also scriptable and non-interactive, and it means the operator
    # has stated the scale, which is what `k_runs_source` then records.
    #
    # --plots-only is exempt: it runs no simulation and reads the scale from the
    # archive's own manifest, so demanding a scale it would ignore would be
    # noise (and would tempt someone into passing a wrong one).
    if getattr(args, "k_runs", None) is None and not getattr(args, "plots_only", False):
        print(
            "error: --k-runs is required.\n"
            "\n"
            "  This command's scale is not defaulted: a bare invocation must "
            "not silently\n"
            "  start the ~47-hour production sweep.\n"
            "\n"
            f"    --k-runs 10     validation pass  (~{len(DEFAULT_GOVERNMENTS) * len(DIFFICULTIES) * 10:,} runs)\n"
            f"    --k-runs {K_RUNS}    the published design "
            f"(~{len(DEFAULT_GOVERNMENTS) * len(DIFFICULTIES) * K_RUNS:,} runs, ~47 hours)\n"
            "\n"
            "  Not sure the pipeline works end to end? Run run_quick_test.py "
            "first;\n"
            "  it needs no flags and finishes in minutes.\n"
            "  Only regenerating figures from an existing archive? Use "
            "--plots-only,\n"
            "  which reads the scale from that archive's manifest and does not "
            "need --k-runs.",
            file=sys.stderr,
        )
        return 2

    try:
        return main_from_config(build_config(), args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
