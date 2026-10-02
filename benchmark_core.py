#!/usr/bin/env python3
"""
benchmark_core.py — the shared sweep engine behind the two entry-point commands.

This module owns *all* of the benchmark logic: the (government × difficulty × seed)
sweep, the per-cell runner, snapshot rendering, CSV/JSON output, and the summary
plots.  It is deliberately parameterised by a single :class:`BenchmarkConfig`
value object rather than module-level constants, so that different entry points
can drive the identical code path with different scales:

  * ``run_full_simulation.py`` — production configuration (reproduces the paper).
  * ``run_quick_test.py``      — tiny configuration for fast end-to-end validation.

Nothing here should be edited to change *what* is run; construct a different
``BenchmarkConfig`` instead.  That keeps the two commands provably identical in
behaviour, so the constants in a runner script cannot drift away from the
methodology the paper reports.

Output layout produced by :func:`run_benchmark`:

  <cwd at invocation>/
  └── sim_<prefix>_<UTC>.log       the sweep log — always written, one per
                                   invocation, deliberately NOT in output_root
                                   so it survives that directory being cleared

  <output_root>/
  ├── manifest.json                what command produced this directory, and
  │                                the per-cell sample sizes it achieved
  ├── sweep.log                    archival copy of the log above
  ├── <gov>/<difficulty>/
  │   ├── run_<k>/
  │   │   ├── health_stats.csv     (per-cycle statistics)
  │   │   ├── final_stats.json     (summary for last cycle)
  │   │   ├── run_detail.jsonl     (structured per-cycle / per-decision trace)
  │   │   └── agent_states.jsonl   (only with --detailed-agents)
  │   └── visualizations/          (only from run 1)
  │       ├── cycle_001.png … cycle_last.png
  ├── combined_final_stats.csv     (all govts × difficulties × runs)
  ├── plots/
  │   ├── normalized_health_score.png        Tukey IQR family — box = IQR,
  │   ├── final_survival_rate.png            whiskers = 1.5xIQR, line = median
  │   ├── final_median_health.png            trend.  Captions must NEVER say
  │   ├── final_health_gini.png              "confidence interval" or "CI".
  │   ├── normalized_health_score_median_ci.png   Median + 95% bootstrap CI
  │   ├── final_survival_rate_median_ci.png       family — the PARALLEL set,
  │   ├── final_median_health_median_ci.png       same metrics, different
  │   ├── final_health_gini_median_ci.png         dispersion statistic.  These
  │   │                                           captions SHOULD say CI.
  │   ├── mean_prediction_error.png    (ADS only — self-calibration accuracy)
  │   ├── calibration_half_split.png   (ADS only — early vs late within a run,
  │   │                                 split derived post-burn-in)
  │   └── calibration_trend_by_cycle.png (ADS only — error vs cycle, pooled
  │                                       sweep-wide, burn-in shaded)
  │       ... each with a companion <name>_alt_text.txt sidecar.
  │
  │   Every figure above can be regenerated from an archived output tree with
  │   no re-simulation:  python3 run_full_simulation.py --plots-only \
  │                             --output-dir <output_root>
  └── analysis/                    (POST-HOC ANALYSIS — no re-run needed)
      ├── analysis_manifest.json   (metadata describing all analysis files)
      ├── per_government_summary.csv
      ├── per_difficulty_summary.csv
      ├── regression_analysis.csv
      ├── pairwise_comparisons.csv
      ├── pairwise_comparisons_by_difficulty.csv  (t-test + bootstrap CI, per difficulty)
      ├── correlation_matrix.csv
      └── per_metric_analysis/
          ├── normalized_health_score_analysis.csv
          ├── final_survival_rate_analysis.csv
          ├── final_median_health_analysis.csv
          └── final_health_gini_analysis.csv

The ``analysis/`` directory holds comprehensive analysis files for post-hoc
investigation, alongside the original per-run and summary outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import getpass
import glob
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import (
    Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple,
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running this file as a script (``python3
# benchmark_core.py``) from the project root while still resolving the flat
# top-level packages (``engine``, ``governments``, ``scenarios``) as plain
# imports, regardless of the caller's own working directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# GOVERNMENT_REGISTRY is the *validation* set (everything runnable by name);
# DEFAULT_GOVERNMENTS is the *default sweep* set (the paper's primary eight).
# The two coincide today — ABLATION_GOVERNMENTS is empty — but they are kept
# distinct so a future opt-in entry is one edit in `governments/__init__.py`.
# Both are re-exported here because the two CLI entry points import their names
# from this module, not from `governments`.
from governments import (
    ABLATION_GOVERNMENTS,
    DEFAULT_GOVERNMENTS,
    GOVERNMENT_REGISTRY,
)
from engine.scenario_plan import (
    derive_seed,
    generate_scenario_plan,
    build_simulation_from_plan,
    make_scheduled_event_dict,
)
from engine.simulation import SimulationConfig
from engine.visualizer import (
    DASHBOARD_DPI,
    FS_SUPTITLE,
    SimulationVisualizer,
)
from engine.events import EventType as _ET
from engine.run_recorder import (
    RunRecorder,
    software_versions,
    utc_timestamp,
)
from scenarios.scenario_base import (
    _auto_base_severity,
    _auto_event_schedule,
    _auto_n_waves,
    _make_citizen_population,
)
from benchmark_logging import (
    DEFAULT_ENGINE_LEVEL,
    DETAILED_AGENTS_MAX_RUNS,
    LOG_LEVEL_NAMES,
    WORKER_START_METHOD,
    bench_log,
    get_context,
    init_worker_logging,
    resolve_log_path,
    set_context,
    setup_parent_logging,
    utc_now,
    verbose_level_warning,
    worker_mp_context,
)

# NOTE: the simulation loggers are NOT silenced here any more.  Their level is
# owned by benchmark_logging (default WARNING), so engine warnings and errors
# now reach the sweep log instead of being discarded.


# ---------------------------------------------------------------------------
# Presentation constants (shared by every entry point)
# ---------------------------------------------------------------------------

# Government display names for plots / legends.
#
# Every lookup below is `.get(name, name)`, so an unlisted government degrades
# to its raw key rather than raising.
#
# `autocracy_lookahead` is one of the eight curves the paper compares, not an
# ablation or control, so its legend label carries no "(ablation)" qualifier.
GOV_DISPLAY: Dict[str, str] = {
    "anarchy":              "Anarchy",
    "democracy":            "Democracy",
    "republic":             "Republic",
    "autocracy":            "Autocracy",
    "autocracy_lookahead":  "Autocracy+Lookahead",
    "oligarchy":            "Oligarchy",
    "federated":            "Federated",
    "ads":                  "ADS",
}

# WCAG AA accessible color palette (3:1+ contrast ratio, color-blind safe)
# Tested against WebAIM contrast checker
GOV_COLORS: Dict[str, str] = {
    "anarchy":   "#D55E00",      # Darker orange (WCAG AA compliant)
    "democracy": "#0072B2",      # Dark blue (high contrast)
    "republic":  "#009E73",      # Dark teal (accessible)
    "autocracy": "#CC79A7",      # Muted purple (WCAG AA)
    # Same hue family as its base regime `autocracy`, so the pair reads as a
    # pair, but darker to stay distinguishable in greyscale as well as colour.
    "autocracy_lookahead": "#7B3F63",
    "oligarchy": "#CC7A00",      # Dark orange (darker than standard)
    "federated": "#005A96",      # Deep blue (accessible)
    "ads":       "#000000",      # Black (maximum contrast)
}

# Line styles for additional color-blind distinction.
#
# The four simple styles ("-", "--", "-.", ":") were already taken before
# `autocracy_lookahead` existed, so it uses an explicit dash tuple rather than
# colliding with `republic`'s dash-dot.  Matplotlib accepts (offset, on/off-seq)
# anywhere a style string is accepted; the value is also written verbatim into
# the accessibility legend file, where a tuple renders fine.
#
# This map is not injective — "-" is shared by anarchy/oligarchy/ads and "--"
# by democracy/federated.  The scheme leans on colour and marker to
# disambiguate.
GOV_LINESTYLES: Dict[str, Any] = {
    "anarchy":   "-",            # solid
    "democracy": "--",           # dashed
    "republic":  "-.",           # dash-dot
    "autocracy": ":",            # dotted
    # densely dash-dotted — echoes autocracy's dotted without duplicating any
    # style already in use
    "autocracy_lookahead": (0, (3, 1, 1, 1)),
    "oligarchy": "-",            # solid
    "federated": "--",           # dashed
    "ads":       "-",            # solid
}

# Markers for additional distinction on point plots
GOV_MARKERS: Dict[str, str] = {
    "anarchy":   "o",            # circle
    "democracy": "s",            # square
    "republic":  "^",            # triangle
    "autocracy": "D",            # diamond
    "autocracy_lookahead": "d",  # thin diamond (echoes autocracy's diamond)
    "oligarchy": "o",            # circle
    "federated": "s",            # square
    "ads":       "x",            # x mark
}

# Final statistics to produce individual plots for
PLOT_STATS: List[Tuple[str, str, str]] = [
    ("normalized_health_score", "Normalized Health Score",
     "Mean total health / initial population"),
    ("final_survival_rate",     "Final Survival Rate",
     "Fraction of agents alive at end of simulation"),
    ("final_median_health",     "Final Median Health",
     "Median individual health score at end of simulation"),
    ("final_health_gini",       "Health Inequality (Gini)",
     "Gini coefficient of health"),
]

# Summary fields carried from each run into the combined CSV / plots.
SUMMARY_FIELDS: Tuple[str, ...] = (
    "normalized_health_score",
    "final_survival_rate",
    "final_median_health",
    "final_health_gini",
    "time_to_50pct_loss",
    "min_survival_rate",
)

DEFAULT_BASE_SEED = 42

#: Runs per cell in the PUBLISHED design.  THE single definition of that number.
#:
#: This is the projection target for the figure layer — it answers questions of
#: the form "this corpus is k=10, but what will the paper's figure look like?".
#: It deliberately does NOT drive any sweep: the sweep's k comes from
#: ``BenchmarkConfig.k_runs`` and nothing else, so editing this cannot change
#: what is run.
#:
#: ``run_full_simulation.py`` imports this constant rather than defining its
#: own copy, so the sweep size and the paper's stated k cannot drift apart.
#:
#: The number a *particular* sweep projects to travels on
#: ``BenchmarkConfig.paper_k_runs`` (defaulted from here) and is recorded in the
#: manifest, so a replot months later reproduces the original sidecar even if
#: this constant has moved in the meantime.  Read the config field, not this
#: global, from anywhere inside the figure layer.
PAPER_K_RUNS = 100

#: Private alias, kept because the test suites import it by this
#: name; new code should use :data:`PAPER_K_RUNS`, and figure code should use
#: ``config.paper_k_runs`` in preference to either.
_PAPER_K_RUNS = PAPER_K_RUNS

#: Provenance vocabulary for ``BenchmarkConfig.k_runs_source``.  Closed set, and
#: validated, so the manifest field can be relied on by a reader rather than
#: grepped hopefully.
K_RUNS_SOURCE_DEFAULT = "default"        # the entry point's own constant
K_RUNS_SOURCE_CLI = "cli-override"       # an explicit --k-runs on the command line
K_RUNS_SOURCE_ARCHIVE = "archive"        # read back out of a manifest (--plots-only)
K_RUNS_SOURCE_UNKNOWN = "unknown"        # a manifest predating this field

K_RUNS_SOURCES: Tuple[str, ...] = (
    K_RUNS_SOURCE_DEFAULT,
    K_RUNS_SOURCE_CLI,
    K_RUNS_SOURCE_ARCHIVE,
    K_RUNS_SOURCE_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when a BenchmarkConfig is internally inconsistent."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Everything that distinguishes one benchmark invocation from another.

    Frozen (and built only from primitives / tuples) so that it can be safely
    shared with worker processes and never mutated mid-sweep.

    Attributes:
        governments:  Government keys to run, in output order.  Must exist in
                      GOVERNMENT_REGISTRY.
        difficulties: Difficulty levels to sweep, 1–100.
        k_runs:       Independent seeds per (government, difficulty) cell.  The
                      AUTHORITATIVE scale of this invocation: figure titles, the
                      manifest and every per-cell n derive from it.
        k_runs_source: Where ``k_runs`` came from — one of
                      :data:`K_RUNS_SOURCES`.  Recorded in the manifest so a
                      future reader can tell a deliberate ``--k-runs`` pilot
                      apart from the published default without having to
                      reconstruct it from ``argv``.
        paper_k_runs: The k this sweep's figures PROJECT TO when they answer
                      "what will the published figure look like?".  Defaults to
                      :data:`PAPER_K_RUNS`.  Distinct from ``k_runs`` on
                      purpose: a k=10 pilot still wants to know what the k=100
                      paper figure will look like, so the projection target must
                      NOT follow ``--k-runs``.  Carried on the config rather
                      than read from the module global so that the value is
                      archived in the manifest and a replot reproduces the
                      original sidecar byte for byte.
        grid_size:    Side length of the square grid.
        n_agents:     Initial population per run.
        max_cycles:   Cycles simulated per run.
        max_steps:    Agent actions permitted per cycle.
        output_root:  Directory that receives all results (created if absent).
        base_seed:    Root seed of the whole sweep.  Every random stream is
                      derived from it by ``engine.scenario_plan.derive_seed``;
                      nothing consumes it directly.  Run ``k`` at difficulty
                      ``d`` faces the environment derived from
                      ``("env", d, k)``, which is the same for every
                      government and different for every ``k``.
        max_workers:  Parallel worker processes.  ``None`` → one per government.
        label:        Human-readable name shown in the console banner.

        log_prefix:   Stem of the generated log filename (``<prefix>_<UTC>.log``).
        log_file:     Explicit log path or directory; ``None`` → CWD.
        engine_log_level: Level for the engine loggers (``sim.<Gov>``,
                      ``sim.events``).  The harness's own progress logger is
                      pinned at INFO and is not affected.
        heartbeat_every:  Cycles between per-run progress lines.
        write_run_detail: Write ``run_detail.jsonl`` in each run directory.
        decision_detail:  Include the (large) ``groupings`` array in decision
                      records.  Disabling drops ~95% of the bytes and keeps
                      everything that says *what was decided*.
        agent_states_every: Cadence for ``agent_states.jsonl``; ``None`` → off.
        clean_output_root: Remove ``output_root`` entirely before the sweep
                      starts.  Opt-in (``--clean``), never automatic — this is
                      destructive and only runs when explicitly requested.
    """

    governments: Tuple[str, ...]
    difficulties: Tuple[int, ...]
    k_runs: int
    grid_size: int
    # NOTE: the two k-provenance fields are declared further down with defaults,
    # not here, because every positional construction of this dataclass in the
    # tree passes grid_size third.  See `k_runs_source` / `paper_k_runs` below.
    n_agents: int
    max_cycles: int
    max_steps: int
    output_root: str
    base_seed: int = DEFAULT_BASE_SEED
    max_workers: Optional[int] = None
    label: str = "Benchmark"
    # -- scale provenance (see the class docstring) ------------------------
    k_runs_source: str = K_RUNS_SOURCE_DEFAULT
    paper_k_runs: int = PAPER_K_RUNS
    # -- logging / structured output (all primitives → still frozen, still
    #    picklable, so a worker receives them unchanged) --------------------
    log_prefix: str = "sim"
    log_file: Optional[str] = None
    engine_log_level: str = DEFAULT_ENGINE_LEVEL
    heartbeat_every: int = 25
    write_run_detail: bool = True
    decision_detail: bool = True
    agent_states_every: Optional[int] = None
    clean_output_root: bool = False

    def __post_init__(self) -> None:
        # Validate eagerly: a bad config discovered three hours into a sweep is
        # far more expensive than one rejected at startup.
        if not self.governments:
            raise ConfigError("governments must not be empty")
        unknown = [g for g in self.governments if g not in GOVERNMENT_REGISTRY]
        if unknown:
            raise ConfigError(
                f"unknown government(s): {', '.join(unknown)}. "
                f"Valid options: {', '.join(GOVERNMENT_REGISTRY)}"
            )
        if len(set(self.governments)) != len(self.governments):
            raise ConfigError(f"duplicate government(s) in {self.governments}")

        if not self.difficulties:
            raise ConfigError("difficulties must not be empty")
        bad = [d for d in self.difficulties if not (1 <= d <= 100)]
        if bad:
            raise ConfigError(f"difficulty values must be in 1–100; got {bad}")
        if len(set(self.difficulties)) != len(self.difficulties):
            raise ConfigError(f"duplicate difficulty levels in {self.difficulties}")

        for name, value, minimum in (
            ("k_runs", self.k_runs, 1),
            ("paper_k_runs", self.paper_k_runs, 1),
            ("grid_size", self.grid_size, 2),
            ("n_agents", self.n_agents, 1),
            ("max_cycles", self.max_cycles, 1),
            ("max_steps", self.max_steps, 1),
        ):
            if value < minimum:
                raise ConfigError(f"{name} must be >= {minimum}; got {value}")

        if self.k_runs_source not in K_RUNS_SOURCES:
            raise ConfigError(
                f"k_runs_source must be one of {', '.join(K_RUNS_SOURCES)}; "
                f"got {self.k_runs_source!r}"
            )

        if self.max_workers is not None and self.max_workers < 1:
            raise ConfigError(f"max_workers must be >= 1; got {self.max_workers}")
        if not str(self.output_root).strip():
            raise ConfigError("output_root must be a non-empty path")

        if self.heartbeat_every < 1:
            raise ConfigError(
                f"heartbeat_every must be >= 1; got {self.heartbeat_every}"
            )
        if str(self.engine_log_level).upper() not in LOG_LEVEL_NAMES:
            raise ConfigError(
                f"engine_log_level must be one of {', '.join(LOG_LEVEL_NAMES)}; "
                f"got {self.engine_log_level!r}"
            )
        if not str(self.log_prefix).strip():
            raise ConfigError("log_prefix must be a non-empty string")

        if self.agent_states_every is not None:
            if self.agent_states_every < 1:
                raise ConfigError(
                    f"--detailed-agents cadence must be >= 1; "
                    f"got {self.agent_states_every}"
                )
            # Hard ceiling, not a warning.  Per-agent dumps are ~1.6 MB per run
            # at production scale; an accidental `--detailed-agents` on the full
            # sweep would write ~2.4 GB and slow it measurably.  The flag's
            # stated purpose is targeted debugging, so require the narrowing
            # rather than trusting that nobody types it at full scale.
            if self.total_runs > DETAILED_AGENTS_MAX_RUNS:
                est_mb = self.total_runs * 1.6
                # Below 1 GB, "about 0.0 GB" reads as self-contradicting a
                # refusal (a sweep just past the run-count ceiling can still be
                # well under a gigabyte). Report in MB until the estimate
                # actually clears 1 GB, then switch to GB.
                if est_mb < 1024:
                    est_size = f"{est_mb:.0f} MB"
                else:
                    est_size = f"{est_mb / 1024:.1f} GB"
                raise ConfigError(
                    f"--detailed-agents is a targeted-debugging option and this "
                    f"sweep has {self.total_runs} runs (limit "
                    f"{DETAILED_AGENTS_MAX_RUNS}). It would write about "
                    f"{est_size}. Narrow the sweep, e.g.:\n"
                    f"    python3 run_full_simulation.py --detailed-agents "
                    f"--governments ads --difficulties 76"
                )

    # -- derived values -------------------------------------------------

    @property
    def effective_workers(self) -> int:
        """Number of worker processes to launch (one per government by default)."""
        return self.max_workers or len(self.governments)

    @property
    def total_cells(self) -> int:
        """Number of (government, difficulty) work units."""
        return len(self.governments) * len(self.difficulties)

    @property
    def total_runs(self) -> int:
        """Total number of individual simulation runs this config implies."""
        return self.total_cells * self.k_runs

    def with_overrides(
        self,
        governments: Optional[Sequence[str]] = None,
        output_root: Optional[str] = None,
        difficulties: Optional[Sequence[int]] = None,
        k_runs: Optional[int] = None,
        log_file: Optional[str] = None,
        engine_log_level: Optional[str] = None,
        heartbeat_every: Optional[int] = None,
        write_run_detail: Optional[bool] = None,
        decision_detail: Optional[bool] = None,
        agent_states_every: Optional[int] = None,
        clean_output_root: Optional[bool] = None,
    ) -> "BenchmarkConfig":
        """
        Return a copy with the CLI-overridable fields replaced.

        ``None`` always means "not overridden", so a flag the user did not pass
        can never silently reset a field.  The copy is built with
        ``dataclasses.replace``, which re-runs ``__post_init__`` — validation is
        never bypassed by an override.

        Overriding ``k_runs`` also stamps ``k_runs_source`` to
        :data:`K_RUNS_SOURCE_CLI`.  The two move together by construction here
        rather than being set by the caller, so there is no way to change the
        scale and forget to record that it was changed — which is the whole
        provenance failure the field exists to prevent.
        """
        changes: Dict[str, object] = {}
        if governments is not None:
            changes["governments"] = tuple(governments)
        if output_root is not None:
            changes["output_root"] = os.path.abspath(os.path.expanduser(output_root))
        if difficulties is not None:
            changes["difficulties"] = tuple(difficulties)
        if k_runs is not None:
            changes["k_runs"] = int(k_runs)
            changes["k_runs_source"] = K_RUNS_SOURCE_CLI
        if log_file is not None:
            changes["log_file"] = log_file
        if engine_log_level is not None:
            changes["engine_log_level"] = str(engine_log_level).upper()
        if heartbeat_every is not None:
            changes["heartbeat_every"] = int(heartbeat_every)
        if write_run_detail is not None:
            changes["write_run_detail"] = bool(write_run_detail)
        if decision_detail is not None:
            changes["decision_detail"] = bool(decision_detail)
        if agent_states_every is not None:
            changes["agent_states_every"] = int(agent_states_every)
        if clean_output_root is not None:
            changes["clean_output_root"] = bool(clean_output_root)
        return replace(self, **changes) if changes else self


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """
    Emit an INFO progress line.

    Thin shim over :func:`bench_log`, kept so the ~40 existing call sites don't
    need updating.  From a worker, this travels through the logging queue to
    the parent's single file + console handler pair rather than racing on the
    inherited stdout; new call sites that need a level or ``%``-style
    arguments should call ``bench_log`` directly.
    """
    bench_log(logging.INFO, msg)


def fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable h m s string."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def progress_bar(done: int, total: int, width: int = 30) -> str:
    """Return an ASCII progress bar like [####------]  40%."""
    pct = done / total if total else 0.0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:5.1%}  ({done}/{total})"


# ---------------------------------------------------------------------------
# Plan generation (one plan per (difficulty, run_idx) — identical for every
# government at that coordinate, different at every run index)
# ---------------------------------------------------------------------------

def build_plan(config: BenchmarkConfig, difficulty: int, run_idx: int):
    """
    Generate the scenario plan for one ``(difficulty, run_idx)`` coordinate.

    The plan is a **pure function** of ``(config, difficulty, run_idx)``: it does
    not read the clock, the process id, the hash seed or any global RNG.  That
    purity is what makes the experiment fair without shipping a shared object
    between processes — every worker that evaluates this function at the same
    coordinate builds a byte-identical plan, and
    :meth:`ScenarioPlan.fingerprint` is recorded per run so the property is
    checkable from the archive rather than merely asserted here.

    Two properties, corresponding to the two experimental requirements:

    * ``run_idx`` is NOT in the plan's key → all eight governments at
      ``(d, k)`` face the identical terrain, resources, placements and event
      schedule.  Government identity is deliberately absent from every seed
      derived here.
    * ``run_idx`` IS in the plan's key → the ``k_runs`` runs of a cell face
      genuinely different environments, so ``n = k_runs`` counts environments
      rather than replicates of one environment.  If the plan were instead
      built once per difficulty and replayed ``k_runs`` times, most regimes
      would report ``n_distinct = 1``.

    ``run_idx`` is a **required** parameter with no default, on purpose: a
    default would let a forgetful call site silently rebuild the old
    one-environment-per-difficulty behaviour and produce data that looks
    perfectly normal but carries one effective observation per cell.
    """
    env_seed = derive_seed(config.base_seed, "env", difficulty, run_idx)
    sim_config = SimulationConfig.from_difficulty(
        difficulty,
        seed=env_seed,
        max_steps_per_cycle=config.max_steps,
        grid_rows=config.grid_size,
        grid_cols=config.grid_size,
        num_agents=config.n_agents,
        max_cycles=config.max_cycles,
    )
    auto_events = _auto_event_schedule(config.max_cycles, difficulty, env_seed)
    scheduled = [
        make_scheduled_event_dict(
            cycle=ev[0],
            event_type=ev[1].value,
            severity=ev[3],
            duration=12 if ev[1] == _ET.EPIDEMIC else 20,
            grid_rows=config.grid_size,
            grid_cols=config.grid_size,
            # Indexed by `i` so each scheduled event gets an independent seed.
            # `make_scheduled_event_dict` samples epidemic origin cells and
            # toxic-spill regions from this seed; without per-event indexing,
            # every epidemic in a run would land on the same grid cell and
            # every toxic spill on the same region.
            seed=derive_seed(
                config.base_seed, "env.events.sched", difficulty, run_idx, i
            ),
        )
        for i, ev in enumerate(auto_events)
    ]
    return generate_scenario_plan(sim_config, scheduled_events=scheduled)


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
#
# GOVERNMENT_REGISTRY keys (e.g. "autocracy_lookahead") are code identifiers,
# not reader-facing prose, and each government class's own `.name` attribute
# is not a substitute: it is either overly formal ("Representative Republic"
# vs. the paper's "Republic") or itself code-style ("AutocracyLookahead", no
# separator).  This mapping is the one place figure/plot text turns a
# registry key into the label the paper's Table 3 actually uses, so the two
# cannot drift apart government-by-government.
GOVERNMENT_DISPLAY_NAMES: Dict[str, str] = {
    "anarchy":             "Anarchy",
    "democracy":           "Direct Democracy",
    "republic":            "Republic",
    "autocracy":           "Autocracy",
    "autocracy_lookahead": "Autocracy + Lookahead (A+L)",
    "oligarchy":           "Oligarchy",
    "federated":           "Federated",
    "ads":                 "ADS",
}


def display_name(gov_name: str) -> str:
    """Reader-facing label for a GOVERNMENT_REGISTRY key.

    Falls back to the raw key for anything not in GOVERNMENT_DISPLAY_NAMES
    (e.g. a future ablation government) rather than raising, since this is
    used on the label-rendering path and a missing entry should degrade to
    the old behaviour, not break a sweep.
    """
    return GOVERNMENT_DISPLAY_NAMES.get(gov_name, gov_name)


# ---------------------------------------------------------------------------
# Snapshot selection
# ---------------------------------------------------------------------------

def snapshot_points(max_cycles: int) -> Dict[int, Tuple[str, str]]:
    """
    Choose which cycles to capture as PNG snapshots.

    Returns {cycle_index: (file_stem, human_title)}.  Cycle indices are
    0-based; titles are 1-based to match how cycles are reported elsewhere.

    For the production length (>100 cycles) this uses cycles 1, 50, 100 and
    the last, so figure paths (e.g. ``visualizations/cycle_last.png``) stay
    stable across runs.  Shorter runs get three evenly spaced snapshots plus
    the last cycle.
    """
    last = max_cycles - 1
    if max_cycles > 100:
        candidates = [0, 49, 99]
    else:
        candidates = sorted({0, last // 3, (2 * last) // 3})

    points: Dict[int, Tuple[str, str]] = {}
    for idx in candidates:
        if 0 <= idx < last:
            points[idx] = (f"cycle_{idx + 1:03d}", f"Cycle {idx + 1}")
    points[last] = ("cycle_last", f"Cycle {max_cycles} (final)")
    return points


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_detail_path(run_dir: str) -> str:
    """Absolute path of a run's structured detail file."""
    return os.path.join(run_dir, "run_detail.jsonl")


def agent_states_path(run_dir: str) -> str:
    """Absolute path of a run's opt-in per-agent state file."""
    return os.path.join(run_dir, "agent_states.jsonl")


def _build_recorder(
    config: BenchmarkConfig,
    gov_name: str,
    difficulty: int,
    run_idx: int,
    run_dir: str,
    env_seed: int,
    pop_seed: int,
    gov_seed: int,
    env_fingerprint: str,
) -> RunRecorder:
    """
    Construct the run's recorder.

    Deliberately called from inside ``_run_one`` — i.e. inside the worker
    process — because the recorder opens its files in ``__init__``.  Opening a
    handle before ``fork`` and writing to it from several children is the
    classic corruption bug this whole design is arranged to avoid.
    """
    meta = {
        "run": {
            "government": gov_name,
            "difficulty": difficulty,
            "run_index": run_idx + 1,
            "run_dir": os.path.relpath(run_dir, config.output_root),
            "run_id": f"{gov_name}-d{difficulty:03d}-run{run_idx + 1:02d}",
            # The full seed lattice for this run, recorded so the methodology
            # is checkable per run instead of being taken on trust.
            #
            #   base_seed  the sweep root; everything below derives from it
            #   env_seed   keyed on (difficulty, run_idx) ONLY — identical for
            #              all eight governments here, different at every run
            #   pop_seed   keyed on (difficulty, run_idx) ONLY — agent starting
            #              state is part of "the same starting conditions"
            #   gov_seed   keyed on (gov_name, difficulty, run_idx) — the one
            #              stream that is deliberately NOT shared, because each
            #              government's own decisions are its own
            #
            # The environment (terrain, resources, event schedule) is built
            # once per difficulty and shared identically by every government at
            # this coordinate; each government's own decision seed (gov_seed)
            # is NOT shared.  env_fingerprint (see ScenarioPlan.fingerprint) is
            # the machine-checkable witness for both halves.
            "base_seed": config.base_seed,
            "env_seed": env_seed,
            "pop_seed": pop_seed,
            "gov_seed": gov_seed,
            "env_fingerprint": env_fingerprint,
        },
        "started_at": utc_timestamp(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "software": software_versions(),
        "heartbeat_every": config.heartbeat_every,
    }
    return RunRecorder(
        run_detail_path(run_dir),
        meta,
        decision_detail=config.decision_detail,
        agent_states_path=(
            agent_states_path(run_dir) if config.agent_states_every else None
        ),
        agent_states_every=config.agent_states_every,
    )


def _decisions_made(government) -> Any:
    """Decision-round count if the government tracks one, else ``'-'``."""
    return getattr(government, "_decisions_made", "-")


def _merge_calibration_summary(
    summary: Dict[str, Any], government, max_cycles: int
) -> None:
    """
    Fold a government's per-run self-calibration summary into *summary* in place.

    Duck-typed on ``get_calibration_summary``, which today only ``AdsGovernment``
    implements.  A government without it contributes nothing and its
    ``final_stats.json`` keeps exactly the keys it had before self-calibration
    existed — that byte-for-byte equivalence for the other seven regimes is the
    point of doing this here rather than in ``MetricsCollector.summary()``,
    which is shared by all of them.

    Deliberately NOT added to :data:`SUMMARY_FIELDS`: that tuple drives
    ``combined_final_stats.csv`` and the per-cell CI aggregation, both
    of which are cross-government schemas.  A column that is a float for one
    regime and absent for seven would make every aggregate over it a trap.  The
    fields live in the per-run JSON, where a pandas read of
    ``ads/*/run_*/final_stats.json`` picks them up without disturbing anything.

    Failure is logged and swallowed.  This is instrumentation attached to a run
    that has already completed successfully; losing the metric is bad, throwing
    away the run's results to report it would be worse.
    """
    getter = getattr(government, "get_calibration_summary", None)
    if not callable(getter):
        return
    try:
        summary.update(getter(max_cycles) or {})
    except Exception as exc:                         # pragma: no cover - defensive
        bench_log(
            logging.ERROR,
            "calibration summary failed for %s: %s: %s",
            getattr(government, "name", "?"), type(exc).__name__, exc,
        )


def _run_one(
    config: BenchmarkConfig,
    gov_name: str,
    difficulty: int,
    run_idx: int,
    capture_viz: bool,
    run_dir: Optional[str] = None,
) -> Tuple[dict, List[dict], Optional[SimulationVisualizer], List[dict]]:
    """
    Run one government simulation.

    Returns (final_stats, csv_rows, visualizer_or_None, health_history).
    ``health_history`` is a list of per-cycle lightweight dicts, populated only
    when ``capture_viz`` is True; it drives the health timeline in snapshots.

    ``run_dir`` is where the structured detail file is written.  It is optional
    so the function stays callable without an output tree (the test suite does
    this); when omitted, no detail file is produced.

    If the run raises, the recorder's ``__exit__`` still writes a footer
    carrying the exception and traceback, then re-raises — so a lost seed leaves
    *more* forensic evidence than a successful one, not less.

    The plan is built HERE rather than received as an argument.  Because the
    plan varies with ``run_idx``, the alternatives were to ship a list of
    ``k_runs`` plans per task (11 MB pickled per task at production scale, for
    a 112 KB object that takes 8 ms to rebuild) or to rebuild it in the
    worker.  Rebuilding costs 8 ms against a ~40 s run — 0.02% — and removes
    the IPC entirely.
    """
    base = config.base_seed
    # The seed lattice.  Which keys each stream is derived from IS the design —
    # see the table in _build_recorder.
    pop_seed = derive_seed(base, "pop", difficulty, run_idx)
    gov_seed = derive_seed(base, "gov", gov_name, difficulty, run_idx)
    eval_seed = derive_seed(base, "gov.eval", gov_name, difficulty, run_idx)

    plan = build_plan(config, difficulty, run_idx)
    # plan.seed is derive_seed(base, "env", difficulty, run_idx); read back off
    # the plan rather than recomputed so there is exactly one derivation site
    # and the recorded value cannot drift from the one actually used.
    env_seed = plan.seed
    env_fingerprint = plan.fingerprint()

    gov_cls = GOVERNMENT_REGISTRY[gov_name]
    government = gov_cls(seed=gov_seed) if gov_name != "anarchy" else gov_cls()

    # ADS carries a second RNG — the evaluator's rollout stream — which is reset
    # on every evaluate() call and so cannot be seeded through the constructor's
    # `seed` alone.  Injected by duck-typing rather than by a constructor kwarg
    # for the same reason `get_calibration_summary` is duck-typed above: the
    # line above constructs all eight regimes uniformly as `gov_cls(seed=...)`,
    # and an ADS-only required kwarg would force a per-government branch here
    # for a concern that belongs to ADS.  A government without the hook is
    # unaffected.
    set_eval_seed = getattr(government, "set_eval_seed", None)
    if callable(set_eval_seed):
        set_eval_seed(eval_seed)

    # Every government — ADS included — runs on the same plain CitizenAgent
    # population.  (scenario_base._make_ads_population is an alias of this
    # function; ADS differs only in its Government implementation.)
    #
    # pop_seed is keyed on (difficulty, run_idx) and NOT on gov_name: "identical
    # starting conditions across governments" includes agent initial state.  The
    # agents' generators are therefore seeded identically for all eight regimes
    # and diverge only because behaviour consumes them in a different order —
    # which is what "same start, different outcome" is supposed to mean.
    agents = _make_citizen_population(config.n_agents, seed=pop_seed)

    # record_audit_trail=False: AuditTrail formats a text log the benchmark
    # never saves, at the cost of a full pass over every living agent every
    # cycle.  run_simulation.py keeps the default and still writes its trail.
    sim = build_simulation_from_plan(
        plan, government, agents, verbose=False, record_audit_trail=False,
    )

    viz: Optional[SimulationVisualizer] = SimulationVisualizer() if capture_viz else None
    capture_cycles = set(snapshot_points(config.max_cycles)) if capture_viz else set()
    health_history: List[dict] = []
    heartbeat = config.heartbeat_every
    last_cycle = config.max_cycles - 1

    with contextlib.ExitStack() as stack:
        recorder: Optional[RunRecorder] = None
        if config.write_run_detail and run_dir:
            recorder = stack.enter_context(
                _build_recorder(
                    config, gov_name, difficulty, run_idx, run_dir,
                    env_seed=env_seed, pop_seed=pop_seed, gov_seed=gov_seed,
                    env_fingerprint=env_fingerprint,
                )
            )
            recorder.write_header(sim)

        for cycle in range(config.max_cycles):
            sim.cycle = cycle
            sim._step(cycle)

            if recorder is not None:
                recorder.record_cycle(sim, cycle, getattr(sim, "last_warnings", ()))

            if capture_viz:
                alive = [a for a in sim.agents if a.alive]
                health_history.append({
                    "cycle": cycle,
                    "nhs": sim.normalized_health_score(),
                    "survival_rate": len(alive) / max(1, len(sim.agents)),
                    "median_health": (
                        float(np.median([a.health for a in alive])) if alive else 0.0
                    ),
                    "has_events": bool(sim.event_system.active_summary(cycle)),
                })
                if cycle in capture_cycles:
                    viz.record_frame(sim, cycle)

            if cycle % heartbeat == 0 or cycle == last_cycle:
                n_alive = sum(1 for a in sim.agents if a.alive)
                mean_health = (
                    sum(a.health for a in sim.agents if a.alive) / n_alive
                    if n_alive else 0.0
                )
                events = [e["type"] for e in sim.event_system.active_summary(cycle)]
                bench_log(
                    logging.INFO,
                    "cycle %3d/%d alive=%d/%d nhs=%.4f health=%.3f "
                    "events=[%s] laws=%d decisions=%s",
                    cycle + 1, config.max_cycles, n_alive, len(sim.agents),
                    sim.normalized_health_score(), mean_health,
                    ",".join(events),
                    sum(1 for l in government.active_laws if l.is_active(cycle)),
                    _decisions_made(government),
                )

        metrics = sim.metrics
        summary = metrics.summary()
        # The fairness/variation witness travels with the RESULTS, not only with
        # the run header.  A replicator receives `benchmark_results/`, not the
        # process that produced it, so the evidence has to survive into
        # final_stats.json and from there into combined_final_stats.csv — which
        # is what makes the fairness and variation assertions re-runnable by
        # someone who did not run the sweep.  It is deliberately NOT in
        # SUMMARY_FIELDS: that tuple
        # drives the per-cell numeric CI aggregation, and a string column there
        # would be a trap for every mean taken over it.
        summary["env_fingerprint"] = env_fingerprint
        _merge_calibration_summary(summary, government, config.max_cycles)
        csv_rows = metrics.to_csv_rows()
        if recorder is not None:
            recorder.set_summary(summary)
            if recorder.errors:
                bench_log(
                    logging.WARNING,
                    "run detail recorder reported %d error(s) — see "
                    "footer.recorder_errors in %s",
                    recorder.errors, run_detail_path(run_dir),
                )

    # Outside the ExitStack: the footer is written and flushed before we return.
    return summary, csv_rows, viz, health_history


# ---------------------------------------------------------------------------
# Per-(government, difficulty) task — executed inside a worker process
# ---------------------------------------------------------------------------

def run_gov_difficulty(
    config: BenchmarkConfig, gov_name: str, difficulty: int
) -> Tuple[List[dict], Optional[List[dict]], List[dict]]:
    """
    Run ``config.k_runs`` runs for (gov_name, difficulty) and write their output.

    Returns (combined_rows, raw_frames, health_history) where raw_frames and
    health_history come from run 1 only and are rendered by the *main process*.
    Matplotlib is deliberately NOT called here — calling it inside a forked
    worker process deadlocks due to its internal threading locks.

    Each run builds its own scenario plan from ``(config, difficulty, k)``; no
    plan crosses the process boundary.  See :func:`build_plan` and
    :func:`_run_one`.
    """
    # Tag every record this worker produces with its cell, so a log interleaving
    # eight concurrent governments stays greppable:
    #     grep '\[ads/D=56' sim_full_*.log
    cell_ctx = f"{gov_name}/D={difficulty}"
    outer_ctx = get_context()
    set_context(cell_ctx)
    try:
        return _run_gov_difficulty_inner(config, gov_name, difficulty, cell_ctx)
    finally:
        set_context(outer_ctx)


def _run_gov_difficulty_inner(
    config: BenchmarkConfig, gov_name: str, difficulty: int, cell_ctx: str
) -> Tuple[List[dict], Optional[List[dict]], List[dict]]:
    """Body of :func:`run_gov_difficulty`, with the log context already set."""
    base_dir = os.path.join(config.output_root, gov_name, str(difficulty))
    os.makedirs(base_dir, exist_ok=True)

    combined_rows: List[dict] = []
    raw_frames: Optional[List[dict]] = None   # frame snapshots from run 1
    raw_health_history: List[dict] = []       # per-cycle scalars from run 1
    failed_runs = 0                           # seeds that raised and were skipped
    cell_t0 = time.time()

    log(f"    ⟶  {gov_name:<12} D={difficulty:3d}  starting {config.k_runs} runs …")

    for k in range(config.k_runs):
        run_dir = os.path.join(base_dir, f"run_{k + 1:02d}")
        os.makedirs(run_dir, exist_ok=True)

        # Capture frame snapshots on the first run so the main process can render them.
        capture_viz = (k == 0)
        run_t0 = time.time()
        set_context(f"{cell_ctx}/k={k + 1}")
        try:
            bench_log(
                logging.INFO,
                "run %d/%d starting  env_seed=%d  gov_seed=%d  dir=%s",
                k + 1, config.k_runs,
                derive_seed(config.base_seed, "env", difficulty, k),
                derive_seed(config.base_seed, "gov", gov_name, difficulty, k),
                run_dir,
            )
            try:
                summary, csv_rows, viz, health_history_data = _run_one(
                    config, gov_name, difficulty, k, capture_viz, run_dir
                )
            except Exception as exc:
                # One bad seed must not abort the whole cell; record and continue
                # so the sweep still produces usable data for the remaining
                # seeds.  The loss is COUNTED, not just logged: a cell that
                # quietly drops seeds would otherwise publish a mean/CI over a
                # smaller n than reported.
                failed_runs += 1
                bench_log(
                    logging.ERROR,
                    "RUN FAILED %s D=%d run %d/%d: %s: %s",
                    gov_name, difficulty, k + 1, config.k_runs,
                    type(exc).__name__, exc,
                )
                # Name the forensic record by path.  The debugging chain becomes
                # exit code -> grep ERROR -> this file -> the failed run's last
                # recorded cycle and the traceback in its footer.  No re-run.
                if config.write_run_detail:
                    bench_log(
                        logging.ERROR,
                        "forensic record -> %s", run_detail_path(run_dir),
                    )
                bench_log(logging.DEBUG, "traceback:\n%s", traceback.format_exc())
                continue

            run_secs = time.time() - run_t0
            nhs = summary.get("normalized_health_score", float("nan"))
            log(
                f"      {gov_name:<12} D={difficulty:3d}  "
                f"run {k+1:>2}/{config.k_runs}  done  "
                f"({run_secs:.1f}s,  health={nhs:.3f})"
            )
        finally:
            set_context(cell_ctx)

        # Save per-cycle health statistics
        if csv_rows:
            health_csv_path = os.path.join(run_dir, "health_stats.csv")
            with open(health_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        # Save final-cycle summary
        final_json_path = os.path.join(run_dir, "final_stats.json")
        with open(final_json_path, "w") as f:
            json.dump(
                {**summary, "run": k + 1, "difficulty": difficulty,
                 "government": gov_name},
                f, indent=2,
            )

        # Stash raw frames and health history from run 1 for the main process
        # to render.  Do NOT call matplotlib here — it is not fork-safe.
        if capture_viz and viz is not None:
            raw_frames = list(viz._frames)
            raw_health_history = health_history_data

        combined_rows.append({
            "government": gov_name,
            "difficulty": difficulty,
            "run": k + 1,
            # Carried explicitly because it is deliberately absent from
            # SUMMARY_FIELDS (a string among numeric metrics).  These rows are
            # what verify_env_fingerprints audits and what
            # combined_final_stats.csv is written from, so the witness has to
            # ride along here or the archive loses it.
            "env_fingerprint": summary.get("env_fingerprint"),
            **{key: val for key, val in summary.items() if key in SUMMARY_FIELDS},
        })

    if failed_runs:
        # WARNING, not INFO, so `grep -E ' (WARNING|ERROR) ' sim_full_*.log` is a
        # complete incident report for the whole sweep.
        bench_log(
            logging.WARNING,
            f"    ✗  {gov_name:<12} D={difficulty:3d}  "
            f"ONLY {len(combined_rows)}/{config.k_runs} runs completed — "
            f"{failed_runs} FAILED and were skipped "
            f"({fmt_duration(time.time() - cell_t0)})"
        )
    else:
        log(
            f"    ✓  {gov_name:<12} D={difficulty:3d}  all {config.k_runs} runs complete "
            f"({fmt_duration(time.time() - cell_t0)})"
        )
    return combined_rows, raw_frames, raw_health_history


# ---------------------------------------------------------------------------
# Visualization (main process only)
# ---------------------------------------------------------------------------

def render_frames(
    config: BenchmarkConfig,
    raw_frames: List[dict],
    viz_dir: str,
    difficulty: int,
    gov_name: str,
    health_history: Optional[List[dict]] = None,
) -> None:
    """
    Render and save snapshot PNGs from raw frame data.

    When health_history is provided, each PNG uses the rich two-panel dashboard
    (grid with legend + health timeline up to that cycle).  Otherwise it falls
    back to the minimal grid-only view.

    Called ONLY from the main process — never from a forked worker — so
    matplotlib's internal locks are safe to use.
    """
    os.makedirs(viz_dir, exist_ok=True)
    frame_by_cycle = {f["cycle"]: f for f in raw_frames}

    viz = SimulationVisualizer()
    for cycle_idx, (fname, cycle_title) in snapshot_points(config.max_cycles).items():
        frame = frame_by_cycle.get(cycle_idx)
        if frame is None:
            continue
        out = os.path.join(viz_dir, f"{fname}.png")
        # The dashboard's point sizes are chosen so they land at 10-12 pt once
        # the PNG is scaled into a 6.5 in LaTeX figure.  That calculation needs
        # the saved width to be exactly DASHBOARD_FIGSIZE[0], so this path must
        # NOT use bbox_inches="tight" — tight bbox resizes the canvas to fit the
        # artists and would make the effective point size vary per frame.  The
        # legend-only fallback below has no such contract and keeps tight bbox.
        save_kwargs: Dict[str, Any] = {"dpi": DASHBOARD_DPI}
        try:
            if health_history:
                fig = viz.render_snapshot_dashboard(frame, health_history)
                # Short: the dashboard is saved without tight bbox, so a
                # suptitle wider than the canvas is clipped rather than
                # accommodated.  The grid panel already reports norm-health, so
                # it is not repeated here.
                fig.suptitle(
                    f"{display_name(gov_name)}  |  Difficulty {difficulty}  |  "
                    f"{cycle_title}",
                    fontsize=FS_SUPTITLE, fontweight="bold",
                )
            else:
                fig = viz._render_frame_dict(frame)
                fig.suptitle(
                    f"{display_name(gov_name)}  Difficulty {difficulty}  "
                    f"{cycle_title}",
                    fontsize=13, fontweight="bold", y=1.01,
                )
                save_kwargs["bbox_inches"] = "tight"
            # DASHBOARD_DPI matters here: the canvas is 9.75 in wide so that
            # point sizes survive the scale-down into a 6.5 in LaTeX figure,
            # and DPI is what puts the pixels back at that size.
            fig.savefig(out, **save_kwargs)
            plt.close(fig)
        except Exception as exc:
            # A failed snapshot is cosmetic — never let it kill a completed sweep.
            log(f"  [WARN] viz save failed ({out}): {exc}")


# ---------------------------------------------------------------------------
# Combined CSV
# ---------------------------------------------------------------------------

def write_combined_csv(all_rows: List[dict], path: str) -> None:
    """Write every (government, difficulty, run) summary row to one CSV."""
    if not all_rows:
        log("\n  [WARN] no rows to write — combined CSV skipped.")
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Union of keys, first-seen order: tolerant of rows with missing/extra fields
    # rather than raising mid-write after a long sweep.
    fieldnames: List[str] = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    log(f"\n  Combined CSV → {path}")


# ---------------------------------------------------------------------------
# Fairness / variation audit
#
# The experiment rests on two claims:
#
#   FAIRNESS   all governments compared at (difficulty, run) faced the SAME
#              environment, so a difference between them is a difference in
#              governance and not in luck of the draw;
#   VARIATION  the runs within a cell faced DIFFERENT environments, so n counts
#              observations rather than replays of one condition.
#
# Both are guaranteed by the purity of build_plan: the same plan is derived
# for every government at (difficulty, run_idx), and a different plan is
# derived for every run_idx within a difficulty.  Purity is a weaker kind of
# guarantee than object
# identity — it holds only as long as nobody threads a stray key into a seed —
# so it is not left as a claim in a docstring.  Every run records the hash of
# the environment it actually ran, and the sweep checks both properties over
# those hashes before it will call itself complete.
# ---------------------------------------------------------------------------

class FingerprintAuditError(AssertionError):
    """
    Raised when the recorded environment fingerprints violate FAIRNESS or
    VARIATION.

    Deliberately fatal.  A sweep whose governments did not face the same
    environment has no comparison in it, and a sweep whose runs all faced the
    same environment has n = 1 per cell however many rows it wrote.  Either way
    the output is not publishable, and the failure mode of a warning is that the
    warning scrolls past and the numbers get used anyway — exactly the failure
    mode this check exists to prevent.

    Subclasses ``AssertionError`` so the intent reads at the catch site, while
    still being a specific type that a test can assert on.
    """


def verify_env_fingerprints(
    all_combined_rows: List[dict],
    k_runs: int,
) -> Dict[str, Any]:
    """
    Assert FAIRNESS and VARIATION over the recorded ``env_fingerprint`` values.

    Parameters
    ----------
    all_combined_rows  Per-run rows carrying ``government``, ``difficulty``,
                       ``run`` (1-based) and ``env_fingerprint``.
    k_runs             The configured runs per cell.  Used for REPORTING only —
                       see the note on the variation check below.

    Returns
    -------
    A summary dict suitable for embedding in ``manifest.json``.

    Raises
    ------
    FingerprintAuditError
        On any violation, after logging every offending coordinate at ERROR so
        the log alone is enough to diagnose it.

    Rows with no fingerprint are counted and reported but not audited: an
    archive that lacks this field entirely should not fail for missing
    something it never recorded.  If NO row carries one, the audit reports
    "not applicable" rather than vacuously passing — a silent pass on an
    empty check is exactly how this kind of guard rots.

    The variation check compares against the number of run indices actually
    RECORDED at a difficulty, not against ``k_runs``.  ``run_gov_difficulty``
    deliberately skips a run that raises so one bad run cannot lose a whole
    cell, so a legitimate partial sweep has fewer runs than ``k_runs`` and
    comparing to the nominal figure would turn an already-reported incomplete
    sample into a spurious fairness failure.  Incomplete samples are the
    ``short_cells`` mechanism's job; this function's job is to prove that the
    runs which DID happen were distinct.  Both counts are reported.
    """
    # (difficulty, run) -> {gov: fingerprint}
    by_coord: Dict[Tuple[Any, Any], Dict[str, str]] = {}
    # difficulty -> {run: fingerprint}  (one arbitrary gov's; fairness makes the
    # choice irrelevant, and fairness is checked first)
    by_difficulty: Dict[Any, Dict[Any, str]] = {}
    n_missing = 0

    for row in all_combined_rows:
        fingerprint = row.get("env_fingerprint")
        if not fingerprint:
            n_missing += 1
            continue
        gov = row.get("government")
        diff = row.get("difficulty")
        run = row.get("run")
        by_coord.setdefault((diff, run), {})[gov] = fingerprint
        by_difficulty.setdefault(diff, {})[run] = fingerprint

    if not by_coord:
        log(
            f"\n  Environment audit: SKIPPED — none of {len(all_combined_rows)} "
            f"row(s) carry env_fingerprint (older dataset without this field?)."
        )
        return {
            "status": "not_applicable",
            "rows_audited": 0,
            "rows_missing_fingerprint": n_missing,
        }

    # --- FAIRNESS: one environment per (difficulty, run) ---------------------
    unfair: List[Tuple[Any, Any, Dict[str, str]]] = [
        (diff, run, per_gov)
        for (diff, run), per_gov in sorted(by_coord.items(), key=repr)
        if len(set(per_gov.values())) != 1
    ]

    # --- VARIATION: one environment per run index, within a difficulty -------
    non_varying: List[Tuple[Any, int, int]] = []
    for diff, per_run in sorted(by_difficulty.items(), key=repr):
        n_runs = len(per_run)
        n_distinct = len(set(per_run.values()))
        if n_distinct != n_runs:
            non_varying.append((diff, n_distinct, n_runs))

    for diff, run, per_gov in unfair:
        bench_log(
            logging.ERROR,
            "FAIRNESS VIOLATION  D=%s run=%s: governments did not share one "
            "environment — %s",
            diff, run,
            ", ".join(f"{g}={f}" for g, f in sorted(per_gov.items())),
        )
    for diff, n_distinct, n_runs in non_varying:
        bench_log(
            logging.ERROR,
            "VARIATION VIOLATION  D=%s: %d distinct environment(s) across %d "
            "recorded run(s) — runs are replicates, not samples",
            diff, n_distinct, n_runs,
        )

    summary: Dict[str, Any] = {
        "status": "pass" if not (unfair or non_varying) else "fail",
        "rows_audited": sum(len(v) for v in by_coord.values()),
        "rows_missing_fingerprint": n_missing,
        "coordinates_checked": len(by_coord),
        "difficulties_checked": len(by_difficulty),
        "k_runs_configured": k_runs,
        "fairness_violations": [
            {"difficulty": diff, "run": run, "fingerprints": per_gov}
            for diff, run, per_gov in unfair
        ],
        "variation_violations": [
            {"difficulty": diff, "distinct_environments": n_distinct,
             "runs_recorded": n_runs}
            for diff, n_distinct, n_runs in non_varying
        ],
    }

    if unfair or non_varying:
        raise FingerprintAuditError(
            f"environment audit FAILED: {len(unfair)} fairness violation(s), "
            f"{len(non_varying)} variation violation(s) — see the ERROR lines "
            f"above.  The dataset is not publishable."
        )

    # Report on success too.  An invariant that is only visible when it breaks
    # cannot be cited in a paper; this line is the positive evidence.
    short = sum(
        1 for per_run in by_difficulty.values() if len(per_run) != k_runs
    )
    log(
        f"\n  Environment audit: PASS."
        f"\n    FAIRNESS  — all governments shared one environment at each of "
        f"{len(by_coord)} (difficulty, run) coordinate(s)."
        f"\n    VARIATION — every recorded run within a difficulty faced a "
        f"distinct environment, across {len(by_difficulty)} difficulty level(s)."
        + (
            f"\n    NOTE: {short} difficulty level(s) recorded fewer than the "
            f"configured {k_runs} runs; variation was checked over what was "
            f"recorded.  See the incomplete-sample warnings above."
            if short else ""
        )
        + (
            f"\n    NOTE: {n_missing} row(s) carried no env_fingerprint and "
            f"were not audited."
            if n_missing else ""
        )
    )
    return summary


def verify_env_fingerprints_from_tree(
    output_root: str, k_runs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Re-run :func:`verify_env_fingerprints` over an archived results directory.

    Exists so the two claims are checkable by someone who has the data but did
    not run the sweep — which is the whole point of recording the fingerprint
    rather than merely asserting fairness in a comment.  Reads
    ``combined_final_stats.csv``; that file is part of every published archive.

    ``k_runs`` defaults to the largest run index present, which is correct for a
    complete archive and reports honestly for an incomplete one.
    """
    csv_path = os.path.join(output_root, "combined_final_stats.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        # DictReader yields strings; the audit groups by value, so difficulty
        # and run must be normalised or "50" and 50 become different cells.
        for key in ("difficulty", "run"):
            try:
                row[key] = int(row[key])
            except (KeyError, TypeError, ValueError):
                pass
    if k_runs is None:
        k_runs = max(
            (r["run"] for r in rows if isinstance(r.get("run"), int)), default=0
        )
    return verify_env_fingerprints(rows, k_runs)


def compute_cell_statistics(
    config: BenchmarkConfig,
    all_combined_rows: List[dict],
    manifest_results: Dict[str, Any],
) -> Tuple[List[dict], Dict[Tuple[str, int], dict]]:
    """
    Compute per-cell CI statistics from per-run aggregate rows.

    Groups all_combined_rows by (government, difficulty), and for each cell,
    computes mean and 95% confidence interval bounds for each metric in
    SUMMARY_FIELDS using the t-distribution.

    Args:
        config: the sweep configuration (for output_root).
        all_combined_rows: list of per-run dicts, each with 'government',
                          'difficulty', and the SUMMARY_FIELDS metrics.
        manifest_results: dict from _run_sweep() with 'short_cells' listing
                         per-cell sample sizes (accounting for lost seeds).

    Returns:
        (all_combined_rows, cell_stats_dict):
        - all_combined_rows: unchanged (not enriched with per-run CI fields)
        - cell_stats_dict: {(gov, difficulty): cell_stats_json_dict}
          keyed by tuple so caller can iterate and write per-cell files.

    The cell_stats_json_dict follows this nested schema:
        {
          "government": str,
          "difficulty": int,
          "n": int (true per-cell sample size from manifest),
          "deterministic": bool (every core metric took ONE value across all
                                 seeds — see below),
          "timestamp": str (UTC when aggregation ran),
          "metrics": {
            "<metric_name>": {
              "mean": float or null,
              "ci95_lower": float or null,
              "ci95_upper": float or null,
              "n": int (count of non-None values for this metric),
              "n_distinct": int (how many DIFFERENT values those n seeds gave),
              "deterministic": bool (n_distinct == 1)
            },
            ...
          }
        }

    Reading ``n`` correctly
    -----------------------
    ``n`` is a count of runs, and each run is a **different
    environment**: the scenario plan (terrain, resources, placements, event
    schedule) is derived per ``(difficulty, run_idx)``, so a cell's ``n`` runs
    sample the environment distribution at that difficulty.  The dispersion a CI
    reports is therefore *dispersion of this regime's performance across
    environments*, not uncertainty about a single environment — a different
    quantity from what a reader may assume, and one the methods section should
    name explicitly.

    All governments share the environment at a given ``(difficulty, run_idx)``
    (audited by :func:`verify_env_fingerprints`), so the design is a textbook
    common-random-numbers experiment and per-run outcomes across governments are
    **paired**.  The unpaired estimators in :func:`_numpy_ttest` and
    :func:`_bootstrap_ci_mean_diff` therefore discard the pairing and overstate
    the standard error of a government-vs-government difference.  Switching them
    to paired estimators is a deliberate follow-up, not an oversight; the
    pairing is recoverable from the archive because ``env_fingerprint`` travels
    with every row.

    If the plan were instead built once per difficulty and replayed for every
    run, ``n`` would count replicates of one condition: most regimes have no
    stochastic decision step, so they would return the identical trajectory
    every time, ``n_distinct`` would be 1, and ``ci95_lower == ci95_upper ==
    mean`` — a zero width that reflects one effective observation, not
    precision.  Because each run instead faces a different environment,
    ``deterministic`` / ``n_distinct`` are expected to be False / > 1 for
    every regime; a cell that still reports ``deterministic`` is a signal
    worth investigating rather than the norm.
    """
    # Build a map of (gov, difficulty) -> true n from manifest_results.short_cells
    short_cell_map: Dict[Tuple[str, int], int] = {}
    for short_cell in manifest_results.get("short_cells", []):
        gov = short_cell["government"]
        diff = short_cell["difficulty"]
        n_recorded = short_cell["runs_recorded"]
        short_cell_map[(gov, diff)] = n_recorded

    # Group rows by (government, difficulty)
    cell_groups: Dict[Tuple[str, int], List[dict]] = {}
    for row in all_combined_rows:
        gov = row.get("government")
        diff = row.get("difficulty")
        if gov and diff is not None:
            key = (gov, diff)
            if key not in cell_groups:
                cell_groups[key] = []
            cell_groups[key].append(row)

    # Compute statistics for each cell
    cell_stats_dict: Dict[Tuple[str, int], dict] = {}
    timestamp = utc_timestamp()

    for (gov, diff), rows in sorted(cell_groups.items()):
        # Determine true n from manifest (or fallback to row count if not in short_cells)
        true_n = short_cell_map.get((gov, diff), len(rows))

        metrics_stats: Dict[str, dict] = {}

        for metric in SUMMARY_FIELDS:
            # Extract values for this metric from all rows in the cell
            values = []
            for row in rows:
                val = row.get(metric)
                if val is not None:
                    # Filter out NaN values
                    try:
                        if not math.isnan(val):
                            values.append(val)
                    except (TypeError, ValueError):
                        # Not a float (e.g., already None or wrong type)
                        pass

            # Compute CI
            if len(values) >= 1:
                mean, lower, upper = ci_95(values)
                # Round to 4 decimals for cleaner JSON
                mean = round(mean, 4) if mean is not None else None
                lower = round(lower, 4) if lower is not None else None
                upper = round(upper, 4) if upper is not None else None
            else:
                mean, lower, upper = None, None, None

            # Effective sample size, reported alongside the nominal one.
            #
            # `n` counts runs that produced a value; each run is a distinct
            # environment, so `n` should also be the number of independent
            # observations.  `n_distinct` is what proves it: if the environment
            # were instead shared across all runs of a difficulty, seven of the
            # eight regimes are deterministic given a fixed environment, so all
            # `n` runs would return one identical value and the cell would
            # carry a single degree of freedom.  When that happens ci95_lower ==
            # ci95_upper == mean, which reads like an infinitely precise
            # estimate rather than "this regime has no run-to-run variance at
            # all".  The guard is kept because it is the only thing that would
            # catch a regression in the run_idx keying from the statistics side.
            #
            # `n_distinct` is counted on the raw float, not the rounded mean,
            # so a genuinely-varying cell whose spread hides below the 4-decimal
            # rounding still reports as varying.
            n_distinct = len(set(values))
            metrics_stats[metric] = {
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n": len(values),
                "n_distinct": n_distinct,
                "deterministic": bool(values) and n_distinct == 1,
            }

        # Cell-level roll-up so a reader does not have to scan every metric to
        # learn whether this cell varied with seed at all.
        core_metrics = [
            m for m in ("normalized_health_score", "final_survival_rate",
                        "final_median_health")
            if m in metrics_stats and metrics_stats[m]["n"] > 0
        ]
        cell_deterministic = bool(core_metrics) and all(
            metrics_stats[m]["deterministic"] for m in core_metrics
        )

        cell_stats_dict[(gov, diff)] = {
            "government": gov,
            "difficulty": diff,
            "n": true_n,
            # True when every core outcome metric took exactly one value across
            # all `n` seeds — i.e. the seeds are replicates of one run, and any
            # dispersion statistic over them (CI, sd, SEM, t-test) is
            # structurally zero rather than small.  Do not quote `n` as a
            # sample size for this cell without also quoting this flag.
            "deterministic": cell_deterministic,
            "timestamp": timestamp,
            "metrics": metrics_stats,
        }

    return all_combined_rows, cell_stats_dict


def write_cell_stats(
    config: BenchmarkConfig,
    cell_stats_dict: Dict[Tuple[str, int], dict],
) -> None:
    """
    Write per-cell CI statistics to JSON files.

    For each (government, difficulty) cell in the dict, creates
    <output_root>/<government>/<difficulty>/cell_stats.json and writes
    the cell_stats dict as pretty-printed JSON.

    Args:
        config: the benchmark configuration (specifies output_root).
        cell_stats_dict: dict from compute_cell_statistics(), keyed by
                        (government, difficulty) tuples.
    """
    total_bytes = 0
    file_count = 0

    for (gov, diff), cell_stats in cell_stats_dict.items():
        cell_dir = os.path.join(config.output_root, gov, str(diff))
        os.makedirs(cell_dir, exist_ok=True)

        cell_stats_path = os.path.join(cell_dir, "cell_stats.json")
        try:
            with open(cell_stats_path, "w") as f:
                json.dump(cell_stats, f, indent=2)
            file_count += 1
            total_bytes += os.path.getsize(cell_stats_path)
        except OSError as exc:
            bench_log(
                logging.ERROR,
                f"  Failed to write cell_stats.json for {gov} D={diff}: {exc}",
            )

    if file_count > 0:
        avg_size = total_bytes / file_count if file_count > 0 else 0
        log(
            f"\n  Cell statistics → {file_count} files, "
            f"total {total_bytes / 1024:.1f} KB, avg {avg_size:.0f} B/cell"
        )

    _log_determinism_summary(config, cell_stats_dict)


def _log_determinism_summary(
    config: BenchmarkConfig,
    cell_stats_dict: Dict[Tuple[str, int], dict],
) -> None:
    """
    Report, per government, how many cells showed no seed-to-seed variance.

    Why this is logged loudly rather than left to be discovered downstream: a
    deterministic cell is indistinguishable from a well-estimated one in every
    artifact that quotes only a mean, and the CI fields actively disguise it
    (they collapse to +-0, which reads as high precision).  Left unnoticed,
    this can invalidate a table of p-values; a line in the sweep log is the
    cheapest place to catch it.

    Because each run faces a *different* environment, a flagged cell means the
    regime's outcome is genuinely invariant to the environment at that
    difficulty — usually total extinction or total survival, where the metric
    saturates — or that the run_idx keying has regressed.  The second
    possibility is separately and loudly excluded by
    :func:`verify_env_fingerprints`, which fails the sweep outright rather than
    logging.  This remains a REPORTING function: determinism is a property to be
    measured, never one to be papered over with injected noise.
    """
    by_gov: Dict[str, List[int]] = {}
    for (gov, _diff), cell_stats in cell_stats_dict.items():
        by_gov.setdefault(gov, []).append(1 if cell_stats.get("deterministic") else 0)

    if not by_gov:
        return

    flagged = {g: v for g, v in by_gov.items() if sum(v)}
    if not flagged:
        log("\n  Seed variance: every cell varied across seeds.")
        return

    log(f"\n  Seed variance check (k_runs={config.k_runs}):")
    for gov in sorted(by_gov):
        det, total = sum(by_gov[gov]), len(by_gov[gov])
        marker = "  <-- no seed variance" if det == total else ""
        log(f"    {gov:22} {det:3d}/{total:<3d} cells deterministic{marker}")
    log(
        "    NOTE: a deterministic cell has ONE effective observation, not "
        f"{config.k_runs}.\n"
        "    Its CI is zero-width by construction and any t-test against it "
        "is undefined\n"
        "    (reported as 'Undefined' in the pairwise CSVs, not as "
        "'significant')."
    )


# ---------------------------------------------------------------------------
# Statistics helpers
#
# t_critical_95 / ci_95 are consumed by compute_cell_statistics (which writes
# cell_stats.json) and by nothing in the plotting layer.  They do not feed
# the median + IQR box plus mean +/- 95% CI figure family, because they
# compute the wrong statistic for it:
#
#   ci_95 is a normal-theory t-interval on the MEAN.
#   make_median_ci_plots draws an interval on the MEDIAN.
#
# Those are different estimands, they are not interchangeable, and a figure
# whose boxes show a median while its band shows the uncertainty of a mean
# would be describing two different quantities in one glyph.  The median
# interval therefore has its own estimator — see :func:`bootstrap_median_ci`.
# Do not "simplify" the two into one helper.
# ---------------------------------------------------------------------------

def t_critical_95(df: int) -> float:
    """
    Approximate two-tailed t critical value for a 95% CI.
    Uses a lookup table for small df and 1.96 (z) for df >= 30.
    """
    _TABLE = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        20: 2.086, 25: 2.060, 29: 2.045,
    }
    if df in _TABLE:
        return _TABLE[df]
    if df >= 30:
        return 1.960
    # Linear interpolation between nearest table entries
    keys = sorted(k for k in _TABLE if k <= df)
    if not keys:
        return 1.96
    lo = keys[-1]
    hi_keys = [k for k in _TABLE if k > df]
    if not hi_keys:
        return _TABLE[lo]
    hi = min(hi_keys)
    frac = (df - lo) / (hi - lo)
    return _TABLE[lo] + frac * (_TABLE[hi] - _TABLE[lo])


def ci_95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) using the t-distribution for small K."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(n)
    margin = t_critical_95(n - 1) * se
    return mean, mean - margin, mean + margin


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on the MEDIAN
#
# This section covers the `*_median_ci.png` figure family.  Deliberately NOT
# built on ci_95 (see the section note above): that is a t-interval on the
# mean, this is a percentile bootstrap on the median.
# ---------------------------------------------------------------------------

#: Resamples per cell.  10,000 is the number the user specified, and it is also
#: where the percentile estimate stops moving materially: the Monte-Carlo
#: standard error of a 2.5% percentile at B resamples falls as 1/sqrt(B), so
#: going to 100,000 buys ~3x less MC noise for 10x the time, on a quantity whose
#: real uncertainty is dominated by n=10 runs per cell, not by B.
_MEDIAN_CI_N_BOOT = 10_000

#: Two-sided coverage.  Named rather than inlined so the sidecar text and the
#: percentile arithmetic cannot disagree about what "95%" meant.
_MEDIAN_CI_LEVEL = 0.95

#: Below this many observations a percentile bootstrap on the median is not
#: reporting sampling uncertainty, it is reporting the arrangement of a handful
#: of points: the interval comes back as exactly [min, max] no matter what the
#: data are — a band whose width is a property of n, not of the cell.  Drawing
#: it would be a lie with error bars on it.
#:
#: WHY SIX.  The percentile endpoints are quantiles of the bootstrap median's
#: distribution, and that distribution is discrete with atoms on the order
#: statistics of the sample.  Whenever
#:
#:     P(bootstrap median == min)  >  alpha/2   ( = 2.5% at the 95% level)
#:
#: the 2.5th percentile cannot move off the smallest observation, because more
#: than 2.5% of the mass already sits there.  The same argument mirrors at the
#: top.  So the interval is pinned to [min, max] for every n where that mass
#: exceeds 2.5%.  Exact enumeration of P(median == min) over all n**n resample
#: index vectors:
#:
#:     n = 2 → 25.000%   n = 3 → 25.926%   n = 4 →  5.078%
#:     n = 5 →  5.792%   n = 6 →  0.870%   n = 7 →  1.015%
#:
#: n = 2, 3, 4 and 5 are all above 2.5%; n = 6 is the first that is not.  Note
#: the sequence is NOT monotone — it alternates by parity, because at even n the
#: median is the midpoint of the two middle order statistics and "median == min"
#: additionally requires *both* of them to be the minimum.  So it is not enough
#: to find the first n below 2.5% and stop: the binding case above six is the
#: odd neighbour n = 7 at 1.015%, and both parity subsequences decrease from
#: there, so 2.5% is never re-crossed.  Six is therefore the true threshold and
#: not merely the first passing value.
#:
#: Confirmed empirically against this function: over 300 random datasets per n
#: with distinct seeds, 0/300 intervals were narrower than [min, max] at
#: n = 2, 3, 4 and 5, and 300/300 were narrower at n >= 6.
#:
#: Raising this threshold only ever moves cells INTO ``insufficient_n``, so it
#: is strictly more conservative and cannot turn an honest band into a false
#: one.
_MEDIAN_CI_MIN_N = 6

#: Status vocabulary for :class:`MedianCI`.  Every one of these is a distinct
#: honest statement about a cell; none of them is "pretend there is a band".
MEDIAN_CI_STATUS_OK = "ok"
MEDIAN_CI_STATUS_EMPTY = "empty"
MEDIAN_CI_STATUS_INSUFFICIENT_N = "insufficient_n"
MEDIAN_CI_STATUS_CONSTANT = "constant"
MEDIAN_CI_STATUS_ZERO_WIDTH = "degenerate_zero_width"


@dataclass(frozen=True)
class MedianCI:
    """One cell's median and its bootstrap interval, plus *why* if there isn't one.

    The status field is the point of this type.  A plain
    ``(median, lower, upper)`` tuple cannot distinguish "the interval is
    genuinely this narrow" from "there were two runs" from "every run returned
    the same number", and all three collapse to ``lower == upper``.  A reader
    of the figure is entitled to know which of those they are looking at, so
    the distinction is carried in the data rather than reconstructed by
    whoever draws it.

    ``lower``/``upper`` are populated for every non-empty status — including the
    degenerate ones, where they equal the median — so a caller that wants the
    numbers can have them.  :attr:`drawable` is the only thing that decides
    whether a band is rendered.
    """

    median: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    n: int
    status: str

    @property
    def drawable(self) -> bool:
        """True only when a band would convey sampling uncertainty."""
        return self.status == MEDIAN_CI_STATUS_OK

    @property
    def half_width(self) -> Optional[float]:
        """``(upper - lower) / 2``, or None when there is no interval at all."""
        if self.lower is None or self.upper is None:
            return None
        return (self.upper - self.lower) / 2.0

    def describe(self) -> str:
        """A short phrase naming what this cell actually is, for alt text."""
        if self.status == MEDIAN_CI_STATUS_EMPTY:
            return "no runs reported this metric"
        if self.status == MEDIAN_CI_STATUS_INSUFFICIENT_N:
            return (f"n = {self.n}, below the {_MEDIAN_CI_MIN_N}-run minimum for "
                    f"a median bootstrap; median plotted without an interval")
        if self.status == MEDIAN_CI_STATUS_CONSTANT:
            return (f"all {self.n} runs returned the identical value, so the "
                    f"interval is zero-width by construction, not by precision")
        if self.status == MEDIAN_CI_STATUS_ZERO_WIDTH:
            return (f"n = {self.n}, values vary, but over 95% of the bootstrap "
                    f"median mass sits on a single order statistic, so the "
                    f"percentile interval collapses to a point")
        return f"n = {self.n}"


def bootstrap_median_ci(
    values: Sequence[float],
    seed: int,
    n_boot: int = _MEDIAN_CI_N_BOOT,
    ci_level: float = _MEDIAN_CI_LEVEL,
) -> MedianCI:
    """Percentile-bootstrap confidence interval for the MEDIAN of *values*.

    Why a bootstrap rather than a closed form
    -----------------------------------------
    There is no small-sample normal theory for the median the way there is for
    the mean.  The order-statistic (binomial) interval exists but at n=10 its
    achievable coverage levels are coarse (the tightest interval with at least
    95% coverage is [x(2), x(9)], ~97.9%), and it cannot interpolate.  The
    percentile bootstrap gives a 95% interval at any n and makes no distributional
    assumption, which matters here: the per-cell outcome distributions in this
    model are routinely bounded, skewed, and piled up against 0.0 or 1.0.

    Known and accepted limitation: the percentile bootstrap of a median is
    *discrete* — every resampled median is an order statistic of *values* (or a
    midpoint of two, at even n), so at n=10 the interval endpoints can only land
    on about ten distinct values.  It therefore under-resolves rather than
    misstates, and it degrades gracefully into the explicitly-flagged
    :data:`MEDIAN_CI_STATUS_ZERO_WIDTH` case instead of silently narrowing.

    Determinism
    -----------
    *seed* is required, not defaulted.  Every figure in this tree must be
    byte-reproducible from a fresh process (see
    :func:`_derive_bootstrap_seed`), and a default seed is how a caller
    accidentally ships a figure that cannot be reproduced.  Production callers
    derive it with ``derive_seed(base_seed, "plot.bootstrap", gov, difficulty)``.

    Degenerate cells are reported, never smoothed
    ---------------------------------------------
    Four non-``ok`` outcomes are possible and each gets its own status; see
    :class:`MedianCI`.  In particular a cell where every run returned the same
    number is NOT the same thing as a cell estimated to infinite precision —
    a zero-width band must never be read as the latter.  See
    :func:`_log_determinism_summary`.
    """
    clean: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            clean.append(fv)

    n = len(clean)
    if n == 0:
        return MedianCI(None, None, None, 0, MEDIAN_CI_STATUS_EMPTY)

    arr = np.asarray(clean, dtype=float)
    median = float(np.median(arr))

    if n < _MEDIAN_CI_MIN_N:
        return MedianCI(median, median, median, n,
                        MEDIAN_CI_STATUS_INSUFFICIENT_N)

    # A constant cell is detected on the raw floats before any resampling.
    # Resampling would reach the same answer, but the *reason* would be lost:
    # "the bootstrap returned zero width" and "there is only one value in this
    # cell" need different sentences in the sidecar.
    if float(arr.min()) == float(arr.max()):
        return MedianCI(median, median, median, n, MEDIAN_CI_STATUS_CONSTANT)

    rng = np.random.default_rng(seed)
    # Single vectorised draw rather than a Python loop over n_boot: this is
    # called once per (metric, government, difficulty) — 672 times for the
    # production grid — and a 10,000-iteration Python loop at that call count
    # would dominate the plotting stage.  Matches _bootstrap_ci_mean_diff.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(arr[idx], axis=1)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_medians, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))

    if lower == upper:
        return MedianCI(median, lower, upper, n, MEDIAN_CI_STATUS_ZERO_WIDTH)
    return MedianCI(median, lower, upper, n, MEDIAN_CI_STATUS_OK)


# ---------------------------------------------------------------------------
# Accessibility utilities for plots (WCAG AA compliance)
# ---------------------------------------------------------------------------

def check_color_contrast(color_hex: str, bg_hex: str = "#FFFFFF") -> float:
    """
    Calculate WCAG contrast ratio between two colors.

    Returns the contrast ratio (1.0 to 21.0).
    WCAG AA requires 3:1 for non-text, 4.5:1 for text.
    WCAG AAA requires 7:1 for non-text, 7:1 for text.

    Parameters:
        color_hex: Foreground color as '#RRGGBB'
        bg_hex: Background color as '#RRGGBB' (default white)

    Returns:
        Contrast ratio as float
    """
    def relative_luminance(hex_color: str) -> float:
        """Calculate relative luminance per WCAG formula."""
        # Convert hex to RGB
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
        # Apply sRGB gamma
        def adjust(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)

    l1 = relative_luminance(color_hex)
    l2 = relative_luminance(bg_hex)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def validate_color_palette(colors: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate a color palette for WCAG AA accessibility.

    Returns a dict with:
        'wcag_aa_compliant': bool (all colors pass 3:1 contrast)
        'wcag_aaa_compliant': bool (all colors pass 7:1 contrast)
        'failures': list of (color_key, contrast_ratio) tuples that fail
    """
    failures = []
    for key, color in colors.items():
        ratio = check_color_contrast(color)
        if ratio < 3.0:  # WCAG AA minimum for non-text
            failures.append((key, ratio))

    return {
        'wcag_aa_compliant': len(failures) == 0,
        'wcag_aaa_compliant': all(ratio >= 7.0 for _, ratio in
                                  [(k, check_color_contrast(colors[k]))
                                   for k in colors]),
        'failures': failures,
    }


def generate_alt_text(
    metric_name: str,
    governments: Sequence[str],
    data_summary: Dict[str, Any],
    max_words: int = 100,
) -> str:
    """
    Generate accessibility alt text for a metric plot.

    The text must describe what a sighted reader actually sees: a Tukey IQR
    boxplot per government at every difficulty level, with a line joining the
    medians.  Two rules apply:

      1. Never mention confidence intervals or CI.  The figures do not draw
         them (see make_plots) and the manuscript's captions are required to
         say "Tukey IQR boxplot" instead.
      2. Never assert which government performs "best".  Hard-coding the
         first government in the list as "the best performer" would
         misdescribe the figure whenever it is not actually best, and "best"
         is metric-dependent anyway (for the Gini panel, lower is better).
         We therefore report the highest and lowest medians neutrally and let
         the reader judge.

    Parameters:
        metric_name: Human-readable metric name (e.g., "Normalized Health Score")
        governments: Display names of the governments shown, in plot order
        data_summary: Accurate summary of the plotted data.  Recognised keys:
            'n_difficulties', 'difficulty_min', 'difficulty_max', 'k_runs',
            'value_min', 'value_max' (range of plotted medians),
            'edge_difficulty', 'highest_gov', 'highest_value',
            'lowest_gov', 'lowest_value' (medians at the hardest level).
        max_words: Maximum words in alt text (default 100, JPART guideline)

    Returns:
        Alt text string suitable for a sidecar file or PNG metadata
    """
    gov_list = ", ".join(governments)

    # Clause list, most important first.  If the text runs long we drop whole
    # trailing clauses rather than truncating mid-sentence, which would leave a
    # dangling fragment in a screen reader.
    clauses: List[str] = [
        f"Tukey IQR boxplot: {metric_name} for {len(governments)} governments "
        f"({gov_list})."
    ]

    n_diffs = data_summary.get("n_difficulties")
    d_min = data_summary.get("difficulty_min")
    d_max = data_summary.get("difficulty_max")
    k_runs = data_summary.get("k_runs")
    if n_diffs and d_min is not None and d_max is not None:
        span = f"Difficulty levels {d_min} to {d_max} ({n_diffs} levels)"
        if k_runs:
            span += f", {k_runs} runs per government per level"
        clauses.append(span + ".")

    clauses.append(
        "Each government has a box spanning the interquartile range with "
        "1.5x IQR whiskers at every difficulty level, and a line joining its "
        "medians across levels."
    )

    # Ties are compared on the *rendered* text, not on the raw floats or the
    # government names.  max()/min() over an all-equal series still return two
    # different governments, which would otherwise produce the self-
    # contradictory "highest is Republic (0.00) and the lowest is ADS (0.00)".
    v_min = data_summary.get("value_min")
    v_max = data_summary.get("value_max")
    if v_min is not None and v_max is not None:
        if f"{v_min:.2f}" == f"{v_max:.2f}":
            clauses.append(f"All medians are {v_min:.2f}.")
        else:
            clauses.append(f"Medians span {v_min:.2f} to {v_max:.2f}.")

    edge = data_summary.get("edge_difficulty")
    hi_gov = data_summary.get("highest_gov")
    lo_gov = data_summary.get("lowest_gov")
    hi_val = data_summary.get("highest_value")
    lo_val = data_summary.get("lowest_value")
    if edge is not None and hi_gov and lo_gov and hi_val is not None and lo_val is not None:
        if f"{hi_val:.2f}" == f"{lo_val:.2f}":
            clauses.append(
                f"At difficulty {edge} every government has a median of {hi_val:.2f}."
            )
        else:
            clauses.append(
                f"At difficulty {edge} the highest median is {hi_gov} ({hi_val:.2f}) "
                f"and the lowest is {lo_gov} ({lo_val:.2f})."
            )

    return _clauses_to_alt_text(clauses, max_words)


def _clauses_to_alt_text(clauses: Sequence[str], max_words: int) -> str:
    """
    Join *clauses* into a single alt-text string inside a word budget.

    Shared by :func:`generate_alt_text` and the ADS calibration figures so
    both enforce the same budget the same way.  The budget rule is a JPART
    accessibility guideline, not a cosmetic preference, so having two
    implementations of it would eventually mean having two budgets.
    """
    clauses = list(clauses)

    # Drop trailing clauses until the whole text fits the word budget.  The
    # first clause always survives: it alone identifies the figure.
    while len(clauses) > 1 and len(" ".join(clauses).split()) > max_words:
        clauses.pop()

    alt = " ".join(clauses)

    # Backstop: even the lead clause can exceed the budget if the government
    # list is very long.  Truncate on a word boundary and close the sentence.
    words = alt.split()
    if len(words) > max_words:
        words = words[:max_words]
        words[-1] = words[-1].rstrip(".,") + "."
        alt = " ".join(words)

    return alt


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def load_cell_stats(output_root: str, government: str, difficulty: int) -> Optional[Dict[str, Any]]:
    """
    Load the per-cell statistics (mean, 95% CI bounds) for a government/difficulty pair.

    Cell stats are computed during aggregation and stored in JSON files at:
      <output_root>/<government>/<difficulty>/cell_stats.json

    NOT USED BY make_plots().  This reader is intentionally retained even though
    it currently has no caller: those figures are IQR-only by decision (see
    make_plots' docstring), but per-cell CI data is still written during
    aggregation, so this is the supported way to read it back for any future
    analysis.  Do not wire it back into make_plots.

    Returns:
      A dict with structure: {metric_name: {mean, ci95_lower, ci95_upper, n}}
      Or None if the file doesn't exist or cannot be parsed.
    """
    cell_stats_path = os.path.join(
        output_root, str(government), str(difficulty), "cell_stats.json"
    )
    if not os.path.exists(cell_stats_path):
        return None

    try:
        with open(cell_stats_path, "r") as f:
            data = json.load(f)
        # Extract the metrics dict, which contains per-metric CI data
        return data.get("metrics", {})
    except (json.JSONDecodeError, IOError, KeyError) as exc:
        # Silently skip files that can't be read; plots still render without CI
        return None


def _style_difficulty_axes(
    ax: Any,
    sorted_diffs: Sequence[int],
    x_range: float,
    title: str,
    ylabel: str,
) -> None:
    """
    Apply the house axis treatment for a difficulty-on-x figure.

    Extracted from :func:`make_plots` alongside :func:`_draw_box_series`.
    Every value here — tick sizes, the 5% x-padding, the y-only
    dashed grid at alpha 0.25, the 11pt bold title — is the one
    :func:`make_plots` already used, and the call order is preserved, so the
    extraction is renderer-identical rather than merely equivalent.

    *title* arrives fully composed.  The caller owns it because the two figure
    families say different things in it, but both are subject to the same
    standing constraint: **the title describes only what is actually drawn.**
    A title that claims something the figure does not draw (e.g. an overlay
    that is not rendered) propagates a false claim into the manuscript's
    captions.  Do not assert anything in a title that a reader cannot see in
    the figure.

    **Frozen by `test_plot_regression.py`'s golden-AST IQR guard** (byte-
    identical to its extraction baseline) — do not change this function's pad
    or any other behaviour to fix a calibration-figure-only defect.
    `calibration_half_split.png`'s D1/D100 box clipping is deliberately fixed
    at the calibration call sites instead, via
    :func:`_widen_axis_for_drawn_boxes` called immediately after this
    function — see that function's docstring for why. If a genuine future
    defect requires changing this function's actual pad behaviour for every
    caller, that is a deliberate change to the guarded contract itself
    (update `_GOLDEN_IQR_AST` and the baseline file's status explicitly), not
    a side effect of a calibration fix.
    """
    # Proportional x-axis — ticks at actual difficulty values
    ax.set_xticks(list(sorted_diffs))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    pad = x_range * 0.05
    ax.set_xlim(sorted_diffs[0] - pad, sorted_diffs[-1] + pad)
    ax.set_xlabel("Difficulty Level", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")


def _widen_axis_for_drawn_boxes(
    ax: Any, sorted_diffs: Sequence[int], x_range: float,
) -> None:
    """
    Widen ``ax``'s x-limits, just set by :func:`_style_difficulty_axes`, so
    every box-plot patch already drawn on it actually fits.

    **Why this is a separate function rather than a change inside
    ``_style_difficulty_axes`` itself.** `_style_difficulty_axes` is one of
    the 18 keys `test_plot_regression.py`'s GROUP 1 guard hashes and
    cross-checks against a baseline file byte-for-byte, specifically so the
    plain Tukey IQR figures (`make_plots`, `PLOT_STATS`) can never silently
    change shape. Editing the shared function's pad formula to fix a
    calibration-only defect would trip that guard for every caller,
    including ones that never had the bug — and
    ``calibration_half_split.png``'s own single-government case never needs
    more than the existing 5% pad (its 2-series half-extent, 1.25x box_width,
    is always comfortably inside 2.0x box_width), so the plain-IQR callers
    have no reason to change at all. Confining the fix to a second call,
    made only by the calibration figures right after `_style_difficulty_axes`
    returns, keeps the guarded function's golden hash and baseline match
    intact while still closing the defect where it actually occurs.

    Derived from ``ax.patches`` — the box-plot rectangles already drawn by
    :func:`_draw_box_series` — rather than from a series-count or offset
    parameter a caller has to remember to pass in and keep in sync: a
    box-plot box (``patch_artist=True``) is the only artifact type these
    figures draw with genuine horizontal data-space extent beyond its own
    tick position, so scanning what is actually on the axes makes a future
    series-count change (a third government, a third half, anything)
    structurally unable to under-pad the axis again — there is no separate
    number left to fall behind, only what was actually drawn.

    A small margin (1% of ``x_range``) is added beyond whatever the boxes
    themselves require, so an outer box's own outline stroke does not sit
    exactly flush on the axis spine — at exact equality, half the stroke
    width would render outside the clipped drawing area and look faintly cut
    even though the box's data extent is not actually clipped.

    No-op (limits only ever widen, never shrink) when every drawn box already
    fits inside what `_style_difficulty_axes` set — which is every existing
    2-series calibration figure, so this changes nothing for them.
    """
    lo, hi = ax.get_xlim()
    margin = x_range * 0.01
    for patch in ax.patches:
        extent = patch.get_path().get_extents()
        if extent.x0 - margin < lo:
            lo = extent.x0 - margin
        if extent.x1 + margin > hi:
            hi = extent.x1 + margin
    ax.set_xlim(lo, hi)


def _draw_box_series(
    ax: Any,
    points: Sequence[Tuple[float, Sequence[float]]],
    color: str,
    box_width: float,
    linestyle: Any = "-",
    label: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    Draw one Tukey IQR series: a box per x-position plus a median trend line.

    Lifted verbatim out of :func:`make_plots`'s inner loop for the ADS
    calibration figure.  The extraction exists so the two
    figure families cannot drift apart visually: box colour, alpha, whisker
    dash, cap weight, flier glyph, median weight and z-ordering are stated once
    here rather than twice.  Changing the house box style means changing this
    function, and every figure in the output tree follows.

    *points* is an ordered sequence of ``(x, values)``.  The caller owns the
    x-positions — it passes the difficulty level directly for a single series,
    or a pre-offset position for side-by-side series — so this function never
    does arithmetic on x and cannot silently shift a figure away from its
    intended integer positions.

    Positions with no values are skipped entirely (no box, no point on the
    trend line); a position with exactly one value gets a marker rather than a
    degenerate box, because a boxplot of n=1 draws a box of zero height that
    reads as a measurement rather than as a single sample.

    Returns ``(median_xs, median_ys)`` — the trend-line vertices, so a caller
    that needs to describe what it drew (e.g. for alt text) derives that from
    the drawing rather than recomputing it.
    """
    median_xs: List[float] = []
    median_ys: List[float] = []

    for x, vals in points:
        if not vals:
            continue

        if len(vals) == 1:
            # Single run: just draw a marker at that value
            ax.plot(x, vals[0], marker="o", color=color,
                    markersize=5, zorder=4)
            median_xs.append(x)
            median_ys.append(float(vals[0]))
        else:
            arr = sorted(vals)
            ax.boxplot(
                arr,
                positions=[x],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                zorder=3,
                boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.2,
                              edgecolor=color),
                medianprops=dict(color=color, linewidth=2.0),
                whiskerprops=dict(color=color, linewidth=1.2,
                                  linestyle="--"),
                capprops=dict(color=color, linewidth=1.5),
                flierprops=dict(marker="x", color=color, markersize=4,
                                markeredgewidth=1.0, alpha=0.6),
            )
            median_xs.append(x)
            median_ys.append(float(np.median(arr)))

    # Connecting line through medians (with color-blind safe line style)
    if len(median_xs) > 1:
        ax.plot(median_xs, median_ys, color=color, linewidth=1.5,
                linestyle=linestyle, alpha=0.75, zorder=2, label=label)

    return median_xs, median_ys


def make_plots(config: BenchmarkConfig, all_rows: List[dict], plots_dir: str) -> None:
    """
    Create one PNG per final statistic.

    At each difficulty level each government gets a box plot (IQR box + whiskers
    + median line) drawn at the difficulty's x-position.  A thin line connects the
    medians across difficulty levels to make trends easy to follow.

    These figures are deliberately IQR-only; no CI overlay is drawn.  At
    k=100 the CI on a mean is roughly two orders of magnitude narrower than
    the IQR, so such a band would collapse into the median trend line and not
    be perceptible in the rendered PNG — yet a title claiming "shaded areas =
    95% CI" would assert something the reader could not verify.  The
    manuscript's figure captions say "Tukey IQR boxplot" and must never say
    "confidence interval" or "CI"; these plots are the source of those
    figures, so they carry the same constraint.

    The underlying ci95_lower / ci95_upper values are still computed and
    written to cell_stats.json during aggregation — only the visual overlay
    is absent.
    """
    if not all_rows:
        log("  [WARN] no rows to plot — plots skipped.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    governments = list(config.governments)
    sorted_diffs = sorted(config.difficulties)

    # Organise data: {stat -> {gov -> {diff -> [values]}}}
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        stat: {g: {d: [] for d in sorted_diffs} for g in governments}
        for stat, _, _ in PLOT_STATS
    }
    known_govs = set(governments)
    for row in all_rows:
        gov = row.get("government")
        if gov not in known_govs:
            continue
        try:
            diff = int(row["difficulty"])
        except (KeyError, TypeError, ValueError):
            continue
        for stat, _, _ in PLOT_STATS:
            val = row.get(stat)
            if val is None:
                continue
            try:
                data[stat][gov][diff].append(float(val))
            except (KeyError, TypeError, ValueError):
                pass

    # All government boxes sit at the exact difficulty value on a proportional
    # x-axis — overlapping is intentional and acceptable.  box_width is in
    # difficulty-space units; ~1% of the full range keeps boxes thin but visible.
    x_range = max(sorted_diffs) - min(sorted_diffs) if len(sorted_diffs) > 1 else 10
    box_width = x_range * 0.010

    for stat, title, ylabel in PLOT_STATS:
        fig, ax = plt.subplots(figsize=(16, 7))

        for gov in governments:
            # One call per government, drawing that government's boxes at every
            # difficulty followed by its median trend line — the same
            # boxes-then-line interleaving the inlined version produced, which
            # the figures' z-ordering depends on.
            _draw_box_series(
                ax,
                [(diff, data[stat][gov][diff]) for diff in sorted_diffs],
                color=GOV_COLORS.get(gov, "#333333"),
                box_width=box_width,
                linestyle=GOV_LINESTYLES.get(gov, "-"),
                label=gov,
            )

        # The title describes only what is actually drawn: a Tukey IQR boxplot
        # and a median trend line.  Do not reintroduce a CI note here.
        _style_difficulty_axes(
            ax, sorted_diffs, x_range,
            title=f"{title} by Difficulty Level",
            ylabel=ylabel,
        )

        # Legend entries carry both the colour and the line style, so the
        # series remain distinguishable without relying on hue alone.
        legend_handles = []
        for g in governments:
            color = GOV_COLORS.get(g, "#333333")
            linestyle = GOV_LINESTYLES.get(g, "-")
            # Create a line element showing both color and style
            line = plt.Line2D(
                [0], [0],
                color=color,
                linewidth=2.5,
                linestyle=linestyle,
                marker="o",
                markersize=6,
                label=GOV_DISPLAY.get(g, g),
            )
            legend_handles.append(line)

        ax.legend(
            handles=legend_handles,
            title="Government",
            title_fontsize=10,
            fontsize=9,
            loc="best",
            framealpha=0.88,
        )

        fig.tight_layout()

        # Save figure and generate alt text
        out_path = os.path.join(plots_dir, f"{stat}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

        # Generate and save alt text for accessibility.  Every number below is
        # derived from the same `data` dict the figure was drawn from, so the
        # sidecar cannot drift from the picture it describes.
        all_medians = [
            float(np.median(data[stat][g][d]))
            for g in governments
            for d in sorted_diffs
            if data[stat][g][d]
        ]
        data_summary: Dict[str, Any] = {
            'n_difficulties': len(sorted_diffs),
            'difficulty_min': sorted_diffs[0],
            'difficulty_max': sorted_diffs[-1],
            'k_runs': config.k_runs,
        }
        if all_medians:
            data_summary['value_min'] = min(all_medians)
            data_summary['value_max'] = max(all_medians)

        # Highest / lowest median at the hardest difficulty level that actually
        # has data, reported neutrally — "best" is metric-dependent (lower Gini
        # is better, higher health is better), so the alt text does not judge.
        edge_medians = [
            (float(np.median(data[stat][g][sorted_diffs[-1]])), g)
            for g in governments
            if data[stat][g][sorted_diffs[-1]]
        ]
        if edge_medians:
            hi_val, hi_gov = max(edge_medians)
            lo_val, lo_gov = min(edge_medians)
            data_summary.update({
                'edge_difficulty': sorted_diffs[-1],
                'highest_gov': GOV_DISPLAY.get(hi_gov, hi_gov),
                'highest_value': hi_val,
                'lowest_gov': GOV_DISPLAY.get(lo_gov, lo_gov),
                'lowest_value': lo_val,
            })

        alt_text = generate_alt_text(
            title,
            [GOV_DISPLAY.get(g, g) for g in governments],
            data_summary,
        )

        # Save alt text to companion .txt file for guaranteed accessibility
        # PNG metadata embedding is fragile across readers; text files are reliable
        alt_text_path = out_path.replace('.png', '_alt_text.txt')
        with open(alt_text_path, 'w') as f:
            f.write(f"Figure: {title}\n")
            f.write(f"Alt text (WCAG 2.1 AA compliant, max 100 words):\n{alt_text}\n\n")
            f.write(f"Figure statistics:\n")
            f.write(f"  Metric: {stat}\n")
            f.write(f"  Governments: {len(governments)}\n")
            f.write(f"  Difficulty levels: {len(sorted_diffs)}\n")
            # Grid/agent/cycle scale is recorded here in the alt-text sidecar
            # rather than in the figure title (titles are kept bare),
            # matching the `Scale:` line the calibration sidecars carry.
            f.write(f"  Scale: grid {config.grid_size}×{config.grid_size}, "
                    f"{config.n_agents} agents, {config.max_cycles} cycles\n")
            f.write("\n")
            f.write(f"Color palette (WCAG AA accessible):\n")
            for g in governments:
                color = GOV_COLORS.get(g, "#333333")
                linestyle = GOV_LINESTYLES.get(g, "-")
                f.write(f"  {GOV_DISPLAY.get(g, g):15} | Color: {color} | Line style: {linestyle}\n")

        plt.close(fig)
        log(f"  Plot saved → {out_path}")
        log(f"  Alt text saved → {alt_text_path}")


# ---------------------------------------------------------------------------
# Median + bootstrap-CI figures  (`*_median_ci.png`)
#
# A SECOND, PARALLEL figure family.  It does not replace the Tukey IQR family
# above and does not share a single line of its drawing code.
#
# WHY IT IS A SEPARATE FAMILY RATHER THAN AN OVERLAY.
# An overlaid CI band on the IQR figures would put two dispersion glyphs in one
# axes, one showing the spread of the sample and one showing the uncertainty of
# a point estimate, which readers reliably conflate.  Two separate figure sets
# let the choice between them be made on the rendered evidence.  So:
#
#   *.png            unchanged Tukey IQR.  Captions must never say "CI".
#   *_median_ci.png  median + 95% bootstrap CI.  Captions SHOULD say CI,
#                    because that is the figure's actual, verifiable content —
#                    the constraint was never about the word, it was about
#                    claiming a band the reader could not see.
#
# WHY make_plots' BINNING LOOP IS DUPLICATED BELOW RATHER THAN EXTRACTED.
# The binding requirement on this figure family is that the IQR figures stay
# provably content-unchanged.  "Provably" is much cheaper to establish for a
# function nobody touches than for one that was refactored into a shared
# helper, however faithful the extraction.  _bin_rows_by_stat is therefore a
# separate function used only by this family, and make_plots keeps its inline
# loop.  If the two ever need to diverge, they can; if a future change makes
# them genuinely identical, merge them then.
# ---------------------------------------------------------------------------

#: Total x-span, as a fraction of the difficulty range, over which the
#: per-government error bars are fanned out around each difficulty tick.
#:
#: The IQR figures overlay every government at the exact difficulty value and
#: accept the overlap — a box has width, so eight of them at one x still read as
#: eight boxes.  An error bar is a *line*; eight collinear lines at one x are one
#: line.  So this family fans them.  At the production range (1..100) the total
#: fan is ~2.2 difficulty units, i.e. ~0.3 units between adjacent governments,
#: against a 5-unit tick spacing and a 4-unit D1->D5 gap — wide enough to
#: separate, far too narrow to let a reader mistake a bar for a different
#: difficulty.
_MEDIAN_CI_FAN_FRACTION = 0.022

#: savefig dpi for this family.  Matches make_plots so the two sets can be
#: compared side by side at the same scale, and so the perceptibility
#: measurement below is reported in the pixels the reader will actually get.
_MEDIAN_CI_DPI = 150


def _bin_rows_by_stat(
    all_rows: Sequence[dict],
    governments: Sequence[str],
    sorted_diffs: Sequence[int],
) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """Group per-run rows into ``{stat: {gov: {difficulty: [values]}}}``.

    Every cell in the grid is pre-created (even when it has no runs) so a caller
    can tell "swept and empty" from "not swept", which is the same distinction
    :func:`load_ads_calibration_series` preserves for its own axis.

    Rows for unknown governments, unparseable difficulties, and non-numeric or
    non-finite metric values are skipped rather than raising: this feeds a
    figure, and a plotting helper must not be able to abort a completed sweep.
    Non-finite values are filtered here and not merely at use — ``json``/``csv``
    round-trips can carry a bare ``NaN`` through, and one would poison
    ``np.median`` silently.

    See the section header for why this is not shared with :func:`make_plots`.
    """
    known = set(governments)
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        stat: {g: {d: [] for d in sorted_diffs} for g in governments}
        for stat, _, _ in PLOT_STATS
    }
    for row in all_rows:
        gov = row.get("government")
        if gov not in known:
            continue
        try:
            diff = int(row["difficulty"])
        except (KeyError, TypeError, ValueError):
            continue
        if diff not in data[PLOT_STATS[0][0]][gov]:
            continue
        for stat, _, _ in PLOT_STATS:
            val = row.get(stat)
            if val is None or val == "":
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fval):
                data[stat][gov][diff].append(fval)
    return data


def _median_ci_perceptibility(
    cis: Sequence[MedianCI],
    iqr_half_widths: Sequence[Optional[float]],
    y_range: float,
    axes_height_px: float,
    target_k: int,
) -> Dict[str, Any]:
    """Quantify whether the CI band on this figure is actually *visible*.

    At production sample sizes the CI on a median can be one to two orders of
    magnitude narrower than the IQR, which means the band can visually
    collapse into the median trend line even though it is mathematically
    nonzero (or, in a degenerate cell, genuinely zero — see the discussion in
    :func:`make_median_ci_plots`). A figure whose band cannot be seen but
    whose title claims one is worse than a figure with no band. This function
    makes band visibility a measured, reported property of every figure
    rather than an unverified assumption.

    Two ratios are reported, because they answer different questions:

    ``ratio``
        CI half-width / IQR half-width, per cell.  Scale-free, comparable
        across metrics, and directly comparable to the "two orders of
        magnitude" (ratio ~0.01) claim above.
    ``px``
        The band's full height in rendered pixels, given this figure's actual
        y-limits and axes geometry.  This is the one that decides visibility: a
        ratio of 0.3 is invisible on an axis whose whole data range is 2 px tall
        and obvious on one that is 800 px tall.

    *target_k* projects to a different runs-per-cell.  The projection is the
    asymptotic ``1/sqrt(n)`` scaling of the median's standard error, applied
    per cell as ``sqrt(n_cell / target_k)``, and it assumes the per-run outcome
    distribution is unchanged at the larger n.  It is an EXTRAPOLATION and is
    labelled as one everywhere it is reported: no k=100 corpus with per-run
    environment variation exists to measure it directly.
    """
    ratios: List[float] = []
    px_now: List[float] = []
    px_target: List[float] = []
    n_drawable = 0

    for ci, iqr_hw in zip(cis, iqr_half_widths):
        if not ci.drawable:
            continue
        hw = ci.half_width
        if hw is None:
            continue
        n_drawable += 1
        if iqr_hw is not None and iqr_hw > 0:
            ratios.append(hw / iqr_hw)
        if y_range > 0:
            px_now.append(2.0 * hw / y_range * axes_height_px)
            scale = math.sqrt(ci.n / target_k) if target_k > 0 else 1.0
            px_target.append(2.0 * hw * scale / y_range * axes_height_px)

    def _summary(vals: Sequence[float]) -> Optional[Dict[str, float]]:
        if not vals:
            return None
        a = np.asarray(vals, dtype=float)
        return {
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }

    return {
        "n_drawable_cells": n_drawable,
        "n_ratio_cells": len(ratios),
        "ratio_ci_over_iqr": _summary(ratios),
        "band_px": _summary(px_now),
        "band_px_projected": _summary(px_target),
        "projected_k": target_k,
        # Share of drawn bands that would render thinner than the ~2 px it takes
        # for a band to be distinguishable from the trend line it sits on.
        "frac_under_2px": (
            float(np.mean(np.asarray(px_now) < 2.0)) if px_now else None
        ),
        "frac_under_2px_projected": (
            float(np.mean(np.asarray(px_target) < 2.0)) if px_target else None
        ),
    }


def _perceptibility_lines(percept: Dict[str, Any], observed_k: int) -> List[str]:
    """Render :func:`_median_ci_perceptibility`'s dict as sidecar prose."""
    lines = [
        "Band perceptibility — is the CI actually visible in this PNG?",
        f"  Cells with a drawable band: {percept['n_drawable_cells']}",
    ]
    ratio = percept.get("ratio_ci_over_iqr")
    if ratio:
        lines.append(
            f"  CI half-width / IQR half-width (n={percept['n_ratio_cells']} "
            f"cells): median {ratio['median']:.3f}, "
            f"10th pct {ratio['p10']:.3f}, 90th pct {ratio['p90']:.3f}."
        )
        lines.append(
            "  For reference, a ratio around 0.01 (two orders of magnitude "
            "narrower than\n"
            "  the IQR) renders as visually indistinguishable from the median "
            "trend line."
        )
    px = percept.get("band_px")
    if px:
        lines.append(
            f"  Rendered band height at this figure's y-limits and "
            f"{_MEDIAN_CI_DPI} dpi: median {px['median']:.1f} px "
            f"(10th pct {px['p10']:.1f}, 90th pct {px['p90']:.1f})."
        )
        if percept.get("frac_under_2px") is not None:
            lines.append(
                f"  Bands thinner than 2 px (indistinguishable from the trend "
                f"line): {percept['frac_under_2px'] * 100:.0f}% of drawn cells."
            )
    proj = percept.get("band_px_projected")
    target_k = percept.get("projected_k")
    if proj and target_k and target_k != observed_k:
        lines.append(
            f"  PROJECTED to k={target_k} runs per cell (this corpus is "
            f"k={observed_k}): median {proj['median']:.1f} px "
            f"(10th pct {proj['p10']:.1f}, 90th pct {proj['p90']:.1f}); "
            f"{(percept['frac_under_2px_projected'] or 0.0) * 100:.0f}% under "
            f"2 px."
        )
        lines.append(
            "  That projection is the asymptotic 1/sqrt(n) scaling of the "
            "median's standard\n"
            "  error, applied per cell. It is an EXTRAPOLATION: no k="
            f"{target_k} corpus with\n"
            "  per-run environment variation exists to measure it directly.\n"
            "  Treat it as an order-of-magnitude answer, not a measurement."
        )
    return lines


def make_median_ci_plots(
    config: BenchmarkConfig,
    all_rows: List[dict],
    plots_dir: str,
) -> List[str]:
    """Create one ``<stat>_median_ci.png`` per statistic, plus alt-text sidecars.

    The parallel set to :func:`make_plots`.  Same metrics, same axis treatment,
    same governments, same colours — and a different dispersion statistic:

      * a marker at each cell's **median**, joined across difficulty by the
        government's trend line (identical estimand to the IQR figure's median
        bar, so the two families can be laid side by side and read together);
      * an error bar spanning the **95% percentile-bootstrap confidence
        interval on that median**, 10,000 resamples, seeded per cell.

    Seeding
    -------
    ``derive_seed(config.base_seed, "plot.bootstrap", gov, difficulty)`` — the
    same lattice that seeds the simulation itself, under its own domain label so
    it cannot alias any simulation stream.  Two consequences worth stating:

    * The figures are byte-reproducible from a fresh process and from an
      archived tree.
    * The key deliberately omits the *metric*, per the specification.  All four
      metrics of one cell therefore draw the identical resample-index matrix.
      That is harmless for the marginal interval each one reports (a bootstrap
      CI does not care what other quantity shared its indices), and it is
      arguably the more defensible choice: the four metrics are four functions
      of the *same* n runs, so resampling the runs once and evaluating all four
      is coherent, whereas an independent draw per metric would imply the runs
      were four separate samples.

    Degenerate cells
    ----------------
    Never smoothed, never silently drawn as a hairline.  A cell that cannot
    support a band gets its median drawn with an **open** marker and no bar,
    and the sidecar says which of the three reasons applies — too few runs, a
    constant cell, or a bootstrap distribution concentrated on one order
    statistic.  See :class:`MedianCI`.

    Band visibility is measured, not assumed
    -----------------------------------------
    A CI band that is mathematically nonzero can still be visually
    indistinguishable from the median trend line when the per-run outcome
    distribution is narrow. A cell where every run of a (government,
    difficulty) pair faces the identical environment is a case worth calling
    out specifically: its true CI half-width is exactly zero, which is a very
    different situation from "the CI is merely too narrow to draw" — the
    pathology :func:`_log_determinism_summary` exists to shout about. Because
    the production sweep gives each run a distinct environment, cells vary
    genuinely rather than degenerately. Rather than assuming a band is
    visible, every figure this function writes carries its own measured
    perceptibility block in its sidecar and logs it; see
    :func:`_median_ci_perceptibility`.

    Returns the paths written, PNG and sidecar, in write order.
    """
    if not all_rows:
        log("  [WARN] no rows to plot — median/CI plots skipped.")
        return []

    os.makedirs(plots_dir, exist_ok=True)
    governments = list(config.governments)
    sorted_diffs = sorted(config.difficulties)
    data = _bin_rows_by_stat(all_rows, governments, sorted_diffs)

    x_range = (max(sorted_diffs) - min(sorted_diffs)) if len(sorted_diffs) > 1 else 10
    fan = x_range * _MEDIAN_CI_FAN_FRACTION
    # Centre the fan on the tick.  A single government sits exactly on it.
    if len(governments) > 1:
        offsets = [
            -fan / 2.0 + fan * i / (len(governments) - 1)
            for i in range(len(governments))
        ]
    else:
        offsets = [0.0]

    # Seeds depend only on (gov, difficulty), so derive each one once and reuse
    # it across metrics rather than re-hashing four times per cell.
    seeds: Dict[Tuple[str, int], int] = {
        (gov, diff): derive_seed(config.base_seed, "plot.bootstrap", gov, diff)
        for gov in governments
        for diff in sorted_diffs
    }

    written: List[str] = []

    for stat, title, ylabel in PLOT_STATS:
        fig, ax = plt.subplots(figsize=(16, 7))

        # Per-figure census, all derived from what was actually drawn.
        #
        # `cell_cis` exists so the sidecar's per-cell table below reuses the
        # intervals the drawing loop already computed instead of recomputing
        # every one of them.  Without this cache, the figure would run the
        # bootstrap twice per cell — 1,344 calls for 672 cells, ~13 s at k=100.
        #
        # Safe because `bootstrap_median_ci` is a pure function of
        # (values, seed) and seeds each call's generator locally with
        # `np.random.default_rng(seed)`.  There is no shared or global RNG
        # stream, so removing a call cannot shift what any other call draws;
        # the cached object is bit-identical to what the second call returned.
        # This is the same "derive once, reuse" move already made for `seeds`
        # above, one level further out.
        cell_cis: Dict[Tuple[str, int], MedianCI] = {}
        all_cis: List[MedianCI] = []
        all_iqr_hw: List[Optional[float]] = []
        status_counts: Dict[str, int] = {}
        cell_ns: List[int] = []
        drawn_medians: List[float] = []
        edge_medians: List[Tuple[float, str]] = []

        for gov, offset in zip(governments, offsets):
            color = GOV_COLORS.get(gov, "#333333")
            linestyle = GOV_LINESTYLES.get(gov, "-")
            marker = GOV_MARKERS.get(gov, "o")

            line_xs: List[float] = []
            line_ys: List[float] = []

            for diff in sorted_diffs:
                vals = data[stat][gov][diff]
                ci = bootstrap_median_ci(vals, seeds[(gov, diff)])
                cell_cis[(gov, diff)] = ci
                status_counts[ci.status] = status_counts.get(ci.status, 0) + 1

                if ci.median is None:
                    # Nothing to draw and nothing to describe; the cell is
                    # absent from the figure exactly as it is from the data.
                    continue

                all_cis.append(ci)
                all_iqr_hw.append(
                    (float(np.percentile(vals, 75))
                     - float(np.percentile(vals, 25))) / 2.0
                    if len(vals) >= 2 else None
                )
                cell_ns.append(ci.n)
                drawn_medians.append(ci.median)
                if diff == sorted_diffs[-1]:
                    edge_medians.append((ci.median, gov))

                x = diff + offset
                line_xs.append(x)
                line_ys.append(ci.median)

                if ci.drawable:
                    ax.errorbar(
                        [x], [ci.median],
                        yerr=[[ci.median - (ci.lower or ci.median)],
                              [(ci.upper or ci.median) - ci.median]],
                        fmt=marker, color=color, markersize=4.5,
                        markerfacecolor=color, markeredgecolor=color,
                        elinewidth=1.4, capsize=2.5, capthick=1.4,
                        zorder=4,
                    )
                else:
                    # Open marker, no bar.  The median is real; the interval is
                    # not, and the glyph says so without needing the legend.
                    ax.plot(
                        x, ci.median, marker=marker, color=color,
                        markersize=4.5, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.2,
                        linestyle="none", zorder=4,
                    )

            if len(line_xs) > 1:
                ax.plot(line_xs, line_ys, color=color, linewidth=1.5,
                        linestyle=linestyle, alpha=0.75, zorder=2)

        n_caption = (
            f"{min(cell_ns)}–{max(cell_ns)}" if cell_ns and min(cell_ns) != max(cell_ns)
            else (str(cell_ns[0]) if cell_ns else "0")
        )
        _style_difficulty_axes(
            ax, sorted_diffs, x_range,
            title=f"{title} by Difficulty Level",
            ylabel=ylabel,
        )

        legend_handles = [
            plt.Line2D([0], [0], color=GOV_COLORS.get(g, "#333333"), linewidth=2.5,
                       linestyle=GOV_LINESTYLES.get(g, "-"),
                       marker=GOV_MARKERS.get(g, "o"), markersize=6,
                       label=GOV_DISPLAY.get(g, g))
            for g in governments
        ]
        ax.legend(handles=legend_handles, title="Government", title_fontsize=10,
                  fontsize=9, loc="best", framealpha=0.88)
        fig.tight_layout()

        # Measured AFTER all drawing and tight_layout, so the geometry is the
        # one the reader gets rather than an assumption about it.  bbox_inches
        # ="tight" on savefig trims margins, not the axes, so the axes height in
        # pixels is preserved to within a rounding error.
        y_lo, y_hi = ax.get_ylim()
        axes_px = ax.get_position().height * fig.get_figheight() * _MEDIAN_CI_DPI
        # Projection target comes from the CONFIG, not the module global, so it
        # is archived in the manifest and a replot reproduces this sidecar even
        # if PAPER_K_RUNS has since moved.  Reading the global directly would
        # make the "byte-identical replot" property conditional on nobody
        # having edited the published-design constant.
        percept = _median_ci_perceptibility(
            all_cis, all_iqr_hw, float(y_hi - y_lo), float(axes_px),
            target_k=config.paper_k_runs,
        )

        ratio = percept.get("ratio_ci_over_iqr")
        px = percept.get("band_px")

        out_path = os.path.join(plots_dir, f"{stat}_median_ci.png")
        fig.savefig(out_path, dpi=_MEDIAN_CI_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

        # -- sidecar ------------------------------------------------------
        clauses = [
            f"Median with 95% bootstrap confidence interval: {title} for "
            f"{len(governments)} governments "
            f"({', '.join(GOV_DISPLAY.get(g, g) for g in governments)})."
        ]
        clauses.append(
            f"Difficulty levels {sorted_diffs[0]} to {sorted_diffs[-1]} "
            f"({len(sorted_diffs)} levels), n = {n_caption} runs per government "
            f"per level."
        )
        clauses.append(
            "Each government has a marker at its median and a vertical bar "
            "spanning the 95% confidence interval of that median, at every "
            "difficulty level, with a line joining its medians across levels."
        )
        if drawn_medians:
            lo, hi = min(drawn_medians), max(drawn_medians)
            clauses.append(
                f"All medians are {lo:.2f}." if f"{lo:.2f}" == f"{hi:.2f}"
                else f"Medians span {lo:.2f} to {hi:.2f}."
            )
        if edge_medians:
            hi_val, hi_gov = max(edge_medians)
            lo_val, lo_gov = min(edge_medians)
            # Same tie rule as generate_alt_text: compare the RENDERED text, so
            # an all-equal series cannot produce "highest is X (0.00) and lowest
            # is Y (0.00)".
            if f"{hi_val:.2f}" == f"{lo_val:.2f}":
                clauses.append(
                    f"At difficulty {sorted_diffs[-1]} every government has a "
                    f"median of {hi_val:.2f}."
                )
            else:
                clauses.append(
                    f"At difficulty {sorted_diffs[-1]} the highest median is "
                    f"{GOV_DISPLAY.get(hi_gov, hi_gov)} ({hi_val:.2f}) and the "
                    f"lowest is {GOV_DISPLAY.get(lo_gov, lo_gov)} ({lo_val:.2f})."
                )

        alt_text_path = out_path.replace(".png", "_alt_text.txt")
        with open(alt_text_path, "w") as f:
            f.write(f"Figure: {title} — median with 95% bootstrap CI\n")
            f.write("Alt text (WCAG 2.1 AA compliant, max 100 words):\n"
                    f"{_clauses_to_alt_text(clauses, 100)}\n\n")
            f.write("Figure statistics:\n")
            f.write(f"  Metric: {stat}\n")
            f.write(f"  Governments: {len(governments)}\n")
            f.write(f"  Difficulty levels: {len(sorted_diffs)}\n")
            f.write(f"  Runs per cell (n): {n_caption}\n")
            f.write(f"  Interval: 95% percentile bootstrap on the MEDIAN, "
                    f"{_MEDIAN_CI_N_BOOT:,} resamples\n")
            f.write(f"  Bootstrap seed: derive_seed(base_seed="
                    f"{config.base_seed}, \"plot.bootstrap\", government, "
                    f"difficulty)\n")
            # See the identical addition in make_plots' sidecar writer for
            # why this line exists.
            f.write(f"  Scale: grid {config.grid_size}×{config.grid_size}, "
                    f"{config.n_agents} agents, {config.max_cycles} cycles\n\n")

            # The per-cell table is the sidecar's bulkiest section and is pure
            # noise when every cell has the same n and every cell is drawable —
            # the common case.  Collapsing it then is not brevity for its own
            # sake: the flagged cells are the only actionable content here, and
            # 168 identical entries is an excellent place to lose eight of them.
            # Indexed, not recomputed: the drawing loop above visited exactly
            # this product (governments x sorted_diffs) and cached each result.
            # Direct indexing is deliberate — if the two loops ever stop
            # covering the same cells, a KeyError naming the missing cell is a
            # far better outcome than a silent recomputation that hides the
            # divergence and makes the sidecar disagree with the figure.
            per_cell: List[Tuple[str, int, int, bool]] = []
            for gov in governments:
                for diff in sorted_diffs:
                    ci = cell_cis[(gov, diff)]
                    per_cell.append((gov, diff, ci.n, ci.drawable))
            distinct_n = {n for _, _, n, _ in per_cell}
            flagged = [(g, d) for g, d, _, ok in per_cell if not ok]

            f.write("Per-cell sample sizes and interval status:\n")
            if len(distinct_n) == 1 and not flagged:
                f.write(f"  Every one of the {len(per_cell)} cells has "
                        f"n = {next(iter(distinct_n))} and a drawable "
                        f"interval.\n\n")
            else:
                if len(distinct_n) == 1:
                    f.write(f"  Every one of the {len(per_cell)} cells has "
                            f"n = {next(iter(distinct_n))}.\n")
                else:
                    for gov in governments:
                        parts = [
                            f"D{d}:n={n}{'' if ok else '*'}"
                            for g, d, n, ok in per_cell if g == gov
                        ]
                        f.write(f"  {GOV_DISPLAY.get(gov, gov):20} "
                                f"{'  '.join(parts)}\n")
                f.write(f"  {len(flagged)} cell(s) drawn with an open marker "
                        f"and NO interval:\n")
                for gov in governments:
                    diffs = [d for g, d in flagged if g == gov]
                    if diffs:
                        f.write(f"    {GOV_DISPLAY.get(gov, gov):20} "
                                f"D{', D'.join(str(d) for d in diffs)}\n")
                f.write("\n")

            f.write("Cells without a drawable interval — what each one is:\n")
            any_degenerate = False
            for status in (MEDIAN_CI_STATUS_INSUFFICIENT_N,
                           MEDIAN_CI_STATUS_CONSTANT,
                           MEDIAN_CI_STATUS_ZERO_WIDTH,
                           MEDIAN_CI_STATUS_EMPTY):
                count = status_counts.get(status, 0)
                if not count:
                    continue
                any_degenerate = True
                probe = next(c for c in all_cis if c.status == status) \
                    if any(c.status == status for c in all_cis) \
                    else MedianCI(None, None, None, 0, status)
                f.write(f"  {status:24} {count:4d} cell(s) — {probe.describe()}\n")
            if not any_degenerate:
                f.write("  None — every cell supports an interval.\n")
            f.write(
                "  A zero-width interval is NOT a precise estimate. It means "
                "the cell has one\n"
                "  effective observation, or a bootstrap distribution "
                "concentrated on a single\n"
                "  order statistic. It is drawn as an open marker rather than "
                "as a band so it\n"
                "  cannot be read as precision.\n\n"
            )

            for line in _perceptibility_lines(percept, config.k_runs):
                f.write(f"{line}\n")
            f.write("\n")

            f.write("Color palette (WCAG AA accessible):\n")
            for g in governments:
                f.write(f"  {GOV_DISPLAY.get(g, g):15} | "
                        f"Color: {GOV_COLORS.get(g, '#333333')} | "
                        f"Line style: {GOV_LINESTYLES.get(g, '-')} | "
                        f"Marker: {GOV_MARKERS.get(g, 'o')}\n")
            f.write("\n")
            # The closing comparison is DERIVED from the ratio measured above,
            # not asserted — a canned claim like "the CI bar is expected to be
            # the narrower of the two" would be false whenever the measured
            # ratio disagrees (at k=10 on this corpus, the ratio is ~1.28, i.e.
            # the bar is WIDER than the box).
            ratio_med = (ratio or {}).get("median")
            if ratio_med is None:
                relation = (
                    "  No cell on this figure supports both an interquartile "
                    "range and an\n"
                    "  interval, so the two cannot be compared here.\n"
                )
            elif ratio_med > 1.05:
                relation = (
                    f"  On this corpus the bar is the WIDER of the two (median "
                    f"ratio {ratio_med:.2f}).\n"
                    "  That is not a contradiction: with few runs per cell the "
                    "median is poorly\n"
                    "  pinned down, so its uncertainty can legitimately exceed "
                    "the spread of the\n"
                    "  runs themselves. The ratio shrinks as roughly "
                    "1/sqrt(n).\n"
                )
            elif ratio_med < 0.95:
                relation = (
                    f"  On this corpus the bar is the NARROWER of the two "
                    f"(median ratio {ratio_med:.2f}),\n"
                    "  which is the usual relationship once n is large enough "
                    "to pin the median down.\n"
                )
            else:
                relation = (
                    f"  On this corpus the two are of comparable width (median "
                    f"ratio {ratio_med:.2f}).\n"
                )
            f.write(
                "Relationship to the Tukey IQR figure of the same name:\n"
                f"  {stat}.png shows the same medians — identical values, "
                "verified against the\n"
                "  same per-run data — with an interquartile box around them "
                "instead of a bar.\n"
                "  That box describes the SPREAD OF THE RUNS. This figure's bar "
                "describes the\n"
                "  UNCERTAINTY OF THE MEDIAN. They are different quantities.\n"
                + relation +
                "  Quote one or the other, never both as though they were the "
                "same band.\n"
            )
        written.append(alt_text_path)

        log(f"  Plot saved → {out_path}")
        log(f"  Alt text saved → {alt_text_path}")
        if ratio and px:
            log(f"    perceptibility: CI/IQR half-width median "
                f"{ratio['median']:.3f}, band {px['median']:.1f} px "
                f"at k={config.k_runs}; projected "
                f"{(percept['band_px_projected'] or {}).get('median', float('nan')):.1f} px "
                f"at k={config.paper_k_runs}")

    return written


def render_summary_plots(
    config: BenchmarkConfig,
    all_rows: List[dict],
    plots_dir: str,
) -> None:
    """Draw BOTH cross-government figure families into *plots_dir*.

    The two families are deliberately parallel and deliberately separate, and a
    caller should essentially never want one without the other — the whole
    reason the median/CI set exists is so the two can be compared side by side
    before one is chosen for the manuscript.  Having a single call site for the
    pair means a new regeneration path (``--plots-only``) cannot silently
    produce half of the output tree.
    """
    make_plots(config, all_rows, plots_dir)
    make_median_ci_plots(config, all_rows, plots_dir)


# ---------------------------------------------------------------------------
# ADS self-calibration figures (ADS-only, sourced from disk)
# ---------------------------------------------------------------------------
#
# WHY THESE DO NOT GO THROUGH make_plots().
#
# make_plots() draws whatever is in PLOT_STATS, and PLOT_STATS can only name
# columns that exist in `all_rows` — which is built from SUMMARY_FIELDS.  The
# self-calibration scalars are deliberately NOT in SUMMARY_FIELDS: they are a
# float for `ads` and absent for the other seven regimes, and a cross-government
# column like that makes every aggregate over it a trap.  The full reasoning is
# in _merge_calibration_summary's docstring and must not be "simplified" by
# adding the fields to SUMMARY_FIELDS.
#
# So an ADS-only figure needs an ADS-only data path, and the natural one is the
# per-run JSON that _run_gov_difficulty_inner already writes unconditionally for
# every run of every regime, regardless of --no-run-detail.  Reading it back
# from disk costs one small file read per run and buys regenerability: these
# figures can be redrawn from an archived output tree alone.

#: Directory name (and government key) that carries the calibration fields.
_ADS_GOV_KEY = "ads"

#: Governments actually drawn on the calibration figures by the two
#: production call sites (the sweep's own plot generation and `--plots-only`),
#: as opposed to `make_ads_calibration_plots`'s own default of `(_ADS_GOV_KEY,)`
#: alone.  Both ADS and `autocracy_lookahead` open a forecast and record
#: self-calibration (ADS on every enactment; A+L on every decision), so both
#: are comparable via the filtered helpers in this section.  Extending this
#: tuple is how a THIRD forecast-carrying government joins the figures -- no
#: other code change required, per `make_ads_calibration_plots`'s `gov_keys`
#: contract.
_CALIBRATION_GOV_KEYS: Tuple[str, ...] = (_ADS_GOV_KEY, "autocracy_lookahead")

#: Calibration figure titles are kept bare -- no per-government "headline"
#: clause and no per-government, per-half ``n = ...`` disclosure.  That
#: information lives in each figure's sidecar instead (see
#: ``_write_calibration_sidecar``).

#: Per-run scalars read back out of ``ads/<difficulty>/run_*/final_stats.json``.
#:
#: ``final_mean_prediction_error`` is written by ``MetricsCollector.summary()``
#: from the last cycle's snapshot, so it is the run's all-time mean
#: ``|predicted - realized|`` over every prediction that closed.  The two
#: ``_half`` fields come from ``AdsGovernment.get_calibration_summary()`` via
#: ``_merge_calibration_summary``.  All three are ``None`` when the run closed
#: no predictions in the relevant window, which is routine for the second half
#: of a short run and is why every read below is null-tolerant.
ADS_CALIBRATION_FIELDS: Tuple[str, ...] = (
    "final_mean_prediction_error",
    "calibration_mean_abs_delta_first_half",
    "calibration_mean_abs_delta_second_half",
)

#: The outcome cells that correspond to what ADS grades.
#:
#: ADS opens a forecast only on an enactment, so its ``outcome`` tag is the
#: constant ``"enacted"``.  ``autocracy_lookahead`` opens one on EVERY decision,
#: so its closures additionally include ``retained``, ``repealed`` and
#: ``no_action``.  Filtering A+L to these two is what makes a cross-arm
#: comparison like-for-like.  (``"replaced"`` is an enactment that displaced an
#: incumbent — an enactment by any reading, and ADS's ``"enacted"`` covers the
#: same case because its candidate filter removes standing law types before
#: scoring.)
CROSS_ARM_OUTCOMES: FrozenSet[str] = frozenset({"enacted", "replaced"})

#: Key of the two-level partition emitted by ``ForecastLedger.as_summary``.
_PARTITION_DELTAS_KEY = "calibration_cycle_deltas_by_outcome_then_scope"
_PARTITION_CLOSURES_KEY = "calibration_closures_by_outcome_then_scope"

#: The scope cell that makes an ADS/A+L comparison scope-matched rather than
#: confounded by ADS's group-targeting search.
#:
#: A+L always legislates the whole population -- ``n_groups`` is hardcoded to
#: 1 (``autocracy_lookahead.py:1184``) -- while ADS *selects* a subgroup from
#: a search over ``GROUP_DISTRIBUTION_COUNTS`` (``ads.py:201``), and 81% of
#: ADS's corpus is the finest-grained ``n_groups == 10`` scheme (A4.8's
#: census: 23,221 of 28,638 enactments). Predicting a fine-grained group's
#: outcome is a different, easier problem than predicting the whole
#: population's, so comparing ADS's FULL corpus against A+L's (necessarily
#: whole-population) corpus mixes that scope difference into whatever gap the
#: figure reports. Restricting ADS to this one scope key -- the same scope
#: A+L's own closures always carry -- removes the confound: both series are
#: then forecasts of the same kind of target, a whole population.
#:
#: This is the *only* scope key ever passed as *scopes* to
#: :func:`filtered_calibration_mean` / :func:`filtered_half_split` /
#: :func:`_filtered_cycle_deltas` in this module; the ``str`` wrapping is
#: because scope keys in the partition are ``str(n_groups)``, not ``int``
#: (readers must ``int(k)`` before sorting, which applies to the AXIS the
#: scope-matched figures draw, not to this lookup).
_SCOPE_MATCHED_SCOPES: FrozenSet[str] = frozenset({"1"})

#: Minimum pooled closures a difficulty level or cycle must carry before the
#: scope-matched figures (:func:`make_scope_matched_calibration_plots`) draw a
#: point for it, rather than a mean over 1-2 observations that reads as a
#: measurement.
#:
#: 5 is not an arbitrary round number: it is the exact floor the ADS-full-
#: corpus trend figure's own docstring already uses to characterise this same
#: archive's ADS ``n_groups == 1`` subset -- "94 distinct cycles (median
#: 4/cycle, 54 of those 94 cycles with fewer than 5 observations)"
#: (see the comment above :func:`make_ads_calibration_plots`). Reusing that
#: exact threshold rather than picking a fresh one means the "thin" cutoff
#: this module already asserts in prose is the same cutoff the figure
#: enforces in pixels. On the real k=10 archive this suppresses 2 of 21 ADS
#: difficulty levels (D10 n=2, D15 n=3) and 54 of 94 ADS cycles; A+L clears it
#: everywhere but 4 of 130 cycles.
_SCOPE_MATCHED_MIN_N = 5


def filtered_calibration_mean(
    final_stats: Mapping[str, Any],
    outcomes: FrozenSet[str] = CROSS_ARM_OUTCOMES,
    scopes: Optional[FrozenSet[str]] = None,
) -> Optional[float]:
    """
    The n-weighted mean ``|predicted - realized|`` over the named outcome cells.

    **THIS IS THE ONLY FUNCTION A FIGURE MAY USE TO OBTAIN AN ARM'S CALIBRATION
    MEAN.**  ``final_mean_prediction_error`` and ``calibration_mean_abs_err`` are
    unfiltered all-time means.  They are correct for ADS *by accident* — its
    ``outcome`` tag is constant, so filtered and unfiltered coincide — and wrong
    for ``autocracy_lookahead``, which grades ``retained``, ``repealed`` and
    ``no_action`` closures that ADS never grades at all.

    The size of that error is not hypothetical.  On a gate fixture
    (``MAX_CYCLES = 90``) A+L closed 115
    forecasts of which only 9 were enactments or replacements: the unfiltered
    mean was 0.1917 and the filtered mean 0.2088, i.e. the unfiltered scalar was
    the smaller of the two on this fixture.  Plotting the unfiltered number
    beside ADS's would be a plausible-looking wrong figure rather than a crash,
    which is why the filtering lives behind one named function with one place to
    review and one place to test.

    NEITHER THE MAGNITUDE NOR THE SIGN IS A PROPERTY OF THE HAZARD -- BOTH ARE
    PROPERTIES OF THE FIXTURE.  At the fixture's previous horizon of 80 the same
    three numbers were 120 closures, 8 enactments, 0.2304 unfiltered / 0.3012
    filtered.  Those two fixtures agree in sign (unfiltered smaller) and
    disagree in magnitude by a factor of nearly four.  A third configuration
    disagrees in sign as well: on ``test_ads_calibration.py``'s own live A+L
    run the unfiltered scalar comes out LARGER — 0.209140 unfiltered vs 0.182314
    filtered.  Whether the enacted/replaced subset is harder or easier to
    forecast than the retained bulk is a property of which laws happened to pass
    in a given run, not of the grading mechanism, so it can go either way.  DO
    NOT describe the unfiltered scalar as "understating" or "overstating" the
    filtered one in a figure caption or in prose — say that they differ, not in
    which direction they differ, unless you are quoting one specific run's own
    numbers. Do not treat any of the figures above as a threshold or a test
    constant either: the invariant worth asserting is that the two numbers
    DISAGREE on this arm and AGREE on ADS to within the ledger's own rounding
    bound.  ``test_ads_calibration.py`` asserts exactly that -- a
    directional (disagree/agree) invariant, not a magnitude or a sign -- and
    deliberately hard-codes none of the figures above.

    The scalars themselves are deliberately NOT filtered at source: changing
    ADS's would break the no-op gate this filtering depends on, and
    changing only A+L's would break the two arms' shape-identity in substance
    while preserving it in form.  Both numbers are in the archive; the figure
    picks the right one.

    *scopes*, when not ``None``, additionally restricts the cells summed to
    those whose scope key (``str(n_groups)``) is a member -- this is the
    scope-matching filter :data:`_SCOPE_MATCHED_SCOPES` exists for.  ``None``
    (the default) sums every scope cell under the named outcomes, exactly the
    pre-existing behaviour: every call site that predates the scope-matched
    figures omits this argument and is therefore untouched by it.

    Returns ``None`` iff no named outcome cell is present — which is a real
    state (an arm that enacted nothing in this run), not an error.
    """
    cells = _outcome_cells(final_stats, outcomes, scopes)
    total_n = sum(n for n, _ in cells)
    if total_n == 0:
        return None
    return sum(n * mean for n, mean in cells) / total_n


def filtered_half_split(
    final_stats: Mapping[str, Any],
    outcomes: FrozenSet[str] = CROSS_ARM_OUTCOMES,
    scopes: Optional[FrozenSet[str]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    ``(first_half_mean, second_half_mean)`` over the named outcome cells.

    The same hazard as :func:`filtered_calibration_mean`, one level further in
    and MORE dangerous.  ``final_mean_prediction_error`` is absent on A+L, so a
    figure reaching for it gets nothing; the two
    ``calibration_mean_abs_delta_{first,second}_half`` fields are **present,
    plausible, and unfiltered**, so a figure reaching for those gets a wrong
    number that looks right.

    Derived from the partition's leaf pairs, which carry the closure cycle, split
    at the run's own emitted ``calibration_half_split_cycle`` rather than a
    recomputed boundary — so the filtered and unfiltered halves are split at the
    identical cycle and remain comparable.  No fourth emitted field is needed:
    the boundary is already in the file and the cycles are already in the leaves.

    *scopes*, when not ``None``, restricts the leaves summed into either half
    to the named scope keys -- same contract as :func:`filtered_calibration_mean`,
    ``None`` reproducing the pre-existing unrestricted behaviour exactly.

    A half with no closures returns ``None`` rather than ``0.0``, matching the
    unfiltered fields' convention.
    """
    boundary = final_stats.get("calibration_half_split_cycle")
    if not isinstance(boundary, (int, float)):
        return None, None
    deltas = final_stats.get(_PARTITION_DELTAS_KEY)
    if not isinstance(deltas, dict):
        return None, None

    first: List[float] = []
    second: List[float] = []
    for outcome, by_scope in deltas.items():
        if outcome not in outcomes or not isinstance(by_scope, dict):
            continue
        for scope_key, pairs in by_scope.items():
            if scopes is not None and scope_key not in scopes:
                continue
            for pair in pairs or ():
                try:
                    cycle, delta = pair[0], float(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                (first if cycle < boundary else second).append(delta)

    def _mean(values: List[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    return _mean(first), _mean(second)


def _outcome_cells(
    final_stats: Mapping[str, Any],
    outcomes: FrozenSet[str],
    scopes: Optional[FrozenSet[str]] = None,
) -> List[Tuple[int, float]]:
    """
    ``[(n, mean_abs_err), ...]`` for every cell under the named outcomes.

    Reads the AGGREGATE partition rather than re-deriving from the series: the
    emitter computes each cell's mean from that cell's own rounded leaf values,
    so the two agree exactly and the aggregate is cheaper.  Cells with no
    closures are omitted by the emitter, so every cell reached here has n >= 1.

    *scopes*, when not ``None``, additionally restricts to cells whose scope
    key (``str(n_groups)``) is a member of it -- see
    :func:`filtered_calibration_mean` for the contract this exists to serve.
    ``None`` walks every scope cell, matching every call site that predates
    scope-matching.
    """
    closures = final_stats.get(_PARTITION_CLOSURES_KEY)
    if not isinstance(closures, dict):
        return []
    out: List[Tuple[int, float]] = []
    for outcome, by_scope in closures.items():
        if outcome not in outcomes or not isinstance(by_scope, dict):
            continue
        for scope_key, cell in by_scope.items():
            if scopes is not None and scope_key not in scopes:
                continue
            if not isinstance(cell, dict):
                continue
            n, mean = cell.get("n"), cell.get("mean_abs_err")
            if (isinstance(n, int) and not isinstance(n, bool) and n > 0
                    and isinstance(mean, (int, float))
                    and not isinstance(mean, bool)
                    and math.isfinite(float(mean))):
                out.append((n, float(mean)))
    return out


# ---------------------------------------------------------------------------
# Per-government routing for the calibration figures
#
# `filtered_calibration_mean` / `filtered_half_split` (above) are the ENGINE:
# they know how to compute an outcome-filtered scalar or half-pair from a raw
# `final_stats` mapping.  They do not know which government they are being
# asked about.  The four functions below are the WIRING: given a `gov_key`,
# each picks ADS's own unfiltered field (correct for ADS by construction) or
# routes through the filtered engine (mandatory for every other government,
# whose `outcome` tag is not constant).  Every loader in this module that
# grew a `gov_key` parameter calls through here rather than re-deciding the
# ADS-vs-everyone-else branch at its own call site, so there is exactly one
# place this decision is made and exactly one place to review it.
# ---------------------------------------------------------------------------

def _run_calibration_mean(
    final_stats: Mapping[str, Any], gov_key: str,
) -> Optional[float]:
    """
    The one all-time calibration scalar one run of *gov_key* contributes to
    ``mean_prediction_error.png``.

    ADS keeps reading its own ``final_mean_prediction_error`` verbatim —
    unfiltered is correct for ADS BY ACCIDENT (its ``outcome`` tag is
    constant; see :func:`filtered_calibration_mean`).  Every other government
    is routed through :func:`filtered_calibration_mean`, which exists
    precisely because its unfiltered fields are ``None`` or present-and-wrong
    for a government that grades outcomes ADS never grades at all.  Do not add
    a branch here that reads an unfiltered scalar for a non-ADS government —
    that is the exact hazard this function exists to close off at one call
    site instead of several.
    """
    value = (
        final_stats.get("final_mean_prediction_error")
        if gov_key == _ADS_GOV_KEY
        else filtered_calibration_mean(final_stats)
    )
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _run_calibration_half_split(
    final_stats: Mapping[str, Any], gov_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    The fixed-split ``(first_half, second_half)`` pair for one run of
    *gov_key*.

    Feeds only the FALLBACK branch of ``calibration_half_split.png`` — the one
    taken when no burn-in breakpoint can be estimated and the figure falls
    back to the sweep's own pre-bucketed halves.  ADS reads its own unfiltered
    ``calibration_mean_abs_delta_{first,second}_half`` fields exactly as
    before; every other government routes through :func:`filtered_half_split`,
    for the reason :func:`_run_calibration_mean` gives.
    """
    if gov_key == _ADS_GOV_KEY:
        return (
            final_stats.get("calibration_mean_abs_delta_first_half"),
            final_stats.get("calibration_mean_abs_delta_second_half"),
        )
    return filtered_half_split(final_stats)


def _filtered_cycle_deltas(
    final_stats: Mapping[str, Any],
    outcomes: FrozenSet[str] = CROSS_ARM_OUTCOMES,
    scopes: Optional[FrozenSet[str]] = None,
) -> List[Tuple[int, float]]:
    """
    ``(cycle, abs_delta)`` pairs for the named outcome cells, unsplit.

    Same source and parsing posture as :func:`filtered_half_split` — reads
    ``calibration_cycle_deltas_by_outcome_then_scope`` rather than the
    unfiltered ``calibration_cycle_deltas`` field — but returns the raw pairs
    instead of pre-splitting at a boundary, for callers that need the whole
    per-cycle series: the trend figure's pooled per-cycle mean, and the
    half-split figure's post-burn-in early/late split, both of which need a
    run's full per-cycle series rather than two pre-computed means.

    *scopes*, when not ``None``, restricts the leaves returned to the named
    scope keys.  This is the primitive :func:`load_scope_matched_calibration_deltas`
    calls, with ``scopes=_SCOPE_MATCHED_SCOPES``, to build the two scope-matched
    figures -- and it is called identically for EVERY government, including
    ADS, rather than through :func:`_run_calibration_cycle_deltas`'s
    ADS-reads-its-own-unfiltered-field branch: the scope tag lives only inside
    the partition, so a scope-matched ADS series must go through the partition
    too, on purpose, or it would silently be the unfiltered full-corpus series.
    ``None`` (the default) walks every scope, reproducing every pre-existing
    call site's behaviour exactly.

    Returns an empty list, never ``None``, when the run's partition is present
    but contains nothing in *outcomes* (and, if given, *scopes*) — a real
    state (this arm enacted or replaced nothing this run, or nothing in the
    requested scope) that must plot as "no contribution", not as a fabricated
    zero.
    """
    deltas = final_stats.get(_PARTITION_DELTAS_KEY)
    if not isinstance(deltas, dict):
        return []
    out: List[Tuple[int, float]] = []
    for outcome, by_scope in deltas.items():
        if outcome not in outcomes or not isinstance(by_scope, dict):
            continue
        for scope_key, pairs in by_scope.items():
            if scopes is not None and scope_key not in scopes:
                continue
            for pair in pairs or ():
                try:
                    cycle, delta = pair[0], float(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if isinstance(cycle, bool) or not isinstance(cycle, int):
                    continue
                if not math.isfinite(delta):
                    continue
                out.append((cycle, delta))
    return out


def _run_calibration_cycle_deltas(
    final_stats: Mapping[str, Any], gov_key: str,
) -> List[Tuple[int, float]]:
    """
    The ``(cycle, abs_delta)`` pairs one run of *gov_key* contributes to the
    trend and half-split-by-cycle figures.

    ADS reads its own unfiltered ``calibration_cycle_deltas`` field exactly as
    before — unfiltered is correct for ADS by construction, and this is the
    field :func:`load_ads_calibration_cycle_deltas_by_run` has always parsed.
    Every other government is filtered to :data:`CROSS_ARM_OUTCOMES` via
    :func:`_filtered_cycle_deltas`, so the two arms' per-cycle series are drawn
    from like-for-like closures rather than mixing in outcomes (``retained``,
    ``repealed``, ``no_action``) that ADS never grades at all.
    """
    if gov_key != _ADS_GOV_KEY:
        return _filtered_cycle_deltas(final_stats)
    series = final_stats.get(_ADS_CALIBRATION_CYCLE_DELTAS_FIELD)
    if not isinstance(series, list):
        return []
    out: List[Tuple[int, float]] = []
    for obs in series:
        if not isinstance(obs, (list, tuple)) or len(obs) != 2:
            continue
        cycle_raw, delta_raw = obs
        if isinstance(cycle_raw, bool) or not isinstance(cycle_raw, int):
            continue
        if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
            continue
        delta = float(delta_raw)
        if not math.isfinite(delta):
            continue
        out.append((cycle_raw, delta))
    return out


def _series_offsets(n: int, box_width: float) -> Tuple[float, ...]:
    """
    Symmetric x-offsets for *n* box series sharing one difficulty tick.

    Generalizes the two-series convention ``calibration_half_split.png``
    already used (``-0.75 x box_width`` / ``+0.75 x box_width`` — adjacent but
    not touching) to any series count.  ``n <= 1`` returns ``(0.0,)``: a
    single series centred exactly on the tick, which is what every
    calibration figure drew before ``gov_keys`` existed, so a single-arm call
    reproduces that placement exactly (default-preserving).
    """
    if n <= 1:
        return (0.0,)
    spacing = box_width * 1.5
    start = -spacing * (n - 1) / 2.0
    return tuple(start + i * spacing for i in range(n))


#: Two-series palette for the half-split figure'S SINGLE-GOVERNMENT case only.
#:
#: Both entries are lifted from GOV_COLORS rather than invented, so they
#: inherit that palette's WCAG AA contrast and colour-blind-safe properties
#: (the pair is Okabe-Ito blue/orange, the canonical deuteranopia-safe pair).
#: Reusing two *government* colours for a non-government distinction is safe
#: only when the figure carries no government legend — true when exactly one
#: government is plotted (its legend then names the two halves of a run) and
#: FALSE the moment a second government is added: from two governments
#: up, ``calibration_half_split.png`` colours by GOV_COLORS per government and
#: distinguishes the two halves by linestyle instead. Do not reuse this pair
#: on any figure — or any branch of
#: this one — that also plots more than one government.
_HALF_SPLIT_COLORS: Tuple[str, str] = ("#0072B2", "#D55E00")
_HALF_SPLIT_LINESTYLES: Tuple[Any, Any] = ("-", "--")


def load_ads_calibration_series(
    output_root: str,
    fields: Sequence[str] = ADS_CALIBRATION_FIELDS,
    *,
    gov_key: str = _ADS_GOV_KEY,
) -> Dict[str, Dict[int, List[float]]]:
    """
    Read per-run calibration scalars back out of an output tree.

    Walks ``<output_root>/<gov_key>/<difficulty>/run_*/final_stats.json`` and
    returns ``{field: {difficulty: [values]}}``.  Every difficulty directory
    found gets a key even if it yielded no usable values, so a caller can tell
    "this level was swept and produced nothing" apart from "this level was
    not swept".

    **Routing by *gov_key*.**  For ``gov_key ==
    _ADS_GOV_KEY`` the three names in *fields* are read verbatim off disk —
    the original behaviour, unchanged, because ADS's unfiltered scalars are
    correct BY ACCIDENT (its ``outcome`` tag is constant; see
    :func:`filtered_calibration_mean`).  For any OTHER government the same
    three field-name KEYS are populated from the routed, outcome-filtered
    equivalents computed by :func:`_run_calibration_mean` /
    :func:`_run_calibration_half_split` instead of read verbatim: reading
    ``calibration_mean_abs_delta_first_half`` and friends straight off disk
    for a non-ADS government would silently plot the unfiltered scalar, which
    is "present, plausible and wrong" for a government that grades outcomes
    ADS never grades (see the two functions' docstrings). The dict keys stay
    the ADS field names on purpose: every downstream consumer (the box-plot
    figure and the half-split fallback path) already keys off them, and
    giving the routed non-ADS numbers a second, parallel set of keys would
    double the surface a future edit has to keep in sync rather than shrink
    it.

    Defensive in the same way :func:`load_cell_stats` is, and for the same
    reason — a plotting helper must not be able to abort a completed sweep:

    * a missing ``<gov_key>/`` directory returns empty series rather than
      raising;
    * an unreadable or non-object JSON file is counted and skipped (WARNING:
      ``final_stats.json`` is written by the sweep itself, so a corrupt one is
      a real signal, not routine);
    * a field that is absent, ``null`` or non-numeric is counted and skipped
      (DEBUG: this is *expected*, see ADS_CALIBRATION_FIELDS);
    * a non-finite float is skipped.  This is not paranoia — ``json.dump``
      writes bare ``NaN``/``Infinity`` tokens for non-finite floats and
      ``json.load`` reads them straight back, so a NaN can survive the round
      trip and would poison ``np.median`` silently.
    """
    result: Dict[str, Dict[int, List[float]]] = {field: {} for field in fields}
    gov_root = os.path.join(output_root, gov_key)
    if not os.path.isdir(gov_root):
        return result

    log_label = GOV_DISPLAY.get(gov_key, gov_key)
    n_files = 0
    n_unreadable = 0
    skipped: Dict[str, int] = {field: 0 for field in fields}

    for entry in sorted(os.listdir(gov_root)):
        # Difficulty directories are named with the bare integer level.  Skip
        # anything else so a stray file or a future sibling directory cannot
        # become a phantom x-tick.
        if not entry.isdigit():
            continue
        diff_dir = os.path.join(gov_root, entry)
        if not os.path.isdir(diff_dir):
            continue
        difficulty = int(entry)
        for field in fields:
            result[field].setdefault(difficulty, [])

        run_files = sorted(
            glob.glob(os.path.join(diff_dir, "run_*", "final_stats.json"))
        )
        for path in run_files:
            n_files += 1
            try:
                with open(path, "r") as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                n_unreadable += 1
                bench_log(logging.WARNING, "  unreadable %s: %s: %s",
                          path, type(exc).__name__, exc)
                continue
            if not isinstance(stats, dict):
                n_unreadable += 1
                bench_log(logging.WARNING,
                          "  %s is valid JSON but not an object — skipped", path)
                continue

            if gov_key == _ADS_GOV_KEY:
                routed: Dict[str, Any] = {field: stats.get(field) for field in fields}
            else:
                first, second = _run_calibration_half_split(stats, gov_key)
                routed = {
                    "final_mean_prediction_error":
                        _run_calibration_mean(stats, gov_key),
                    "calibration_mean_abs_delta_first_half": first,
                    "calibration_mean_abs_delta_second_half": second,
                }

            for field in fields:
                value = routed.get(field)
                # bool is a subclass of int; a bool here would be nonsense and
                # would plot as 0.0/1.0 rather than announcing itself.
                if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                    skipped[field] += 1
                    continue
                value = float(value)
                if not math.isfinite(value):
                    skipped[field] += 1
                    continue
                result[field][difficulty].append(value)

    n_levels = len(result[fields[0]]) if fields else 0
    log(f"  {log_label} calibration: read {n_files} run file(s) across "
        f"{n_levels} difficulty level(s) under {gov_root}")
    if n_unreadable:
        bench_log(logging.WARNING,
                  "  %s calibration: %d run file(s) were unreadable and "
                  "contributed nothing to the figures", log_label, n_unreadable)
    for field in fields:
        if skipped[field]:
            bench_log(logging.DEBUG,
                      "  %s calibration: %s absent/null in %d of %d run(s)",
                      log_label, field, skipped[field], n_files)
    return result


#: Per-run list-valued field read back by :func:`load_ads_calibration_cycle_deltas`.
#: Kept out of :data:`ADS_CALIBRATION_FIELDS` deliberately: that tuple's reader
#: validates every field as a bare int/float scalar, and this one is a list of
#: ``[cycle, abs_delta]`` pairs — a different shape needs a different reader,
#: not an `isinstance` special case bolted onto the scalar one.
_ADS_CALIBRATION_CYCLE_DELTAS_FIELD = "calibration_cycle_deltas"


def load_ads_calibration_cycle_deltas(
    output_root: str, *, gov_key: str = _ADS_GOV_KEY,
) -> Dict[int, List[float]]:
    """
    Read the full per-cycle calibration-error series back out of an output tree.

    Walks ``<output_root>/<gov_key>/<difficulty>/run_*/final_stats.json`` and
    returns ``{cycle: [abs_delta, abs_delta, ...]}``, POOLED ACROSS EVERY
    DIFFICULTY LEVEL AND EVERY RUN.

    Routes through :func:`load_ads_calibration_cycle_deltas_by_run`'s own
    *gov_key* handling: unfiltered for ADS, filtered to
    :data:`CROSS_ARM_OUTCOMES` for every other government. See that function
    and :func:`_run_calibration_cycle_deltas` for why.

    Deliberately NOT keyed by difficulty, unlike :func:`load_ads_calibration_series`.
    That function's box plots need one column per difficulty level, so pooling
    would destroy the thing they exist to show. This reader feeds a
    calibration-error-vs-cycle *trend* figure instead, whose whole point is
    statistical power at each cycle: a single 150-cycle run only closes a
    modest number of predictions (open predictions at the end are the norm,
    not the exception — see ``calibration_open_predictions_at_end``), so
    splitting that already-thin per-run series further by difficulty would
    leave most (cycle, difficulty) cells with one or two observations, too few
    to plot a meaningful mean or CI. Pooling every run of every difficulty
    into one series per cycle is what makes a per-cycle mean and 95% CI worth
    drawing at all.

    Defensive in the same way :func:`load_ads_calibration_series` is, and for
    the same reasons:

    * a missing ``ads/`` directory returns an empty dict rather than raising;
    * an unreadable or non-object JSON file is counted and skipped (WARNING —
      ``final_stats.json`` is written by the sweep itself, so a corrupt one is
      a real signal, not routine);
    * a ``calibration_cycle_deltas`` field that is absent, ``null``, or not a
      list is counted and skipped (DEBUG — expected for every non-ADS regime,
      and for any ``ads`` archive that predates this field);
    * within a list, an entry that is not a 2-element ``[cycle, abs_delta]``
      pair, whose cycle is not an integer, or whose delta is not a finite
      number is counted and skipped (DEBUG) rather than aborting the whole
      file — one malformed observation should not discard the rest of a run's
      series;
    * a non-finite ``abs_delta`` is skipped for the same reason
      :func:`load_ads_calibration_series` skips non-finite scalars: bare
      ``NaN``/``Infinity`` tokens survive a ``json.dump``/``json.load`` round
      trip and would poison ``np.mean``/``np.std`` silently.

    A malformed record is never allowed to raise and abort a whole
    regeneration pass — the same posture :func:`load_ads_calibration_series`
    takes.

    The walking and validation live in
    :func:`load_ads_calibration_cycle_deltas_by_run`, and this function is a
    pooling view over it.  The half-split figure needed the same records grouped
    by ``(difficulty, run)`` instead of pooled by cycle, and two independent
    parsers for one field is how the two views eventually disagree about which
    records are malformed.  Pooling is the cheap operation; parsing is the one
    with the judgement calls in it, so parsing is what is shared.
    """
    by_run = load_ads_calibration_cycle_deltas_by_run(output_root, gov_key=gov_key)
    result: Dict[int, List[float]] = {}
    for runs in by_run.values():
        for series in runs:
            for cycle, delta in series:
                result.setdefault(cycle, []).append(delta)

    n_obs = sum(len(v) for v in result.values())
    log(f"  {GOV_DISPLAY.get(gov_key, gov_key)} calibration (per-cycle): "
        f"{n_obs} observation(s) pooled across {len(result)} distinct "
        f"cycle(s)")
    return result


def load_ads_calibration_cycle_deltas_by_run(
    output_root: str, *, gov_key: str = _ADS_GOV_KEY,
) -> Dict[int, List[List[Tuple[int, float]]]]:
    """Read the per-cycle calibration series grouped by difficulty and run.

    Returns ``{difficulty: [run_series, run_series, ...]}`` where each
    ``run_series`` is a list of ``(cycle, abs_delta)`` pairs for one run, in the
    order the run recorded them.  Runs that recorded the field but contributed
    no usable observation appear as an empty list, so "this run closed nothing"
    stays distinguishable from "this run was never read".

    **Routing by *gov_key*.**  For ``gov_key ==
    _ADS_GOV_KEY`` this parses the unfiltered ``calibration_cycle_deltas``
    field exactly as before.  For any other government it instead reads
    ``calibration_cycle_deltas_by_outcome_then_scope`` and filters to
    :data:`CROSS_ARM_OUTCOMES`, via :func:`_run_calibration_cycle_deltas` — see
    that function for why an unfiltered read would be wrong for a government
    whose ``outcome`` tag is not constant.

    Why this grouping exists, on top of the pooled view
    ---------------------------------------------------
    The half-split figure asks a within-run question — is this run's forecast
    better late than early — and answering it requires knowing which
    observations belong to the same run.  Pooling destroys that.  It is also
    what makes the split point a *choice* rather than something frozen into the
    corpus: the ``calibration_mean_abs_delta_{first,second}_half`` scalars in
    ``final_stats.json`` were bucketed at cycle 75 while the sweep ran and
    cannot be re-bucketed, but these raw pairs can be re-split at any cycle
    without touching the simulation.  See :func:`estimate_burn_in_breakpoint`
    for the re-derived split that replaces the fixed cycle-75 boundary.

    This is the parser both cycle-delta views share.  Its defensive posture is
    the one :func:`load_ads_calibration_series` documents, item for item:

    * a missing ``ads/`` directory returns an empty dict rather than raising;
    * an unreadable or non-object JSON file is counted and skipped (WARNING —
      ``final_stats.json`` is written by the sweep itself, so a corrupt one is
      a real signal, not routine);
    * a ``calibration_cycle_deltas`` field that is absent, ``null``, or not a
      list is counted and skipped (DEBUG — expected for every non-ADS regime,
      and for any ``ads`` archive that predates this field);
    * within a list, an entry that is not a 2-element ``[cycle, abs_delta]``
      pair, whose cycle is not an integer, or whose delta is not a finite
      number is counted and skipped (DEBUG) rather than aborting the whole
      file — one malformed observation should not discard the rest of a run's
      series;
    * a non-finite ``abs_delta`` is skipped because bare ``NaN``/``Infinity``
      tokens survive a ``json.dump``/``json.load`` round trip and would poison
      ``np.mean``/``np.std`` silently.
    """
    result: Dict[int, List[List[Tuple[int, float]]]] = {}
    gov_root = os.path.join(output_root, gov_key)
    if not os.path.isdir(gov_root):
        return result

    log_label = GOV_DISPLAY.get(gov_key, gov_key)
    source_field = (
        _ADS_CALIBRATION_CYCLE_DELTAS_FIELD if gov_key == _ADS_GOV_KEY
        else _PARTITION_DELTAS_KEY
    )
    n_files = 0
    n_unreadable = 0
    n_files_without_field = 0
    n_obs_kept = 0
    n_entries_skipped = 0

    for entry in sorted(os.listdir(gov_root)):
        # Same difficulty-directory filter as load_ads_calibration_series, for
        # the same reason: skip anything that is not a bare integer directory
        # name so a stray file cannot masquerade as a run source.
        if not entry.isdigit():
            continue
        diff_dir = os.path.join(gov_root, entry)
        if not os.path.isdir(diff_dir):
            continue
        difficulty = int(entry)
        result.setdefault(difficulty, [])

        run_files = sorted(
            glob.glob(os.path.join(diff_dir, "run_*", "final_stats.json"))
        )
        for path in run_files:
            n_files += 1
            try:
                with open(path, "r") as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                n_unreadable += 1
                bench_log(logging.WARNING, "  unreadable %s: %s: %s",
                          path, type(exc).__name__, exc)
                continue
            if not isinstance(stats, dict):
                n_unreadable += 1
                bench_log(logging.WARNING,
                          "  %s is valid JSON but not an object — skipped", path)
                continue

            if gov_key == _ADS_GOV_KEY:
                # Unfiltered read — see _run_calibration_cycle_deltas for why
                # this branch stays a literal duplicate of that function's ADS
                # arm rather than calling through it: the per-entry skip/kept
                # counters below are diagnostic-only, but they must stay
                # byte-identical to the original for a default (ADS-only)
                # invocation.
                series = stats.get(_ADS_CALIBRATION_CYCLE_DELTAS_FIELD)
                if not isinstance(series, list):
                    # Expected, not an error: every non-ads run has no such
                    # field at all, and any ads archive that predates this
                    # field lacks it too.
                    n_files_without_field += 1
                    continue
                run_series: List[Tuple[int, float]] = []
                for obs in series:
                    if not isinstance(obs, (list, tuple)) or len(obs) != 2:
                        n_entries_skipped += 1
                        continue
                    cycle_raw, delta_raw = obs
                    # bool is a subclass of int; a bool cycle would be nonsense.
                    if isinstance(cycle_raw, bool) or not isinstance(cycle_raw, int):
                        n_entries_skipped += 1
                        continue
                    if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
                        n_entries_skipped += 1
                        continue
                    delta = float(delta_raw)
                    if not math.isfinite(delta):
                        n_entries_skipped += 1
                        continue
                    run_series.append((cycle_raw, delta))
                    n_obs_kept += 1
                result[difficulty].append(run_series)
            else:
                # Filtered read: the partition must be PRESENT for the
                # run to be counted at all — absence means "this run predates
                # the partition field", the same "not yet read" state the ADS
                # branch above tracks via `n_files_without_field`.  Once the
                # partition is present, an empty filtered result (this arm
                # enacted/replaced nothing this run) is a real, appendable
                # `[]`, not an absence -- see _filtered_cycle_deltas.
                if not isinstance(stats.get(_PARTITION_DELTAS_KEY), dict):
                    n_files_without_field += 1
                    continue
                run_series = _run_calibration_cycle_deltas(stats, gov_key)
                n_obs_kept += len(run_series)
                result[difficulty].append(run_series)

    n_runs = sum(len(v) for v in result.values())
    log(f"  {log_label} calibration (per-cycle): read {n_files} run file(s), "
        f"kept {n_obs_kept} observation(s) from {n_runs} run(s) across "
        f"{len(result)} difficulty level(s) under {gov_root}")
    if n_unreadable:
        bench_log(logging.WARNING,
                  "  %s calibration (per-cycle): %d run file(s) were "
                  "unreadable and contributed nothing to the trend figure",
                  log_label, n_unreadable)
    if n_files_without_field:
        bench_log(logging.DEBUG,
                  "  %s calibration (per-cycle): %s absent/null in %d of "
                  "%d run(s)",
                  log_label, source_field, n_files_without_field, n_files)
    if n_entries_skipped:
        bench_log(logging.DEBUG,
                  "  %s calibration (per-cycle): %d malformed observation(s) "
                  "skipped across all run(s)", log_label, n_entries_skipped)
    return result


# ---------------------------------------------------------------------------
# Scope-matched loading -- ADS(n_groups == 1) vs A+L, the apples-to-apples
# figures that complement the full-corpus comparison. See the module
# docstring above make_scope_matched_calibration_plots for why the
# full-corpus comparison alone is confounded, and its census for the
# measured size of the confound.
# ---------------------------------------------------------------------------

def load_scope_matched_calibration_deltas(
    output_root: str,
    *,
    gov_key: str,
    outcomes: FrozenSet[str] = CROSS_ARM_OUTCOMES,
    scopes: FrozenSet[str] = _SCOPE_MATCHED_SCOPES,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Read the scope-matched per-cycle deltas for one government, pooled by
    difficulty.

    Returns ``{difficulty: [(cycle, abs_delta), ...]}`` — every closure any run
    at that difficulty recorded under an outcome in *outcomes* and a scope in
    *scopes*, run identity discarded.  Difficulty directories present but
    contributing nothing still get a key mapped to ``[]``, matching
    :func:`load_ads_calibration_series`'s convention of distinguishing "swept,
    empty" from "not swept at all".

    **Deliberately gov_key-uniform, unlike every other loader in this module.**
    ``load_ads_calibration_series`` / ``load_ads_calibration_cycle_deltas_by_run``
    both special-case ``gov_key == _ADS_GOV_KEY`` to read ADS's own *unfiltered*
    field, because the unfiltered field is correct for ADS's FULL-CORPUS
    figures (its ``outcome`` tag is constant, so filtered and unfiltered
    coincide there). That branch is exactly wrong here: the ``n_groups`` scope
    tag exists ONLY inside the partition
    (``calibration_cycle_deltas_by_outcome_then_scope``), so a scope-matched
    ADS series has no unfiltered equivalent to fall back to — it MUST come
    from the partition, via :func:`_filtered_cycle_deltas`, exactly the same
    call for ADS as for A+L or any future forecast-carrying government. This
    is why this function calls ``_filtered_cycle_deltas`` directly rather than
    routing through :func:`_run_calibration_cycle_deltas`: that router's whole
    purpose is to take the ADS-reads-its-own-field branch this function must
    never take. ``test_ads_calibration.py``'s scope-filter regression test
    guards exactly this — see its docstring for the failure mode (a silently
    unfiltered series that looks plausible) this uniformity closes off.

    Pooling posture (unreadable file, non-object JSON, missing partition,
    malformed pair, non-finite delta) is identical to
    :func:`load_ads_calibration_cycle_deltas_by_run` — same tree walk, same
    per-run parser (``_filtered_cycle_deltas``), just grouped by difficulty
    instead of by ``(difficulty, run)`` because the scope-matched figures pool
    closures directly rather than computing a per-run statistic (A4.8: a
    per-run statistic is not viable on this subset — median ~2 whole-population
    closures per run, a quarter of runs at zero).
    """
    result: Dict[int, List[Tuple[int, float]]] = {}
    gov_root = os.path.join(output_root, gov_key)
    if not os.path.isdir(gov_root):
        return result

    log_label = GOV_DISPLAY.get(gov_key, gov_key)
    n_files = 0
    n_unreadable = 0
    n_files_without_field = 0
    n_obs_kept = 0

    for entry in sorted(os.listdir(gov_root)):
        if not entry.isdigit():
            continue
        diff_dir = os.path.join(gov_root, entry)
        if not os.path.isdir(diff_dir):
            continue
        difficulty = int(entry)
        result.setdefault(difficulty, [])

        for path in sorted(glob.glob(os.path.join(diff_dir, "run_*", "final_stats.json"))):
            n_files += 1
            try:
                with open(path, "r") as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                n_unreadable += 1
                bench_log(logging.WARNING, "  unreadable %s: %s: %s",
                          path, type(exc).__name__, exc)
                continue
            if not isinstance(stats, dict):
                n_unreadable += 1
                bench_log(logging.WARNING,
                          "  %s is valid JSON but not an object — skipped", path)
                continue
            if not isinstance(stats.get(_PARTITION_DELTAS_KEY), dict):
                # Predates the partition (or the partition was never emitted
                # for this run) -- "not yet read", not "read and empty".
                n_files_without_field += 1
                continue
            pairs = _filtered_cycle_deltas(stats, outcomes, scopes)
            n_obs_kept += len(pairs)
            result[difficulty].extend(pairs)

    log(f"  {log_label} calibration (scope-matched, scopes="
        f"{sorted(scopes)}): read {n_files} run file(s), kept {n_obs_kept} "
        f"observation(s) across {len(result)} difficulty level(s) under "
        f"{gov_root}")
    if n_unreadable:
        bench_log(logging.WARNING,
                  "  %s calibration (scope-matched): %d run file(s) were "
                  "unreadable and contributed nothing", log_label, n_unreadable)
    if n_files_without_field:
        bench_log(logging.DEBUG,
                  "  %s calibration (scope-matched): partition absent in %d "
                  "of %d run(s)", log_label, n_files_without_field, n_files)
    return result


# ---------------------------------------------------------------------------
# Burn-in detection for the calibration-vs-cycle series
#
# The published claim "calibration error GROWS with cycle (+6.20e-04/cycle,
# p=0.0000)" is an
# artifact of fitting ONE straight line through a SATURATING curve.  The
# plotted quantity is abs(predicted - realized) on mean per-agent health, which
# is bounded above by 1.0 and pinned near its ceiling for the first tens of
# cycles of every run: at cycle 0 the population sits at health ~0.97-0.99, so
# both the projection and the outcome are near 1.0 and the absolute gap is near
# zero BY CONSTRUCTION, independent of forecast skill.  The metric only acquires
# dynamic range once health disperses.
#
# A line fitted across that regime change measures the regime change, not a
# trend.  Restricted past the burn-in the slope flips sign; meanwhile the thing
# the figure is supposed to be about — the parameter estimator — converges
# monotonically in all ten cycle bins (drain_mult relative error 0.2613 ->
# 0.0516, a 5.1x reduction).  The estimator is not degrading.
#
# The fix is to find where the burn-in ends FROM THE DATA rather than asserting
# a round number, and to report the fit on each side of it.
# ---------------------------------------------------------------------------

#: Minimum pooled observations on each side of a candidate breakpoint.  A
#: breakpoint with a handful of points on one side is fitting noise, and the
#: argmin would happily run to the edge of the cycle range to find it.
_BURN_IN_MIN_SEGMENT_OBS = 200

#: Cycles trimmed from each end of the candidate grid, on top of the sample-size
#: floor above.  Keeps the estimate interior: a "breakpoint" at the first or
#: last observed cycle is not a breakpoint, it is a failure to find one.
_BURN_IN_EDGE_MARGIN_CYCLES = 10


@dataclass(frozen=True)
class BurnInFit:
    """A continuous two-segment fit to the calibration-error-vs-cycle series.

    ``breakpoint`` is the cycle at which the two segments join.  ``slope_pre``
    and ``slope_post`` are the segment slopes; the model is continuous, so they
    meet rather than jumping.

    ``ssr_improvement`` is the fraction of residual sum of squares the segmented
    model removes relative to a single straight line.  **Read it before quoting
    the breakpoint.**  A shallow improvement means the SSR profile is flat and
    the breakpoint is weakly identified — the *existence* of a sign change can
    be solid while its *location* is uncertain to tens of cycles.  That is
    exactly the situation on the current corpus, and pretending otherwise would
    replace one over-confident number with another.

    ``profile`` is the SSR at a coarse grid of candidate breakpoints, carried so
    the sidecar can show the reader how flat the minimum is instead of asking
    them to take the argmin on trust.
    """

    breakpoint: int
    slope_pre: float
    slope_post: float
    ssr_segmented: float
    ssr_linear: float
    ssr_improvement: float
    n_pre: int
    n_post: int
    profile: Tuple[Tuple[int, float], ...]


def estimate_burn_in_breakpoint(
    cycles: Sequence[float],
    values: Sequence[float],
) -> Optional[BurnInFit]:
    """Locate the end of the burn-in by continuous two-segment least squares.

    The model is the standard continuous piecewise-linear (segmented
    regression) form::

        y = a + b*x + c*max(0, x - k)

    fitted by ordinary least squares at each candidate breakpoint ``k``, with
    the ``k`` minimising the residual sum of squares selected.  Because the
    model is linear in ``(a, b, c)`` for fixed ``k``, this grid search is the
    exact profile-likelihood estimate of ``k`` under Gaussian residuals — not an
    approximation of one.  Pre- and post-breakpoint slopes are ``b`` and
    ``b + c``.

    Why this, rather than the alternatives
    --------------------------------------
    * **A round number** (cycle 75) is unjustifiable: it has no basis in the
      data.
    * **Eyeballing the mean curve** cannot be reproduced by the next reader.
    * **A discontinuous two-piece fit** would let the estimate exploit a jump
      that the underlying process does not have — the error curve saturates
      smoothly, so continuity is the honest constraint and it also costs a
      parameter less.
    * **Choosing by t-statistic sign flip** (scan for where the restricted slope
      changes sign) optimises the quantity being reported, which is circular.
      SSR does not look at the sign at all.

    Returns ``None`` when the series is too short or too degenerate to support
    two segments — fewer than ``2 * _BURN_IN_MIN_SEGMENT_OBS`` observations, a
    cycle range too narrow to place an interior breakpoint, or a singular
    design.  A ``None`` return is a normal outcome for a smoke-test-scale sweep
    and callers must handle it by falling back to the single-line fit and
    SAYING SO, not by inventing a breakpoint.
    """
    x = np.asarray(cycles, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size != y.size or x.size < 2 * _BURN_IN_MIN_SEGMENT_OBS:
        return None

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 * _BURN_IN_MIN_SEGMENT_OBS:
        return None

    lo = int(math.ceil(x.min())) + _BURN_IN_EDGE_MARGIN_CYCLES
    hi = int(math.floor(x.max())) - _BURN_IN_EDGE_MARGIN_CYCLES
    if hi <= lo:
        return None

    # Single-line baseline, for the improvement figure.
    x_mean, y_mean = float(x.mean()), float(y.mean())
    ss_xx = float(((x - x_mean) ** 2).sum())
    if ss_xx <= 0:
        return None
    b_lin = float(((x - x_mean) * (y - y_mean)).sum()) / ss_xx
    a_lin = y_mean - b_lin * x_mean
    ssr_linear = float(((y - (a_lin + b_lin * x)) ** 2).sum())
    if not math.isfinite(ssr_linear) or ssr_linear <= 0:
        return None

    ones = np.ones_like(x)
    best: Optional[Tuple[float, int, np.ndarray, int, int]] = None
    profile: List[Tuple[int, float]] = []

    for k in range(lo, hi + 1):
        n_pre = int((x < k).sum())
        n_post = int((x >= k).sum())
        if n_pre < _BURN_IN_MIN_SEGMENT_OBS or n_post < _BURN_IN_MIN_SEGMENT_OBS:
            continue
        design = np.column_stack([ones, x, np.maximum(0.0, x - k)])
        try:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        ssr = float(((y - design @ coef) ** 2).sum())
        if not math.isfinite(ssr):
            continue
        profile.append((k, ssr))
        if best is None or ssr < best[0]:
            best = (ssr, k, coef, n_pre, n_post)

    if best is None:
        return None

    ssr_seg, k_best, coef, n_pre, n_post = best
    # Thin the profile to a readable grid; the full per-cycle curve is hundreds
    # of rows and the point of showing it is the SHAPE, not the resolution.
    step = max(1, len(profile) // 12)
    thinned = tuple(profile[::step])

    return BurnInFit(
        breakpoint=k_best,
        slope_pre=float(coef[1]),
        slope_post=float(coef[1] + coef[2]),
        ssr_segmented=ssr_seg,
        ssr_linear=ssr_linear,
        ssr_improvement=1.0 - ssr_seg / ssr_linear,
        n_pre=n_pre,
        n_post=n_post,
        profile=thinned,
    )


def _manifest_scale_caption(output_root: str) -> str:
    """
    ``"grid 20×20, 80 agents, 80 cycles"`` from the sweep's own manifest.

    Returns ``""`` when the manifest is missing or unparseable, in which case
    the caller simply omits the clause.  The manifest is written with
    ``status="running"`` *before* the sweep starts (see :func:`run_benchmark`),
    so it is on disk by the time any plot is drawn — which is what lets a live
    sweep and a months-later regeneration from the same directory produce a
    byte-identical caption.  Reading the scale from disk rather than taking a
    BenchmarkConfig parameter is the whole reason that identity holds.

    (``MANIFEST_NAME`` is defined further down the module, in the sweep-driver
    section.  Module globals resolve at call time, so the forward reference is
    fine; it is noted here only so the next reader does not go looking.)
    """
    path = os.path.join(output_root, MANIFEST_NAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get("config") or {}
        grid = cfg["grid_size"]
        return (f"grid {grid}×{grid}, {cfg['n_agents']} agents, "
                f"{cfg['max_cycles']} cycles")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError,
            AttributeError, KeyError, TypeError) as exc:
        bench_log(logging.DEBUG, "  no scale caption from %s: %s: %s",
                  path, type(exc).__name__, exc)
        return ""


#: Maps a half-split window's stable `key` (as stored in `halves_meta`) to a
#: short, human-readable tag for the n-caption at ~benchmark_core.py:5680.
#: The re-derived split uses "early"/"late" directly; the fixed-split
#: fallback keys on the long field name instead (see `make_ads_calibration_plots`'s
#: `halves_meta` construction), so it needs its own short word.  Keying off
#: `key` rather than `label` is deliberate: `label` is bare
#: ("Cycles 81-114" / "Cycles 115-149") and both windows' labels start with
#: the same word, so a label substring cannot distinguish them -- `key` is
#: the window's stable identity.
_HALF_WINDOW_TAGS: Dict[str, str] = {
    "early": "early",
    "late": "late",
    "calibration_mean_abs_delta_first_half": "first",
    "calibration_mean_abs_delta_second_half": "second",
}


def _half_window_tag(key: str) -> str:
    """Short window word for a half-split n-caption; see `_HALF_WINDOW_TAGS`."""
    return _HALF_WINDOW_TAGS.get(key, key)


def _n_per_cell_caption(per_diff: Dict[int, List[float]]) -> str:
    """
    ``"n = 5 runs per cell"``, or ``"n = 3–5 runs per cell"`` when it varies.

    Counts runs that actually *reported* the field, not ``config.k_runs``.  A
    run that closed no predictions contributes no value, and captioning it as
    though it had would overstate the sample behind every box in the figure.
    """
    counts = sorted({len(vals) for vals in per_diff.values() if vals})
    if not counts:
        return "no runs reported this field"
    lo, hi = counts[0], counts[-1]
    span = f"{lo}" if lo == hi else f"{lo}–{hi}"
    unit = "run" if (lo == hi == 1) else "runs"
    return f"n = {span} {unit} per cell"


def _trend_direction_clause(slope: float, p_value: float) -> str:
    """One sentence stating which way a fitted trend actually goes.

    Sign-conditional on purpose.  Hardcoded boilerplate that asserts a
    direction unconditionally is fragile: if it claims "a negative,
    statistically significant slope indicates calibration improving" while
    the sweep actually fits a POSITIVE slope, the alt text tells a
    screen-reader user the exact opposite of what the figure plots, and
    nothing in the code can catch it, because the sentence never looks at the
    number it describes.

    Rule this encodes: never describe the direction of a fitted quantity
    without branching on its actual sign, and treat "not significant" as its
    own third case rather than silently folding it into one of the two
    directions.

    This function is a pure function of ``(slope, p_value)`` and asserts
    nothing about which quantity was fitted, so it stays correct even when
    the metric it describes changes meaning underneath it (e.g. raw error at
    law expiry vs. calibrated error at a fixed T+20 review) — unlike the
    prose caveats around it, which do need updating when that happens.
    """
    if not math.isfinite(slope) or not math.isfinite(p_value):
        return ("Lower is a better forecast. No trend direction is reported: "
                "the least-squares fit did not return a finite slope.")
    if p_value >= 0.05:
        return (f"Lower is a better forecast. The fitted slope "
                f"({slope:+.2e} per cycle) is not statistically significant "
                f"(p = {p_value:.4f}), so error shows no detectable trend "
                f"across a run.")
    if slope > 0:
        return (f"Lower is a better forecast. The slope is POSITIVE "
                f"({slope:+.2e} per cycle, p = {p_value:.4f}): error GROWS "
                f"as a run progresses — predictions get worse, not better.")
    return (f"Lower is a better forecast. The slope is NEGATIVE "
            f"({slope:+.2e} per cycle, p = {p_value:.4f}): error SHRINKS as "
            f"a run progresses — predictions get better.")


def _half_sample_size_clause(n_first: int, n_second: int) -> str:
    """One sentence stating which half of the split actually has more closures.

    Direction-conditional on purpose, for exactly the reason
    :func:`_trend_direction_clause` is: hardcoded boilerplate that asserts a
    direction unconditionally is fragile.  For example, asserting "the second
    half is systematically thinner because predictions opened in the last 20
    cycles of a run never reach their review date" is false whenever the
    second half is actually larger — as it is in 205 of the 210 ADS runs in
    the production corpus (9,509 first-half closures against 15,745
    second-half: the second half was 66% LARGER).  A sentence that never
    looks at the numbers it describes can tell a screen-reader user the
    opposite of the figure.

    The mechanism such a sentence would name is real but can be outweighed.
    Closures begin at cycle 20 (the T+20 review horizon), so under an unequal
    split (e.g. cycle 75) the first half spans more close-cycles than the
    second; that wider window can more than offset end-of-run censoring.
    Under an equal-width split the two windows are equal by construction, so
    any remaining imbalance is a property of the data rather than of the
    arithmetic — which is why it is worth stating explicitly.

    Rule this encodes, restated from :func:`_trend_direction_clause`: never
    describe the direction of a measured quantity without branching on its
    actual value.
    """
    if n_first <= 0 or n_second <= 0:
        return ("One of the two halves closed no predictions at all, so the "
                "halves are not comparable; read the per-half sample sizes "
                "before drawing any conclusion.")
    bigger, smaller = max(n_first, n_second), min(n_first, n_second)
    # Expressed as "the larger is X% LARGER than the smaller", i.e. relative to
    # the smaller.  Stated deliberately rather than left to the reader: the
    # same pair of counts also supports "the smaller is Y% smaller" with a
    # different Y (66% larger is 40% smaller), and a sentence that does not say
    # which direction it is quoting is the same genus of defect as the one this
    # function replaced.
    pct = 100.0 * (bigger - smaller) / smaller
    if pct < 5.0:
        return (f"The two halves carry comparable samples "
                f"({n_first} closures in the first, {n_second} in the second — "
                f"the larger exceeds the smaller by {pct:.0f}%), so the "
                f"medians can be compared directly.")
    thicker = "second" if n_second > n_first else "first"
    return (f"The {thicker} half is the THICKER of the two on this corpus "
            f"({n_first} closures in the first half, {n_second} in the second "
            f"— the {thicker} half is {pct:.0f}% LARGER), so compare the two "
            f"halves only alongside their sample sizes.")


#: Appended to the cycle-trend figure's sidecar only.  Lives outside the
#: 100-word alt-text budget deliberately: it is too long to fit, and dropping
#: it would be worse than dropping any of the descriptive clauses.
_CALIBRATION_TREND_SCOPE_CAVEAT = (
    "  BURN-IN — READ THIS BEFORE READING ANY SLOPE:\n"
    "  The plotted quantity is bounded above by 1.0 and is PINNED NEAR ITS\n"
    "  CEILING for the first tens of cycles of every run.  At cycle 0 the\n"
    "  population sits at health ~0.97-0.99 with std ~0.01-0.03, so both the\n"
    "  projection and the outcome are near 1.0 and the absolute gap between\n"
    "  them is near zero BY CONSTRUCTION, independent of forecast skill.  The\n"
    "  metric only acquires dynamic range once health disperses.  The curve is\n"
    "  therefore SATURATING, not linear, and a single straight line fitted\n"
    "  across the whole cycle range measures that regime change rather than a\n"
    "  trend -- treating it as one would give the misleading impression that\n"
    "  'error GROWS as a run progresses -- predictions get worse, not\n"
    "  better'.  Past the burn-in the slope changes sign.  The breakpoint\n"
    "  drawn on this figure is estimated from the data by continuous\n"
    "  two-segment least squares, and the headline slope quoted in the title\n"
    "  and alt text is the POST-burn-in fit; the full-range fit is drawn too,\n"
    "  greyed and labelled, only so a reader can see why it is the wrong\n"
    "  estimator.\n"
    "  WHAT THIS DOES *NOT* LICENCE.  A negative post-burn-in slope is weak\n"
    "  evidence of improvement, not proof of it: the segmented model's residual\n"
    "  improvement over a straight line is small, so the breakpoint's LOCATION\n"
    "  is only loosely identified even where the sign change is robust, and\n"
    "  error also tracks the SCALE of the predicted quantity (mean abs error is\n"
    "  0.018 at difficulty 1, where health never leaves the ceiling, against\n"
    "  0.210 at difficulty 100, where it collapses) rather than forecast\n"
    "  quality alone.  Mortality is a real but secondary contributor.  The\n"
    "  direct evidence that the estimator works is NOT this figure at all -- it\n"
    "  is parameter_estimates in run_detail.jsonl, where drain_mult relative\n"
    "  error falls monotonically across all ten cycle bins, 0.2613 -> 0.0516.\n"
    "\n"
    "  SCOPE — what this trend can and cannot show:\n"
    "  The plotted quantity is abs(predicted - realized) for the ADS\n"
    "  evaluator's T+20 forecast: its own projection of mean health per agent\n"
    "  over the evaluated group, differenced against what those same agents\n"
    "  actually had exactly 20 cycles later.  Prediction and outcome span the\n"
    "  same interval, so this is forecast error rather than horizon mismatch.\n"
    "  THERE IS NO PAIRED NULL WITHIN A RUN: there is no second number to\n"
    "  difference against.  A multiplicative correction on this quantity --\n"
    "  and the raw fields it would need -- is not used, for a structural\n"
    "  reason: a multiplicative correction cannot serve a quantity pinned\n"
    "  near its own ceiling, so it would learn the large upward correction\n"
    "  that small sick groups demand and then apply it to healthy ones where\n"
    "  no upward room exists.\n"
    "  THE MECHANISM IS UPSTREAM.  Accuracy is expected to improve over a\n"
    "  run because ADS infers the environment-dynamics parameters the rollout\n"
    "  CONSUMES — drain_mult, regen_mult, max_steps_per_cycle — from evidence\n"
    "  it accumulates as cycles pass, and the projection is reported exactly\n"
    "  as the evaluator produced it.  The within-run comparison is therefore a\n"
    "  FIRST-HALF/SECOND-HALF SPLIT (see calibration_half_split.png), not a\n"
    "  cal-vs-raw subtraction — but only once the burn-in above is EXCLUDED\n"
    "  from it: a split that leaves the burn-in inside the first half\n"
    "  reproduces the artifact instead of testing it.\n"
    "  The\n"
    "  inferred parameters themselves are persisted per cycle as\n"
    "  parameter_estimates in run_detail.jsonl, so a reader can check whether\n"
    "  a flat error trend reflects an estimator that never converged.\n"
    "  The level is still NOT a skill score: the projection remains a\n"
    "  deliberately simplified forward model (no movement except storm\n"
    "  sheltering, resource collection pinned to the origin cell), and the\n"
    "  realized outcome is read for the originally-evaluated agents whether\n"
    "  or not the originating law survived the 20 cycles.\n"
    "  DO NOT POOL ACROSS GENERATIONS.  Three incompatible quantities have\n"
    "  worn these field names.  Identify a run by its schema key:\n"
    "    no key                         raw projection at law expiry\n"
    "                                   against a 30-cycle forecast\n"
    "    calibration_schema_version: 3  abs(c * raw - realized)\n"
    "    ads_forecast_schema_version: 4 abs(predicted - realized)\n"
)


def _burn_in_evidence_lines(burn_in: Optional[BurnInFit]) -> List[str]:
    """Show the reader how well-identified the burn-in breakpoint actually is.

    The breakpoint is an argmin, and an argmin over a flat surface is a number
    with no precision behind it.  Printing the SSR profile is the cheapest
    honest way to say so: a reader who can see that moving the breakpoint 20
    cycles barely changes the fit will not quote its location to the cycle, and
    one who can see a sharp minimum can.  Neither can be inferred from the
    argmin alone.
    """
    if burn_in is None:
        return [
            "Burn-in breakpoint: NOT ESTIMATED.",
            "  Too few pooled observations, or too narrow a cycle range, to fit "
            "two segments.",
            "  Any statement about within-run improvement on this corpus is "
            "unsupported.",
        ]
    lines = [
        f"Burn-in breakpoint: cycle {burn_in.breakpoint} "
        f"(continuous two-segment least squares).",
        f"  Pre-breakpoint slope  {burn_in.slope_pre:+.4e} per cycle "
        f"(n = {burn_in.n_pre})",
        f"  Post-breakpoint slope {burn_in.slope_post:+.4e} per cycle "
        f"(n = {burn_in.n_post})",
        f"  Residual sum of squares: {burn_in.ssr_segmented:.4f} segmented vs "
        f"{burn_in.ssr_linear:.4f} for one straight line",
        f"  Improvement over a single line: "
        f"{burn_in.ssr_improvement * 100:.2f}%",
    ]
    if burn_in.ssr_improvement < 0.05:
        lines.append(
            "  READ THAT IMPROVEMENT AS A WARNING. It is small, which means the "
            "SSR surface is\n"
            "  nearly flat and the breakpoint's LOCATION is only loosely "
            "identified. The sign\n"
            "  change between the two segments can be robust while the cycle it "
            "happens at is\n"
            "  uncertain to tens of cycles. Do not quote the breakpoint as a "
            "precise quantity."
        )
    lines.append("  SSR by candidate breakpoint (flatness is the point):")
    for cycle, ssr in burn_in.profile:
        marker = "  <-- selected" if cycle == burn_in.breakpoint else ""
        lines.append(f"    cycle {cycle:4d}  SSR {ssr:.4f}{marker}")
    return lines


def _half_split_caveat(
    split_window: Optional[Tuple[int, int, int]],
    burn_in: Optional[BurnInFit],
    n_early: int,
    n_late: int,
    halves: Sequence[Tuple[str, str]],
) -> str:
    """Interpretation caveat for the early/late figure.

    Direction-conditional for the same reason :func:`_half_sample_size_clause`
    is: asserting "the second half is systematically thinner because laws
    enacted near the end of a run never expire and so never close" is false
    once closures are graded on a fixed T+20 review rather than law expiry —
    the second half is the *larger* of the two in 205 of 210 runs.
    """
    lines: List[str] = [
        "Interpretation caveat — how this figure's split was chosen:\n",
    ]
    if split_window is not None:
        w_lo, w_mid, w_hi = split_window
        lines.append(
            f"  The split is NOT a round number and NOT the one the sweep "
            f"recorded.\n"
            f"  Cycles below {w_lo} are discarded as burn-in; the remaining "
            f"window {w_lo}-{w_hi}\n"
            f"  is divided at its midpoint, cycle {w_mid}, giving two windows "
            f"of equal width.\n"
            f"  WHY: the plotted error is bounded above by 1.0 and pinned near "
            f"its ceiling\n"
            f"  early in every run, so a split that leaves the burn-in inside "
            f"the first half\n"
            f"  compares a ceiling-pinned window against a dispersed one and "
            f"reproduces the\n"
            f"  artifact described in calibration_trend_by_cycle's sidecar "
            f"instead of testing\n"
            f"  it. An unequal split at a fixed cycle (e.g. 75) does exactly "
            f"that. Equal window\n"
            f"  widths also remove the arithmetic asymmetry that would "
            f"otherwise make an\n"
            f"  unconditional sample-size sentence wrong.\n"
        )
    else:
        lines.append(
            "  THIS IS THE FIXED SPLIT AND IT IS NOT THE HONEST COMPARISON.\n"
            "  The archive did not carry enough per-cycle records to re-derive "
            "a burn-in\n"
            "  breakpoint, so the figure falls back to the fixed split the "
            "sweep bucketed at\n"
            "  run time. That split leaves the ceiling-pinned burn-in inside "
            "the first half.\n"
            "  Do not read this figure as evidence about whether the estimator "
            "learns within a\n"
            "  run — in this state it cannot answer that.\n"
        )
    lines.append("\n")
    lines.append(f"  Closures per window: {halves[0][0]} = {n_early}; "
                 f"{halves[1][0]} = {n_late}.\n")
    lines.append(f"  {_half_sample_size_clause(n_early, n_late)}\n")
    lines.append("\n")
    for line in _burn_in_evidence_lines(burn_in):
        lines.append(f"  {line}\n")
    return "".join(lines)


def _calibration_governments_line(gov_keys: Sequence[str]) -> str:
    """
    The sidecar's "Governments: ..." line, honest about what was actually drawn.

    Asserting "1 (ADS only — no other regime records self-calibration)"
    unconditionally would be false the moment a second forecast-carrying
    government joins ``gov_keys``: ``autocracy_lookahead`` also records
    self-calibration, on every decision it makes rather than only on
    enactments.  Recomputed from what was actually plotted rather than
    asserted as a constant, so this sentence cannot go stale.
    """
    names = [GOV_DISPLAY.get(g, g) for g in gov_keys]
    if len(names) == 1:
        return f"Governments: 1 ({names[0]} only)"
    return f"Governments: {len(names)} ({', '.join(names)})"


#: Appended to a calibration sidecar only when more than one government is
#: plotted on it.  Lives outside the alt-text word budget, alongside
#: `_CALIBRATION_TREND_SCOPE_CAVEAT`, for the same reason: it is interpretation
#: guidance, not a figure description, and is too long to fit in 100 words.
_CROSS_ARM_CALIBRATION_CAVEAT = (
    "  CROSS-ARM COMPARISON — read before comparing the two series:\n"
    "  Every non-ADS government's number on this figure is FILTERED to the\n"
    "  'enacted'/'replaced' outcome cells — the closures ADS also grades — via\n"
    "  benchmark_core.filtered_calibration_mean / filtered_half_split.  A\n"
    "  government such as autocracy_lookahead opens a forecast on EVERY\n"
    "  decision, including 'retained', 'repealed' and 'no_action' closures ADS\n"
    "  never grades at all; pooling those into the comparison would compare\n"
    "  different populations rather than a like-for-like one. See\n"
    "  CROSS_ARM_OUTCOMES and the two functions' docstrings for the measured\n"
    "  size of that hazard.\n"
    "  THE SCALE CAVEAT ABOVE APPLIES ACROSS ARMS TOO.  This metric tracks the\n"
    "  scale of the predicted quantity as well as forecast skill, so a gap\n"
    "  between one government's error and another's is not necessarily a gap\n"
    "  in forecasting skill — it can also reflect a difference in which\n"
    "  situations the two governments chose to grade, or in the underlying\n"
    "  distribution of outcomes each one faced.  Do not read this figure as a\n"
    "  leaderboard.\n"
)


def _scope_matched_pointer_caveat(counterpart_png: Optional[str]) -> str:
    """
    Interpretation caveat appended to a FULL-CORPUS calibration sidecar,
    pointing the reader at :func:`make_scope_matched_calibration_plots`'s
    apples-to-apples comparison.

    Every full-corpus calibration figure's ADS series mixes ADS's
    fine-grained group-targeting search (81% of its corpus is the
    finest-grained ``n_groups == 10`` scheme) in with A+L's
    necessarily whole-population one, so a reader who opens only one figure
    family must not be left to assume it is the scope-matched comparison, or
    fail to notice a scope-matched comparison exists at all.

    *counterpart_png* names the scope-matched figure this sidecar's own figure
    corresponds to.  Pass ``None`` for ``calibration_half_split.png``, which
    has no scope-matched counterpart — its unit of analysis is the per-run
    half-split mean, and ADS's ``n_groups == 1`` closures pool to a median of
    ~2 per run with a quarter of runs at zero, too thin to define
    that statistic at all, let alone split it in half.
    """
    common = (
        "  SCOPE NOTE — this figure's ADS series is ADS's FULL CORPUS, not\n"
        "  scope-matched to A+L.  ADS selects a subgroup from a search over\n"
        "  GROUP_DISTRIBUTION_COUNTS (ads.py:201) while A+L always legislates\n"
        "  the whole population (n_groups hardcoded to 1,\n"
        "  autocracy_lookahead.py:1184); 81% of ADS's corpus is the\n"
        "  finest-grained n_groups == 10 scheme, a different and easier\n"
        "  prediction problem than A+L's whole-population grading.\n"
    )
    if counterpart_png is None:
        return common + (
            "  This figure's per-run half-split statistic has NO scope-matched\n"
            "  counterpart, and none is drawn: ADS's n_groups == 1 closures pool\n"
            "  to a median of ~2 per run with a quarter of runs at zero, too\n"
            "  thin to define a per-run statistic at all.  See\n"
            "  mean_prediction_error_scope_matched.png and\n"
            "  calibration_trend_by_cycle_scope_matched.png for the two\n"
            "  scope-matched figures that do exist.\n"
        )
    return common + (
        f"  For the apples-to-apples comparison — ADS restricted to\n"
        f"  n_groups == 1, the same scope A+L always carries — see\n"
        f"  {counterpart_png}.\n"
    )


def _write_calibration_sidecar(
    png_path: str,
    title: str,
    metric: str,
    alt_text: str,
    n_levels: int,
    palette_lines: Sequence[str],
    axis_label: str = "Difficulty levels",
    extra_caveat: str = "",
    gov_keys: Sequence[str] = (_ADS_GOV_KEY,),
    scale: str = "",
) -> str:
    """
    Write the accessibility sidecar for a calibration figure.

    Deliberately parallel to the sidecar :func:`make_plots` writes — same file
    naming, same four leading sections — so a reader of ``plots/`` does not hit
    two formats.  It adds one section the government figures have no need for:
    an interpretation caveat.  Returns the path written.

    *axis_label* names whatever *n_levels* is counting.  Both original callers
    (the accuracy-by-difficulty and half-split-by-difficulty figures) count
    difficulty levels and rely on the default, so their sidecar output is
    unchanged.  The cycle-trend figure has a cycle axis instead of a
    difficulty axis, and passes ``axis_label="Distinct cycles with data"``
    rather than forcing a fictitious "Difficulty levels" line onto a figure
    that has no difficulty axis at all.

    *gov_keys* is the government(s) actually drawn on THIS figure — not
    necessarily all of ``make_ads_calibration_plots``'s ``gov_keys``, since a
    government with no usable data for one particular figure is silently
    omitted from that figure specifically.  Feeds the "Governments:" line
    (:func:`_calibration_governments_line`) and, when more than one government
    is present, appends :data:`_CROSS_ARM_CALIBRATION_CAVEAT` after
    *extra_caveat*.
    """
    txt_path = png_path.replace(".png", "_alt_text.txt")
    with open(txt_path, "w") as f:
        f.write(f"Figure: {title}\n")
        f.write(f"Alt text (WCAG 2.1 AA compliant, max 100 words):\n{alt_text}\n\n")
        f.write("Figure statistics:\n")
        f.write(f"  Metric: {metric}\n")
        f.write(f"  {_calibration_governments_line(gov_keys)}\n")
        f.write(f"  {axis_label}: {n_levels}\n")
        # Run scale lives here, not in the figure title: titles carry only the
        # plotted quantity.
        if scale:
            f.write(f"  Scale: {scale}\n")
        f.write("\n")
        f.write("Color palette (WCAG AA accessible):\n")
        for line in palette_lines:
            f.write(f"  {line}\n")
        f.write("\n")
        # This caveat is the single most important thing in the sidecar: the
        # figure invites a reading it does not support, and misreading it is
        # easy to fall into.
        #
        # This paragraph names "ADS" only where the wording is specific to
        # ADS.  Two clauses are conditional on whether more than
        # one government is actually on the figure, so the single-ADS-arm
        # (default) call site reproduces the ORIGINAL wording verbatim — the
        # substitutions below are a no-op string-for-string when
        # `gov_keys == (_ADS_GOV_KEY,)`.  The fourth paragraph (schema-version
        # field names) is left untouched: it is a footnote about ADS's own
        # on-disk schema history and stays true regardless of how many
        # governments are plotted alongside it.
        single_arm = tuple(gov_keys) == (_ADS_GOV_KEY,)
        _evaluator_clause = "the ADS" if single_arm else "each plotted government's own"
        _infers_clause = (
            "ADS infers" if single_arm else "each government infers"
        )
        _overread_clause = (
            "ADS forecasts are 15% wrong" if single_arm
            else "these forecasts are 15% wrong"
        )
        f.write("Interpretation caveat — read before quoting a number from this "
                "figure:\n")
        f.write(f"  The plotted quantity is the mean absolute gap between {_evaluator_clause}\n"
                "  evaluator's 20-cycle projection and the health the same agents\n"
                "  actually had exactly 20 cycles later.  Prediction and outcome span\n"
                "  the same interval, so this is forecast error rather than\n"
                "  review-horizon mismatch.  Lower is a better forecast.\n"
                "  NOTHING CORRECTS THIS NUMBER.  The projection is reported exactly as\n"
                "  the evaluator produced it; the accuracy mechanism is upstream, in the\n"
                f"  environment-dynamics parameters {_infers_clause} from accumulated evidence\n"
                "  and feeds INTO the rollout.  A flat trend means the inference is not\n"
                "  helping, not that a correction is under-damped — check\n"
                "  parameter_estimates in run_detail.jsonl to tell the two apart.\n"
                "  The level is NOT a skill score: the projection is a deliberately\n"
                "  simplified forward model, so a nonzero floor is expected.  Reading\n"
                f"  0.15 as \"{_overread_clause}\" over-reads it.\n"
                "  DO NOT POOL ACROSS GENERATIONS: three different quantities have worn\n"
                "  these field names.  Runs carrying ads_forecast_schema_version hold\n"
                "  abs(predicted - realized); runs carrying calibration_schema_version\n"
                "  hold abs(c * raw - realized); runs carrying neither are older still.\n")
        if extra_caveat:
            f.write("\n")
            f.write(extra_caveat)
        if len(gov_keys) > 1:
            f.write("\n")
            f.write(_CROSS_ARM_CALIBRATION_CAVEAT)
    return txt_path


def _manifest_base_seed(output_root: str, default: int = 0) -> int:
    """``config.base_seed`` from the sweep's manifest, or *default*."""
    try:
        with open(os.path.join(output_root, MANIFEST_NAME), "r",
                  encoding="utf-8") as f:
            return int((json.load(f).get("config") or {})["base_seed"])
    except (json.JSONDecodeError, OSError, UnicodeDecodeError,
            AttributeError, KeyError, TypeError, ValueError):
        return default


def _draw_half_split_median_ci(
    output_root: str,
    plots_dir: str,
    *,
    plotted2: Sequence[str],
    halves_meta: Sequence[Tuple[str, str]],
    half_series: Mapping[Tuple[str, str], Mapping[int, List[float]]],
    half_diffs: Sequence[int],
    single2: bool,
    re_split: bool,
    title: str,
    legend_title: str,
    split_note: str,
    scale: str,
    gov_label: Callable[[str], str],
) -> List[str]:
    """Write ``calibration_half_split_median_ci.png`` and its sidecar.

    Same data as ``calibration_half_split.png``; median + 95% bootstrap CI
    markers instead of IQR boxes.  Visual grammar (fan offsets, error bars,
    open markers, trend line) is copied from :func:`make_median_ci_plots` so
    the two CI figure families read the same way.  Colour follows the
    half-split figure's rule: window colours when one government is drawn,
    government colours (window carried by line style and marker) otherwise.

    *re_split* is ``split_window is not None`` at
    the call site: whether the window boundary was re-derived from the
    archive's own per-cycle burn-in estimate, or the sidecar falls back to
    the fixed split the sweep bucketed at run time (see
    ``make_ads_calibration_plots``'s own branching, which this mirrors).  The
    sidecar's ``Metric:`` line must say which one actually produced
    *half_series*, since asserting "re-split post-burn-in" unconditionally
    would be false on the fixed-split path.
    """
    base_seed = _manifest_base_seed(output_root)
    sorted_diffs = list(half_diffs)
    x_range = (max(sorted_diffs) - min(sorted_diffs)) if len(sorted_diffs) > 1 else 10
    n_series = len(plotted2) * len(halves_meta)
    fan = x_range * _MEDIAN_CI_FAN_FRACTION
    offsets = ([-fan / 2.0 + fan * i / (n_series - 1) for i in range(n_series)]
               if n_series > 1 else [0.0])
    half_markers = ("o", "s")

    fig, ax = plt.subplots(figsize=(16, 7))
    handles = []
    palette_lines: List[str] = []
    status_counts: Dict[str, int] = {}
    drawn_medians: List[float] = []
    cell_ns: List[int] = []
    flagged: List[str] = []
    idx = 0
    for g in plotted2:
        for half_i, (label, key) in enumerate(halves_meta):
            offset = offsets[idx]
            idx += 1
            if single2:
                color = _HALF_SPLIT_COLORS[half_i]
                plabel = label
            else:
                color = GOV_COLORS.get(g, "#333333")
                plabel = f"{gov_label(g)}, {label.lower()}"
            linestyle = _HALF_SPLIT_LINESTYLES[half_i]
            marker = half_markers[half_i % len(half_markers)]

            line_xs: List[float] = []
            line_ys: List[float] = []
            for d in sorted_diffs:
                vals = list(half_series[(g, key)].get(d, []))
                ci = bootstrap_median_ci(
                    vals, derive_seed(base_seed, "plot.bootstrap.half_split",
                                      g, key, d))
                status_counts[ci.status] = status_counts.get(ci.status, 0) + 1
                if ci.median is None:
                    continue
                drawn_medians.append(ci.median)
                cell_ns.append(ci.n)
                x = d + offset
                line_xs.append(x)
                line_ys.append(ci.median)
                if ci.drawable:
                    ax.errorbar(
                        [x], [ci.median],
                        yerr=[[ci.median - (ci.lower or ci.median)],
                              [(ci.upper or ci.median) - ci.median]],
                        fmt=marker, color=color, markersize=4.5,
                        markerfacecolor=color, markeredgecolor=color,
                        elinewidth=1.4, capsize=2.5, capthick=1.4, zorder=4,
                    )
                else:
                    flagged.append(f"{gov_label(g)} {label} D{d}: {ci.describe()}")
                    ax.plot(
                        x, ci.median, marker=marker, color=color,
                        markersize=4.5, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.2,
                        linestyle="none", zorder=4,
                    )
            if len(line_xs) > 1:
                ax.plot(line_xs, line_ys, color=color, linewidth=1.5,
                        linestyle=linestyle, alpha=0.75, zorder=2)
            handles.append(plt.Line2D(
                [0], [0], color=color, linewidth=2.5, linestyle=linestyle,
                marker=marker, markersize=6, label=plabel,
            ))
            palette_lines.append(
                f"{plabel:34} | Color: {color} | Line style: {linestyle} | "
                f"Marker: {marker}"
            )

    if not drawn_medians:
        plt.close(fig)
        return []

    _style_difficulty_axes(
        ax, sorted_diffs, x_range,
        title=title,
        ylabel="Mean |predicted − realized| health per agent",
    )
    ax.legend(handles=handles, title=legend_title, title_fontsize=10,
              fontsize=9, loc="best", framealpha=0.88)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    png = os.path.join(plots_dir, "calibration_half_split_median_ci.png")
    fig.savefig(png, dpi=_MEDIAN_CI_DPI, bbox_inches="tight")
    plt.close(fig)

    n_caption = (f"{min(cell_ns)}–{max(cell_ns)}" if min(cell_ns) != max(cell_ns)
                 else str(cell_ns[0]))
    govs = " vs ".join(gov_label(g) for g in plotted2)
    clauses = [
        f"Median with 95% bootstrap confidence interval: {govs} "
        f"self-calibration error in an early and a late window within each "
        f"run, {n_series} series per difficulty level.",
        f"Window: {split_note}.",
        f"Difficulty levels {sorted_diffs[0]} to {sorted_diffs[-1]} "
        f"({len(sorted_diffs)} levels); n = {n_caption} runs per series per "
        f"level.",
        "Lower is a better forecast.",
    ]
    lo, hi = min(drawn_medians), max(drawn_medians)
    clauses.append(f"All medians are {lo:.3f}." if f"{lo:.3f}" == f"{hi:.3f}"
                   else f"Medians span {lo:.3f} to {hi:.3f}.")
    caveat = [
        "Same data as calibration_half_split.png (one per-run mean per window "
        "per run); only the summary drawn differs.\n",
        f"  Interval: 95% percentile bootstrap on the MEDIAN, "
        f"{_MEDIAN_CI_N_BOOT:,} resamples, seed derive_seed(base_seed="
        f"{base_seed}, \"plot.bootstrap.half_split\", government, window, "
        f"difficulty).\n",
        "  Open marker = median drawn without an interval.\n",
        "  Interval status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())) + "\n",
    ]
    if flagged:
        caveat.append("  Cells without an interval:\n")
        caveat.extend(f"    {line}\n" for line in flagged)
    txt = _write_calibration_sidecar(
        png,
        title=f"{title} — median with 95% bootstrap CI",
        # Branches exactly like the box figure's own `metric=` (single-ADS vs
        # multi-government, re-split vs fixed split) instead of asserting
        # "per-run window means" for every archive unconditionally -- that
        # phrase is true for the re-split path but not the fixed-split one.
        metric=(
            (
                "calibration_cycle_deltas, per-run window means, re-split "
                "post-burn-in"
                if re_split else
                "calibration_mean_abs_delta_first_half vs "
                "calibration_mean_abs_delta_second_half (fixed split)"
            ) if single2 else
            (
                "calibration_cycle_deltas (ADS, unfiltered) / "
                "calibration_cycle_deltas_by_outcome_then_scope (other "
                "governments, filtered), per-run window means, re-split "
                "post-burn-in"
                if re_split else
                "calibration_mean_abs_delta_first_half/_second_half (ADS) / "
                "filtered_half_split (other governments) — fixed split"
            )
        ),
        alt_text=_clauses_to_alt_text(clauses, 100),
        n_levels=len(sorted_diffs),
        palette_lines=palette_lines,
        extra_caveat="".join(caveat),
        gov_keys=list(plotted2),
        scale=scale,
    )
    log(f"  Plot saved → {png}")
    log(f"  Alt text saved → {txt}")
    return [png, txt]


def make_ads_calibration_plots(
    output_root: str,
    plots_dir: str,
    *,
    gov_keys: Sequence[str] = (_ADS_GOV_KEY,),
) -> List[str]:
    """
    Render the self-calibration figures from an existing output tree.

    **Takes no BenchmarkConfig and reads no in-memory sweep state.**  Every
    value and every caption comes off disk: the per-run scalars from
    ``<gov_key>/<difficulty>/run_*/final_stats.json``, the scale clause from
    the sweep's own ``manifest.json``.  That is the point of the signature,
    not an accident of it — it makes "these figures can be regenerated from
    an archived output directory alone" a structural property rather than a
    convention, because there is no in-memory input available to withhold.

    **``gov_keys``.**  Generalizes the ADS-only default (gated by the
    module-level scalar ``_ADS_GOV_KEY``) to any tuple of forecast-carrying
    governments; the default ``(_ADS_GOV_KEY,)`` reproduces the ADS-only
    figures byte-for-byte.  Every non-ADS government's
    number is routed through :func:`filtered_calibration_mean` /
    :func:`filtered_half_split` (via :func:`_run_calibration_mean` and
    friends) rather than read off the unfiltered fields ADS itself uses —
    see those functions' docstrings for why an unfiltered read is wrong for
    any government whose ``outcome`` tag is not constant.  A government
    present in *gov_keys* but absent from *output_root*, or present but with
    no usable data for one particular figure, is silently omitted from that
    figure rather than raising — the other requested government(s), if any,
    still plot.

    **Burn-in breakpoint is estimated ONCE and shared across every plotted
    government** (Figures 2 and 3), from ADS's own pooled series when ADS is
    among *gov_keys*, or from the first active government otherwise.  This is
    deliberate, not an oversight: ceiling-pinned burn-in
    (see the comment block above :func:`estimate_burn_in_breakpoint`) is a
    property of the SIMULATION's health dynamics near cycle 0, common to
    every government, not of which one is forecasting — and a non-ADS arm's
    filtered series (restricted to ``enacted``/``replaced`` closures) is far
    thinner than ADS's own pooled series, often too sparse to independently
    clear the estimator's ``_BURN_IN_MIN_SEGMENT_OBS`` floor.  One physically
    motivated, well-powered breakpoint beats several noisy ones that a reader
    could not tell apart on the same axis.  Stated in the sidecar whenever
    more than one government is plotted.

    **Scope-matched comparison is a SEPARATE function, not a series on this
    one.**  A four-series overlay directly on ``calibration_trend_by_cycle.png``
    is not attempted: ``calibration_half_split.png`` is already at its
    readability limit with two governments times two halves, and a
    per-difficulty scope-matched series has no home on
    ``calibration_trend_by_cycle.png`` at all (that figure's x-axis is cycle,
    not difficulty). See
    :func:`make_scope_matched_calibration_plots` for the two figures this
    produces (``mean_prediction_error_scope_matched.png`` and
    ``calibration_trend_by_cycle_scope_matched.png``) and for why a
    scope-matched ``calibration_half_split.png`` counterpart is NOT drawn
    at all, not even as a fourth figure.

    That function reuses this docstring's own thinness measurement rather than
    re-deriving it: ADS's ``n_groups == 1`` ``enacted``/``replaced`` closures
    on the real k=10 archive pool to only 515 observations across 94 distinct
    cycles (median 4/cycle, 54 of those 94 cycles with fewer than 5
    observations) and 21 difficulty levels (~25/level, 2 levels under 10) —
    independently re-counted from
    ``benchmark_results/ads/*/run_*/final_stats.json``'s
    ``calibration_cycle_deltas_by_outcome_then_scope["enacted"]["1"]`` entries.
    Thin enough that :data:`_SCOPE_MATCHED_MIN_N` suppresses a per-cell mean
    rather than drawing one over 1-4 observations, but not so thin that the
    pooled, closure-as-unit comparison this docstring's own figures already
    use for ``calibration_trend_by_cycle.png`` is unusable; only a PER-RUN
    statistic (this figure's
    ``mean_prediction_error.png``/``calibration_half_split.png`` box-plot
    shape) is foreclosed on this subset.  The 515/94/54 numbers above were
    measured directly against THIS figure's own viability bar, which is a
    different, stricter test than the pooled closure-as-unit view's bar.

    Three figures:

    ``mean_prediction_error.png``
        One Tukey IQR box per difficulty over each run's all-time mean
        ``|predicted - realized|``.  This is the figure that was asked for.

    ``calibration_half_split.png``
        The same error split into two halves of each run, two boxes per
        difficulty — the more direct "is the forecast actually getting better
        within a run" view.

        **The split point is derived from the data and the burn-in is
        excluded.**  A round number like cycle 75, chosen without regard for
        the shape of the series, risks putting the entire ceiling-pinned
        burn-in inside the first half — reproducing the saturation artifact
        instead of providing an independent check.  The window instead
        starts at the breakpoint :func:`estimate_burn_in_breakpoint` finds
        and is split at its midpoint, giving two equal-width post-burn-in
        windows.

        Two consequences of that, both deliberate:

        * The figure reads the raw ``calibration_cycle_deltas`` pairs rather
          than the ``calibration_mean_abs_delta_*_half`` scalars: the scalars
          are bucketed at cycle 75 and cannot be re-bucketed, while the raw
          pairs can — which is what makes this a replot rather than a
          resweep.  The scalars are still written and read by nothing else;
          they remain the fallback for an archive too old or too thin to
          carry the pairs.
        * The sample-size sentence in the sidecar is computed, not asserted.
          Asserting "the second half is systematically thinner because laws
          enacted near the end of a run never expire and so never close"
          would not hold once closures are graded on a fixed T+20 review
          rather than law expiry: that conclusion is **false in 205 of 210
          runs**.  See :func:`_half_sample_size_clause`.

    ``calibration_trend_by_cycle.png``
        The first two figures cannot show a within-run trend: the first
        collapses each run to one scalar, and the second is a coarse
        two-bucket split. This one plots the per-cycle mean ``|predicted -
        realized|`` (with a 95% CI band) against cycle number, pooled across
        every difficulty level and every run in the sweep. Sourced from
        ``calibration_cycle_deltas`` in ``final_stats.json`` via
        :func:`load_ads_calibration_cycle_deltas`.
        Deliberately pooled rather than faceted by difficulty — see that
        function's docstring.

        **The headline fit is the POST-burn-in one, not a single full-range
        trend line.**  The burn-in region is shaded; the full-range fit is
        still drawn, greyed and explicitly labelled as not interpretable,
        because the reader needs to see *why* it is wrong rather than be
        told: a single full-range fit would report "error GROWS as a run
        progresses, predictions get worse, not better", which does not
        survive scrutiny.  See the module-level comment above
        :func:`estimate_burn_in_breakpoint`.

    Consequences of the disk-only contract, all deliberate:

    * The x-axis is discovered from the ``ads/<difficulty>/`` directories that
      exist, not from ``config.difficulties``.  A narrowed or partly failed
      sweep plots exactly the levels it has data for rather than reserving
      empty ticks — so this figure's axis can legitimately be narrower than
      ``normalized_health_score.png``'s from the same sweep.
    * The per-cell n in the caption counts runs that reported a value.

    Skips quietly (INFO, no stub file, no exception) when the sweep contains no
    ``ads`` output — the normal case for ``--governments anarchy,autocracy``.

    Returns the paths written, PNG and sidecar, in write order.
    """
    active_govs = [g for g in gov_keys if os.path.isdir(os.path.join(output_root, g))]
    if not active_govs:
        # INFO, not WARNING: a sweep that excludes every requested government
        # is a supported, ordinary invocation, and a warning here would train
        # readers to ignore warnings in the sweep log.
        if tuple(gov_keys) == (_ADS_GOV_KEY,):
            log(f"  No {_ADS_GOV_KEY}/ output under {output_root} — "
                f"ADS calibration plots skipped (expected when the sweep "
                f"excludes {_ADS_GOV_KEY}).")
        else:
            log(f"  No {'/'.join(gov_keys)}/ output under {output_root} — "
                f"calibration plots skipped (expected when the sweep "
                f"excludes every requested government).")
        return []
    for g in gov_keys:
        if g not in active_govs:
            log(f"  No {g}/ output under {output_root} — its calibration "
                f"series is skipped; the other requested government(s), if "
                f"any, still plot.")

    # Read once per active government and reused by both Figure 1 (the
    # `final_mean_prediction_error` key) and Figure 2's fallback branch (the
    # two `calibration_mean_abs_delta_*_half` keys) — the same structure the
    # single-government version used for its one `series` dict.
    series_by_gov = {
        g: load_ads_calibration_series(output_root, gov_key=g) for g in active_govs
    }
    written: List[str] = []
    scale = _manifest_scale_caption(output_root)
    scale_clause = f", {scale}" if scale else ""

    def _axis_geometry(diffs: List[int]) -> Tuple[float, float]:
        """(x_range, box_width) using make_plots' own rule for x_range."""
        x_range = (max(diffs) - min(diffs)) if len(diffs) > 1 else 10
        # make_plots uses 1% of the range because eight government boxes are
        # deliberately overlaid at the same x and any wider would be a smear.
        # These figures draw at most a handful of series, so a 1% box would be
        # a hairline for no benefit.  Only the WIDTH differs; every other
        # aspect of the box comes from the shared _draw_box_series, which is
        # what actually carries the visual consistency.
        return x_range, x_range * 0.025

    def _gov_label(g: str) -> str:
        return GOV_DISPLAY.get(g, g)

    def _gov_join(govs: Sequence[str]) -> str:
        return " vs ".join(_gov_label(g) for g in govs)

    # -- Figure 1: all-time mean absolute prediction error -------------------
    per_gov_mean = {
        g: series_by_gov[g]["final_mean_prediction_error"] for g in active_govs
    }
    plotted1 = [g for g in active_govs if any(per_gov_mean[g].values())]
    if not plotted1:
        log("  No run recorded final_mean_prediction_error (or its filtered "
            "equivalent) for any requested government — "
            "mean_prediction_error.png skipped.")
    else:
        single1 = plotted1 == [_ADS_GOV_KEY]
        sorted_diffs = sorted({
            d for g in plotted1 for d, vals in per_gov_mean[g].items() if vals
        })
        x_range, box_width = _axis_geometry(sorted_diffs)
        offsets = _series_offsets(len(plotted1), box_width)

        fig, ax = plt.subplots(figsize=(16, 7))
        handles = []
        caption_by_gov: Dict[str, str] = {}
        all_median_ys: List[float] = []
        for g, offset in zip(plotted1, offsets):
            color = GOV_COLORS.get(g, "#333333")
            linestyle = GOV_LINESTYLES.get(g, "-")
            per_diff = per_gov_mean[g]
            _, median_ys = _draw_box_series(
                ax,
                [(d + offset, per_diff.get(d, [])) for d in sorted_diffs],
                color=color,
                box_width=box_width,
                linestyle=linestyle,
                label=g,
            )
            all_median_ys.extend(median_ys)
            caption_by_gov[g] = _n_per_cell_caption(per_diff)
            handles.append(plt.Line2D(
                [0], [0], color=color, linewidth=2.5, linestyle=linestyle,
                marker="o", markersize=6, label=_gov_label(g),
            ))

        distinct_captions = {caption_by_gov[g] for g in plotted1}
        n_caption = (
            next(iter(distinct_captions)) if len(distinct_captions) == 1
            else "; ".join(f"{_gov_label(g)} {caption_by_gov[g]}"
                           for g in plotted1)
        )
        gov_title_word = "ADS" if single1 else _gov_join(plotted1)

        _style_difficulty_axes(
            ax, sorted_diffs, x_range,
            title=f"{gov_title_word} Calibration Error by Difficulty Level",
            ylabel="Mean |predicted − realized| health per agent",
        )
        # See _widen_axis_for_drawn_boxes's docstring: this is a no-op for the
        # single-government case (len(plotted1) <= 2), and only widens beyond
        # _style_difficulty_axes's guarded 5% pad when a series count this
        # figure has never needed before would otherwise clip.
        _widen_axis_for_drawn_boxes(ax, sorted_diffs, x_range)
        ax.legend(
            handles=handles,
            title="Government",
            title_fontsize=10, fontsize=9, loc="best", framealpha=0.88,
        )
        fig.tight_layout()

        # Created here rather than on entry so that a call which draws nothing
        # leaves nothing behind — a standalone regeneration pointed at a tree
        # with no usable data should not conjure an empty plots/ directory.
        os.makedirs(plots_dir, exist_ok=True)
        png = os.path.join(plots_dir, "mean_prediction_error.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(png)

        # Every number in the sidecar is derived from `all_median_ys`, the
        # trend line(s) the figure actually drew, so the description cannot
        # drift from the picture.
        clauses = [
            f"Tukey IQR boxplot: {gov_title_word} self-calibration accuracy, "
            "the mean absolute difference between the 20-cycle "
            "prediction and the realized agent health 20 cycles later."
        ]
        clauses.append(
            f"Difficulty levels {sorted_diffs[0]} to {sorted_diffs[-1]} "
            f"({len(sorted_diffs)} levels), "
            f"{n_caption}."
        )
        clauses.append(
            "One box per difficulty level spanning the interquartile range "
            "with 1.5x IQR whiskers, and a line joining the medians. Lower "
            "is a better forecast."
        )
        if all_median_ys:
            lo, hi = min(all_median_ys), max(all_median_ys)
            clauses.append(
                f"All medians are {lo:.3f}." if f"{lo:.3f}" == f"{hi:.3f}"
                else f"Medians span {lo:.3f} to {hi:.3f}."
            )
        txt = _write_calibration_sidecar(
            png,
            title=f"{gov_title_word} Calibration Error",
            metric=(
                "final_mean_prediction_error" if single1 else
                "final_mean_prediction_error (ADS, unfiltered) / "
                "filtered_calibration_mean (other governments, outcomes="
                "{enacted,replaced})"
            ),
            alt_text=_clauses_to_alt_text(clauses, 100),
            n_levels=len(sorted_diffs),
            palette_lines=[
                f"{_gov_label(g):15} | Color: {GOV_COLORS.get(g, '#333333')} | "
                f"Line style: {GOV_LINESTYLES.get(g, '-')}"
                for g in plotted1
            ],
            extra_caveat=(
                _scope_matched_pointer_caveat("mean_prediction_error_scope_matched.png")
                if _ADS_GOV_KEY in plotted1 else ""
            ),
            gov_keys=plotted1,
            scale=scale,
        )
        written.append(txt)
        log(f"  Plot saved → {png}")
        log(f"  Alt text saved → {txt}")

    # -- Shared input for Figures 2 and 3 ------------------------------------
    #
    # Both remaining figures are views of the same per-cycle records, and both
    # need to know where the burn-in ends, so the tree is walked ONCE PER
    # GOVERNMENT and the breakpoint is estimated ONCE overall (see the
    # function docstring's "Burn-in breakpoint is estimated ONCE" section for
    # why it is shared rather than re-estimated per government).  Estimating
    # it more than once would not merely be wasteful: the half-split's window
    # and the trend figure's shading would be free to disagree across figures
    # AND across governments, and a reader comparing them would have no way to
    # tell that they had.
    by_run_per_gov = {
        g: load_ads_calibration_cycle_deltas_by_run(output_root, gov_key=g)
        for g in active_govs
    }
    primary_gov = _ADS_GOV_KEY if _ADS_GOV_KEY in active_govs else active_govs[0]
    primary_label = _gov_label(primary_gov)
    by_run = by_run_per_gov[primary_gov]
    pooled_x: List[float] = []
    pooled_y: List[float] = []
    for _runs in by_run.values():
        for _series in _runs:
            for _c, _v in _series:
                pooled_x.append(float(_c))
                pooled_y.append(_v)

    burn_in = (
        estimate_burn_in_breakpoint(pooled_x, pooled_y) if pooled_x else None
    )
    if burn_in is not None:
        log(f"  {primary_label} calibration: burn-in breakpoint estimated at "
            f"cycle {burn_in.breakpoint} (segmented least squares; pre-slope "
            f"{burn_in.slope_pre:+.3e}, post-slope {burn_in.slope_post:+.3e}, "
            f"SSR improvement over a single line "
            f"{burn_in.ssr_improvement * 100:.1f}%)")
        if len(active_govs) > 1:
            log(f"    breakpoint shared across all plotted governments: "
                f"{', '.join(_gov_label(g) for g in active_govs)}")
    elif pooled_x:
        bench_log(logging.INFO,
                  "  %s calibration: too few pooled observations (%d over "
                  "cycles %d-%d) to estimate a burn-in breakpoint; figures "
                  "fall back to the unsegmented view and say so.",
                  primary_label, len(pooled_x), int(min(pooled_x)),
                  int(max(pooled_x)))

    # The post-burn-in window, and the cycle that splits it in two.  Both
    # figures read these; ``None`` means "no defensible split exists on this
    # corpus", which is a normal outcome at smoke-test scale and must be
    # reported rather than papered over with a round number.
    split_window: Optional[Tuple[int, int, int]] = None
    if burn_in is not None and pooled_x:
        _lo = burn_in.breakpoint
        _hi = int(max(pooled_x))
        if _hi - _lo >= 2:
            split_window = (_lo, _lo + (_hi - _lo) // 2, _hi)

    # -- Figure 2: post-burn-in early vs late half (beyond the literal ask) --
    #
    # This figure is the design's nominated "honest within-run comparison".
    # At a fixed cycle-75 split it would not be one: the ceiling-pinned
    # burn-in would sit entirely inside the first half, reproducing the
    # saturation artifact instead of testing it.  The split is re-derived
    # from disk instead — see the function docstring and
    # :func:`estimate_burn_in_breakpoint`.
    #
    # This figure stays PRIMARY-SERIES ONLY — no scope-matched
    # (``n_groups == 1``) secondary series is drawn here for any
    # government.  What generalizes is the number of PRIMARY series: one full
    # early/late pair per plotted government, keyed by ``(gov, half_key)``
    # rather than by ``half_key`` alone.
    half_series: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    half_closures: Dict[Tuple[str, str], int] = {}
    halves_meta: Tuple[Tuple[str, str], ...]
    split_note: str

    if split_window is not None:
        w_lo, w_mid, w_hi = split_window
        early_label = f"Cycles {w_lo}–{w_mid - 1}"
        late_label = f"Cycles {w_mid}–{w_hi}"
        halves_meta = ((early_label, "early"), (late_label, "late"))
        for g in active_govs:
            for _, key in halves_meta:
                half_series[(g, key)] = {}
                half_closures[(g, key)] = 0
            for difficulty, runs in by_run_per_gov[g].items():
                for _, key in halves_meta:
                    half_series[(g, key)].setdefault(difficulty, [])
                for run_series in runs:
                    early = [v for c, v in run_series if w_lo <= c < w_mid]
                    late = [v for c, v in run_series if w_mid <= c <= w_hi]
                    # A run contributes one point per window, and only for a
                    # window in which it actually closed something.  Averaging
                    # over an empty window would silently invent a zero — this
                    # is also how a run whose arm enacted/replaced nothing
                    # (the 5-of-210 A+L case observed on the production
                    # corpus) degrades
                    # gracefully: it contributes to neither window rather than
                    # a fabricated one.
                    if early:
                        half_series[(g, "early")][difficulty].append(
                            float(np.mean(early)))
                        half_closures[(g, "early")] += len(early)
                    if late:
                        half_series[(g, "late")][difficulty].append(
                            float(np.mean(late)))
                        half_closures[(g, "late")] += len(late)
        split_note = (
            f"split at cycle {w_mid}, burn-in (cycles < {w_lo}) EXCLUDED"
        )
    else:
        # Fallback: the archive cannot support a re-derived split, so use the
        # scalars the sweep bucketed at its own fixed cycle — ADS's own
        # unfiltered fields, and every other government's filtered equivalent
        # via `series_by_gov` (see `load_ads_calibration_series`'s routing).
        # Labelled as the fixed split everywhere it appears — this path must
        # never be mistaken for the re-derived one.
        halves_meta = (
            ("First half of run",
             "calibration_mean_abs_delta_first_half"),
            ("Second half of run",
             "calibration_mean_abs_delta_second_half"),
        )
        for g in active_govs:
            for _, field in halves_meta:
                half_series[(g, field)] = series_by_gov[g][field]
                half_closures[(g, field)] = sum(
                    len(v) for v in series_by_gov[g][field].values()
                )
        split_note = (
            "FIXED split recorded by the sweep — the burn-in is NOT excluded"
        )

    plotted2 = [
        g for g in active_govs
        if any(vals for _, key in halves_meta
               for vals in half_series[(g, key)].values())
    ]
    half_diffs = sorted({
        d
        for g in plotted2
        for _, key in halves_meta
        for d, vals in half_series[(g, key)].items() if vals
    })
    if not half_diffs:
        log("  No run recorded a half-split calibration error for any "
            "requested government — calibration_half_split.png skipped.")
        return written

    single2 = plotted2 == [_ADS_GOV_KEY]
    x_range, box_width = _axis_geometry(half_diffs)
    n_series2 = len(plotted2) * len(halves_meta)
    # Place series side by side at each tick, adjacent but not touching.  For
    # a single government this is the original ±0.75×box_width convention
    # (`_series_offsets(2, box_width)` reproduces that pair exactly); for
    # several governments it spreads all `n_series2` boxes symmetrically.
    offsets = _series_offsets(n_series2, box_width)

    fig, ax = plt.subplots(figsize=(16, 7))
    handles = []
    palette_lines2: List[str] = []
    series_captions: List[Tuple[str, str, str, str]] = []  # (gov, half_label, half_key, caption)
    series_idx = 0
    for g in plotted2:
        for half_i, (label, key) in enumerate(halves_meta):
            offset = offsets[series_idx]
            series_idx += 1
            if single2:
                # Two non-government
                # colours distinguishing the halves, licensed by this figure
                # carrying no government legend when only one government is
                # plotted (see the comment above `_HALF_SPLIT_COLORS`).
                color = _HALF_SPLIT_COLORS[half_i]
                linestyle = _HALF_SPLIT_LINESTYLES[half_i]
                plabel = label
            else:
                # Once more than one
                # government is on the figure, colour MUST mean government
                # (GOV_COLORS) and the half is carried by linestyle instead.
                color = GOV_COLORS.get(g, "#333333")
                linestyle = _HALF_SPLIT_LINESTYLES[half_i]
                plabel = f"{_gov_label(g)}, {label.lower()}"
            per_half = half_series[(g, key)]
            caption = _n_per_cell_caption(per_half)
            series_captions.append((g, label, key, caption))
            _draw_box_series(
                ax,
                [(d + offset, per_half.get(d, [])) for d in half_diffs],
                color=color,
                box_width=box_width,
                linestyle=linestyle,
                label=key,
            )
            handles.append(plt.Line2D(
                [0], [0], color=color, linewidth=2.5, linestyle=linestyle,
                marker="o", markersize=6, label=plabel,
            ))
            pcm_label = label if single2 else f"{_gov_label(g)} {label}"
            palette_lines2.append(
                f"{pcm_label:20} | Color: {color} | Line style: {linestyle}"
            )

    # The series usually share a sample size, in which case saying it
    # repeatedly in the sidecar is noise; when they diverge, say each.
    # `n_caption` lives in the sidecar's alt-text clauses rather than the
    # title (see below).
    #
    # Keyed off `_half_window_tag(key)` -- the window's stable identity --
    # rather than a substring of `label` (e.g. `label.split()[0].lower()`):
    # both windows' labels can start with the same word ("Cycles 81-114" /
    # "Cycles 115-149"), which would make the per-window n this line reports
    # unattributable.
    distinct_n = {c for _, _, _, c in series_captions}
    if len(distinct_n) == 1:
        n_caption = next(iter(distinct_n))
    elif single2:
        n_caption = "; ".join(
            f"{_half_window_tag(key)} half {caption}"
            for _, _label, key, caption in series_captions
        )
    else:
        n_caption = "; ".join(
            f"{_gov_label(g)} {_half_window_tag(key)} {caption}"
            for g, _label, key, caption in series_captions
        )
    gov_title_word2 = "ADS" if single2 else _gov_join(plotted2)
    # `n_caption` (the per-government, per-half n disclosure) lives in the
    # sidecar (see the clauses built below), not in the title, so
    # `half_title` carries only the plotted quantity.
    half_title = (f"{gov_title_word2} Calibration Error by Difficulty "
                  f"Level, Early vs Late Cycles")
    half_legend_title = "Window" if single2 else "Government, window"
    _style_difficulty_axes(
        ax, half_diffs, x_range,
        title=half_title,
        ylabel="Mean |predicted − realized| health per agent",
    )
    # n_series2 reaches 4 (2 governments x 2 halves) once a second
    # government is plotted, past the 2-series geometry
    # _style_difficulty_axes's guarded 5% pad was sized for. See
    # _widen_axis_for_drawn_boxes's docstring for why the fix lives here
    # rather than in that guarded function.
    _widen_axis_for_drawn_boxes(ax, half_diffs, x_range)
    ax.legend(handles=handles, title=half_legend_title,
              title_fontsize=10, fontsize=9, loc="best", framealpha=0.88)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)   # see the note at the figure above
    png = os.path.join(plots_dir, "calibration_half_split.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(png)

    # Clause ORDER is load-bearing here for the same reason it is on Figure 3:
    # _clauses_to_alt_text drops clauses from the END to meet the 100-word
    # budget, so what a screen-reader user must not lose goes early.  What they
    # must not lose on this figure is that the burn-in was excluded — without it
    # the reader has no way to know this is not the discredited cycle-75 view.
    if single2:
        clauses = [
            "Tukey IQR boxplot: ADS self-calibration error split into an "
            "early and a late window within each run, two boxes per "
            "difficulty level."
        ]
    else:
        clauses = [
            f"Tukey IQR boxplot: {gov_title_word2} self-calibration error "
            f"split into an early and a late window within each run, "
            f"{n_series2} boxes per difficulty level."
        ]
    clauses.append(f"Window: {split_note}.")
    clauses.append(
        f"Difficulty levels {half_diffs[0]} to {half_diffs[-1]} "
        f"({len(half_diffs)} levels); {n_caption}."
    )
    # Computed from the closure counts, never asserted: asserting the second
    # half is "systematically thinner" would be false in 205 of 210 runs.
    if single2:
        n_early = half_closures.get((plotted2[0], halves_meta[0][1]), 0)
        n_late = half_closures.get((plotted2[0], halves_meta[1][1]), 0)
        clauses.append("Lower is a better forecast. "
                       + _half_sample_size_clause(n_early, n_late))
        extra_caveat2 = _half_split_caveat(
            split_window, burn_in, n_early, n_late, halves_meta)
    else:
        clauses.append(
            "Lower is a better forecast. Per-government closure counts for "
            "each window are in the sidecar's interpretation caveat below."
        )
        caveat_lines = [
            "Interpretation caveat — how this figure's split was chosen, and "
            "per-government closure counts:\n"
        ]
        if split_window is not None:
            w_lo, w_mid, w_hi = split_window
            caveat_lines.append(
                f"  The split is SHARED across every plotted government: "
                f"cycles below {w_lo} are discarded\n"
                f"  as burn-in (estimated from {primary_label}'s pooled "
                f"series — see the function\n"
                f"  docstring's 'Burn-in breakpoint is estimated ONCE' "
                f"note), and the remaining window\n"
                f"  {w_lo}-{w_hi} is divided at its midpoint, cycle "
                f"{w_mid}.  See calibration_trend_by_cycle's\n"
                f"  sidecar for why the burn-in must be excluded before "
                f"comparing halves.\n"
            )
        else:
            caveat_lines.append(
                "  THIS IS THE FIXED SPLIT AND IT IS NOT THE HONEST "
                "COMPARISON.\n"
                "  No burn-in breakpoint could be re-derived on this "
                "archive.\n"
            )
        caveat_lines.append("\n  Closures per window:\n")
        for g in plotted2:
            for label, key in halves_meta:
                caveat_lines.append(
                    f"    {_gov_label(g)} — {label}: "
                    f"{half_closures.get((g, key), 0)}\n"
                )
        caveat_lines.append("\n")
        for line in _burn_in_evidence_lines(burn_in):
            caveat_lines.append(f"  {line}\n")
        extra_caveat2 = "".join(caveat_lines)
    if _ADS_GOV_KEY in plotted2:
        extra_caveat2 = extra_caveat2 + "\n" + _scope_matched_pointer_caveat(None)
    txt = _write_calibration_sidecar(
        png,
        title=f"{gov_title_word2} Calibration Error — Early vs Late Within a Run",
        metric=(
            (
                "calibration_cycle_deltas, re-split post-burn-in"
                if split_window is not None else
                "calibration_mean_abs_delta_first_half vs "
                "calibration_mean_abs_delta_second_half (fixed split)"
            ) if single2 else
            (
                "calibration_cycle_deltas (ADS, unfiltered) / "
                "calibration_cycle_deltas_by_outcome_then_scope (other "
                "governments, filtered), re-split post-burn-in"
                if split_window is not None else
                "calibration_mean_abs_delta_first_half/_second_half (ADS) / "
                "filtered_half_split (other governments) — fixed split"
            )
        ),
        alt_text=_clauses_to_alt_text(clauses, 100),
        n_levels=len(half_diffs),
        palette_lines=palette_lines2,
        extra_caveat=extra_caveat2,
        gov_keys=plotted2,
        scale=scale,
    )
    written.append(txt)
    log(f"  Plot saved → {png}")
    log(f"  Alt text saved → {txt}")
    if single2:
        log(f"    half-split window: {split_note}; closures "
            f"{n_early} early / {n_late} late")
    else:
        log(f"    half-split window: {split_note}")

    # -- Figure 2b: the same early/late data as median + 95% bootstrap CI ----
    #
    # calibration_half_split.png overlays up to four boxes per tick (2
    # governments x 2 windows) and is too crowded to read.  This sibling draws
    # the IDENTICAL `half_series` the way make_median_ci_plots draws the health
    # metrics: a marker at each cell's median, a bar spanning the 95%
    # percentile-bootstrap CI on that median, open marker = no defensible
    # interval (see MedianCI), and a line joining the medians.  Seeds are
    # derived per (government, window, difficulty) from the sweep's own
    # base_seed under a dedicated domain label, so the figure is
    # byte-reproducible from the archive alone.
    written.extend(_draw_half_split_median_ci(
        output_root, plots_dir,
        plotted2=plotted2, halves_meta=halves_meta, half_series=half_series,
        half_diffs=half_diffs, single2=single2,
        re_split=(split_window is not None), title=half_title,
        legend_title=half_legend_title, split_note=split_note, scale=scale,
        gov_label=_gov_label,
    ))

    # -- Figure 3: calibration error vs cycle, pooled across the whole sweep -
    #
    # Neither of the two figures above resolves how forecast error moves
    # *within* a run: Figure 1 collapses each run to one all-time scalar, and
    # Figure 2's two-bucket halves are too coarse to see a trend, only a
    # before/after snapshot. This figure uses the full per-cycle series
    # persisted in ``calibration_cycle_deltas`` instead of discarding it
    # after bucketing.
    #
    # Whether "the trend shows calibration improving" is a valid reading
    # depends on which schema generation produced the archive, because three
    # incompatible quantities have worn this field name:
    #
    #   * Runs carrying neither ``ads_forecast_schema_version`` nor
    #     ``calibration_schema_version`` differenced the RAW projection
    #     (``_record_prediction(predicted=base_norm)``) against the realized
    #     outcome, while the learned EWMA bias moved only the additive
    #     ranking score.  The graded prediction was never touched by the
    #     feedback loop, so any trend was a property of ``EvaluatorNode``'s
    #     fidelity, not of calibration.
    #   * Runs carrying ``calibration_schema_version`` used a multiplicative
    #     correction that fed ranking directly: the single value
    #     ``c * raw`` was simultaneously what candidates were ranked on, what
    #     was recorded as the forecast, and what was differenced here, so the
    #     trend genuinely reflected calibration.
    #   * Runs carrying ``ads_forecast_schema_version`` (current) have no
    #     multiplicative correction, so the value differenced here is again
    #     the evaluator's own untouched projection.  What makes the trend
    #     meaningful here is that the projection's INPUTS are
    #     inferred from accumulating evidence rather than read from ground
    #     truth: the series responds to a learning mechanism that sits
    #     UPSTREAM of ``evaluate()``, so a flat slope indicts the parameter
    #     ESTIMATOR rather than an under-damped correction.
    #     ``parameter_estimates`` in run_detail.jsonl is what distinguishes
    #     the two.
    #
    # One thing that remains true and bounds the claim: the projection is
    # still a simplified forward model, so a residual is expected on
    # fidelity grounds alone.  On the current schema there is no paired null
    # to difference against — ``calibration_cycle_deltas_raw`` is gone along
    # with the multiplicative correction it was the null for — so the
    # honest within-run read is the first-half/second-half split.
    #
    # Archives from an earlier generation hold a different quantity under
    # this same field name and must not be pooled with these.  Identify a
    # run by its schema key: ``ads_forecast_schema_version`` (current),
    # ``calibration_schema_version`` (multiplicative-correction generation),
    # or neither (oldest).
    #
    # Pooled across every difficulty level and every run — deliberately not
    # faceted by difficulty. See load_ads_calibration_cycle_deltas's own
    # docstring for why: a single run closes only a modest number of
    # predictions over ~150 cycles, so splitting further by difficulty would
    # leave most (cycle, difficulty) cells too thin to plot.
    #
    # This figure fits the POST-burn-in slope as the headline rather than one
    # least-squares line across the whole cycle range: a single straight
    # line is the wrong estimator for a saturating curve whose first tens of
    # cycles are ceiling-pinned, and would produce the sentence "error GROWS
    # as a run progresses, predictions get worse, not better" from a slope
    # that is itself computed correctly but describes the wrong regime.  See
    # the comment block above estimate_burn_in_breakpoint for the mechanism.
    # The figure shades the burn-in, fits each side of it, and quotes the
    # POST-burn-in slope as the headline.  The full-range fit is still drawn,
    # greyed and labelled, because a reader who is only told the full-range
    # line is wrong has to take that on trust, whereas a reader who can see
    # it cutting across a visibly bent curve does not.
    cycle_deltas_by_gov: Dict[str, Dict[int, List[float]]] = {}
    for g in active_govs:
        cd: Dict[int, List[float]] = {}
        for _runs in by_run_per_gov[g].values():
            for _series in _runs:
                for _c, _v in _series:
                    cd.setdefault(_c, []).append(_v)
        cycle_deltas_by_gov[g] = cd

    plotted3 = [g for g in active_govs if cycle_deltas_by_gov[g]]
    if not plotted3:
        log("  No run recorded calibration_cycle_deltas (or its filtered "
            "equivalent) for any requested government — "
            "calibration_trend_by_cycle.png skipped.")
        return written
    single3 = plotted3 == [_ADS_GOV_KEY]

    # Per-arm regression inputs and fits.  Computed once per government and
    # cached in `per_gov_stats`, then drawn in the loop below — the same
    # arithmetic the single-government version did, just per `g`.
    per_gov_stats: Dict[str, Dict[str, Any]] = {}
    for g in plotted3:
        cd = cycle_deltas_by_gov[g]
        sorted_cycles = sorted(cd.keys())
        n_per_cycle = [len(cd[c]) for c in sorted_cycles]
        means = np.array([float(np.mean(cd[c])) for c in sorted_cycles])

        # Per-cycle 95% CI, same "guard n<2" posture as per_government_summary
        # in generate_analysis_files (benchmark_core.py ~line 2650-2660): a
        # cycle that closed only one prediction anywhere in the sweep gets a
        # zero-width band rather than a divide-by-zero or a fabricated spread.
        cis = np.zeros(len(sorted_cycles))
        for i, c in enumerate(sorted_cycles):
            vals = cd[c]
            if len(vals) > 1:
                std = float(np.std(vals, ddof=1))
                sem = std / np.sqrt(len(vals))
                cis[i] = 1.96 * sem

        # The regression is fit on every individual pooled observation, NOT on
        # the per-cycle means computed above — fitting on means would let a
        # sparsely-observed cycle (n=1) carry the same leverage as a densely
        # observed one, which is exactly backwards for a trend claim.
        x_raw: List[float] = []
        y_raw: List[float] = []
        for c in sorted_cycles:
            for delta in cd[c]:
                x_raw.append(float(c))
                y_raw.append(delta)
        x_arr = np.array(x_raw)
        y_arr = np.array(y_raw)
        slope, intercept, r_value, p_value, std_err = _numpy_linregress(x_arr, y_arr)
        n_total = len(x_raw)
        cycle_arr = np.array(sorted_cycles, dtype=float)
        fitted = slope * cycle_arr + intercept

        # Post-burn-in fit: the headline.  Restricted to cycles at or past the
        # SHARED estimated breakpoint (see the docstring's burn-in section),
        # and left as None when no breakpoint could be estimated at all — in
        # which case the figure reports the full-range fit but says plainly
        # that it cannot separate burn-in from trend.
        post_slope: Optional[float] = None
        post_p: Optional[float] = None
        post_n = 0
        post_x: Optional[np.ndarray] = None
        post_fitted: Optional[np.ndarray] = None
        if burn_in is not None:
            mask = x_arr >= burn_in.breakpoint
            post_n = int(mask.sum())
            if post_n > 2:
                ps, pi, _pr, pp, _pse = _numpy_linregress(x_arr[mask], y_arr[mask])
                post_slope, post_p = ps, pp
                post_x = cycle_arr[cycle_arr >= burn_in.breakpoint]
                post_fitted = ps * post_x + pi

        per_gov_stats[g] = dict(
            sorted_cycles=sorted_cycles, n_per_cycle=n_per_cycle, means=means,
            cis=cis, slope=slope, p_value=p_value, n_total=n_total,
            cycle_arr=cycle_arr, fitted=fitted, post_slope=post_slope,
            post_p=post_p, post_n=post_n, post_x=post_x, post_fitted=post_fitted,
        )

    # The full-range-fit "artifact" colour/style is shared across every
    # government — grey, dotted, thin, drawn underneath everything else — and
    # deliberately not a government colour: it is on the figure as an exhibit
    # ("this is what you get if you don't exclude the burn-in"), not as a
    # per-arm result, and the visual hierarchy has to say so on its own even
    # when more than one arm's artifact line is present.
    artifact_color = "#767676"          # >= 4.5:1 on white, WCAG AA for text
    artifact_linestyle = ":"

    fig, ax = plt.subplots(figsize=(16, 7))

    if burn_in is not None:
        overall_min_cycle = min(
            per_gov_stats[g]["sorted_cycles"][0] for g in plotted3
        )
        ax.axvspan(
            float(overall_min_cycle), float(burn_in.breakpoint),
            color="#BBBBBB", alpha=0.25, linewidth=0, zorder=0,
            label=f"Burn-in (cycles < {burn_in.breakpoint})",
        )

    palette_lines3: List[str] = []
    for g in plotted3:
        st = per_gov_stats[g]
        gov_label = _gov_label(g)
        mean_color = GOV_COLORS.get(g, "#333333")
        mean_linestyle = GOV_LINESTYLES.get(g, "-")
        if single3:
            # The post-burn-in fit uses
            # the half-split figure's non-government orange rather than a
            # government colour, licensed by this figure carrying no
            # government legend when only one government is plotted.
            trend_color = _HALF_SPLIT_COLORS[1]
        else:
            # Once more than one
            # government is on the figure, every non-artifact line must use
            # GOV_COLORS so colour keeps meaning "government" consistently
            # across the whole figure set.
            trend_color = mean_color
        trend_linestyle = "--"

        mean_series_label = "Mean" if single3 else f"{gov_label} mean"
        ax.plot(st["sorted_cycles"], st["means"], color=mean_color,
                linestyle=mean_linestyle, linewidth=2.0, marker="o",
                markersize=4, zorder=3, label=mean_series_label)
        ci_label = "95% CI" if single3 else f"{gov_label} 95% CI"
        ax.fill_between(st["sorted_cycles"], st["means"] - st["cis"],
                         st["means"] + st["cis"], color=mean_color, alpha=0.18,
                         linewidth=0, zorder=2, label=ci_label)
        artifact_label = ("OLS, all cycles" if single3
                          else f"{gov_label} OLS, all cycles")
        ax.plot(st["cycle_arr"], st["fitted"], color=artifact_color,
                linestyle=artifact_linestyle, linewidth=1.5, zorder=1,
                label=artifact_label)
        if (st["post_x"] is not None and st["post_fitted"] is not None
                and st["post_slope"] is not None):
            post_label = (
                f"OLS, cycles ≥ {burn_in.breakpoint}" if single3 else
                f"{gov_label} OLS, cycles ≥ {burn_in.breakpoint}"
            )
            ax.plot(st["post_x"], st["post_fitted"], color=trend_color,
                    linestyle=trend_linestyle, linewidth=2.6, zorder=4,
                    label=post_label)

        pcm_label = "Per-cycle mean" if single3 else f"{gov_label} per-cycle mean"
        pfr_label = "Full-range OLS" if single3 else f"{gov_label} full-range OLS"
        ppb_label = "Post-burn-in OLS" if single3 else f"{gov_label} post-burn-in OLS"
        palette_lines3.append(
            f"{pcm_label:22} | Color: {mean_color} | Line style: {mean_linestyle}"
        )
        palette_lines3.append(
            f"{pfr_label:22} | Color: {artifact_color} | "
            f"Line style: {artifact_linestyle}  (ARTIFACT — shown, not claimed)"
        )
        if st["post_slope"] is not None:
            palette_lines3.append(
                f"{ppb_label:22} | Color: {trend_color} | "
                f"Line style: {trend_linestyle}  (the interpretable fit)"
            )

    def _headline_for(g: str) -> str:
        st = per_gov_stats[g]
        prefix = "" if single3 else f"{_gov_label(g)}: "
        if st["post_slope"] is not None:
            return (
                f"{prefix}post-burn-in slope = {st['post_slope']:+.2e} per "
                f"cycle (p = {st['post_p']:.4f}, cycles ≥ {burn_in.breakpoint}); "
                f"full-range {st['slope']:+.2e} is a burn-in artifact"
            )
        return (
            f"{prefix}slope = {st['slope']:+.2e} per cycle, "
            f"p = {st['p_value']:.4f} — BURN-IN NOT SEPARATED, see sidecar "
            f"before reading this as a trend"
        )

    headline = "; ".join(_headline_for(g) for g in plotted3)
    n_total_all = sum(per_gov_stats[g]["n_total"] for g in plotted3)
    all_cycles = sorted({
        c for g in plotted3 for c in per_gov_stats[g]["sorted_cycles"]
    })
    gov_title_word3 = "ADS" if single3 else _gov_join(plotted3)
    # `headline` (the per-government slope/p-value clause built above) lives
    # in the sidecar caveat text below, not in the title.
    title = f"{gov_title_word3} Calibration Error by Cycle"
    ax.set_xlabel("Cycle", fontsize=13)
    ax.set_ylabel("Mean |predicted − realized| health per agent",
                  fontsize=13)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.legend(loc="best", fontsize=9, framealpha=0.88)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)   # see the note at the figure above
    png = os.path.join(plots_dir, "calibration_trend_by_cycle.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(png)

    # Clause ORDER is load-bearing.  _clauses_to_alt_text enforces the 100-word
    # WCAG budget by dropping clauses from the END, so anything appended last is
    # the first thing a screen-reader user loses.  The direction of the trend is
    # the single most important fact about this figure, so it goes SECOND —
    # immediately after the clause that identifies the figure — not last.  It
    # was last, and at smoke-test scale it silently vanished from the sidecar
    # entirely.
    #
    # The direction clause is computed from the POST-burn-in fit, not the
    # full-range one.  _trend_direction_clause itself is a pure function of
    # (slope, p_value); what matters is which slope it is handed.
    clauses = [
        f"Line plot: {gov_title_word3} self-calibration error, mean "
        "|predicted - realized| health per agent, as a function "
        "of cycle number, pooled across every difficulty level and run in "
        "the sweep."
    ]
    if single3:
        g0 = plotted3[0]
        st0 = per_gov_stats[g0]
        if st0["post_slope"] is not None and st0["post_p"] is not None:
            clauses.append(
                f"The first {burn_in.breakpoint} cycles are a ceiling-pinned "
                f"burn-in and are shaded, not fitted. "
                + _trend_direction_clause(st0["post_slope"], st0["post_p"])
            )
            clauses.append(
                f"A single line fitted across the whole range instead gives "
                f"{st0['slope']:+.2e} per cycle; that is an artifact of the "
                f"burn-in and is retracted, not a finding."
            )
        else:
            clauses.append(
                "No burn-in breakpoint could be estimated on this corpus, so "
                "no trend direction is reported: the early cycles of this "
                "metric are ceiling-pinned and a single fit across them is "
                "not interpretable."
            )
        clauses.append(
            f"Cycles {st0['sorted_cycles'][0]} to {st0['sorted_cycles'][-1]} "
            f"({len(st0['sorted_cycles'])} distinct cycles), "
            f"n = {st0['n_total']} closed predictions pooled "
            f"({min(st0['n_per_cycle'])}-{max(st0['n_per_cycle'])} per cycle)."
        )
    else:
        for g in plotted3:
            st = per_gov_stats[g]
            label = _gov_label(g)
            if st["post_slope"] is not None and st["post_p"] is not None:
                clauses.append(
                    f"{label} post-burn-in fit — "
                    + _trend_direction_clause(st["post_slope"], st["post_p"])
                )
            else:
                clauses.append(
                    f"{label}: no burn-in breakpoint could be estimated on "
                    "this corpus, so no trend direction is reported."
                )
        clauses.append(
            f"Cycles {all_cycles[0]} to {all_cycles[-1]} "
            f"({len(all_cycles)} distinct cycles across {len(plotted3)} "
            f"government(s)), n = {n_total_all} closed predictions pooled."
        )
    clauses.append(
        "Shaded band is a 95% confidence interval (1.96 x SEM) around the "
        "per-cycle mean."
    )
    trend_caveat = (
        f"  Fitted slopes: {headline}\n\n"
        + "".join(f"  {line}\n" for line in _burn_in_evidence_lines(burn_in))
        + "\n" + _CALIBRATION_TREND_SCOPE_CAVEAT
    )
    if not single3:
        trend_caveat += (
            "\n  SHARED BREAKPOINT ACROSS ARMS: the burn-in breakpoint "
            f"above was estimated once, from {primary_label}'s pooled "
            "series, and reused for every other plotted government rather "
            "than re-estimated per arm.  Ceiling-pinning is a property of "
            "the simulation's health dynamics near cycle 0, common to every "
            "government, so one physically motivated breakpoint was judged "
            "preferable to per-arm estimates that would often be too sparse "
            "on a filtered (enacted/replaced-only) series to clear the "
            "estimator's own minimum-segment-size floor "
            "(_BURN_IN_MIN_SEGMENT_OBS).\n"
        )
    if _ADS_GOV_KEY in plotted3:
        trend_caveat += "\n" + _scope_matched_pointer_caveat(
            "calibration_trend_by_cycle_scope_matched.png")
    txt = _write_calibration_sidecar(
        png,
        title=f"{gov_title_word3} Calibration Error vs Cycle, Pooled Across Sweep",
        metric=(
            "calibration_cycle_deltas" if single3 else
            "calibration_cycle_deltas (ADS, unfiltered) / "
            "calibration_cycle_deltas_by_outcome_then_scope (other "
            "governments, filtered to outcomes={enacted,replaced})"
        ),
        alt_text=_clauses_to_alt_text(clauses, 100),
        extra_caveat=trend_caveat,
        n_levels=len(all_cycles),
        palette_lines=palette_lines3,
        axis_label="Distinct cycles with data",
        gov_keys=plotted3,
        scale=scale,
    )
    written.append(txt)
    log(f"  Plot saved → {png}")
    log(f"  Alt text saved → {txt}")
    for g in plotted3:
        st = per_gov_stats[g]
        if st["post_slope"] is not None:
            log(f"    {_gov_label(g)} trend: full-range {st['slope']:+.4e}/cycle "
                f"(artifact) vs post-burn-in {st['post_slope']:+.4e}/cycle "
                f"(p={st['post_p']:.4f}, n={st['post_n']}) from cycle "
                f"{burn_in.breakpoint}")
    return written


def _n_closures_per_cell_caption(per_cell_n: Mapping[Any, int]) -> str:
    """
    ``"n = 12-55 closures per cell (min 5)"`` for a scope-matched figure.

    Deliberately says "closures", not "runs": :func:`_n_per_cell_caption`
    counts runs contributing a per-run statistic, but the scope-matched
    figures pool individual closures directly (A4.8: a per-run statistic is
    not viable on the ADS ``n_groups == 1`` subset — median ~2 closures/run,
    a quarter of runs at zero), so the unit this caption names is different
    from the sibling function's and must say so. *per_cell_n* is expected to
    contain only cells that already cleared :data:`_SCOPE_MATCHED_MIN_N` —
    see :func:`make_scope_matched_calibration_plots` — so this describes
    exactly what is drawn, not what was available before suppression.
    """
    counts = sorted(set(per_cell_n.values()))
    if not counts:
        return "no cell cleared the minimum-n threshold"
    lo, hi = counts[0], counts[-1]
    span = f"{lo}" if lo == hi else f"{lo}-{hi}"
    unit = "closure" if (lo == hi == 1) else "closures"
    return f"n = {span} {unit} per cell (min {_SCOPE_MATCHED_MIN_N})"


def make_scope_matched_calibration_plots(
    output_root: str,
    plots_dir: str,
    *,
    gov_keys: Sequence[str] = (_ADS_GOV_KEY,),
    min_n: int = _SCOPE_MATCHED_MIN_N,
) -> List[str]:
    """
    Render the ADS(``n_groups == 1``)-vs-A+L SCOPE-MATCHED calibration figures.

    **Why this function exists, separately from :func:`make_ads_calibration_plots`.**
    That function's three figures compare ADS's FULL corpus against A+L. That
    comparison is scope-confounded: A+L always legislates the whole population
    (``n_groups`` hardcoded to 1, ``autocracy_lookahead.py:1184``), while ADS
    *selects* a subgroup from a search over ``GROUP_DISTRIBUTION_COUNTS``
    (``ads.py:201``), and 81% of ADS's corpus is the finest-grained
    ``n_groups == 10`` scheme (23,221 of 28,638
    enactments) — a different, easier prediction problem than A+L's
    whole-population grading. Measured on the real k=10 archive: ADS's FULL
    corpus mean |predicted - realized| is 0.0999 (n=25,254); restricted to
    ``n_groups == 1`` it is 0.1251 (n=515); A+L's is 0.1647 (n=2,961). ADS
    still forecasts better either way, but the scope-matched gap is 39%
    smaller than the full-corpus one — the full-corpus comparison alone
    overstates the advantage that is attributable to *forecasting skill* as
    opposed to *scope*.

    This function draws the apples-to-apples counterpart: ADS restricted to
    the SAME scope A+L always carries, via a scope-restricted read of the
    exact same partition (`calibration_cycle_deltas_by_outcome_then_scope`)
    ``make_ads_calibration_plots`` already reads for every non-ADS government
    — see :func:`load_scope_matched_calibration_deltas` and
    :data:`_SCOPE_MATCHED_SCOPES`. It is a SEPARATE function producing
    SEPARATE figures, not additional series bolted onto the three existing
    ones, for two independent reasons: (1) ``calibration_half_split.png`` is
    already at four box series (two governments x two halves), close to
    that figure's readability limit; (2) a per-difficulty scope-matched
    series has no home on ``calibration_trend_by_cycle.png`` at all, since
    that figure's x-axis is cycle, not difficulty — the two figures answer
    different questions and each needs its own scope-matched counterpart.

    **What this draws, and what it deliberately does not.**

    ``mean_prediction_error_scope_matched.png`` — the counterpart to
    ``mean_prediction_error.png``. **Not the same chart type.** The original
    is a per-run Tukey IQR boxplot; the measurement above established that a
    per-run statistic is not viable on the ADS ``n_groups == 1`` subset at
    all (median ~2
    whole-population closures per run, a quarter of runs at zero — most runs
    cannot report a per-run mean for this subset even once). This figure
    instead plots one POOLED CLOSURE mean with a 95% CI per difficulty level,
    the same "pool the closures directly, closure is the unit of analysis"
    treatment ``calibration_trend_by_cycle.png`` already uses for its
    per-cycle statistic — extended here to a per-difficulty axis. Both series
    on the figure (ADS restricted, and A+L) use this same treatment, even
    though A+L's own corpus would support a per-run boxplot: mixing box
    series for one arm and point/CI series for the other on one axis would
    imply the two arms carry different statistical weight, which they must
    not appear to for a scope-matched comparison to be honest.

    ``calibration_trend_by_cycle_scope_matched.png`` — the counterpart to
    ``calibration_trend_by_cycle.png``: per-cycle pooled mean with 95% CI,
    same shape as the original, restricted to the scope-matched subset on
    both arms. **No OLS trend is fitted** on this series (unlike the
    full-corpus figure's post-burn-in fit): ADS's ``n_groups == 1`` closures
    pool to a median of 4 observations per cycle with 54 of 94 cycles below
    :data:`_SCOPE_MATCHED_MIN_N`, and fitting a regression through a series
    that thin would manufacture a precision the data does not have. The
    burn-in region IS still shaded, for visual continuity with the
    full-corpus trend figure, but the breakpoint is the SAME one estimated
    from ADS's full pooled corpus (recomputed here, not threaded through as a
    parameter — see the "shared burn-in" note below), not independently
    re-derived on this thinner series: ceiling-pinning is a property of the
    simulation's health dynamics near cycle 0, common to every scope, so one
    physically motivated breakpoint is preferred to one this series is too
    thin to estimate reliably on its own.

    **``calibration_half_split_scope_matched.png`` is NOT drawn, and this is
    a judgement call made explicit rather than a silent gap.** That figure's
    unit of analysis is a per-run half-split mean — exactly the per-run
    statistic A4.8 measured as not viable on this subset (see above). Every
    call to this function logs an INFO line stating this omission and why,
    so "no half-split scope-matched figure exists" is discoverable from the
    sweep log, not just from this docstring.

    **Minimum-n suppression.** A cell (difficulty level or cycle) whose
    pooled scope-matched closure count is below *min_n* (default
    :data:`_SCOPE_MATCHED_MIN_N` = 5) is omitted rather than drawn — a mean
    over 1-4 observations reads as a measurement it is not. On the real k=10
    archive this suppresses 2 of 21 ADS difficulty levels (D10 n=2, D15 n=3)
    and 54 of 94 ADS cycles; A+L clears the floor everywhere but 4 of 130
    cycles. Suppressed cells are enumerated in the sidecar, and the pooled
    total (every scope-matched closure, before suppression) is reported
    there too, per the honesty requirement that a thin per-cell view must not
    obscure an adequately powered pooled one.

    **Shared burn-in, recomputed rather than threaded through.**
    ``make_ads_calibration_plots`` already estimates this exact breakpoint
    from ADS's full pooled corpus for its own Figures 2 and 3. This function
    recomputes it independently from the same source
    (``load_ads_calibration_cycle_deltas_by_run(output_root,
    gov_key=_ADS_GOV_KEY)`` pooled, then :func:`estimate_burn_in_breakpoint`)
    rather than receiving it as a parameter, because
    ``make_ads_calibration_plots``'s return type is ``List[str]`` (the
    written paths) with no channel for an internal statistic, and widening
    that signature to thread one through would be a second, unrelated change
    to a function this pass otherwise leaves untouched. The recomputation is
    deterministic given the same archive, so the two figures' shaded regions
    agree by construction, at the cost of walking the ADS tree a second time
    — an acceptable trade for a ``--plots-only`` regeneration that is not
    performance-sensitive.

    Takes no ``BenchmarkConfig`` and reads no in-memory sweep state, for the
    same "regenerable from an archived output directory alone" reason
    :func:`make_ads_calibration_plots` gives.

    Skips quietly (INFO, no stub file, no exception) when ADS itself is not
    among the active governments — the scope filter is defined relative to
    ADS's own partition, so there is nothing to scope-match without it — or
    when no cell of either figure clears *min_n* for any active government.

    Returns the paths written, PNG and sidecar, in write order.
    """
    active_govs = [g for g in gov_keys if os.path.isdir(os.path.join(output_root, g))]
    if not active_govs:
        log(f"  No {'/'.join(gov_keys)}/ output under {output_root} — "
            f"scope-matched calibration plots skipped.")
        return []
    if _ADS_GOV_KEY not in active_govs:
        log("  ADS is not among the active governments (not requested, or "
            "its directory is absent) — scope-matched calibration plots are "
            "defined relative to ADS's own n_groups partition and are "
            "skipped without it.")
        return []
    for g in gov_keys:
        if g not in active_govs:
            log(f"  No {g}/ output under {output_root} — its scope-matched "
                f"series is skipped; the other requested government(s), if "
                f"any, still plot.")

    written: List[str] = []
    scale = _manifest_scale_caption(output_root)
    scale_clause = f", {scale}" if scale else ""

    def _gov_label(g: str) -> str:
        return GOV_DISPLAY.get(g, g)

    def _gov_join(govs: Sequence[str]) -> str:
        return " vs ".join(_gov_label(g) for g in govs)

    def _pooled_mean_ci(values: Sequence[float]) -> Tuple[int, float, float]:
        """``(n, mean, 95% CI half-width)``; CI is 0.0 for n <= 1."""
        n = len(values)
        mean = float(np.mean(values))
        if n > 1:
            std = float(np.std(values, ddof=1))
            ci = 1.96 * std / np.sqrt(n)
        else:
            ci = 0.0
        return n, mean, ci

    # {gov: {difficulty: [(cycle, delta), ...]}}, scope == "1" only, for EVERY
    # active government including ADS -- see load_scope_matched_calibration_deltas's
    # docstring for why ADS must go through the partition here rather than its
    # own unfiltered field.
    by_diff_raw: Dict[str, Dict[int, List[Tuple[int, float]]]] = {
        g: load_scope_matched_calibration_deltas(output_root, gov_key=g)
        for g in active_govs
    }

    # ==== Figure A: per-difficulty pooled mean +/- 95% CI ===================
    per_gov_diff_stats: Dict[str, Dict[int, Tuple[int, float, float]]] = {}
    per_gov_diff_suppressed: Dict[str, List[Tuple[int, int]]] = {}
    for g in active_govs:
        stats_by_diff: Dict[int, Tuple[int, float, float]] = {}
        suppressed: List[Tuple[int, int]] = []
        for diff, pairs in by_diff_raw[g].items():
            values = [v for _, v in pairs]
            if not values:
                continue
            n, mean, ci = _pooled_mean_ci(values)
            if n < min_n:
                suppressed.append((diff, n))
                continue
            stats_by_diff[diff] = (n, mean, ci)
        per_gov_diff_stats[g] = stats_by_diff
        per_gov_diff_suppressed[g] = sorted(suppressed)

    plottedA = [g for g in active_govs if per_gov_diff_stats[g]]
    if not plottedA:
        log(f"  No difficulty level cleared the minimum-n threshold "
            f"(min_n={min_n}) for any requested government — "
            f"mean_prediction_error_scope_matched.png skipped.")
    else:
        sorted_diffsA = sorted({d for g in plottedA for d in per_gov_diff_stats[g]})
        xA_range = (max(sorted_diffsA) - min(sorted_diffsA)) if len(sorted_diffsA) > 1 else 10
        boxA_width = xA_range * 0.025
        offsetsA = _series_offsets(len(plottedA), boxA_width)

        fig, ax = plt.subplots(figsize=(16, 7))
        handles = []
        caption_by_gov: Dict[str, str] = {}
        for g, offset in zip(plottedA, offsetsA):
            color = GOV_COLORS.get(g, "#333333")
            linestyle = GOV_LINESTYLES.get(g, "-")
            diffs = sorted(per_gov_diff_stats[g])
            xs = [d + offset for d in diffs]
            means = [per_gov_diff_stats[g][d][1] for d in diffs]
            cis = [per_gov_diff_stats[g][d][2] for d in diffs]
            ax.errorbar(
                xs, means, yerr=cis, marker="o", markersize=6, linewidth=2.0,
                linestyle=linestyle, color=color, capsize=4, zorder=3,
            )
            caption_by_gov[g] = _n_closures_per_cell_caption(
                {d: per_gov_diff_stats[g][d][0] for d in diffs})
            handles.append(plt.Line2D(
                [0], [0], color=color, linewidth=2.5, linestyle=linestyle,
                marker="o", markersize=6, label=_gov_label(g),
            ))

        distinct_captions = {caption_by_gov[g] for g in plottedA}
        nA_caption = (
            next(iter(distinct_captions)) if len(distinct_captions) == 1
            else "; ".join(f"{_gov_label(g)} {caption_by_gov[g]}" for g in plottedA)
        )
        gov_title_wordA = _gov_join(plottedA)
        n_suppressed_ads_diff = len(per_gov_diff_suppressed.get(_ADS_GOV_KEY, []))

        _style_difficulty_axes(
            ax, sorted_diffsA, xA_range,
            title=f"{gov_title_wordA} Calibration Error by Difficulty Level",
            ylabel="Mean |predicted − realized| health per agent",
        )
        ax.legend(
            handles=handles, title="Government",
            title_fontsize=10, fontsize=9, loc="best", framealpha=0.88,
        )
        fig.tight_layout()

        os.makedirs(plots_dir, exist_ok=True)
        pngA = os.path.join(plots_dir, "mean_prediction_error_scope_matched.png")
        fig.savefig(pngA, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(pngA)

        clausesA = [
            f"Point plot: {gov_title_wordA} self-calibration accuracy, "
            "SCOPE-MATCHED -- ADS restricted to n_groups == 1 closures, the "
            "same scope A+L's own decisions always carry, so the comparison "
            "is apples-to-apples rather than confounded by ADS's group-"
            "targeting search."
        ]
        clausesA.append(
            f"Difficulty levels {sorted_diffsA[0]} to {sorted_diffsA[-1]} "
            f"({len(sorted_diffsA)} levels shown), {nA_caption}."
        )
        clausesA.append(
            "One point per difficulty level: pooled closure mean with 95% "
            "CI whiskers (1.96 x SEM). Lower is a better forecast."
        )
        if n_suppressed_ads_diff:
            clausesA.append(
                f"{n_suppressed_ads_diff} ADS difficulty level(s) with fewer "
                f"than {min_n} scope-matched closures are omitted rather "
                "than plotted from 1-4 observations."
            )
        pooled_linesA = []
        for g in plottedA:
            all_vals = [v for pairs in by_diff_raw[g].values() for _, v in pairs]
            if all_vals:
                pooled_linesA.append(
                    f"    {_gov_label(g)}: n = {len(all_vals)} pooled "
                    f"closures, mean = {float(np.mean(all_vals)):.4f}"
                )
        suppressed_linesA = [
            f"    {_gov_label(g)} D{d}: n={n}"
            for g in plottedA
            for d, n in per_gov_diff_suppressed.get(g, [])
        ]
        extra_caveatA = (
            "Interpretation caveat — this is the SCOPE-MATCHED comparison "
            "(read before quoting a number from this figure):\n"
            "  ADS is restricted here to n_groups == 1 closures — the same\n"
            "  scope A+L's own decisions always carry (n_groups is hardcoded\n"
            "  to 1, autocracy_lookahead.py:1184) — filtered to\n"
            "  outcomes={enacted, replaced} exactly as every other\n"
            "  calibration figure filters A+L. This is the apples-to-apples\n"
            "  contrast; mean_prediction_error.png plots ADS's FULL corpus\n"
            "  instead, which is 81% n_groups == 10 (a different, easier\n"
            "  prediction problem) and is NOT scope-matched to A+L — see\n"
            "  that figure's own sidecar, which now points back here.\n"
            "\n"
            f"  MINIMUM-N THRESHOLD: cells below n={min_n} are suppressed\n"
            "  rather than drawn — a mean over 1-4 observations reads as a\n"
            "  measurement it is not.\n"
            + (
                "\n  Suppressed cells:\n" + "\n".join(suppressed_linesA) + "\n"
                if suppressed_linesA else ""
            )
            + "\n  Pooled totals (every scope-matched closure, before "
            "per-difficulty suppression):\n"
            + "\n".join(pooled_linesA) + "\n"
        )
        txt = _write_calibration_sidecar(
            pngA,
            title=f"{gov_title_wordA} Calibration Error — Scope-Matched (ADS n_groups == 1)",
            metric=(
                "calibration_closures_by_outcome_then_scope, scope=='1', "
                "outcomes={enacted,replaced} (both arms, via "
                "load_scope_matched_calibration_deltas / _filtered_cycle_deltas)"
            ),
            alt_text=_clauses_to_alt_text(clausesA, 100),
            n_levels=len(sorted_diffsA),
            palette_lines=[
                f"{_gov_label(g):15} | Color: {GOV_COLORS.get(g, '#333333')} | "
                f"Line style: {GOV_LINESTYLES.get(g, '-')}"
                for g in plottedA
            ],
            extra_caveat=extra_caveatA,
            gov_keys=plottedA,
            scale=scale,
        )
        written.append(txt)
        log(f"  Plot saved → {pngA}")
        log(f"  Alt text saved → {txt}")

    # ==== NOT drawn: scope-matched calibration_half_split counterpart ======
    log("  calibration_half_split_scope_matched.png intentionally NOT drawn: "
        "its unit of analysis is a per-run half-split mean, and ADS's "
        "n_groups == 1 closures pool to a median of ~2 per run with a "
        "quarter of runs at zero — too thin to define "
        "that statistic at all. See make_scope_matched_calibration_plots's "
        "own docstring.")

    # ==== Figure B: per-cycle pooled mean +/- 95% CI, pooled across every ===
    # ==== difficulty level (mirrors calibration_trend_by_cycle.png's shape) =
    per_gov_cycle_raw: Dict[str, Dict[int, List[float]]] = {}
    for g in active_govs:
        cd: Dict[int, List[float]] = {}
        for pairs in by_diff_raw[g].values():
            for c, v in pairs:
                cd.setdefault(c, []).append(v)
        per_gov_cycle_raw[g] = cd

    # Shared burn-in breakpoint, recomputed from ADS's FULL pooled corpus --
    # see the function docstring's "Shared burn-in" note for why this is
    # recomputed here rather than threaded through as a parameter. Purely for
    # shading: no OLS trend is fitted on the scope-matched series itself.
    full_ads_by_run = load_ads_calibration_cycle_deltas_by_run(
        output_root, gov_key=_ADS_GOV_KEY)
    full_pooled_x: List[float] = []
    full_pooled_y: List[float] = []
    for _runs in full_ads_by_run.values():
        for _series in _runs:
            for _c, _v in _series:
                full_pooled_x.append(float(_c))
                full_pooled_y.append(_v)
    burn_in = (
        estimate_burn_in_breakpoint(full_pooled_x, full_pooled_y)
        if full_pooled_x else None
    )

    per_gov_cycle_stats: Dict[str, Dict[int, Tuple[int, float, float]]] = {}
    per_gov_cycle_suppressed_n: Dict[str, int] = {}
    for g in active_govs:
        stats_by_cycle: Dict[int, Tuple[int, float, float]] = {}
        n_suppressed = 0
        for c, values in per_gov_cycle_raw[g].items():
            n, mean, ci = _pooled_mean_ci(values)
            if n < min_n:
                n_suppressed += 1
                continue
            stats_by_cycle[c] = (n, mean, ci)
        per_gov_cycle_stats[g] = stats_by_cycle
        per_gov_cycle_suppressed_n[g] = n_suppressed

    plottedB = [g for g in active_govs if per_gov_cycle_stats[g]]
    if not plottedB:
        log(f"  No cycle cleared the minimum-n threshold (min_n={min_n}) "
            f"for any requested government — "
            f"calibration_trend_by_cycle_scope_matched.png skipped.")
        return written

    all_cyclesB = sorted({c for g in plottedB for c in per_gov_cycle_stats[g]})
    gov_title_wordB = _gov_join(plottedB)

    fig, ax = plt.subplots(figsize=(16, 7))
    if burn_in is not None:
        overall_min_cycle = min(all_cyclesB[0], burn_in.breakpoint)
        ax.axvspan(
            float(overall_min_cycle), float(burn_in.breakpoint),
            color="#BBBBBB", alpha=0.25, linewidth=0, zorder=0,
            label=f"Burn-in (cycles < {burn_in.breakpoint})",
        )

    palette_linesB: List[str] = []
    n_total_B = 0
    for g in plottedB:
        color = GOV_COLORS.get(g, "#333333")
        linestyle = GOV_LINESTYLES.get(g, "-")
        cycles = sorted(per_gov_cycle_stats[g])
        n_total_B += sum(per_gov_cycle_stats[g][c][0] for c in cycles)
        # NaN-gapped: a cycle this government has no KEPT point for (either
        # no data at all, or suppressed below min_n) breaks the line rather
        # than being bridged over, so the figure never implies continuity it
        # does not have.
        full_range = list(range(min(cycles), max(cycles) + 1))
        cycle_set = set(cycles)
        gapped_y = np.array([
            per_gov_cycle_stats[g][c][1] if c in cycle_set else np.nan
            for c in full_range
        ])
        gapped_ci = np.array([
            per_gov_cycle_stats[g][c][2] if c in cycle_set else np.nan
            for c in full_range
        ])
        ax.plot(full_range, gapped_y, color=color, linestyle=linestyle,
                linewidth=2.0, marker="o", markersize=4, zorder=3,
                label=f"{_gov_label(g)} mean")
        ax.fill_between(full_range, gapped_y - gapped_ci, gapped_y + gapped_ci,
                         color=color, alpha=0.18, linewidth=0, zorder=2,
                         label=f"{_gov_label(g)} 95% CI")
        palette_linesB.append(
            f"{_gov_label(g):22} | Color: {color} | Line style: {linestyle}"
        )

    ax.set_xlabel("Cycle", fontsize=13)
    ax.set_ylabel("Mean |predicted − realized| health per agent",
                  fontsize=13)
    ax.set_title(
        f"{gov_title_wordB} Calibration Error by Cycle",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    ax.legend(loc="best", fontsize=9, framealpha=0.88)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    pngB = os.path.join(plots_dir, "calibration_trend_by_cycle_scope_matched.png")
    fig.savefig(pngB, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(pngB)

    clausesB = [
        f"Line plot: {gov_title_wordB} self-calibration error, mean "
        "|predicted - realized| health per agent, as a function "
        "of cycle, SCOPE-MATCHED -- ADS restricted to n_groups == 1 "
        "closures, the same scope A+L's decisions always carry."
    ]
    clausesB.append(
        "No trend line is fitted: the pooled series is too thin (median "
        "~4 observations per cycle for ADS) to support a segmented OLS "
        "estimate reliably; see calibration_trend_by_cycle.png for the "
        "full-corpus trend analysis."
    )
    clausesB.append(
        (
            f"Cycles below {burn_in.breakpoint} are shaded as burn-in, "
            "shared with the full-corpus figure's estimate rather than "
            "independently re-derived on this thinner series."
        ) if burn_in is not None else
        "No burn-in breakpoint is shaded: none could be estimated on ADS's "
        "full corpus for this archive."
    )
    clausesB.append(
        f"Cycles {all_cyclesB[0]} to {all_cyclesB[-1]} "
        f"({len(all_cyclesB)} distinct cycles shown), n = {n_total_B} "
        f"scope-matched observations pooled; cycles below {min_n} are "
        "omitted rather than plotted."
    )
    clausesB.append(
        "Shaded band is a 95% confidence interval (1.96 x SEM) around the "
        "per-cycle mean. Lower is a better forecast."
    )

    pooled_linesB = []
    for g in plottedB:
        all_vals = [v for pairs in by_diff_raw[g].values() for _, v in pairs]
        if all_vals:
            pooled_linesB.append(
                f"    {_gov_label(g)}: n = {len(all_vals)} pooled closures, "
                f"mean = {float(np.mean(all_vals)):.4f}"
            )
    al_suppressed_clause = (
        f"; A+L: {per_gov_cycle_suppressed_n.get('autocracy_lookahead', 0)} "
        f"of {len(per_gov_cycle_raw.get('autocracy_lookahead', {}))} "
        "cycles with any data suppressed below this floor."
        if "autocracy_lookahead" in active_govs else "."
    )
    trend_caveatB = (
        "Interpretation caveat — this is the SCOPE-MATCHED comparison "
        "(read before quoting a number from this figure):\n"
        "  ADS is restricted here to n_groups == 1 closures — the same\n"
        "  scope A+L's own decisions always carry (n_groups is hardcoded\n"
        "  to 1, autocracy_lookahead.py:1184) — filtered to\n"
        "  outcomes={enacted, replaced} exactly as every other calibration\n"
        "  figure filters A+L. This is the apples-to-apples contrast;\n"
        "  calibration_trend_by_cycle.png plots ADS's FULL corpus instead,\n"
        "  which is 81% n_groups == 10 (a different, easier prediction\n"
        "  problem) and is NOT scope-matched to A+L — see that figure's own\n"
        "  sidecar, which now points back here.\n"
        "\n"
        f"  MINIMUM-N THRESHOLD: cycles below n={min_n} are suppressed\n"
        "  (gapped in the line) rather than drawn — a mean over 1-4\n"
        "  observations reads as a measurement it is not.\n"
        f"  ADS: {per_gov_cycle_suppressed_n.get(_ADS_GOV_KEY, 0)} of "
        f"{len(per_gov_cycle_raw.get(_ADS_GOV_KEY, {}))} cycles with any "
        f"data suppressed below this floor{al_suppressed_clause}\n"
        "\n  Pooled totals (every scope-matched closure, before per-cycle "
        "suppression):\n"
        + "\n".join(pooled_linesB) + "\n"
    )
    txt = _write_calibration_sidecar(
        pngB,
        title=f"{gov_title_wordB} Calibration Error vs Cycle — Scope-Matched (ADS n_groups == 1)",
        metric=(
            "calibration_cycle_deltas_by_outcome_then_scope, scope=='1', "
            "outcomes={enacted,replaced} (both arms, via "
            "load_scope_matched_calibration_deltas)"
        ),
        alt_text=_clauses_to_alt_text(clausesB, 100),
        extra_caveat=trend_caveatB,
        n_levels=len(all_cyclesB),
        palette_lines=palette_linesB,
        axis_label="Distinct cycles with data",
        gov_keys=plottedB,
        scale=scale,
    )
    written.append(txt)
    log(f"  Plot saved → {pngB}")
    log(f"  Alt text saved → {txt}")
    return written


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

MANIFEST_NAME = "manifest.json"
ARCHIVED_LOG_NAME = "sweep.log"


def _entry_point_name() -> str:
    """Best-effort name of the script that launched this process."""
    return os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "<unknown>"


def manifest_path(config: BenchmarkConfig) -> str:
    """Absolute path of the sweep's provenance manifest."""
    return os.path.join(config.output_root, MANIFEST_NAME)


def _schedule_block() -> Dict[str, Any]:
    """Which difficulty schedule produced this archive.

    Archives generated under different tail-slope values are otherwise
    indistinguishable.  ``schedule_digest``
    is a sha256 over TWO difficulty-scaled channels (format v3), each
    covering every difficulty level 1..100, so it changes iff EITHER channel
    changed:

      1. ``SimulationConfig.from_difficulty`` — every field the sweep reads
         directly off the config object (resource density, event
         frequency/severity, warning cycles, regen rate, metabolic rate,
         ambient hazard, ...).  NOTE: ``SimulationConfig`` has no drain field —
         ``difficulty_multiplier`` (the drain multiplier) is NOT read off
         this object and is NOT covered here; it is covered by channel 2
         below.
      2. Everything difficulty-scaled that is NOT read off that config
         object, because it is computed elsewhere from
         ``effective_difficulty()`` directly: the auto event schedule's wave
         count and base severity (``scenarios.scenario_base._auto_n_waves`` /
         ``_auto_base_severity``, consumed by ``_auto_event_schedule`` at
         sweep time — see this module's ``_auto_event_schedule`` import), the
         grid depletion scale (``engine/grid.py``'s
         ``_compute_depletion_rate_per_unit``, ``diff_scale = 0.20 +
         (d_eff / 100.0) * 1.60``), the epidemic infection percentage
         (``engine/events.py``, ``infection_pct = 0.01 + (d_eff - 1) / 99 *
         0.09``), the health recovery-cycle count
         (``engine/simulation.py``'s ``_apply_health_dynamics``,
         ``recovery_cycles = int(15 + (d_eff - 1) / 99 * 35)`` — the same
         expression ``test_difficulty_schedule.py::_check_site6_recovery_cycles``
         extracts and pins directly from source by AST, independently of this
         digest), and (channel-2 addition, v3) the drain multiplier
         itself, ``engine.difficulty.difficulty_multiplier(d)`` — called
         directly, not transcribed, since it is a public function.

    ``effective_difficulty()`` — the one quantity that actually varies with
    ``DIFFICULTY_TAIL_SLOPE`` — is called from the real function, not
    re-derived; the three one-line scale expressions above (diff_scale,
    infection_pct, recovery_cycles) are transcribed from their call sites
    (cited above) because none of them exposes a standalone function to call,
    and each is a single arithmetic expression with no branch or clamp,
    unlike the severity ceiling, which the next paragraph covers separately.

    This digest deliberately covers both channels rather than just channel 1.
    A regression confined to channel 2 alone — such as a reverted
    event-severity ceiling — would leave a channel-1-only digest unchanged,
    so a "verify the manifest's schedule_digest matches the piloted one
    before launch" check would pass over the regression. Covering both
    channels means any schedule change, in either channel, changes the
    digest value.

    This is the FINAL digest format (v3) from here to publication — the v2
    table above, with ``difficulty_multiplier(d)`` appended as the last
    column of channel 2.  No slope, knee or ``DIFFICULTY_T_MAX`` constant
    enters the payload; the digest describes values only, and the constants
    are recorded beside it (below) as informational fields outside the hash.
    v1 (config-only) and v2 (this table before the drain column) digests are
    earlier formats and are not comparable to v3.

    ``validate_schedule()`` runs first, so a sweep cannot write a manifest for
    a schedule whose ``difficulty_t`` extrapolates past
    ``DIFFICULTY_T_MAX`` — this is one of the
    call sites that must re-validate after a constant mutation.
    """
    from dataclasses import asdict
    import engine.difficulty as _difficulty
    from engine.difficulty import (
        DIFFICULTY_T_MAX,
        difficulty_multiplier,
        difficulty_t,
        effective_difficulty,
        validate_schedule,
    )

    validate_schedule()

    config_table = [(d, asdict(SimulationConfig.from_difficulty(d, seed=0)))
                    for d in range(1, 101)]

    event_and_engine_table = []
    for d in range(1, 101):
        d_eff = effective_difficulty(d)
        diff_scale = 0.20 + (d_eff / 100.0) * 1.60          # grid.py:338
        infection_pct = 0.01 + (d_eff - 1) / 99 * 0.09       # events.py:309
        recovery_cycles = int(15 + (d_eff - 1) / 99 * 35)    # simulation.py
        event_and_engine_table.append((
            d, _auto_n_waves(d), _auto_base_severity(d),
            diff_scale, infection_pct, recovery_cycles,
            difficulty_multiplier(d),                        # v3
        ))

    digest_payload = {
        "config": config_table,
        "event_and_engine": event_and_engine_table,
    }
    return {
        "difficulty_knee_level": _difficulty.DIFFICULTY_KNEE_LEVEL,
        "difficulty_tail_slope": _difficulty.DIFFICULTY_TAIL_SLOPE,
        "difficulty_t_max": DIFFICULTY_T_MAX,                    # informational (A10.2)
        "difficulty_t_at_100": difficulty_t(100),                # informational (A10.2)
        "schedule_digest": hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=repr).encode()
        ).hexdigest(),
    }


def build_manifest(
    config: BenchmarkConfig,
    log_path: str,
    status: str,
    started_at: str,
    ended_at: Optional[str] = None,
    wall_seconds: Optional[float] = None,
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the manifest describing *what command produced this directory*.

    Note for anyone publishing the dataset: this file records the invoking
    ``user``, ``host`` and ``argv``.  For these entry points ``argv`` contains
    only sweep parameters, but the username and hostname do travel with the data.
    """
    try:
        user = getpass.getuser()
    except Exception:                                # pragma: no cover - defensive
        user = None
    return {
        # 2: adds the "schedule" block (difficulty knee / tail slope / digest).
        "schema_version": 2,
        "status": status,
        "entry_point": _entry_point_name(),
        "argv": list(sys.argv),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": (round(wall_seconds, 1)
                         if wall_seconds is not None else None),
        "host": socket.gethostname(),
        "user": user,
        "cwd": os.getcwd(),
        "log_file": log_path,
        "software": software_versions(),
        "config": {
            "governments": list(config.governments),
            "difficulties": list(config.difficulties),
            # The AUTHORITATIVE record of this sweep's scale.  `argv` above is
            # not a substitute: an archive can record
            # `argv: ['run_full_simulation.py']` with no flags at all if
            # `k_runs` was set by editing the module's constant rather than by
            # passing a flag.  These three fields say what ran, where that
            # number came from, and what the figures projected to — none of
            # which requires trusting argv.
            "k_runs": config.k_runs,
            "k_runs_source": config.k_runs_source,
            "paper_k_runs": config.paper_k_runs,
            "grid_size": config.grid_size,
            "n_agents": config.n_agents,
            "max_cycles": config.max_cycles,
            "max_steps": config.max_steps,
            "base_seed": config.base_seed,
            "output_root": config.output_root,
            "heartbeat_every": config.heartbeat_every,
            "engine_log_level": config.engine_log_level,
            "write_run_detail": config.write_run_detail,
            "decision_detail": config.decision_detail,
            "agent_states_every": config.agent_states_every,
            "clean_output_root": config.clean_output_root,
        },
        "schedule": _schedule_block(),
        "results": results if results is not None else {},
    }


def _write_manifest(config: BenchmarkConfig, manifest: Dict[str, Any]) -> None:
    """
    Write the manifest, never failing the sweep because of it.

    Written twice — once at start with ``status="running"``, once at the end
    with the outcome — so a killed sweep still leaves its provenance behind.
    """
    path = manifest_path(config)
    try:
        os.makedirs(config.output_root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
    except OSError as exc:
        bench_log(logging.WARNING, "could not write %s: %s", path, exc)


def _clean_output_root(config: BenchmarkConfig) -> None:
    """
    Remove ``output_root`` entirely before the sweep starts.

    Opt-in via ``--clean`` (``BenchmarkConfig.clean_output_root``) — never
    automatic.  This is the deliberate-fresh-run path: it exists so choosing a
    fresh dataset doesn't require a manual ``rm -rf`` (and the mistakes that
    invites), but it is destructive, so it only ever runs when explicitly
    requested by the caller.

    Called from :func:`run_benchmark` before ``output_root`` is recreated and
    before :func:`_warn_on_existing_output` inspects it, so a ``--clean`` run
    starts from a directory that is provably absent rather than merely
    "believed empty" — and, as a direct consequence, the stale-output warning
    correctly does not fire for content that was just removed.  It fires as
    normal if anything else recreates content in ``output_root`` afterwards,
    or if the removal itself only partially succeeds.
    """
    root = config.output_root
    if not os.path.exists(root):
        return
    if not os.path.isdir(root):
        raise ConfigError(
            f"--clean: output_root {root!r} exists and is not a directory; "
            f"refusing to remove it automatically."
        )
    bench_log(logging.WARNING, "--clean: removing existing output directory %s", root)
    shutil.rmtree(root)


def _warn_on_existing_output(config: BenchmarkConfig) -> None:
    """
    Warn (never delete) when the output tree already holds government results.

    A re-run with a different difficulty sequence leaves orphan directories
    behind, so the shipped tree misrepresents the design.  This function never
    deletes anything itself — deletion is `--clean` / `_clean_output_root`'s
    job, and it is opt-in.  Called unconditionally (after any `--clean` has
    already run), so the warning fires whenever stale content is actually
    present, regardless of whether the caller remembered the flag.
    """
    try:
        existing = [
            name for name in os.listdir(config.output_root)
            if name in GOVERNMENT_REGISTRY
            and os.path.isdir(os.path.join(config.output_root, name))
        ]
    except OSError:
        return
    if not existing:
        return

    stale = 0
    wanted = {str(d) for d in config.difficulties}
    for gov in existing:
        gov_dir = os.path.join(config.output_root, gov)
        try:
            stale += sum(
                1 for name in os.listdir(gov_dir)
                if name.isdigit() and name not in wanted
                and os.path.isdir(os.path.join(gov_dir, name))
            )
        except OSError:                              # pragma: no cover - defensive
            continue
    bench_log(
        logging.WARNING,
        "  output directory already contains results for %d government(s)"
        "%s.  Nothing is deleted: results are merged into the existing tree, so "
        "directories from a previous sweep will remain alongside this one.",
        len(existing),
        f" including {stale} difficulty director(y|ies) not in this sweep"
        if stale else "",
    )


def _log_difficulty_ranking(
    all_rows: List[dict], difficulty: int, governments: Sequence[str]
) -> None:
    """
    Log the government ranking by mean NHS at one difficulty.

    This is the substantive status line: not "am I making progress" but "is the
    result taking the expected shape".  A sweep whose ranking suddenly inverts
    is worth interrupting; one that is merely slow is not.
    """
    means: List[Tuple[str, float]] = []
    for gov in governments:
        values = [
            float(row["normalized_health_score"])
            for row in all_rows
            if row.get("government") == gov
            and row.get("difficulty") == difficulty
            and row.get("normalized_health_score") is not None
        ]
        if values:
            means.append((gov, sum(values) / len(values)))
    if not means:
        return
    means.sort(key=lambda kv: -kv[1])
    bench_log(
        logging.INFO,
        "  D=%d mean NHS: %s",
        difficulty,
        " │ ".join(f"{gov} {mean:.3f}" for gov, mean in means),
    )


def _print_banner(config: BenchmarkConfig, log_path: str) -> None:
    diffs = list(config.difficulties)
    log("=" * 72)
    log(f"  Decision-Making at Scale — {config.label}")
    log("=" * 72)
    log(f"  Governments  : {', '.join(config.governments)}")
    log(f"  Difficulties : {diffs[0]}–{diffs[-1]}  ({len(diffs)} levels)")
    log(f"                 {diffs}")
    # The provenance suffix is not decoration: it is the last chance to notice,
    # before hours of CPU time, that the scale is not the one that was meant.
    # "(published design)" vs "(--k-runs override; published design is N)" reads
    # differently at a glance, which is the point.
    if config.k_runs_source == K_RUNS_SOURCE_CLI:
        k_provenance = (f"  [--k-runs override; published design is "
                        f"{config.paper_k_runs}]")
    elif config.k_runs == config.paper_k_runs:
        k_provenance = "  [published design]"
    else:
        k_provenance = (f"  [entry-point default; published design is "
                        f"{config.paper_k_runs}]")
    log(f"  Runs/cell    : {config.k_runs}  ×  {len(config.governments)} govts  "
        f"×  {len(diffs)} difficulties  =  {config.total_runs} total simulation runs")
    log(f"                 k = {config.k_runs}{k_provenance}")
    log(f"  Grid         : {config.grid_size}×{config.grid_size},  "
        f"{config.n_agents} agents,  {config.max_cycles} cycles,  "
        f"max-steps {config.max_steps}")
    log(f"  Seeding      : base seed {config.base_seed}; every stream derived via "
        f"blake2b domain separation")
    log(f"                 environment keyed on (difficulty, run) — identical for "
        f"all {len(config.governments)} governments, different for each of the "
        f"{config.k_runs} runs")
    log(f"                 government RNG keyed on (government, difficulty, run); "
        f"audited per run by env_fingerprint")
    log(f"  Parallelism  : {config.effective_workers} processes "
        f"(true CPU parallelism via ProcessPoolExecutor)")
    log(f"  Output       : {config.output_root}")
    detail_state = "ON" if config.write_run_detail else "OFF"
    groupings_state = "ON" if config.decision_detail else "OFF"
    agents_state = (
        f"every {config.agent_states_every} cycles"
        if config.agent_states_every else "OFF"
    )
    log(f"  Detail files : run_detail.jsonl {detail_state}  "
        f"(decision groupings {groupings_state};  agent states {agents_state})")
    log(f"  Log level    : {config.engine_log_level} (engine);  "
        f"INFO (harness);  heartbeat every {config.heartbeat_every} cycles")
    log(f"  Log file     : {log_path}")
    log(f"  Follow with  : tail -f {log_path}")
    log(f"  Started (UTC): {utc_timestamp()}   host={socket.gethostname()}   "
        f"pid={os.getpid()}")
    versions = software_versions()
    log("  Software     : " + ",  ".join(
        f"{name} {ver}" for name, ver in versions.items() if ver
    ))
    log(f"  Command      : {' '.join(sys.argv)}")
    log("=" * 72)

    warning = verbose_level_warning(
        config.engine_log_level, config.total_runs, config.max_cycles, config.n_agents
    )
    if warning:
        bench_log(logging.WARNING, "  %s", warning)


def _numpy_linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Linear regression using numpy only (replaces scipy.stats.linregress)."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    slope = ss_xy / ss_xx if ss_xx != 0 else 0
    intercept = y_mean - slope * x_mean
    r_value = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    mse = ss_res / (n - 2) if n > 2 else 0
    std_err = np.sqrt(mse / ss_xx) if ss_xx > 0 else 0

    if n > 2 and mse > 0:
        t_stat = slope / std_err if std_err > 0 else 0
        from math import erf
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    else:
        p_value = 1.0

    return slope, intercept, r_value, p_value, std_err


#: Absolute floor on the pooled standard error below which a t-test is
#: reported as degenerate rather than evaluated.  Catches the exactly-zero
#: case and anything within a few ULPs of it.
_TTEST_MIN_SE_ABS = 1e-12

#: Relative floor, as a fraction of the larger group mean.  A pooled SE this
#: far below the scale of the quantity being compared is floating-point
#: residue from summing identical values, not sampling variability.  1e-9 sits
#: ~7 orders of magnitude above float64 epsilon at unit scale, so it cannot
#: swallow a real effect, and ~9 orders below the smallest sd this model
#: produces when a cell genuinely varies.
_TTEST_MIN_SE_REL = 1e-9


def _numpy_ttest(y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float, bool]:
    """Welch-style t-test using numpy only (replaces ``scipy.stats.ttest_ind``).

    Returns ``(t_stat, p_value, degenerate)``.  When ``degenerate`` is True the
    comparison is UNDEFINED, ``t_stat`` and ``p_value`` are both ``nan``, and
    the caller must not report significance either way.

    Why the degeneracy check exists
    -------------------------------
    Seven of the eight government regimes are deterministic given a fixed
    environment.  If the environment were shared across every run of a
    difficulty (rather than varying per run, as it does here), every run in
    such a cell would return the identical value and the within-cell
    variance would be either exactly 0 or floating-point residue on the
    order of 1e-18.  A naive guard like ``if pooled_se > 0`` is ``True`` at
    1e-18, so a difference of 0.0002 between two constants divided by 1e-18
    would produce t-statistics up to 7.3e15 with p = 0.0 and zero-width
    bootstrap CIs — every such row would be flagged "significant".

    A zero-variance cell is not evidence of an infinitely precise estimate; it
    is evidence that the cell carries one degree of freedom.  The honest report
    is "undefined", not a p-value.

    Each run here faces a different environment, so degeneracy is rare
    rather than the norm — but the guard stays, and its meaning sharpens: a
    difference vector that is identically zero says two regimes behaved
    identically on *every* sampled environment, which is a real and
    interesting finding rather than "both are deterministic".

    KNOWN LIMITATION, not a bug in this function: all governments face the
    same environment at the same ``run_idx``, so government-vs-government
    samples are **paired** and this unpaired Welch test discards that pairing.
    It is conservative (it overstates the standard error), so it will not
    manufacture significance — but it throws away the variance reduction the
    common-random-numbers design exists to buy.  The correct estimator is a
    paired test on the per-run difference vector.  Deliberately deferred.
    """
    n1, n2 = len(y1), len(y2)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), True

    mean1, mean2 = float(np.mean(y1)), float(np.mean(y2))
    var1, var2 = float(np.var(y1, ddof=1)), float(np.var(y2, ddof=1))
    pooled_se = float(np.sqrt(var1 / n1 + var2 / n2))

    # Scale-aware floor: compare the SE against the magnitude of the values it
    # is meant to be the sampling error OF.  A purely absolute floor would
    # misjudge metrics on very different scales (survival rate ~1.0 vs
    # time_to_50pct_loss ~150).
    se_floor = max(
        _TTEST_MIN_SE_ABS,
        _TTEST_MIN_SE_REL * max(abs(mean1), abs(mean2)),
    )
    if not math.isfinite(pooled_se) or pooled_se <= se_floor:
        return float("nan"), float("nan"), True

    t_stat = (mean1 - mean2) / pooled_se

    df = n1 + n2 - 2
    if df > 0:
        from math import erf
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    else:
        p_value = 1.0

    return t_stat, p_value, False


# Fixed base constant folded into every derived bootstrap seed (see
# ``_derive_bootstrap_seed``).  Arbitrary but fixed: changing it would change
# every bootstrap CI in every future analysis pass, so it is not meant to be
# tuned — it exists only so the derived seed space is distinguishable from a
# bare SHA-256 truncation, not to add real entropy.
_BOOTSTRAP_SEED_BASE = 1_664_525


def _derive_bootstrap_seed(gov1: str, gov2: str, difficulty: int, metric: str) -> int:
    """Deterministic, per-call seed for :func:`_bootstrap_ci_mean_diff`.

    A full analysis pass calls the bootstrap ~2,000+ times (government-pairs
    x metrics x difficulties); each call must get its own seed so the
    resamples aren't identical across cells, and that seed must be
    reproducible byte-for-byte across process restarts — the same
    reproducibility bar established for the calibration plots.  Python's
    built-in ``hash()`` on
    strings is salted per-process via ``PYTHONHASHSEED`` and would silently
    violate that bar, so the key is hashed with SHA-256 instead (stable
    across processes and interpreter versions) and XORed with a fixed base
    constant.
    """
    key = f"{gov1}|{gov2}|{difficulty}|{metric}".encode("utf-8")
    digest_int = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return _BOOTSTRAP_SEED_BASE ^ digest_int


def _bootstrap_ci_mean_diff(
    y1: np.ndarray, y2: np.ndarray, n_boot: int = 10000,
    ci_level: float = 0.95, seed: int = _BOOTSTRAP_SEED_BASE,
) -> Tuple[float, float]:
    """Percentile-method bootstrap CI for mean(y1) - mean(y2).

    numpy-only, matching this module's existing scipy/pandas-free constraint
    (see :func:`generate_analysis_files`'s docstring). Vectorized: all
    ``n_boot`` resamples are drawn in a single ``rng.integers`` fancy-index
    call per side rather than looping in Python over ``n_boot``, because this
    is called ~2,000+ times in a full sweep (government-pairs x metrics x
    difficulties) and a Python-level loop at that call count would dominate
    ``generate_analysis_files``'s runtime.

    The caller is responsible for passing a distinct, deterministic ``seed``
    per call (see :func:`_derive_bootstrap_seed`) so that a full analysis
    pass is byte-reproducible from a fresh process, matching this project's
    reproducibility bar. The default ``seed`` here is only a
    fallback for standalone/ad-hoc use of this function; production call
    sites in :func:`generate_analysis_files` always pass an explicit,
    per-cell seed.
    """
    n1 = len(y1)
    n2 = len(y2)
    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))
    boot_means1 = y1[idx1].mean(axis=1)
    boot_means2 = y2[idx2].mean(axis=1)
    boot_diffs = boot_means1 - boot_means2

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_diffs, lower_pct))
    ci_upper = float(np.percentile(boot_diffs, upper_pct))
    return ci_lower, ci_upper


def generate_analysis_files(
    config: BenchmarkConfig, all_combined_rows: List[dict]
) -> None:
    """
    Generate comprehensive analysis files for post-hoc investigation without re-running.
    Uses numpy and csv only—no pandas or scipy dependencies.
    """
    output_root = config.output_root
    analysis_dir = os.path.join(output_root, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    metrics = [
        "normalized_health_score", "final_survival_rate",
        "final_median_health", "final_health_gini"
    ]

    by_gov_metric = {}
    by_diff_metric = {}
    # Keyed (government, difficulty, metric) -> list of values.  Unlike
    # by_gov_metric (pools every difficulty together), this preserves the
    # difficulty axis so pairwise comparisons can be stratified by difficulty
    # -- see the "5b. Pairwise comparisons, stratified by difficulty" block
    # below.
    by_gov_diff_metric = {}

    for row in all_combined_rows:
        gov = row["government"]
        diff = row["difficulty"]
        for metric in metrics:
            val = row.get(metric)
            if val is not None:
                key = (gov, metric)
                if key not in by_gov_metric:
                    by_gov_metric[key] = []
                by_gov_metric[key].append(val)

                key_diff = (diff, metric)
                if key_diff not in by_diff_metric:
                    by_diff_metric[key_diff] = {}
                if gov not in by_diff_metric[key_diff]:
                    by_diff_metric[key_diff][gov] = []
                by_diff_metric[key_diff][gov].append(val)

                key_gov_diff = (gov, diff, metric)
                if key_gov_diff not in by_gov_diff_metric:
                    by_gov_diff_metric[key_gov_diff] = []
                by_gov_diff_metric[key_gov_diff].append(val)

    # 1. Per-government summary
    gov_summary = []
    for gov in config.governments:
        for metric in metrics:
            key = (gov, metric)
            values = np.array(by_gov_metric.get(key, []))
            if len(values) > 0:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
                ci = 1.96 * sem
                gov_summary.append({
                    "government": gov,
                    "metric": metric,
                    "n_obs": len(values),
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "median": round(float(np.median(values)), 6),
                    "min": round(float(np.min(values)), 6),
                    "max": round(float(np.max(values)), 6),
                    "ci_lower": round(mean - ci, 6),
                    "ci_upper": round(mean + ci, 6),
                })
    with open(os.path.join(analysis_dir, "per_government_summary.csv"), "w", newline="") as f:
        if gov_summary:
            writer = csv.DictWriter(f, fieldnames=gov_summary[0].keys())
            writer.writeheader()
            writer.writerows(gov_summary)

    # 2. Per-difficulty summary with rankings
    diff_summary = []
    for diff in config.difficulties:
        for metric in metrics:
            key = (diff, metric)
            gov_metrics = by_diff_metric.get(key, {})
            if gov_metrics:
                mean_per_gov = [(gov, np.mean(vs)) for gov, vs in gov_metrics.items()]
                mean_per_gov.sort(key=lambda x: x[1], reverse=True)
                for rank, (gov, mean_val) in enumerate(mean_per_gov, 1):
                    diff_summary.append({
                        "difficulty": diff,
                        "metric": metric,
                        "rank": rank,
                        "government": gov,
                        "mean": round(float(mean_val), 6),
                    })
    with open(os.path.join(analysis_dir, "per_difficulty_summary.csv"), "w", newline="") as f:
        if diff_summary:
            writer = csv.DictWriter(f, fieldnames=diff_summary[0].keys())
            writer.writeheader()
            writer.writerows(diff_summary)

    # 3. Per-metric analysis files
    metrics_dir = os.path.join(analysis_dir, "per_metric_analysis")
    os.makedirs(metrics_dir, exist_ok=True)
    for metric in metrics:
        metric_data = []
        for row in all_combined_rows:
            gov = row["government"]
            diff = row["difficulty"]
            val = row.get(metric)
            if val is not None:
                metric_data.append({
                    "government": gov,
                    "difficulty": diff,
                    "run": row.get("run", ""),
                    metric: round(float(val), 6),
                })
        if metric_data:
            with open(os.path.join(metrics_dir, f"{metric}_analysis.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metric_data[0].keys())
                writer.writeheader()
                writer.writerows(metric_data)

    # 4. Regression analysis
    regression_rows = []
    for gov in config.governments:
        for metric in metrics:
            key = (gov, metric)
            values = by_gov_metric.get(key, [])
            if len(values) > 2:
                x_vals = []
                y_vals = []
                for row in all_combined_rows:
                    if row["government"] == gov and metric in row and row[metric] is not None:
                        x_vals.append(row["difficulty"])
                        y_vals.append(row[metric])
                if len(x_vals) > 2:
                    x = np.array(x_vals)
                    y = np.array(y_vals)
                    slope, intercept, r_value, p_value, std_err = _numpy_linregress(x, y)
                    regression_rows.append({
                        "government": gov,
                        "metric": metric,
                        "n_obs": len(y),
                        "slope": round(slope, 6),
                        "intercept": round(intercept, 6),
                        "r_squared": round(r_value ** 2, 6),
                        "p_value": round(p_value, 6),
                        "std_err": round(std_err, 6),
                    })
    with open(os.path.join(analysis_dir, "regression_analysis.csv"), "w", newline="") as f:
        if regression_rows:
            writer = csv.DictWriter(f, fieldnames=regression_rows[0].keys())
            writer.writeheader()
            writer.writerows(regression_rows)

    # 5. Pairwise comparisons
    pairwise_rows = []
    gov_list = list(config.governments)
    for i, gov1 in enumerate(gov_list):
        for gov2 in gov_list[i+1:]:
            for metric in metrics:
                y1 = np.array(by_gov_metric.get((gov1, metric), []))
                y2 = np.array(by_gov_metric.get((gov2, metric), []))
                if len(y1) > 1 and len(y2) > 1:
                    t_stat, p_val, degenerate = _numpy_ttest(y1, y2)
                    mean_diff = float(np.mean(y1)) - float(np.mean(y2))
                    pairwise_rows.append({
                        "government_1": gov1,
                        "government_2": gov2,
                        "metric": metric,
                        "n1": len(y1),
                        "n2": len(y2),
                        "mean_1": round(float(np.mean(y1)), 6),
                        "mean_2": round(float(np.mean(y2)), 6),
                        "mean_difference": round(mean_diff, 6),
                        # Empty cells rather than "nan" text: a degenerate
                        # comparison has no t or p, and a reader scanning the
                        # CSV should see an absence, not a number-shaped token
                        # they might paste into a table.
                        "t_statistic": "" if degenerate else round(t_stat, 6),
                        "p_value": "" if degenerate else round(p_val, 6),
                        "significant_at_0_05": (
                            "Undefined" if degenerate
                            else ("Yes" if p_val < 0.05 else "No")
                        ),
                    })
    with open(os.path.join(analysis_dir, "pairwise_comparisons.csv"), "w", newline="") as f:
        if pairwise_rows:
            writer = csv.DictWriter(f, fieldnames=pairwise_rows[0].keys())
            writer.writeheader()
            writer.writerows(pairwise_rows)

    # 5b. Pairwise comparisons, stratified by difficulty.  Additive: does NOT
    # replace pairwise_comparisons.csv above, which is an established artifact
    # other files/README may reference and which several downstream readers
    # expect to be pooled across difficulty.  This file answers a question the
    # pooled one structurally cannot -- "is government X significantly ahead
    # of government Y at difficulty D" (e.g. the crossover-point claim near
    # difficulty ~75) -- because by_gov_metric collapses every difficulty
    # level together before the t-test runs.
    pairwise_by_diff_rows = []
    for i, gov1 in enumerate(gov_list):
        for gov2 in gov_list[i+1:]:
            for metric in metrics:
                for diff in config.difficulties:
                    y1 = np.array(by_gov_diff_metric.get((gov1, diff, metric), []))
                    y2 = np.array(by_gov_diff_metric.get((gov2, diff, metric), []))
                    if len(y1) > 1 and len(y2) > 1:
                        t_stat, p_val, degenerate = _numpy_ttest(y1, y2)
                        mean_diff = float(np.mean(y1)) - float(np.mean(y2))
                        boot_seed = _derive_bootstrap_seed(gov1, gov2, diff, metric)
                        ci_lower, ci_upper = _bootstrap_ci_mean_diff(y1, y2, seed=boot_seed)
                        # The bootstrap is degenerate for exactly the same
                        # reason the t-test is: resampling constants with
                        # replacement returns the same constant, so the CI
                        # collapses to zero width and "excludes 0" for any
                        # non-zero difference, however tiny.  Suppress it on
                        # the same condition rather than inventing a second,
                        # independently-drifting criterion.
                        ci_significant = (
                            "Undefined" if degenerate
                            else ("Yes" if (ci_lower > 0 or ci_upper < 0) else "No")
                        )
                        pairwise_by_diff_rows.append({
                            "difficulty": diff,
                            "government_1": gov1,
                            "government_2": gov2,
                            "metric": metric,
                            "n1": len(y1),
                            "n2": len(y2),
                            "mean_1": round(float(np.mean(y1)), 6),
                            "mean_2": round(float(np.mean(y2)), 6),
                            "mean_difference": round(mean_diff, 6),
                            "t_statistic": "" if degenerate else round(t_stat, 6),
                            "p_value": "" if degenerate else round(p_val, 6),
                            "significant_at_0_05": (
                                "Undefined" if degenerate
                                else ("Yes" if p_val < 0.05 else "No")
                            ),
                            "ci_lower_diff": "" if degenerate else round(ci_lower, 6),
                            "ci_upper_diff": "" if degenerate else round(ci_upper, 6),
                            "ci_significant": ci_significant,
                        })
    with open(os.path.join(analysis_dir, "pairwise_comparisons_by_difficulty.csv"), "w", newline="") as f:
        if pairwise_by_diff_rows:
            writer = csv.DictWriter(f, fieldnames=pairwise_by_diff_rows[0].keys())
            writer.writeheader()
            writer.writerows(pairwise_by_diff_rows)

    # 6. Correlation matrix
    corr_matrix = {}
    for m1 in metrics:
        corr_matrix[m1] = {}
        vals1 = np.array([row.get(m1) for row in all_combined_rows if row.get(m1) is not None])
        for m2 in metrics:
            vals2 = np.array([row.get(m2) for row in all_combined_rows if row.get(m2) is not None])
            if len(vals1) > 0 and len(vals2) > 0:
                corr = float(np.corrcoef(vals1, vals2)[0, 1]) if len(vals1) > 1 else 1.0
                corr_matrix[m1][m2] = round(corr, 6)
            else:
                corr_matrix[m1][m2] = ""
    with open(os.path.join(analysis_dir, "correlation_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + metrics)
        for m1 in metrics:
            writer.writerow([m1] + [corr_matrix[m1].get(m2, "") for m2 in metrics])

    # 7. Manifest
    manifest = {
        "analysis_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": output_root,
        "analysis_directory": analysis_dir,
        "files": {
            "per_government_summary.csv": {
                "description": "Summary statistics per government and metric",
                "use_case": "Compare performance across governments",
            },
            "per_difficulty_summary.csv": {
                "description": "Metrics ranked per difficulty level",
                "use_case": "See which government performs best at each difficulty",
            },
            "per_metric_analysis/": {
                "description": "Individual CSV for each metric",
                "use_case": "Detailed analysis of individual metrics",
            },
            "regression_analysis.csv": {
                "description": "Difficulty effects (slope, R², p-value) per government×metric",
                "use_case": "Understand difficulty sensitivity",
            },
            "pairwise_comparisons.csv": {
                "description": "T-test results between government pairs",
                "use_case": "Statistical significance tests",
            },
            "pairwise_comparisons_by_difficulty.csv": {
                "description": (
                    "T-test AND bootstrap-CI results between government pairs, "
                    "stratified by difficulty (companion to pairwise_comparisons.csv, "
                    "which pools all difficulty levels together)"
                ),
                "use_case": (
                    "Answer difficulty-specific significance questions, e.g. "
                    "'is government X significantly ahead of government Y at "
                    "difficulty 75' -- the crossover-point claim. Use this file "
                    "instead of pairwise_comparisons.csv whenever the question is "
                    "tied to a specific difficulty level rather than pooled overall "
                    "performance."
                ),
            },
            "correlation_matrix.csv": {
                "description": "Correlations between metrics",
                "use_case": "Understand metric relationships",
            },
        },
        "metrics": metrics,
        "governments": list(config.governments),
        "difficulties": list(config.difficulties),
        "notes": [
            "All analysis can be performed without re-running the simulation",
            "CI = 95% confidence interval (1.96 * SEM)",
            "Pairwise comparisons use independent t-tests; p < 0.05 = significant",
            "Regression and t-test p-values use numpy with error function approximation",
            (
                "pairwise_comparisons_by_difficulty.csv adds a percentile-method "
                "bootstrap confidence interval (n_boot=10000, 95% level, "
                "np.random.default_rng seeded deterministically per "
                "government-pair/difficulty/metric so results are reproducible "
                "across runs) for the mean difference between the two governments "
                "at that difficulty; ci_significant = 'Yes' when that CI excludes "
                "zero. ci_significant is a distinct check from significant_at_0_05 "
                "(t-test p-value) -- the two should usually agree but are computed "
                "independently"
            ),
            (
                "significant_at_0_05 and ci_significant may both read "
                "'Undefined', with t_statistic / p_value / ci_lower_diff / "
                "ci_upper_diff left blank. That means the comparison is "
                "DEGENERATE: at least one of the two cells has zero (or "
                "floating-point-residue) within-cell variance, i.e. every run "
                "in it returned the identical value despite each run facing a "
                "different environment. Such a pair has no "
                "sampling distribution, so it has no t-statistic and no "
                "p-value -- treat it as 'not tested', never as 'not "
                "different'. Compare the cell means directly instead, and see "
                "cell_stats.json's n_distinct / deterministic fields for how "
                "many genuinely distinct values the cell contained"
            ),
        ],
    }

    with open(os.path.join(analysis_dir, "analysis_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run_benchmark(config: BenchmarkConfig) -> int:
    """
    Execute the full sweep described by ``config``.

    Returns a process exit code: 0 only if every (government, difficulty) cell
    completed *and* every cell produced its full ``k_runs`` seeds; 1 if any cell
    failed outright or silently lost individual runs.  Output is written
    incrementally, so a partial failure still leaves the successful cells on disk.

    The distinction matters for a published dataset: a cell that yields 4 of 10
    seeds still writes plausible-looking output, and every downstream mean and
    confidence interval would then be computed over an under-reported n.  Lost
    runs are therefore counted per cell, reported in the final banner, and made
    to fail the exit code rather than being left to log archaeology.

    This function owns the logging session for the whole sweep: the queue and
    its listener are created once here, before the first pool, and torn down
    only after the last one, so no worker record is lost at a pool boundary.

    If ``config.clean_output_root`` is set (``--clean``), ``output_root`` is
    removed before anything else touches it — see :func:`_clean_output_root`.
    Cleaning is always opt-in; the startup check in
    :func:`_warn_on_existing_output` runs unconditionally afterwards and warns
    whenever stale results remain, whether or not ``--clean`` was used, so a
    caller who forgets the flag still gets a loud signal instead of silently
    contaminated output.
    """
    # The log goes to the CWD at invocation, NOT to output_root: it must survive
    # someone clearing the results directory, and it must stay observable if the
    # output mount stalls.  A copy is archived into output_root at the end so an
    # archived dataset is still self-contained.
    log_path = resolve_log_path(
        config.log_prefix, config.log_file, utc_now(), os.getcwd()
    )
    started_at = utc_timestamp()
    wall_start = time.time()

    try:
        with setup_parent_logging(
            log_path, engine_level=config.engine_log_level
        ) as session:
            _print_banner(config, log_path)
            if config.clean_output_root:
                _clean_output_root(config)
            os.makedirs(config.output_root, exist_ok=True)
            _warn_on_existing_output(config)
            _write_manifest(
                config, build_manifest(config, log_path, "running", started_at)
            )

            try:
                exit_code, results = _run_sweep(
                    config, session, wall_start, log_path
                )
            except FingerprintAuditError as exc:
                # The audit is fatal by design — but the manifest is the
                # forensic record of WHY, and losing it because the exception
                # escaped before it was written would make the loudest failure
                # in the system also the least diagnosable.  Record, then let it
                # propagate: the process still exits non-zero with a traceback,
                # which is what "hard failure, not a warning" has to mean.
                bench_log(logging.ERROR, "SWEEP ABORTED — %s", exc)
                _write_manifest(config, build_manifest(
                    config, log_path,
                    status="failed_environment_audit",
                    started_at=started_at,
                    ended_at=utc_timestamp(),
                    wall_seconds=time.time() - wall_start,
                    results={
                        "exit_code": 1,
                        "environment_audit_error": str(exc),
                    },
                ))
                raise

            _write_manifest(config, build_manifest(
                config, log_path,
                status="complete" if exit_code == 0 else "complete_with_losses",
                started_at=started_at,
                ended_at=utc_timestamp(),
                wall_seconds=time.time() - wall_start,
                results=results,
            ))
            archive_target = os.path.join(config.output_root, ARCHIVED_LOG_NAME)
            log(f"  Log archived → {archive_target}")
    finally:
        # In the `finally` so an aborted sweep archives its log too — that log
        # is where the per-coordinate violation lines are.  Outside the session
        # context, so the handler is closed and the copy is complete rather than
        # truncated at the last flush.  `_archive_log` swallows its own OSError,
        # so this cannot mask the exception that brought us here.
        _archive_log(log_path, config.output_root)
    return exit_code


def _archive_log(log_path: str, output_root: str) -> None:
    """
    Copy the sweep log into ``output_root`` so the dataset is self-contained.

    The original stays in the CWD (that is where the user was told to look, and
    it must outlive the results directory).  The copy costs ~1.6 MB at
    production scale and stops the log from being orphaned from its data when
    the directory is archived or shared; ``manifest.json`` names the original
    path either way.
    """
    target = os.path.join(output_root, ARCHIVED_LOG_NAME)
    try:
        if os.path.abspath(log_path) == os.path.abspath(target):
            return
        shutil.copyfile(log_path, target)
    except OSError as exc:
        # Purely an archival convenience; never fail a completed sweep for it.
        print(f"  [WARN] could not archive log to {target}: {exc}", flush=True)


def _run_sweep(
    config: BenchmarkConfig,
    session,
    wall_start: float,
    log_path: str,
) -> Tuple[int, Dict[str, Any]]:
    """Body of the sweep.  Returns ``(exit_code, manifest_results)``."""
    governments = list(config.governments)
    difficulties = list(config.difficulties)
    n_govs = len(governments)
    n_diffs = len(difficulties)
    total_tasks = config.total_cells

    all_combined_rows: List[dict] = []
    completed_tasks = 0
    failed_cells: List[Tuple[str, int]] = []
    # Cells that returned but with fewer than k_runs seeds: (gov, difficulty, got, expected)
    short_cells: List[Tuple[str, int, int, int]] = []
    diff_elapsed_times: List[float] = []

    for diff_idx, difficulty in enumerate(difficulties):
        diffs_remaining = n_diffs - diff_idx
        if diff_elapsed_times:
            avg_per_diff = sum(diff_elapsed_times) / len(diff_elapsed_times)
            eta_str = fmt_duration(avg_per_diff * diffs_remaining)
        else:
            eta_str = "estimating…"

        log("")
        log("─" * 72)
        log(f"  DIFFICULTY {difficulty:>3d} / {difficulties[-1]}"
            f"   |   Level {diff_idx + 1} of {n_diffs}"
            f"   |   Elapsed: {fmt_duration(time.time() - wall_start)}"
            f"   |   ETA: {eta_str}")
        log(f"  Overall: {progress_bar(completed_tasks, total_tasks)}")
        log("─" * 72)

        # No plan is built here any more.  Plans are per (difficulty, run_idx)
        # and are derived inside each worker (see build_plan / _run_one), so the
        # parent ships no environment and the 11 MB-per-task pickling that a
        # list of k_runs plans would cost never happens.
        log(
            f"  [D={difficulty:3d}] Launching {n_govs} worker process(es) — each "
            f"runs {config.k_runs} simulations on its own CPU core, deriving its "
            f"own per-run scenario plan."
        )

        diff_status: Dict[str, str] = {g: "running" for g in governments}
        diff_t0 = time.time()
        futures: Dict = {}
        gov_start_times: Dict[str, float] = {}

        # ProcessPoolExecutor: each government gets its own OS process →
        # no GIL contention, full CPU utilisation on all cores.
        #
        # `mp_context` is load-bearing, not a tuning knob.  Under `fork` a
        # worker created while the parent's QueueListener thread is inside
        # FileHandler.emit() inherits the stream's C-level buffer lock in a held
        # state, with no thread alive to release it, and hangs before it reaches
        # its task loop — taking the whole sweep with it at shutdown().  It is
        # passed explicitly rather than left to the process-global default so
        # that a caller who never calls configure_multiprocessing() still gets a
        # correct pool.  See benchmark_logging."Why spawn".
        #
        # `initializer` gives each worker a QueueHandler pointing back at the
        # parent's listener, before any task executes there, so exactly one
        # process ever writes the log file.
        with ProcessPoolExecutor(
            max_workers=config.effective_workers,
            mp_context=worker_mp_context(),
            initializer=init_worker_logging,
            initargs=session.worker_initargs(),
        ) as executor:
            for gov_name in governments:
                gov_start_times[gov_name] = time.time()
                futures[executor.submit(
                    run_gov_difficulty, config, gov_name, difficulty
                )] = gov_name

            log(f"  [D={difficulty:3d}] All {n_govs} process(es) launched. "
                f"Progress will print as each run completes …")

            for fut in as_completed(futures):
                gov_name = futures[fut]
                gov_secs = time.time() - gov_start_times[gov_name]
                completed_tasks += 1

                try:
                    rows, raw_frames, health_history = fut.result()
                except Exception as exc:
                    diff_status[gov_name] = "error"
                    failed_cells.append((gov_name, difficulty))
                    bench_log(
                        logging.ERROR,
                        "  ✗  CELL FAILED %-12s D=%d: %s: %s",
                        gov_name, difficulty, type(exc).__name__, exc,
                    )
                    bench_log(
                        logging.ERROR, "traceback:\n%s",
                        "".join(traceback.format_exception(
                            type(exc), exc, exc.__traceback__
                        )),
                    )
                    continue

                all_combined_rows.extend(rows)
                diff_status[gov_name] = "done"

                # A cell can return successfully while having silently dropped
                # seeds (run_gov_difficulty skips a run that raises).  Compare
                # what came back against what was asked for, so an incomplete
                # sample is visible here rather than only in the run log.
                if len(rows) < config.k_runs:
                    short_cells.append(
                        (gov_name, difficulty, len(rows), config.k_runs)
                    )

                elapsed_now = time.time() - wall_start
                avg_secs_task = elapsed_now / completed_tasks
                eta_remaining = avg_secs_task * (total_tasks - completed_tasks)
                done_in_diff = sum(1 for s in diff_status.values() if s == "done")
                mark = "✓" if len(rows) == config.k_runs else "!"
                log(
                    f"  {mark}  {gov_name:<12}  D={difficulty}  "
                    f"({len(rows)}/{config.k_runs} runs, {fmt_duration(gov_secs)})  "
                    f"│  D-level: {done_in_diff}/{n_govs} done  "
                    f"│  Overall: {completed_tasks / total_tasks * 100:.1f}%  "
                    f"│  ETA: {fmt_duration(eta_remaining)}"
                )

                # Render visualizations in the MAIN PROCESS (not in workers) so
                # matplotlib's internal locks are never used inside a fork.
                if raw_frames:
                    viz_dir = os.path.join(
                        config.output_root, gov_name, str(difficulty), "visualizations"
                    )
                    try:
                        render_frames(
                            config, raw_frames, viz_dir, difficulty, gov_name,
                            health_history=health_history or None,
                        )
                    except Exception as exc:
                        bench_log(
                            logging.WARNING,
                            "  viz failed for %s D=%d: %s", gov_name, difficulty, exc,
                        )

        diff_elapsed_times.append(time.time() - diff_t0)
        done_govs = [g for g in governments if diff_status[g] == "done"]
        error_govs = [g for g in governments if diff_status[g] == "error"]
        log(f"\n  D={difficulty} complete in {fmt_duration(diff_elapsed_times[-1])}"
            f"  │  ✓ {len(done_govs)} passed  │  ✗ {len(error_govs)} failed")
        _log_difficulty_ranking(all_combined_rows, difficulty, governments)
        if error_govs:
            bench_log(logging.WARNING, "  Failed governments: %s", error_govs)

    total_elapsed = time.time() - wall_start
    lost_runs = sum(expected - got for _, _, got, expected in short_cells)
    expected_runs = config.total_runs
    log("")
    log("=" * 72)
    log(f"  All simulation runs complete in {fmt_duration(total_elapsed)}")
    log(f"  Tasks completed: {completed_tasks}/{total_tasks}")
    log(f"  Runs recorded  : {len(all_combined_rows)}/{expected_runs}")
    log("=" * 72)

    write_combined_csv(
        all_combined_rows, os.path.join(config.output_root, "combined_final_stats.csv")
    )

    # Audited AFTER the CSV is on disk and BEFORE anything derived from it is
    # computed.  Order is deliberate in both directions: on failure the evidence
    # a diagnosis needs is already written, and no plot, statistic or analysis
    # file is produced from a dataset that has been shown to be unfair.
    fingerprint_audit = verify_env_fingerprints(all_combined_rows, config.k_runs)

    exit_code = 1 if (failed_cells or short_cells) else 0
    # Serialised into manifest.json so a downstream analysis can *assert* n=10
    # rather than trust it — the per-cell sample size travels with the data.
    results = {
        "total_cells": total_tasks,
        "cells_completed": completed_tasks,
        "total_runs_expected": expected_runs,
        "total_runs_recorded": len(all_combined_rows),
        "failed_cells": [
            {"government": gov, "difficulty": diff} for gov, diff in failed_cells
        ],
        "short_cells": [
            {"government": gov, "difficulty": diff,
             "runs_recorded": got, "runs_expected": expected}
            for gov, diff, got, expected in short_cells
        ],
        # Positive evidence for the two claims the comparison rests on, carried
        # in the manifest so an archived dataset states them rather than
        # requiring the reader to trust the code that produced it.
        "environment_audit": fingerprint_audit,
        "exit_code": exit_code,
    }

    # Compute and write per-cell CI statistics.
    #
    # Ordering relative to make_plots() is not load-bearing: make_plots() does
    # not read these files back (the figures are Tukey IQR boxplots only —
    # see make_plots' docstring).  The statistics are still computed and
    # written for downstream analysis.
    _, cell_stats_dict = compute_cell_statistics(config, all_combined_rows, results)
    write_cell_stats(config, cell_stats_dict)

    log("\n  Generating summary plots …")
    plots_dir = os.path.join(config.output_root, "plots")
    render_summary_plots(config, all_combined_rows, plots_dir)

    # Fed from the per-run JSON on disk rather than from `all_combined_rows` —
    # the calibration fields are deliberately outside SUMMARY_FIELDS, so they
    # never reach those rows.  See the section header above
    # make_ads_calibration_plots.  This is the sweep's plot-generation call
    # site, shared by run_full_simulation.py and run_quick_test.py (both route
    # here through main_from_config -> run_benchmark -> _run_sweep), so wiring
    # it here covers both.  :func:`replot_from_output_root` (``--plots-only``)
    # is a second call site that regenerates the same figures from an
    # archived tree without simulating; the two must stay in step, which is
    # why they share `_CALIBRATION_GOV_KEYS` rather than each hardcoding its
    # own government list.  It is called
    # unconditionally: the function itself decides which requested government
    # actually has output to draw, which keeps the "regenerate from a
    # directory alone" contract honest — the gate lives with the data, not
    # with the config.
    make_ads_calibration_plots(
        config.output_root, plots_dir, gov_keys=_CALIBRATION_GOV_KEYS)
    # Scope-matched (ADS n_groups == 1 vs A+L) counterparts to the two figures
    # above that support a pooled, closure-as-unit statistic — see
    # make_scope_matched_calibration_plots's own docstring for why this is a
    # separate function/pair of figures rather than more series on the ones
    # above, and for why calibration_half_split has no scope-matched
    # counterpart. Same gov_keys, same call-unconditionally posture, for the
    # same reason.
    make_scope_matched_calibration_plots(
        config.output_root, plots_dir, gov_keys=_CALIBRATION_GOV_KEYS)

    log("\n" + "=" * 72)
    log(f"  {config.label} complete.  Total wall time: {fmt_duration(total_elapsed)}")
    log(f"  Results → {config.output_root}")
    log(f"  Log     → {log_path}")
    if failed_cells:
        bench_log(
            logging.WARNING,
            f"  WARNING: {len(failed_cells)} cell(s) failed outright: {failed_cells}",
        )
    if short_cells:
        bench_log(
            logging.WARNING,
            f"  WARNING: {len(short_cells)} cell(s) produced an INCOMPLETE sample "
            f"({lost_runs} run(s) lost).  Any mean or confidence interval for "
            f"these cells is computed over fewer seeds than the design specifies:",
        )
        for gov_name, difficulty, got, expected in short_cells:
            bench_log(
                logging.WARNING,
                f"           {gov_name:<12} D={difficulty:<4} {got}/{expected} runs",
            )
    if not failed_cells and not short_cells:
        log(f"  All {expected_runs} runs completed — dataset is complete.")
    log("=" * 72)

    # Generate comprehensive analysis files for post-hoc investigation
    log("\n  Generating analysis files for post-hoc investigation …")
    generate_analysis_files(config, all_combined_rows)

    return exit_code, results


# ---------------------------------------------------------------------------
# Replot from an archived output tree  (``--plots-only``)
#
# Without this path, the only way to change a figure would be to re-run the
# sweep that produced it — 47 hours at production scale — even though every
# figure in the tree is already a pure function of files on disk.  Figure and
# caption defects have no simulation component at all, so re-running the
# sweep to fix one would be pure overhead this path removes.
#
# The contract: --plots-only reads ONLY the archived tree.  It does not import
# the entry point's scale constants, it does not accept a k_runs override, and
# it refuses to run rather than guess.  That is not caution for its own sake —
# every figure title in this module embeds k_runs, grid size, agent count and
# cycle count, so a replot that silently used the caller's defaults against
# someone else's archive would produce a figure captioned with numbers that
# never described its data.
# ---------------------------------------------------------------------------

def load_combined_rows(csv_path: str) -> List[dict]:
    """Read ``combined_final_stats.csv`` back into per-run dicts.

    The inverse of :func:`write_combined_csv`, and shaped to be interchangeable
    with the in-memory ``all_combined_rows`` the sweep passes to the plot
    functions: ``difficulty`` and ``run`` come back as ints, every
    :data:`SUMMARY_FIELDS` metric comes back as a float or ``None``, and
    everything else stays a string.

    Blank cells become ``None`` rather than ``""`` deliberately.  ``csv``
    writes a missing value as an empty field and there is no way to tell that
    apart from an empty string on the way back in, but every consumer here
    treats a metric as either a number or absent — so the conversion happens
    once, at the boundary, instead of at each of the four places that would
    otherwise need a ``val == ""`` check.

    Raises ``ConfigError`` when the file is missing or has no header, because
    the caller asked to replot a specific directory and a silent empty result
    would be indistinguishable from a sweep that produced nothing.
    """
    if not os.path.isfile(csv_path):
        raise ConfigError(
            f"no combined_final_stats.csv at {csv_path} — --plots-only needs "
            f"the per-run summary table that the sweep writes, and this "
            f"directory does not have one"
        )
    rows: List[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ConfigError(f"{csv_path} has no header row")
        metric_fields = set(SUMMARY_FIELDS)
        for raw in reader:
            row: Dict[str, Any] = {}
            for key, val in raw.items():
                if key is None:
                    continue
                if val is None or val == "":
                    row[key] = None
                elif key in metric_fields:
                    try:
                        parsed = float(val)
                    except (TypeError, ValueError):
                        parsed = None
                    row[key] = parsed if (
                        parsed is not None and math.isfinite(parsed)
                    ) else None
                elif key in ("difficulty", "run"):
                    try:
                        row[key] = int(val)
                    except (TypeError, ValueError):
                        row[key] = None
                else:
                    row[key] = val
            rows.append(row)
    return rows


def config_from_manifest(output_root: str) -> BenchmarkConfig:
    """Rebuild the :class:`BenchmarkConfig` that produced *output_root*.

    Sourced entirely from the archived ``manifest.json``.  This is what makes a
    replot honest: the figure captions then describe the sweep that actually
    ran, not the sweep the person doing the replot happened to have configured.

    ``output_root`` is the ONE field taken from the caller rather than the
    manifest, and it has to be.  The manifest records an absolute path on the
    machine that ran the sweep, and these archives travel — the production
    corpus in this tree records
    ``/bb/mbige/.../Simulation/benchmark_results`` from the host it was swept
    on.  Honouring that path would write the regenerated figures somewhere that
    does not exist, or worse, somewhere that does.

    Raises ``ConfigError`` rather than falling back to defaults when the
    manifest is missing, unreadable, or incomplete.  A fallback here would mean
    captioning someone else's data with this process's scale constants, which
    is the specific defect class this module keeps having to repair.
    """
    path = os.path.join(output_root, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise ConfigError(
            f"no {MANIFEST_NAME} in {output_root} — --plots-only reads the "
            f"sweep's own scale (k_runs, grid, agents, cycles) from the "
            f"manifest so the regenerated captions describe the archived data. "
            f"Without it the figures would be captioned with this process's "
            f"defaults, which may describe a different sweep entirely."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {type(exc).__name__}: {exc}")

    cfg = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} has no 'config' object")

    required = ("governments", "difficulties", "k_runs", "grid_size",
                "n_agents", "max_cycles", "max_steps")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ConfigError(
            f"{path} config is missing required field(s): {', '.join(missing)}"
        )

    # `paper_k_runs` and `k_runs_source` are read but NOT required, so that an
    # archive written before these fields existed still replots.  They are
    # deliberately
    # absent from `required` above: adding them there would make every existing
    # manifest in the tree unreadable, which is a worse failure than falling
    # back to the current published-design constant for the projection target.
    #
    # The fallbacks are honest about themselves.  A manifest that predates
    # these fields gets `k_runs_source = "unknown"`, not "default" — we
    # genuinely cannot tell whether its k came from the entry point or from a
    # hand edit, and claiming "default" would assert something we cannot
    # verify.
    raw_source = cfg.get("k_runs_source")
    if raw_source is None:
        k_runs_source = K_RUNS_SOURCE_UNKNOWN
    elif raw_source in K_RUNS_SOURCES:
        k_runs_source = str(raw_source)
    else:
        raise ConfigError(
            f"{path} config has k_runs_source={raw_source!r}, which is not one "
            f"of {', '.join(K_RUNS_SOURCES)}"
        )

    try:
        return BenchmarkConfig(
            governments=tuple(cfg["governments"]),
            difficulties=tuple(int(d) for d in cfg["difficulties"]),
            k_runs=int(cfg["k_runs"]),
            grid_size=int(cfg["grid_size"]),
            n_agents=int(cfg["n_agents"]),
            max_cycles=int(cfg["max_cycles"]),
            max_steps=int(cfg["max_steps"]),
            output_root=os.path.abspath(output_root),
            base_seed=int(cfg.get("base_seed", DEFAULT_BASE_SEED)),
            k_runs_source=k_runs_source,
            paper_k_runs=int(cfg.get("paper_k_runs", PAPER_K_RUNS)),
            label=f"Replot of {output_root}",
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} config is not usable: {exc}")


def replot_from_output_root(output_root: str) -> int:
    """Regenerate every figure in *output_root* from the files already there.

    No simulation, no worker pool, no log session — this is a read-mostly pass
    that rewrites ``<output_root>/plots/``.  It regenerates all three families:

    * the Tukey IQR figures (:func:`make_plots`),
    * the median + bootstrap-CI figures (:func:`make_median_ci_plots`),
    * the ADS self-calibration figures (:func:`make_ads_calibration_plots`).

    What it deliberately does NOT regenerate is ``cell_stats.json`` and
    ``analysis/``.  Those are statistical products, not figures; a flag
    called ``--plots-only`` that quietly rewrote the analysis CSVs would be
    misnamed, and the two have different correctness requirements.

    Returns a process exit code: 0 on success.  Errors surface as
    :class:`ConfigError` for the CLI to render, rather than a traceback.
    """
    output_root = os.path.abspath(os.path.expanduser(output_root))
    if not os.path.isdir(output_root):
        raise ConfigError(f"--plots-only: {output_root} is not a directory")

    config = config_from_manifest(output_root)
    rows = load_combined_rows(
        os.path.join(output_root, "combined_final_stats.csv")
    )

    log("=" * 72)
    log(f"  Replotting from {output_root}")
    log(f"  Archived sweep: {len(config.governments)} government(s) x "
        f"{len(config.difficulties)} difficulty level(s) x k_runs="
        f"{config.k_runs}; grid {config.grid_size}x{config.grid_size}, "
        f"{config.n_agents} agents, {config.max_cycles} cycles")
    log(f"  Rows read from combined_final_stats.csv: {len(rows)}")
    if not rows:
        raise ConfigError(
            f"combined_final_stats.csv in {output_root} has a header but no "
            f"data rows — nothing to plot"
        )
    # Loud, not fatal.  A partial archive is a legitimate thing to replot; a
    # partial archive mistaken for a complete one is not.
    expected = config.total_runs
    if len(rows) != expected:
        bench_log(
            logging.WARNING,
            "  --plots-only: %d row(s) present but the manifest describes a "
            "%d-run sweep. The figures will be drawn from what is on disk and "
            "their per-cell n will reflect that, but the title's k_runs comes "
            "from the manifest and will read %d.",
            len(rows), expected, config.k_runs,
        )
    log("=" * 72)

    plots_dir = os.path.join(output_root, "plots")
    log("\n  Generating summary plots …")
    render_summary_plots(config, rows, plots_dir)
    make_ads_calibration_plots(
        output_root, plots_dir, gov_keys=_CALIBRATION_GOV_KEYS)
    make_scope_matched_calibration_plots(
        output_root, plots_dir, gov_keys=_CALIBRATION_GOV_KEYS)
    log(f"\n  Replot complete → {plots_dir}")
    return 0


# ---------------------------------------------------------------------------
# Entry-point plumbing shared by the two CLI scripts
# ---------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser,
                         default_output: str) -> argparse.ArgumentParser:
    """Register the overrides both entry points accept."""
    parser.add_argument(
        "--governments",
        metavar="LIST",
        help=("Comma-separated subset of governments to run "
              f"(default: {', '.join(DEFAULT_GOVERNMENTS)})."
              + (f" Opt-in only, never run by default: "
                 f"{', '.join(sorted(ABLATION_GOVERNMENTS))}."
                 if ABLATION_GOVERNMENTS else "")),
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help=f"Directory for results (default: {default_output}).",
    )
    parser.add_argument(
        "--difficulties",
        metavar="LIST",
        help=(
            "Comma-separated difficulty levels to run instead of the default "
            "sweep, e.g. --difficulties 56,76. Useful for regenerating a "
            "single figure's snapshot at a specific difficulty without "
            "re-running the full sweep."
        ),
    )
    parser.add_argument(
        "--k-runs",
        metavar="N",
        type=int,
        help=(
            "Independent seeds per (government, difficulty) cell. This is the "
            "sweep's SCALE and it dominates the runtime: cost is linear in N. "
            "The value used is recorded in manifest.json as k_runs, together "
            "with k_runs_source=cli-override, so the archive says what ran "
            "without anyone having to infer it from the command line. Figure "
            "titles and per-cell n follow this value; the 'projected to k=...' "
            "line in the median/CI sidecars does NOT — that tracks the "
            "published design so a small pilot can still tell you what the "
            "full-scale figure will look like."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Do not simulate anything. Regenerate every figure in "
            "--output-dir from the data already archived there "
            "(combined_final_stats.csv, the per-run final_stats.json files, "
            "and manifest.json) and exit. The sweep's scale is read from its "
            "own manifest, so the regenerated captions describe the archived "
            "sweep rather than this command's defaults; every other option "
            "except --output-dir is ignored. Leaves cell_stats.json and "
            "analysis/ untouched."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove --output-dir entirely before the sweep starts, so a "
            "deliberate fresh run doesn't require a manual 'rm -rf'. NOT the "
            "default — omitting this flag merges into whatever is already "
            "there (a startup warning fires either way when stale results are "
            "found). Destructive: everything currently in the output "
            "directory is deleted, with no confirmation prompt."
        ),
    )

    # -- logging ---------------------------------------------------------
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help=("Write the sweep log here instead of an auto-named file in the "
              "current directory.  A directory is accepted, in which case the "
              "generated name is placed inside it.  A log is always written; "
              "there is no way to turn it off."),
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=list(LOG_LEVEL_NAMES),
        help=(f"Verbosity of the ENGINE loggers (default "
              f"{DEFAULT_ENGINE_LEVEL}).  The harness's own progress reporting "
              f"is always INFO and is unaffected.  DEBUG on a large sweep emits "
              f"tens of gigabytes — narrow the sweep first."),
    )
    parser.add_argument(
        "--heartbeat-every",
        metavar="N",
        type=int,
        help="Cycles between per-run progress lines in the log.",
    )

    # -- structured per-run detail ---------------------------------------
    parser.add_argument(
        "--no-run-detail",
        action="store_true",
        help=("Do not write run_detail.jsonl.  Reproduces the older, "
              "structured-detail-free output (e.g. when re-running only to "
              "regenerate plots)."),
    )
    parser.add_argument(
        "--no-decision-detail",
        action="store_true",
        # NOTE the doubled %%: argparse runs every help string through
        # `help % dict(...)` when formatting, so a literal percent sign must be
        # escaped or `--help` itself raises.  This is not cosmetic — an
        # unescaped % here makes the *entire* help output of both entry points
        # unreachable, which is how it shipped once already.
        help=("Drop the per-group `groupings` array from decision records "
              "(~95%% of the bytes).  Evidence, summary, winner, enacted laws "
              "and rejection counts are still written."),
    )
    parser.add_argument(
        "--detailed-agents",
        metavar="N",
        nargs="?",
        const=10,
        type=int,
        help=("Also write agent_states.jsonl every N cycles (default 10 when "
              "the flag is given without a value).  Targeted debugging only: "
              f"refused above {DETAILED_AGENTS_MAX_RUNS} total runs, so narrow "
              "the sweep with --governments / --difficulties first."),
    )
    return parser


def parse_difficulty_list(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    """
    Parse a ``--difficulties`` value into a validated tuple, or None if absent.

    Raises ConfigError on non-integer or empty entries so the CLI fails fast
    instead of silently running a different sweep than requested.
    """
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ConfigError("--difficulties was given but contained no values")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        raise ConfigError(f"--difficulties must be integers; got {raw!r}")
    return values


def parse_k_runs(raw: Optional[int]) -> Optional[int]:
    """
    Validate a ``--k-runs`` value, or return None if the flag was absent.

    ``BenchmarkConfig.__post_init__`` also rejects ``k_runs < 1``, but it does
    so with a message about a config field.  Catching it here lets the CLI name
    the *flag* the operator actually typed, and — more importantly — fails
    before any output directory is created or cleaned.  ``--clean --k-runs 0``
    should not delete the previous results and then refuse to run.
    """
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ConfigError(f"--k-runs must be >= 1; got {value}")
    return value


def parse_government_list(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
    """
    Parse a ``--governments`` value into a validated tuple, or None if absent.

    Raises ConfigError on unknown or empty entries so the CLI can fail fast with
    a clear message instead of silently running a subset.
    """
    if raw is None:
        return None
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not names:
        raise ConfigError("--governments was given but contained no names")
    unknown = [n for n in names if n not in GOVERNMENT_REGISTRY]
    if unknown:
        raise ConfigError(
            f"unknown government(s): {', '.join(unknown)}. "
            f"Valid options: {', '.join(GOVERNMENT_REGISTRY)}"
        )
    # Preserve registry order for deterministic output regardless of CLI order.
    return tuple(g for g in GOVERNMENT_REGISTRY if g in set(names))


def configure_multiprocessing() -> None:
    """
    Pin the process-global multiprocessing start method to ``spawn``.

    The pool and the logging session already pass ``worker_mp_context()``
    explicitly, so this is not what makes the sweep correct — it is belt and
    braces, ensuring that any *incidental* child process created anywhere in
    this interpreter is fork-free too.

    ``fork`` is forbidden here: a worker forked while the log listener thread
    holds a stream buffer lock inherits that lock permanently.  See
    benchmark_logging's "Why ``spawn``" section.

    Safe to call more than once; ``force=True`` makes it idempotent.
    """
    multiprocessing.set_start_method(WORKER_START_METHOD, force=True)


def main_from_config(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Apply CLI overrides to ``config``, then run the sweep.  Returns exit code."""
    # --plots-only short-circuits before any override is applied, and before
    # the multiprocessing start method is pinned.  Deliberately so: it must not
    # inherit the caller's k_runs / grid / agents / cycles, because those go
    # straight into the figure captions and the archive being replotted may
    # have been produced by a completely different configuration (the
    # production corpus in this tree was swept at k=10 by a tree whose
    # run_full_simulation.py says K_RUNS = 100).  The
    # only thing it takes from the caller is where to look.
    if getattr(args, "plots_only", False):
        output_root = getattr(args, "output_dir", None) or config.output_root
        return replot_from_output_root(output_root)

    # `or None` on the store_true flags: an absent flag must mean "leave the
    # config alone", not "set this field to False".
    config = config.with_overrides(
        governments=parse_government_list(getattr(args, "governments", None)),
        output_root=getattr(args, "output_dir", None),
        difficulties=parse_difficulty_list(getattr(args, "difficulties", None)),
        k_runs=parse_k_runs(getattr(args, "k_runs", None)),
        log_file=getattr(args, "log_file", None),
        engine_log_level=getattr(args, "log_level", None),
        heartbeat_every=getattr(args, "heartbeat_every", None),
        write_run_detail=(False if getattr(args, "no_run_detail", False) else None),
        decision_detail=(
            False if getattr(args, "no_decision_detail", False) else None
        ),
        agent_states_every=getattr(args, "detailed_agents", None),
        clean_output_root=(True if getattr(args, "clean", False) else None),
    )
    configure_multiprocessing()
    return run_benchmark(config)
