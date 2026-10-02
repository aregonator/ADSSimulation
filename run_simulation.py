#!/usr/bin/env python3
"""
run_simulation.py — entry point for the Decision-Making Survival Simulation.

Usage examples:
  python run_simulation.py                                         # auto mode, difficulty=25
  python run_simulation.py --difficulty 50 --governments all
  python run_simulation.py --scenario epidemic --government ads
  python run_simulation.py --scenario climate_crisis --governments all
  python run_simulation.py --difficulty 75 --grid-size 30 --cycles 300
  python run_simulation.py --difficulty 40 --save-plan plan.json
  python run_simulation.py --plan-file plan.json --governments all
  python run_simulation.py --list-scenarios
  python run_simulation.py --list-governments
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

# Ensure package is importable: add current directory to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scenarios.scenario_base import (
    build_auto_simulation, SCENARIO_REGISTRY,
    _make_citizen_population, _make_ads_population,
)
from governments import (
    ABLATION_GOVERNMENTS, DEFAULT_GOVERNMENTS, GOVERNMENT_REGISTRY,
)
from engine.metrics import MultiRunComparison
from engine.scenario_plan import (
    generate_scenario_plan, build_simulation_from_plan, ScenarioPlan,
)

#: What ``--governments all`` expands to: the paper's eight primary regimes.
#:
#: Still sourced from DEFAULT_GOVERNMENTS rather than from GOVERNMENT_REGISTRY,
#: even though the two currently agree: ``all`` should mean "the default sweep",
#: and if a future entry is registered but excluded from that sweep, ``all``
#: must follow the exclusion automatically rather than by a second edit here.
ALL_GOVERNMENTS = list(DEFAULT_GOVERNMENTS)

#: Every runnable name, for ``--list-governments`` and for validation.
AVAILABLE_GOVERNMENTS = list(GOVERNMENT_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision-Making at Scale — Survival Grid Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON config file containing simulation parameters. "
            "Explicit CLI flags take precedence over values in the file. "
            "Run --generate-only to inspect the resulting config before simulating."
        ),
    )
    parser.add_argument(
        "--scenario", "-s",
        default=None,
        choices=SCENARIO_REGISTRY,
        help=(
            "Scenario to run. If omitted, auto mode runs a comprehensive "
            "difficulty-driven simulation covering all event types."
        ),
    )
    parser.add_argument(
        "--government", "-g",
        default=None,
        help="Single government type. Use --governments for multiple.",
    )
    parser.add_argument(
        "--governments",
        default=None,
        help="Comma-separated list of government types, or 'all'.",
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=None,
        help="Override max cycles (default: difficulty-based).",
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=int,
        default=None,
        metavar="1-100",
        help="Difficulty level (integer 1–100; ~12=easy, ~25=normal, ~50=hard, 100=extreme). Default: 25.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        metavar="N",
        help="Grid dimensions as NxN (default: difficulty-based). Overrides both rows and cols.",
    )
    parser.add_argument(
        "--event-severity",
        type=float,
        default=None,
        help="Override event severity multiplier.",
    )
    parser.add_argument(
        "--agents", "-n",
        type=int,
        default=None,
        help="Override number of agents.",
    )
    parser.add_argument(
        "--plan-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a scenario plan JSON file. "
            "Use --save-plan to generate, or provide an existing file for fair comparison."
        ),
    )
    parser.add_argument(
        "--save-plan",
        default=None,
        metavar="PATH",
        help="Generate and save a scenario plan to PATH, then run using it.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization plots after each run.",
    )
    parser.add_argument(
        "--save-viz",
        default=None,
        metavar="DIR",
        help="Save visualization images to this directory.",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        metavar="DIR",
        help="Save MP4 video of simulation progression to this directory (5 fps, two-panel).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=5,
        help="Frames per second for the output video. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-cycle progress.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to write CSV metrics files.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--list-governments",
        action="store_true",
        help="List available government types and exit.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Maximum MOVE actions an agent may take per cycle (default: 1). "
            "Higher values let agents cross the map faster to reach distant resources."
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Generate and display the grid map and scenario plan without running "
            "the simulation. Combines well with --save-plan to inspect a plan first."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "NONE"],
        default="NONE",
        help="Logging level for simulation trace logs (one file per government in --output-dir).",
    )
    return parser.parse_args()


def resolve_governments(args: argparse.Namespace) -> List[str]:
    """
    Resolve the requested government names, or fail with a usable message.

    ``all`` means the eight primary regimes (see ``ALL_GOVERNMENTS``).  Unknown
    names are rejected here rather than surfacing later as a bare ``KeyError``
    from inside a worker thread — with names like ``autocracy_lookahead`` in
    play, a typo is likely and a bare ``KeyError`` from a worker thread gives
    the user nothing to act on.
    """
    if args.governments:
        if args.governments.strip().lower() == "all":
            return list(ALL_GOVERNMENTS)
        names = [g.strip() for g in args.governments.split(",") if g.strip()]
    elif args.government:
        names = [args.government.strip()]
    else:
        return ["anarchy", "democracy", "ads"]

    unknown = [n for n in names if n not in GOVERNMENT_REGISTRY]
    if unknown:
        raise SystemExit(
            f"error: unknown government(s): {', '.join(unknown)}. "
            f"Valid options: {', '.join(AVAILABLE_GOVERNMENTS)} "
            f"(or 'all' for the eight primary regimes)."
        )
    return names


def build_config_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.difficulty is not None:
        d = max(1, min(100, args.difficulty))
        overrides["difficulty"] = d
    if args.event_severity:
        overrides["event_severity"] = args.event_severity
    if args.agents:
        overrides["num_agents"] = args.agents
    if getattr(args, "max_steps", None) is not None:
        overrides["max_steps_per_cycle"] = max(1, args.max_steps)
    return overrides


def save_csv(government_name: str, metrics, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{government_name.replace(' ', '_')}.csv")
    rows = metrics.to_csv_rows()
    if not rows:
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Metrics saved to {filename}")


def save_audit_trail(government_name: str, sim, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    safe_name = government_name.replace(" ", "_")
    filename = os.path.join(output_dir, f"{safe_name}_trail.log")
    sim.audit.save(filename)
    print(f"  → Audit trail saved to {filename}")


# SimulationConfig fields that can appear in a config JSON file.
_SIM_CONFIG_FIELDS = {
    "grid_rows", "grid_cols", "resource_density", "terrain_variety",
    "num_agents", "initial_health", "event_frequency", "event_severity",
    "event_warning_cycles", "visibility_radius",
    "max_steps_per_cycle", "max_cycles", "record_every",
}

# CLI arg names whose JSON counterparts map directly onto args attributes.
# Tuple: (json_key, args_attr, args_default)
_CLI_MAPPINGS = [
    ("scenario",       "scenario",       None),
    ("governments",    "governments",    None),
    ("government",     "government",     None),
    ("difficulty",     "difficulty",     None),
    ("grid_size",      "grid_size",      None),
    ("cycles",         "cycles",         None),
    ("seed",           "seed",           None),
    ("verbose",        "verbose",        False),
    ("agents",         "agents",         None),
    ("event_severity", "event_severity", None),
    ("max_steps",      "max_steps",      None),
    ("visualize",      "visualize",      False),
    ("output_dir",     "output_dir",     None),
    ("save_plan",      "save_plan",      None),
    ("plan_file",      "plan_file",      None),
]


def _apply_config_file(args: argparse.Namespace, config_path: str) -> dict:
    """
    Load *config_path* (JSON) and apply its fields to *args* and a returned
    overrides dict.

    Rules:
    - Fields that map to CLI args are written to *args* only when the current
      arg value is still at its default (i.e. the user did not explicitly pass
      that flag on the command line). Value-typed flags (``--difficulty``,
      ``--seed``, ...) use ``None`` as their argparse default specifically so
      that an explicit CLI value equal to the *documented* default (e.g.
      ``--difficulty 25``) is still distinguishable from "not passed" and
      always wins over the file. Boolean ``store_true`` flags need no such
      sentinel: their unset value (``False``) can never be produced by an
      explicit flag, so the plain default comparison is unambiguous for them.
    - Fields that are SimulationConfig parameters but not CLI args are returned
      in the overrides dict; the caller merges them before building the config.
    - Unknown keys in the JSON are silently ignored.

    Returns a dict of SimulationConfig overrides read from the file.
    """
    with open(config_path) as fh:
        cfg = json.load(fh)

    for json_key, attr, default in _CLI_MAPPINGS:
        if json_key not in cfg:
            continue
        current = getattr(args, attr, None)
        if current == default or current is None:
            val = cfg[json_key]
            # "governments" may be a list in JSON; the arg expects a string.
            if attr == "governments" and isinstance(val, list):
                val = ",".join(val)
            setattr(args, attr, val)

    return {k: cfg[k] for k in cfg if k in _SIM_CONFIG_FIELDS}


_TERRAIN_CHAR = {
    "plains":   ".",
    "forest":   "F",
    "water":    "W",
    "mountain": "M",
    "wasteland":"X",
}


def _print_grid_ascii(grid, agent_placements=None) -> None:
    """Print a compact ASCII map of the grid to stdout."""
    from engine.grid import Terrain

    agent_set = set(tuple(p) for p in (agent_placements or []))
    cols = grid.cols
    col_header = "".join(str(c % 10) for c in range(cols))
    print(f"\n  Grid ({grid.rows}×{cols})")
    print(f"  Legend: . plains  F forest  W water  M mountain  X wasteland  @ agent-start")
    print(f"     {col_header}")
    for r in range(grid.rows):
        row_str = ""
        for c in range(cols):
            cell = grid.cell(r, c)
            ch = _TERRAIN_CHAR.get(cell.terrain.value, "?")
            if (r, c) in agent_set:
                ch = "@"
            row_str += ch
        print(f"  {r:2d} {row_str}")
    print()

    terrain_counts: dict = {}
    for cell in grid.all_cells():
        t = cell.terrain.value
        terrain_counts[t] = terrain_counts.get(t, 0) + 1
    total = grid.rows * grid.cols
    print("  Terrain distribution:")
    for t, n in sorted(terrain_counts.items(), key=lambda x: -x[1]):
        bar = "#" * round(n / total * 40)
        print(f"    {t:<10} {n:4d} ({n/total*100:4.1f}%)  {bar}")
    print()


def _build_auto_mode_config(difficulty, seed, config_overrides, grid_size, max_cycles, agents):
    """
    Build the SimulationConfig for auto mode (no ``--scenario``) — the one
    path shared by an actual run and ``--generate-only``.

    Precedence, applied in this order so each later step wins over the one
    before it: (1) the difficulty-based baseline, (2) ``--config`` file /
    merged overrides, (3) the CLI-explicit ``--grid-size``/``--cycles``/
    ``--agents`` flags. Both callers build through this function so
    ``--generate-only`` (with or without ``--save-plan``) previews exactly
    the environment the real run would use for the same arguments.
    """
    from engine.simulation import SimulationConfig

    config = SimulationConfig.from_difficulty(difficulty, seed=seed)
    for key, val in config_overrides.items():
        if hasattr(config, key) and key != "difficulty":
            setattr(config, key, val)
    if grid_size is not None:
        config.grid_rows = grid_size
        config.grid_cols = grid_size
    if max_cycles is not None:
        config.max_cycles = max_cycles
    if agents is not None:
        config.num_agents = agents
    return config


def _run_generate_only(args, config_overrides, difficulty, grid_size, max_cycles) -> None:
    """Generate a grid + scenario plan and print without running simulation."""
    from engine.simulation import SimulationConfig
    from engine.grid import Grid
    from engine.scenario_plan import generate_scenario_plan, make_scheduled_event_dict
    from scenarios.scenario_base import _auto_event_schedule, SCENARIOS

    print(f"\n{'='*60}")
    print(f"  Generate-Only Mode")
    print(f"  Difficulty: {difficulty}  Seed: {args.seed}")
    print(f"{'='*60}")

    if args.scenario:
        scenario = SCENARIOS.get(args.scenario, {})
        cfg_dict = dict(scenario.get("config", {}))
        cfg_dict.update(config_overrides)
        cfg_dict["seed"] = args.seed
        cfg_dict["initial_health"] = 1.0
        d = cfg_dict.pop("difficulty", difficulty)
        if isinstance(d, str):
            mapping = {"easy": 12, "normal": 25, "hard": 50, "extreme": 100}
            d = mapping.get(d.lower(), 25)
        d = max(1, min(100, int(d)))
        cfg_dict["difficulty"] = d
        if grid_size is not None:
            cfg_dict["grid_rows"] = grid_size
            cfg_dict["grid_cols"] = grid_size
        if max_cycles is not None:
            cfg_dict["max_cycles"] = max_cycles
        config = SimulationConfig(**cfg_dict)
        scheduled = [
            make_scheduled_event_dict(
                cycle=entry[0], event_type=entry[1].value, severity=entry[3],
                duration=20, grid_rows=config.grid_rows, grid_cols=config.grid_cols,
                seed=args.seed,
            )
            for entry in scenario.get("scheduled_events", [])
        ]
    else:
        config = _build_auto_mode_config(
            difficulty, args.seed, config_overrides, grid_size, max_cycles, args.agents
        )
        auto_events = _auto_event_schedule(config.max_cycles, difficulty, args.seed)
        from engine.events import EventType as _ET
        scheduled = [
            make_scheduled_event_dict(
                cycle=entry[0], event_type=entry[1].value, severity=entry[3],
                duration=12 if entry[1] == _ET.EPIDEMIC else 20,
                grid_rows=config.grid_rows, grid_cols=config.grid_cols,
                seed=args.seed,
            )
            for entry in auto_events
        ]

    plan = generate_scenario_plan(config, scheduled_events=scheduled)

    # Build a Grid from the plan to display it
    grid = Grid(
        rows=plan.rows, cols=plan.cols,
        resource_density=config.resource_density,
        terrain_variety=config.terrain_variety,
        seed=config.seed,
        ambient_hazard=config.ambient_hazard,
    )
    from engine.scenario_plan import _restore_grid
    _restore_grid(grid, plan.grid_cells)

    _print_grid_ascii(grid, agent_placements=plan.agent_placements)

    print(f"  Scenario Plan Summary:")
    print(f"  {plan.summary()}")
    print()
    print(f"  Config parameters:")
    for k, v in plan.config_params.items():
        print(f"    {k:<25} {v}")

    print(f"\n  Events ({len(plan.events_plan)} total):")
    for ev in plan.events_plan[:20]:
        src = ev.get("source", "")
        print(f"    cycle {ev['cycle']:4d}  {ev['event_type']:<15} severity={ev['severity']:.2f}  "
              f"duration={ev.get('duration',0):3d}  [{src}]")
    if len(plan.events_plan) > 20:
        print(f"    ... and {len(plan.events_plan) - 20} more events")

    save_plan = getattr(args, "save_plan", None)
    if save_plan:
        plan.save(save_plan)
        print(f"\n  Plan saved to: {save_plan}")

    print()


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _attach_log_handler(gov_name: str, output_dir: str, level_str: str) -> logging.FileHandler:
    """Configure a FileHandler on the sim.* logger hierarchy for one government run.

    Returns the handler so the caller can remove it after the run completes.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = gov_name.replace(" ", "_")
    log_path = os.path.join(output_dir, f"{safe_name}_sim.log")
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(getattr(logging, level_str))
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    # Attach to the root "sim" logger so all sim.* child loggers inherit it.
    sim_logger = logging.getLogger("sim")
    sim_logger.setLevel(getattr(logging, level_str))
    sim_logger.addHandler(handler)
    return handler


def _detach_log_handler(handler: logging.FileHandler) -> None:
    """Flush, close, and remove the handler from the sim logger."""
    sim_logger = logging.getLogger("sim")
    sim_logger.removeHandler(handler)
    handler.flush()
    handler.close()


def main() -> None:
    args = parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        from scenarios.scenario_base import SCENARIOS
        for name, defn in SCENARIOS.items():
            print(f"  {name:<22} — {defn['description']}")
        print("\n  (no --scenario)         — Auto mode: difficulty-driven, all event types")
        print()
        return

    if args.list_governments:
        # Every runnable name, not just what `--governments all` expands to.
        # The label below fires only for names genuinely excluded from the
        # default sweep; ABLATION_GOVERNMENTS is empty today, so nothing is
        # labelled and all eight print plain.  `autocracy_lookahead` is no
        # longer flagged — it is a primary regime now, not a control.
        print("\nAvailable government types:")
        for name in AVAILABLE_GOVERNMENTS:
            if name in ABLATION_GOVERNMENTS:
                print(f"  {name:<22} [excluded from the default sweep — "
                      f"opt-in, not included in 'all']")
            else:
                print(f"  {name}")
        print(f"\n  'all' runs the {len(ALL_GOVERNMENTS)} primary regimes: "
              f"{', '.join(ALL_GOVERNMENTS)}.")
        if ABLATION_GOVERNMENTS:
            print("  Run an excluded condition by naming it, e.g. "
                  f"--governments {sorted(ABLATION_GOVERNMENTS)[0]}")
        print()
        return

    # Load --config JSON before resolving anything else so its values can be
    # overridden by explicit CLI flags that were already set by parse_args().
    json_overrides: dict = {}
    if getattr(args, "config", None):
        json_overrides = _apply_config_file(args, args.config)

    # Resolve the documented defaults now that --config precedence has been
    # settled. args.difficulty/args.seed are None here only if neither the
    # CLI nor the config file set them.
    if args.difficulty is None:
        args.difficulty = 25
    if args.seed is None:
        args.seed = 42

    governments = resolve_governments(args)
    # CLI overrides take precedence over JSON; merge JSON first, then CLI on top.
    config_overrides = {**json_overrides, **build_config_overrides(args)}
    difficulty = max(1, min(100, args.difficulty))
    grid_size = args.grid_size
    max_cycles = args.cycles

    if getattr(args, "generate_only", False):
        _run_generate_only(args, config_overrides, difficulty, grid_size, max_cycles)
        return

    mode = args.scenario if args.scenario else "auto"
    print(f"\n{'='*60}")
    print(f"  Mode:        {mode}")
    print(f"  Difficulty:  {difficulty}")
    print(f"  Governments: {governments}")
    print(f"  Seed:        {args.seed}")
    if grid_size:
        print(f"  Grid size:   {grid_size}×{grid_size}")
    if max_cycles:
        print(f"  Max cycles:  {max_cycles}")
    if config_overrides:
        extras = {k: v for k, v in config_overrides.items() if k not in ("difficulty",)}
        if extras:
            print(f"  Overrides:   {extras}")
    print(f"{'='*60}\n")

    comparison = MultiRunComparison()
    plan = _load_or_generate_plan(args, governments, config_overrides, difficulty, grid_size, max_cycles)

    viz_dir = getattr(args, "save_viz", None)
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
    video_dir = getattr(args, "save_video", None)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    gov_sims: Dict[str, object] = {}
    log_level = getattr(args, "log_level", "NONE")
    _print_lock = threading.Lock()

    def _run_one_government(gov_name: str):
        """Run a single government simulation. Returns (gov_name, metrics, sim) or raises."""
        _log_handler: Optional[logging.FileHandler] = None
        if log_level != "NONE" and args.output_dir:
            _log_handler = _attach_log_handler(gov_name, args.output_dir, log_level)
        try:
            gov_cls = GOVERNMENT_REGISTRY[gov_name]
            seed = args.seed
            government = (gov_cls(seed=seed) if gov_name != "anarchy" else gov_cls())
            agents = (
                _make_ads_population(plan.config_params["num_agents"], seed=seed)
                if gov_name == "ads"
                else _make_citizen_population(plan.config_params["num_agents"], seed=seed)
            )
            sim = build_simulation_from_plan(plan, government, agents, verbose=args.verbose)

            do_viz = getattr(args, "visualize", False) or viz_dir or video_dir
            if do_viz:
                from engine.visualizer import SimulationVisualizer
                viz = SimulationVisualizer()
                for cycle in range(sim.config.max_cycles):
                    sim.cycle = cycle
                    sim._step(cycle)
                    viz.record_frame(sim, cycle)
                metrics = sim.metrics
            else:
                metrics = sim.run()

            return gov_name, metrics, sim, do_viz, (viz if do_viz else None)
        finally:
            if _log_handler is not None:
                _detach_log_handler(_log_handler)

    # Run each government in its own thread for parallel execution
    with _print_lock:
        print(f"  Running {len(governments)} governments in parallel (one thread each)...")

    futures = {}
    with ThreadPoolExecutor(max_workers=len(governments)) as executor:
        for gov_name in governments:
            t0 = time.time()
            future = executor.submit(_run_one_government, gov_name)
            futures[future] = (gov_name, t0)

        for future in as_completed(futures):
            gov_name, t0 = futures[future]
            elapsed = time.time() - t0
            try:
                gov_name, metrics, sim, do_viz, viz = future.result()
                with _print_lock:
                    print(f"  Completed in {elapsed:.1f}s")
                    metrics.print_report(gov_name)
                comparison.add(gov_name, metrics)
                gov_sims[gov_name] = sim

                if args.output_dir:
                    save_csv(gov_name, metrics, args.output_dir)
                    save_audit_trail(gov_name, sim, args.output_dir)

                if do_viz and viz is not None:
                    _save_or_show_viz(viz, sim, gov_name, viz_dir,
                                      getattr(args, "visualize", False),
                                      video_dir=video_dir,
                                      video_fps=getattr(args, "video_fps", 5))
            except Exception as e:
                with _print_lock:
                    print(f"  [ERROR] {gov_name} failed after {elapsed:.1f}s: {e}")
                import traceback
                traceback.print_exc()

    if len(governments) > 1:
        comparison.print_comparison()
        best = comparison.best_by("normalized_health_score")
        if best:
            print(f"  Best normalized health score: {best}\n")

        if (getattr(args, "visualize", False) or viz_dir) and gov_sims:
            try:
                from engine.visualizer import SimulationVisualizer
                fig = SimulationVisualizer.compare_governments(
                    {n: gov_sims[n].metrics for n in gov_sims}
                )
                if viz_dir:
                    fig.savefig(os.path.join(viz_dir, "comparison.png"),
                                dpi=150, bbox_inches="tight")
                    print(f"  → Comparison chart saved to {viz_dir}/comparison.png")
                if getattr(args, "visualize", False):
                    import matplotlib.pyplot as plt
                    plt.show()
            except Exception as e:
                print(f"  [WARN] Comparison chart failed: {e}")

    print("Done.")


def _load_or_generate_plan(args, governments, config_overrides, difficulty, grid_size, max_cycles):
    """Always return a ScenarioPlan for fair cross-government comparison."""
    plan_file = getattr(args, "plan_file", None)
    save_plan = getattr(args, "save_plan", None)

    if plan_file:
        plan = ScenarioPlan.load(plan_file)
        print(f"  Loaded scenario plan: {plan.summary()}")
        return plan

    from engine.simulation import SimulationConfig
    from scenarios.scenario_base import _auto_event_schedule, SCENARIOS
    from engine.scenario_plan import make_scheduled_event_dict

    if args.scenario:
        scenario = SCENARIOS.get(args.scenario, {})
        cfg_dict = dict(scenario.get("config", {}))
        cfg_dict.update(config_overrides)
        cfg_dict["seed"] = args.seed
        cfg_dict["initial_health"] = 1.0
        d = cfg_dict.pop("difficulty", difficulty)
        if isinstance(d, str):
            mapping = {"easy": 12, "normal": 25, "hard": 50, "extreme": 100}
            d = mapping.get(d.lower(), 25)
        d = max(1, min(100, int(d)))
        cfg_dict["difficulty"] = d
        if grid_size is not None:
            cfg_dict["grid_rows"] = grid_size
            cfg_dict["grid_cols"] = grid_size
        if max_cycles is not None:
            cfg_dict["max_cycles"] = max_cycles
        config = SimulationConfig(**cfg_dict)
        scheduled = [
            make_scheduled_event_dict(
                cycle=entry[0],
                event_type=entry[1].value,
                severity=entry[3],
                duration=20,
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                seed=args.seed,
            )
            for entry in scenario.get("scheduled_events", [])
        ]
    else:
        # Same helper _run_generate_only uses, so a real run and
        # --generate-only always build an identical environment for the
        # same arguments (grid size, agents, every difficulty-scaled field).
        config = _build_auto_mode_config(
            difficulty, args.seed, config_overrides, grid_size, max_cycles, args.agents
        )
        auto_events = _auto_event_schedule(config.max_cycles, difficulty, args.seed)
        from engine.events import EventType as _ET
        scheduled = [
            make_scheduled_event_dict(
                cycle=entry[0],
                event_type=entry[1].value,
                severity=entry[3],
                duration=12 if entry[1] == _ET.EPIDEMIC else 20,
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                seed=args.seed,
            )
            for entry in auto_events
        ]

    plan = generate_scenario_plan(config, scheduled_events=scheduled)
    if save_plan:
        plan.save(save_plan)
        print(f"  Scenario plan saved to {save_plan}: {plan.summary()}")
    print(f"  Generated scenario plan: {plan.summary()}")
    return plan


def _save_or_show_viz(viz, sim, gov_name, viz_dir, show, video_dir=None, video_fps=5):
    """Save or display visualizations for a single government run."""
    safe_name = gov_name.replace(" ", "_")

    if viz_dir or show:
        try:
            import matplotlib.pyplot as plt
            fig = viz.render_final(sim, government_name=gov_name)
            if viz_dir:
                path = os.path.join(viz_dir, f"{safe_name}_dashboard.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  → Dashboard saved to {path}")
            if show:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Dashboard failed for {gov_name}: {e}")

    if video_dir and len(viz._frames) > 1:
        try:
            video_path = os.path.join(video_dir, f"{safe_name}.mp4")
            viz.save_video(video_path, fps=video_fps, government_name=gov_name)
        except Exception as e:
            # MP4 failed (no ffmpeg) — fall back to GIF
            print(f"  [WARN] MP4 failed ({e}), trying GIF ...")
            try:
                gif_path = os.path.join(video_dir, f"{safe_name}.gif")
                viz.save_video(gif_path, fps=video_fps, government_name=gov_name)
            except Exception as e2:
                print(f"  [WARN] Video export failed: {e2}")


if __name__ == "__main__":
    main()
