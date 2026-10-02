#!/usr/bin/env python3
"""
Benchmark harness for heuristic optimization.

Runs simulations across multiple terrain configs and difficulty levels,
captures normalized_health_score for each government, and prints a
comparison table. Use --save-baseline to save the current results to
benchmark_baseline.json, written next to this script (i.e. the repository
root, regardless of the caller's current working directory), then run again
after changes to see the delta against that saved baseline (pass
--no-compare to skip the comparison).
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from engine.grid import Grid, Terrain
from engine.events import EventType, ActiveEvent
from engine.simulation import Simulation, SimulationConfig
from engine.scenario_plan import generate_scenario_plan, build_simulation_from_plan, make_scheduled_event_dict
from agents.citizen import CitizenAgent
from governments import GOVERNMENT_REGISTRY
from scenarios.scenario_base import _make_citizen_population

GOVERNMENTS = ["anarchy", "democracy", "republic", "autocracy", "oligarchy", "federated"]
DIFFICULTIES = [10, 25, 40, 60]
SEEDS = [42, 123, 7]
GRID_SIZE = 12
N_AGENTS = 20
MAX_CYCLES = 100

BASELINE_FILE = os.path.join(_HERE, "benchmark_baseline.json")


def build_scenario_events(difficulty, seed):
    config = SimulationConfig.from_difficulty(difficulty, seed=seed,
                                              grid_rows=GRID_SIZE, grid_cols=GRID_SIZE,
                                              num_agents=N_AGENTS, max_cycles=MAX_CYCLES)
    from scenarios.scenario_base import _auto_event_schedule
    auto_events = _auto_event_schedule(MAX_CYCLES, difficulty, seed)
    scheduled = [
        make_scheduled_event_dict(
            cycle=entry[0], event_type=entry[1].value, severity=entry[3],
            duration=20, grid_rows=GRID_SIZE, grid_cols=GRID_SIZE, seed=seed,
        )
        for entry in auto_events
    ]
    return config, scheduled


def run_single(gov_name, difficulty, seed):
    config, scheduled = build_scenario_events(difficulty, seed)
    plan = generate_scenario_plan(config, scheduled_events=scheduled)
    gov_cls = GOVERNMENT_REGISTRY[gov_name]
    government = gov_cls(seed=seed) if gov_name != "anarchy" else gov_cls()
    agents = _make_citizen_population(plan.config_params["num_agents"], seed=seed)
    sim = build_simulation_from_plan(plan, government, agents, verbose=False)
    sim.run()
    return sim.normalized_health_score()


def run_benchmark():
    results = {}
    total = len(GOVERNMENTS) * len(DIFFICULTIES) * len(SEEDS)
    done = 0
    t0 = time.time()

    for gov in GOVERNMENTS:
        results[gov] = {}
        for diff in DIFFICULTIES:
            scores = []
            for seed in SEEDS:
                try:
                    score = run_single(gov, diff, seed)
                    scores.append(score)
                except Exception as e:
                    print(f"  ERROR: {gov} d={diff} seed={seed}: {e}")
                    scores.append(0.0)
                done += 1
            avg = sum(scores) / len(scores)
            results[gov][str(diff)] = round(avg, 4)
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] {gov:12s} d={diff:2d}  avg={avg:.4f}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return results


def print_table(results, baseline=None):
    print(f"\n{'='*90}")
    header = f"{'Government':<14}"
    for d in DIFFICULTIES:
        header += f"  d={d:<4}"
        if baseline:
            header += f" {'delta':>7}"
    header += f"  {'AVG':>6}"
    if baseline:
        header += f"  {'dAVG':>6}"
    print(header)
    print("-" * 90)

    for gov in GOVERNMENTS:
        row = f"{gov:<14}"
        vals = []
        deltas = []
        for d in DIFFICULTIES:
            val = results.get(gov, {}).get(str(d), 0.0)
            vals.append(val)
            row += f"  {val:.4f}"
            if baseline:
                base_val = baseline.get(gov, {}).get(str(d), 0.0)
                delta = val - base_val
                deltas.append(delta)
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:+.4f}"
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f"  {avg:.4f}"
        if baseline and deltas:
            avg_delta = sum(deltas) / len(deltas)
            sign = "+" if avg_delta >= 0 else ""
            row += f"  {sign}{avg_delta:+.4f}"
        print(row)
    print(f"{'='*90}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark heuristic optimization")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as baseline")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip baseline comparison")
    args = parser.parse_args()

    print(f"Benchmark: {len(GOVERNMENTS)} govs × {len(DIFFICULTIES)} difficulties "
          f"× {len(SEEDS)} seeds = {len(GOVERNMENTS)*len(DIFFICULTIES)*len(SEEDS)} runs")
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}, Agents: {N_AGENTS}, Cycles: {MAX_CYCLES}\n")

    results = run_benchmark()

    baseline = None
    if not args.no_compare and os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
        print("\n  (Comparing against saved baseline)")

    print_table(results, baseline)

    if args.save_baseline:
        with open(BASELINE_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Baseline saved to {BASELINE_FILE}")


if __name__ == "__main__":
    main()
