# System Architecture

How the code is organised, how one simulation cycle executes, and how the benchmark sweep turns individual runs into the published output archive.

---

## Repository Layout

All paths are relative to the repository root. There is no enclosing package: scripts are run from the root, which is placed on `sys.path` so that `engine`, `governments`, `agents` and `scenarios` import as top-level packages.

```
./
├── README.md                    overview, installation, quick start
├── requirements.txt             numpy, matplotlib (everything else is stdlib)
├── config.json                  sample single-run configuration for run_simulation.py --config
│
├── run_full_simulation.py       publication sweep entry point (8 governments × 21 difficulties × k runs)
├── run_quick_test.py            end-to-end smoke test (8 × 3 difficulties × 2 runs, small grid)
├── run_simulation.py            single-scenario / ad-hoc CLI (scenarios, plans, videos, audit trails)
├── benchmark_core.py            sweep engine: configuration, workers, output files, statistics, figures
├── benchmark_logging.py         multiprocess-safe logging for the sweep
│
├── test_ads_calibration.py      parameter inference, forecast ledger, evaluator invariants
├── test_default_governments.py  default sweep = the eight primary governments
├── test_difficulty_schedule.py  difficulty schedule and its consumers
├── test_median_ci.py            bootstrap median CI and its seeding
├── test_plot_regression.py      figure layer and --plots-only
├── test_seed_derivation.py      seed lattice and environment fingerprint
├── test_terrain_consistency.py  cell hazard, shelter and regen agree with cell terrain
│
├── engine/                      simulation machinery (no government logic)
│   ├── agent.py                 Agent base class, Action / ActionType, movement helpers
│   ├── grid.py                  Grid, Cell, Terrain, regeneration, global depletion
│   ├── events.py                EventSystem, ActiveEvent, EventWarning, observed event statistics
│   ├── difficulty.py            the difficulty schedule (single source of truth)
│   ├── simulation.py            SimulationConfig, Simulation loop, health dynamics
│   ├── metrics.py               MetricsCollector, per-cycle snapshots, run summary
│   ├── scenario_plan.py         derive_seed, ScenarioPlan, plan generation and replay
│   ├── run_recorder.py          per-run run_detail.jsonl / agent_states.jsonl writer
│   ├── visualizer.py            grid snapshots and dashboards (matplotlib)
│   └── audit.py                 human-readable per-cycle audit trail (run_simulation.py)
│
├── governments/
│   ├── __init__.py              GOVERNMENT_REGISTRY, DEFAULT_GOVERNMENTS (import-time check: exactly 8)
│   ├── base.py                  Government base class, Law, law enforcement, shared helpers
│   ├── anarchy.py
│   ├── democracy.py
│   ├── republic.py
│   ├── autocracy.py
│   ├── autocracy_lookahead.py   autocracy + the ADS evaluator
│   ├── oligarchy.py
│   ├── federated.py
│   ├── ads.py                   evidence nodes, hypothesis generator, EvaluatorNode, AdsGovernment
│   ├── parameter_inference.py   ParameterEstimator (drain, regen, movement) and EnvironmentModel
│   └── forecast_ledger.py       ForecastLedger: predicted-vs-realized grading
│
├── agents/
│   └── citizen.py               CitizenAgent, used by every government
│
├── scenarios/
│   └── scenario_base.py         citizen population factory, difficulty-driven event schedule, named scenarios
│
└── Docs/                        this documentation (see index.md)
```

Running a sweep creates an output directory (`benchmark_results/` or `quick_test_results/` by default) and a log file in the working directory. Both are generated artifacts; see [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md).

---

## Dependency Direction

```
engine/difficulty.py            leaf: no intra-package imports
engine/grid.py, events.py       import difficulty
engine/agent.py, metrics.py
engine/simulation.py            imports grid, events, agent, metrics, audit, difficulty
engine/scenario_plan.py         imports grid, events, simulation
engine/run_recorder.py          imports metrics only (no numpy, no matplotlib)

governments/base.py
governments/parameter_inference.py   imports engine.grid, engine.events
governments/forecast_ledger.py       imports nothing from governments/
governments/ads.py                   imports base, parameter_inference, forecast_ledger
governments/autocracy_lookahead.py   imports autocracy, ads (EvaluatorNode), parameter_inference, forecast_ledger
other governments                    import base only

agents/citizen.py                    imports engine.agent
scenarios/scenario_base.py           imports engine, agents
benchmark_core.py                    imports everything above plus benchmark_logging
```

Rules the code follows:

- `engine/` does not import `governments/`, with one exception: `Simulation._do_move` checks for `FederatedGovernment` to apply its cross-region movement penalty.
- `parameter_inference` and `forecast_ledger` never import `ads`. This lets ADS and Autocracy+Lookahead share one estimator class, one ledger class and one evaluator without either government importing the other's internals. They share classes, never instances.
- `run_recorder` avoids numpy and matplotlib so it is safe inside worker processes.

---

## Core Objects

### SimulationConfig and the difficulty schedule

`SimulationConfig` (`engine/simulation.py`) is a plain dataclass holding every run parameter. The benchmark builds it with `SimulationConfig.from_difficulty(d, **overrides)`: grid size, agent count, cycle count, movement steps and seed come from the overrides; every difficulty-scaled field comes from the schedule in `engine/difficulty.py`. The schedule is a linear interpolation on `difficulty_t(d) = (effective_difficulty(d) − 1) / 99`, where `effective_difficulty` rises one step per level up to the knee at D90 and `DIFFICULTY_TAIL_SLOPE = 1.75` steps per level above it, so D95 and D100 extrapolate beyond the D1–D90 line (`difficulty_t(100) = 1.0758`). `validate_schedule()` runs at import and rejects any setting that would push `difficulty_t` above `DIFFICULTY_T_MAX = 1.25`. The emitted values are tabulated in [ENVIRONMENT.md](ENVIRONMENT.md#difficulty-schedule).

### Agents

```
Agent (engine/agent.py, abstract)
│   health, food_stock, water_stock, medicine_stock, hunger, thirst,
│   epidemic_ids, position, alive, age, memory
│   observe(grid, government, cycle) -> dict
│   act(obs, government, cycle) -> List[Action]
│
└── CitizenAgent (agents/citizen.py)
        self-interested survival heuristics; votes in democracy/republic elections;
        has its own seeded RNG used for tie-breaks
```

Every government, ADS included, governs the same `CitizenAgent` population. Governments differ only in how they create and enforce laws.

### Governments and laws

```
Government (governments/base.py, abstract)
│   active_laws: List[Law], rng
│   tick(cycle)                         abstract; called once per cycle after agents act
│   filter_actions(agent, actions, cycle)   injects law-directed actions
│   can_move(agent, r, c, cycle)            quarantine / shelter movement rules
│   food/water/medicine_collection_limit    ration caps
│   receive_vote, receive_proposal, receive_event_warnings
│   _enact_law(...), _expire_laws(cycle)
│   get_audit_info, get_decision_record, get_params, get_log_info
│
├── AnarchyGovernment
├── DemocracyGovernment
├── RepublicGovernment
├── AutocracyGovernment
│   └── AutocracyLookaheadGovernment
├── OligarchyGovernment
├── FederatedGovernment
└── AdsGovernment
```

A `Law` carries `law_id`, `law_type`, `params`, `enacted_cycle`, `duration`, `description`, `applies_to` (`None` means everyone), `event_id`, `event_type` and `source`. A law lapses when its duration elapses, when its linked event type is no longer active, or when no living agent still carries its linked epidemic; `_expire_laws` records which. Law types and their enforcement are listed in [GOVERNMENTS.md](GOVERNMENTS.md#law-types-and-enforcement).

Optional hooks, discovered by `getattr` so that governments without them are unaffected:

| Hook | Implemented by | Consumer |
|---|---|---|
| `get_decision_record(cycle)` | ADS, Autocracy+Lookahead | `decision` records in `run_detail.jsonl` |
| `get_calibration_record(cycle)` | ADS, Autocracy+Lookahead | `calibration` block of `cycle` records |
| `get_calibration_summary(max_cycles)` | ADS, Autocracy+Lookahead | forecast-accuracy fields in `final_stats.json` |
| `get_ads_metrics()` | ADS | `mean_prediction_error` in `health_stats.csv` and `final_mean_prediction_error` in `final_stats.json` |
| `set_eval_seed(seed)` | ADS, Autocracy+Lookahead | seeds the evaluator's rollout streams |
| `get_params()` | ADS, Autocracy+Lookahead | `government_params` in the run header |

### ScenarioPlan

`engine/scenario_plan.py` generates an immutable `ScenarioPlan` (grid cells, agent start positions, every scheduled and random event, and the config values needed to rebuild the simulation). `build_simulation_from_plan` reconstructs a `Simulation` from it with the random-event generator switched off, so every event comes from the plan. `ScenarioPlan.fingerprint()` is a 16-hex-digit SHA-256 of the plan's canonical JSON.

---

## One Simulation Cycle

`Simulation._step(cycle)` in `engine/simulation.py`:

```
1. Events       event_system.update(cycle, grid, agents)
                  - warnings for scheduled events inside the warning window
                  - start due events, apply active ones, retire expired ones
                government.receive_event_warnings(warnings, cycle)

2. Regenerate   grid.regenerate(drought_factor, base_regen_mult=config.regen_mult)
                  - apply pending global depletion, then per-cell regrowth up to terrain caps

3. Observe      for each living agent: agent.observe(...) plus warnings and active events

4. Act          for each living agent:
                  actions = agent.act(obs, government, cycle)
                  actions = government.filter_actions(agent, actions, cycle)
                  execute: up to max_steps_per_cycle MOVE actions (each checked by can_move),
                           every other action type at most once (collect within ration caps,
                           eat, drink, treat, law-directed sharing, rest, vote, propose)

5. Health       metabolic stock consumption, hunger/thirst build-up,
                drains (hunger, thirst, per-epidemic disease, cell hazard
                [terrain + ambient], unsheltered storm)
                all multiplied by drain_mult, contaminated-cell infection,
                infection recovery, death at health 0

6. Government   government.tick(cycle)

7. Metrics      metrics.record(...); audit.record(...) when the audit trail is enabled
```

Two observation channels are refreshed during steps 4–5 for governments that infer environment parameters: `Simulation.last_health_drain` (realized drain per agent this cycle) and `Simulation.last_move_counts` (completed moves per agent this cycle). `Grid.collected_cells_this_cycle` serves the same purpose for regrowth. Mechanics of the grid, agents and events are described in [ENVIRONMENT.md](ENVIRONMENT.md).

---

## The Benchmark Sweep

`run_full_simulation.py` and `run_quick_test.py` each build a `BenchmarkConfig` (a frozen, validated dataclass in `benchmark_core.py`) and hand it to `main_from_config`, which applies CLI overrides and calls `run_benchmark`. The two commands therefore execute identical code at different scales. `run_simulation.py` is independent of this path.

### Seed lattice

Every random stream is derived from one root, `base_seed` (42), by `derive_seed(base_seed, domain, *keys)`, a BLAKE2b hash truncated to 63 bits:

| Stream | Keys | Shared across governments? |
|---|---|---|
| `env` | difficulty, run | yes: grid, placements, random events |
| `env.events.sched` | difficulty, run, event index | yes: location of each scheduled event |
| `pop` | difficulty, run | yes: agent RNGs |
| `gov` | government, difficulty, run | no: the government's own decisions |
| `gov.eval` | government, difficulty, run | no: evaluator rollouts (ADS, Autocracy+Lookahead) |
| `plot.bootstrap` | government, difficulty | figure-layer bootstrap only |

All eight governments at a given (difficulty, run) therefore face the identical environment and starting population, and the runs within a cell face different environments. The design is a common-random-numbers comparison.

### Execution

```
run_benchmark(config)
  resolve log path (CWD), start the logging session, print the banner
  optional --clean: delete output_root
  warn if output_root already contains government results (never deletes)
  write manifest.json with status "running"

  for each difficulty:
      ProcessPoolExecutor(spawn), one worker per government by default
      worker: run_gov_difficulty(config, government, difficulty)
          for k in 1..k_runs:
              build_plan(config, difficulty, k)        plan rebuilt in the worker
              construct government (seed = gov stream), set_eval_seed if present
              build citizen population (seed = pop stream)
              simulate max_cycles cycles with a RunRecorder attached
              write health_stats.csv, final_stats.json (run_detail.jsonl via the recorder)
              a run that raises is logged, counted and skipped
          return summary rows + run-1 snapshot frames
      parent: render run-1 snapshots to visualizations/ (matplotlib only in the parent)
      log the per-difficulty ranking by mean normalized health score

  write combined_final_stats.csv
  verify_env_fingerprints: FAIRNESS (one fingerprint per (difficulty, run) across
      governments) and VARIATION (distinct fingerprints across runs); a violation
      aborts the sweep and records status "failed_environment_audit"
  compute and write cell_stats.json per (government, difficulty)
  write the figure families to plots/
  write analysis/
  rewrite manifest.json with the outcome; copy the log to output_root/sweep.log
```

The exit code is 0 only when every cell completed with its full `k_runs`; lost runs are recorded as `short_cells` in the manifest.

### Logging

Workers never write the log file. `benchmark_logging.py` gives each worker a `QueueHandler` feeding a `QueueListener` in the parent, which owns the single file handler and the console handler. Workers are started with the `spawn` method because a forked child can inherit a stream lock held by the listener thread and hang. Every record carries a context tag (`sweep`, `gov/D=d` or `gov/D=d/k=k`) so an interleaved eight-process log can be filtered per cell. The engine loggers default to WARNING (`--log-level`); the harness's progress logger is always INFO.

### Regenerating figures

`--plots-only --output-dir DIR` rebuilds every figure in `DIR/plots/` from `manifest.json`, `combined_final_stats.csv` and the per-run `final_stats.json` files without simulating. The sweep's scale is read from the manifest so that captions describe the archived data. `cell_stats.json` and `analysis/` are not touched.

---

## Tests

The seven root-level `test_*.py` files are standalone scripts (no pytest dependency). Each exits 0 on success:

```bash
python test_seed_derivation.py
python test_terrain_consistency.py
python test_default_governments.py
python test_median_ci.py
python test_difficulty_schedule.py
python test_plot_regression.py
python test_ads_calibration.py
```

`run_quick_test.py` is the end-to-end check that the full pipeline runs. See [RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md).

---

## Performance Notes

- Cost is dominated by the lookahead evaluator. Each ADS decision round scores every proposal for every group of six grouping schemes (typically 70–80 candidates at 500 agents), and each score is a 20-cycle rollout over the group's agents. ADS decides every 5 cycles, every 2 during an active event, and again whenever a crisis warning arrives.
- Autocracy+Lookahead scores at most four candidates per event type per round, over the whole population.
- One worker process per government means a difficulty level takes as long as its slowest government; ADS usually sets the pace.
- `record_audit_trail=False` in the benchmark skips the text audit trail, which would otherwise walk every agent every cycle.

---

## Extension Points

**Adding a government.** Subclass `Government`, implement `tick`, and register the class in `GOVERNMENT_REGISTRY` in `governments/__init__.py`. Registry entries are part of the default sweep unless listed in `ABLATION_GOVERNMENTS`; the import-time check that the default set has exactly eight members must be updated deliberately if the design changes. Add display name, colour, line style and marker to the `GOV_*` tables in `benchmark_core.py`. Implement `get_decision_record` / `get_params` for structured detail, and `get_calibration_record` / `get_calibration_summary` if the government opens forecasts.

**Adding a metric to the cross-government outputs.** Add it to `MetricsCollector.summary()` and to `SUMMARY_FIELDS` in `benchmark_core.py`; it then appears in `combined_final_stats.csv` and `cell_stats.json`. To plot it, add it to `PLOT_STATS`. Metrics that exist for only some governments belong in `final_stats.json` via `get_calibration_summary`, not in `SUMMARY_FIELDS`.

**Adding a named scenario.** Add an entry to `SCENARIOS` in `scenarios/scenario_base.py`; it becomes available to `run_simulation.py --scenario`. The benchmark sweep uses the difficulty-driven auto schedule, not named scenarios.

---

## See Also

- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md): output archive, file schemas, figures
- [GOVERNMENTS.md](GOVERNMENTS.md): the eight governments and the law system
- [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md): ADS internals, parameter inference, forecast grading
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md): parameters, difficulty schedule, metrics, statistics
- [ENVIRONMENT.md](ENVIRONMENT.md): grid, agents, events
