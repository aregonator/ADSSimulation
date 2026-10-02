# Documentation

Reference documentation for the ADS simulation. Start with the
[README](../README.md) for installation, the smoke test and the command that
reproduces the paper's results.

## Documents

| Document | Read it to… |
|---|---|
| [RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md) | run the sweeps and single simulations: every command-line option, the named scenarios, configuration files, monitoring |
| [ENVIRONMENT.md](ENVIRONMENT.md) | understand the simulated world: grid, resources, agents, events and the difficulty schedule |
| [GOVERNMENTS.md](GOVERNMENTS.md) | compare the eight governments and how each makes decisions |
| [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md) | follow the ADS decision cycle in detail |
| [ARCHITECTURE.md](ARCHITECTURE.md) | find your way around the code: modules, data flow, the sweep engine |
| [METRICS_REFERENCE.md](METRICS_REFERENCE.md) | look up a parameter or outcome metric |
| [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) | interpret the files a sweep writes |

## At a glance

- **Governments:** `anarchy`, `democracy`, `republic`, `autocracy`,
  `autocracy_lookahead`, `oligarchy`, `federated`, `ads`.
- **Difficulty:** an integer from 1 to 100; the benchmark sweeps 21 levels
  (1, 5, 10, …, 100).
- **Published design:** 8 governments × 21 difficulties × 100 runs = 16,800
  runs, each on a 50×50 grid with 500 agents for 150 cycles
  (`python run_full_simulation.py --k-runs 100`).
- **Primary metric:** `normalized_health_score`, the total health of surviving
  agents divided by the initial population.

## Common questions

| Question | Where to look |
|---|---|
| Does my installation work? | [README: Quick start](../README.md#quick-start) |
| How long does the full sweep take? | [RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md#the-benchmark-sweeps) |
| How do I redraw the figures without re-running? | [RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md#redrawing-figures) |
| What does difficulty change? | [ENVIRONMENT.md](ENVIRONMENT.md#difficulty-schedule) |
| What does `normalized_health_score` measure? | [METRICS_REFERENCE.md](METRICS_REFERENCE.md) |
| What is in `manifest.json` or `run_detail.jsonl`? | [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) |
| How are the eight governments different? | [GOVERNMENTS.md](GOVERNMENTS.md) |
| How does the ADS choose its laws? | [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md) |
