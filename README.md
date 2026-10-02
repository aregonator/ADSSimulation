[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23092993.svg)](https://doi.org/10.5281/zenodo.23092993)

# ADS Simulation

Simulation code accompanying a paper currently under review. It generates the
benchmark data and the figures reported in the paper's simulation evaluation,
which compares the Adaptive Decision System (ADS) with seven other forms of
government under increasing survival pressure.

## What the simulation models

- **World.** A square grid of cells with five terrain types (plains, forest,
  water, mountain, wasteland). Cells hold food, water and medicine, which
  regenerate each cycle and are depleted by harvesting.
- **Agents.** A population of self-interested survival agents. Every
  government runs on the same agent code: agents move, collect resources, eat,
  drink and treat infections. Health drains with hunger, thirst, disease,
  cell hazard (terrain hazard plus a uniform ambient hazard) and storms; an
  agent dies when its health reaches 0.
- **Events.** Drought, storm, epidemic, toxic spill and resource rush. Each run
  combines a schedule of event waves with randomly timed events, and scheduled
  events are announced in advance.
- **Governments.** Eight regimes that shape agent behaviour through laws:

  | Name | Decision mechanism |
  |---|---|
  | `anarchy` | No collective decisions |
  | `democracy` | Direct democracy: all agents vote on proposals |
  | `republic` | Citizens elect parties; a parliament passes laws |
  | `autocracy` | A single leader decides |
  | `autocracy_lookahead` | `autocracy` plus a 20-cycle forward projection of candidate laws |
  | `oligarchy` | A small elite rules in its own interest |
  | `federated` | The grid is split into regions, each of which votes on its own laws |
  | `ads` | Evidence nodes, risk-banded agent groups, and a 20-cycle lookahead evaluation of candidate laws |

  See [Docs/GOVERNMENTS.md](Docs/GOVERNMENTS.md) and
  [Docs/ADS_DEEP_DIVE.md](Docs/ADS_DEEP_DIVE.md).
- **Difficulty.** An integer from 1 to 100 that sets resource density,
  starting stocks, regeneration, metabolic cost, event frequency, severity and
  number of waves, warning lead time, the health-drain multiplier, and the
  ambient hazard. The schedule is linear from difficulty 1 to 90 and rises
  1.75 times as fast per level above 90 (`DIFFICULTY_KNEE_LEVEL = 90`,
  `DIFFICULTY_TAIL_SLOPE = 1.75` in `engine/difficulty.py`); the ambient
  hazard rises from 0 at difficulty 1 to 0.008 at difficulty 20 and is
  constant above. Grid size, population and run length do not
  depend on difficulty. See [Docs/ENVIRONMENT.md](Docs/ENVIRONMENT.md).
- **Primary outcome.** `normalized_health_score`: the total health of surviving
  agents divided by the initial population, from 0 to 1.

**Fair comparison.** Within a sweep, all eight governments at a given
(difficulty, run) face the same grid, starting positions, event schedule and
agent population; different runs face different environments. Every random
stream is derived from one base seed (42), and each run records an environment
fingerprint so this can be checked from the output alone.

## Requirements

- Python 3 with `venv` and `pip`
- `numpy >= 1.24` and `matplotlib >= 3.7` (installed from `requirements.txt`)
- Optional: `imageio` (plus `imageio[ffmpeg]` for MP4) for `run_simulation.py
  --save-video`; see the commented lines in `requirements.txt`. PIL/Pillow is
  already covered by matplotlib.

Tested with Python 3.12.3, numpy 2.5.3 and matplotlib 3.11.2 on Linux.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the virtual environment is active and are run from
the repository root.

## Quick start

```bash
python run_quick_test.py
```

This runs the complete benchmark pipeline at a tiny scale: all 8 governments,
difficulties 1, 50 and 100, 2 runs each (48 runs) on a 10×10 grid with 20
agents for 30 cycles. It finishes in one to a few minutes, writes its results to
`quick_test_results/` and a log file `sim_quick_<UTC timestamp>.log` to the
current directory. The numbers it produces only confirm that the pipeline
works; they are not an experiment.

## Reproducing the paper's results

```bash
python run_full_simulation.py --k-runs 100
```

The published design is 8 governments × 21 difficulty levels (1, 5, 10, …, 100)
× 100 runs = 16,800 runs, each on a 50×50 grid with 500 agents for 150 cycles.
`--k-runs` sets the number of runs per (government, difficulty) cell and is
required; without it the command prints the two options below and exits.

| Command | Runs | Approximate runtime |
|---|---|---|
| `python run_full_simulation.py --k-runs 10` | 1,680 | 4–5 hours |
| `python run_full_simulation.py --k-runs 100` | 16,800 | 47 hours (about 2 days) |

Runtimes were measured on a multi-core Linux server and depend on hardware. Difficulty levels run one after another; within a
level, each government runs in its own process, so the sweep uses up to 8 CPU
cores. To keep a long run going after the terminal closes:

```bash
nohup python run_full_simulation.py --k-runs 100 > full_sweep.log 2>&1 &
```

Results are written to `benchmark_results/`, and the progress log to
`sim_full_<UTC timestamp>.log` in the current directory.

The paper's figures come from these files:

| Paper figure | File |
|---|---|
| Normalized health score by difficulty | `benchmark_results/plots/normalized_health_score.png` |
| Final survival rate by difficulty | `benchmark_results/plots/final_survival_rate.png` |
| Final median health by difficulty | `benchmark_results/plots/final_median_health.png` |
| ADS grid at the final cycle, difficulty 75 | `benchmark_results/ads/75/visualizations/cycle_last.png` |

To redraw every figure from an existing results directory without running any
simulations (seconds, not hours):

```bash
python run_full_simulation.py --plots-only --output-dir benchmark_results
```

`--plots-only` reads the sweep's scale from the directory's `manifest.json`, so
it does not need `--k-runs`.

A subset of the sweep can be run with `--governments` and `--difficulties`, for
example:

```bash
python run_full_simulation.py --k-runs 10 --governments ads,autocracy,autocracy_lookahead --difficulties 50,75,100
```

All options are listed in
[Docs/RUNNING_SIMULATIONS.md](Docs/RUNNING_SIMULATIONS.md).

## Outputs

```
benchmark_results/
├── manifest.json               sweep configuration, scale and provenance
├── combined_final_stats.csv    one row per run: final metrics
├── <government>/<difficulty>/
│   ├── cell_stats.json         summary across the cell's runs
│   ├── run_01/ … run_<k>/      per-run statistics and structured trace
│   └── visualizations/         grid snapshots from the first run
├── plots/                      summary figures, each with an alt-text file
└── analysis/                   summary tables, regressions and pairwise comparisons
```

File formats and every column are documented in
[Docs/OUTPUT_STRUCTURE.md](Docs/OUTPUT_STRUCTURE.md); metric definitions are in
[Docs/METRICS_REFERENCE.md](Docs/METRICS_REFERENCE.md).

`manifest.json` records the invoking `user` and `host` alongside `argv` for
provenance. If you intend to publish a results archive alongside this code,
scrub or redact those two fields first.

## Single simulations

`run_simulation.py` runs one or more governments once, side by side, on a
shared environment, and prints a report for each. It supports named scenarios,
a JSON configuration file (`config.json` is an example) and optional
visualizations:

```bash
python run_simulation.py --governments ads,democracy --difficulty 50
python run_simulation.py --config config.json
python run_simulation.py --list-scenarios
```

It is intended for exploration; the paper's results come from
`run_full_simulation.py`. See
[Docs/RUNNING_SIMULATIONS.md](Docs/RUNNING_SIMULATIONS.md).

## Tests

The top-level tests are standalone scripts; pytest is not required. Each exits
with status 0 on success and 1 on failure, and each prints its own pass/fail
count instead of using a test framework's collection or reporting:

```bash
python test_default_governments.py   # the default sweep is the eight governments above
python test_seed_derivation.py       # seed derivation and environment fingerprints
python test_terrain_consistency.py   # each cell's hazard, shelter and regen match its terrain
python test_difficulty_schedule.py   # the difficulty schedule and its knee
python test_median_ci.py             # the bootstrap median confidence intervals
python test_plot_regression.py       # the figures and --plots-only
python test_ads_calibration.py       # the ADS parameter inference and forecast grading
```

### tests/ subtree

`tests/unit/` and `tests/system/` hold additional regression suites in the
same standalone style (no pytest, no fixtures or decorators — each file is a
script with its own `run_all()`/`main()` and prints a pass/fail count). Run
them from the repository root:

```bash
python tests/unit/test_scenarios.py          # grid/agent/government mechanics (54 checks)
python tests/unit/test_decision_making.py    # agent and government decision rationality (32 checks)
python tests/system/test_stage1_refactor.py  # production scale constants, BenchmarkConfig
                                              # validation, end-to-end pipeline smoke
python tests/system/test_stage2_logging_detail.py  # sweep logging and per-run JSONL detail
                                                    # output (142 checks)
python tests/system/test_stage3_ci_statistics.py   # per-cell CI statistics, including an
                                                    # integration run of run_quick_test.py
```

The `tests/system/` suites build and run real `BenchmarkConfig` sweeps with
multiprocessing worker fan-out (including a deadlock guard that deliberately
repeats a fan-out several times), so they take noticeably longer than the
top-level suites above or the `tests/unit/` suites, which each run in well
under a minute. Measured on this project's development machine:
`test_stage2_logging_detail.py` finishes in well under a minute;
`test_stage1_refactor.py` takes on the order of 10-20 minutes, dominated by
its repeated-fan-out deadlock guard (5 iterations at roughly 90-100s each);
`test_stage3_ci_statistics.py` takes on a similar order, dominated by several
full invocations of `run_quick_test.py` (each itself several minutes). Both
figures scale with filesystem and process-spawn overhead — they were
measured on a Windows-mounted drive under WSL2 and will likely be faster on
a native Linux filesystem.

Do not run these under pytest even if it is installed. Each `test_*` function
records its checks into a shared `TestResults` object (`results.ok(...)` /
`results.fail(...)`) instead of raising `AssertionError`, so pytest would
collect and execute every function successfully and report 100% passed
regardless of what `results.fail(...)` recorded internally. The pass/fail
count that matters is the one each script prints itself when run directly, as
shown above: `results.summary()` prints it and returns whether every check
passed, and the script's own `sys.exit(...)` turns that into the process exit
code (0 = all passed, 1 = at least one failure).

## Repository layout

```
.
├── README.md
├── requirements.txt
├── config.json                 example configuration for run_simulation.py
├── run_full_simulation.py      the paper's sweep
├── run_quick_test.py           small smoke test of the same pipeline
├── run_simulation.py           single simulations and scenarios
├── run_benchmark.py            deprecated forwarding shim to run_full_simulation.py
├── benchmark_core.py           sweep engine, output files, figures and analysis
├── benchmark_logging.py        logging across worker processes
├── benchmark_heuristics.py     standalone tuning tool, not part of the benchmark pipeline
├── engine/                     grid, agents, events, simulation loop, difficulty
│                               schedule, metrics, scenario plans, visualization
├── governments/                the eight governments, their shared base class,
│                               and the ADS parameter inference and forecast ledger
├── agents/                     the citizen agent used under every government
├── scenarios/                  named scenarios and the difficulty-driven event schedule
├── test_ads_calibration.py
├── test_default_governments.py
├── test_difficulty_schedule.py
├── test_median_ci.py
├── test_plot_regression.py
├── test_seed_derivation.py
├── test_terrain_consistency.py
├── tests/                      additional regression suites (see "tests/ subtree" above)
│   ├── unit/                   grid/agent/government and decision-rationality checks
│   └── system/                 end-to-end BenchmarkConfig sweeps, logging, CI statistics
└── Docs/                       detailed documentation
```

## Documentation

| Document | Contents |
|---|---|
| [Docs/index.md](Docs/index.md) | Overview of the documentation |
| [Docs/RUNNING_SIMULATIONS.md](Docs/RUNNING_SIMULATIONS.md) | All command-line options, scenarios, monitoring |
| [Docs/ENVIRONMENT.md](Docs/ENVIRONMENT.md) | Grid, agents, events and the difficulty schedule |
| [Docs/GOVERNMENTS.md](Docs/GOVERNMENTS.md) | The eight governments |
| [Docs/ADS_DEEP_DIVE.md](Docs/ADS_DEEP_DIVE.md) | The ADS decision cycle |
| [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) | Code structure and data flow |
| [Docs/METRICS_REFERENCE.md](Docs/METRICS_REFERENCE.md) | Parameter and metric definitions |
| [Docs/OUTPUT_STRUCTURE.md](Docs/OUTPUT_STRUCTURE.md) | Output files and their formats |
