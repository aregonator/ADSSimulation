# Running Simulations

This page covers the three entry points, every command-line option, the named
scenarios, and how to monitor a long sweep. Installation is described in the
[README](../README.md#installation); all commands assume the virtual
environment is active and are run from the repository root.

## Contents

1. [Entry points](#entry-points)
2. [The benchmark sweeps](#the-benchmark-sweeps)
3. [Sweep options](#sweep-options)
4. [Single simulations: run_simulation.py](#single-simulations-run_simulationpy)
5. [Scenarios](#scenarios)
6. [Monitoring a sweep](#monitoring-a-sweep)
7. [Developer tools](#developer-tools)
8. [Troubleshooting](#troubleshooting)

---

## Entry points

| Script | Purpose |
|---|---|
| `run_quick_test.py` | Smoke test of the full benchmark pipeline at a tiny scale |
| `run_full_simulation.py` | The paper's benchmark sweep |
| `run_simulation.py` | One or more governments run once, side by side, for exploration |

`run_quick_test.py` and `run_full_simulation.py` share all of their logic
(`benchmark_core.py`); they differ only in the scale they pass to it, so a
passing quick test exercises exactly the code the full sweep runs.

---

## The benchmark sweeps

A sweep runs every selected government at every selected difficulty level,
`k` times per (government, difficulty) cell. Run `k` at difficulty `d` uses an
environment (grid, starting positions, event schedule, agent population) that
is identical for all governments and different for every `k`.

| | `run_quick_test.py` | `run_full_simulation.py` |
|---|---|---|
| Governments | all 8 | all 8 |
| Difficulties | 1, 50, 100 | 1, 5, 10, …, 100 (21 levels) |
| Runs per cell (`k`) | 2 (default) | set by `--k-runs` (required); 100 in the paper |
| Grid | 10×10 | 50×50 |
| Agents | 20 | 500 |
| Cycles per run | 30 | 150 |
| Moves per agent per cycle (maximum) | 3 | 10 |
| Total runs | 48 | 16,800 at `--k-runs 100` |
| Output directory | `quick_test_results/` | `benchmark_results/` |
| Log file (current directory) | `sim_quick_<UTC>.log` | `sim_full_<UTC>.log` |
| Runtime | one to a few minutes | 4–5 hours at `--k-runs 10`; about 47 hours at `--k-runs 100` |

Grid size, population, cycles and the difficulty list are fixed by each entry
point; only the options below change what runs. Difficulty levels run one after
another, and within a level each government runs in its own worker process, so
a sweep uses up to 8 CPU cores. Runtimes depend on hardware.

### The full sweep

```bash
python run_full_simulation.py --k-runs 10     # validation pass, 1,680 runs
python run_full_simulation.py --k-runs 100    # the published design, 16,800 runs
```

Without `--k-runs` (and without `--plots-only`) the command prints both options
and exits with status 2 before touching the output directory.

For long runs, detach from the terminal:

```bash
nohup python run_full_simulation.py --k-runs 100 > full_sweep.log 2>&1 &
```

Nothing in the sweep prompts for input, so it runs unattended.

### Redrawing figures

```bash
python run_full_simulation.py --plots-only --output-dir benchmark_results
```

This regenerates every figure in `benchmark_results/plots/` from the data
already in the directory (`combined_final_stats.csv`, the per-run
`final_stats.json` files and `manifest.json`) and runs no simulations. The scale
shown in the figure captions is read from the directory's `manifest.json`;
`--k-runs` is not needed, and every option other than `--output-dir` is
ignored. `cell_stats.json` files and `analysis/` are left unchanged.

### Figure captions and `k`

Figure titles and the per-cell `n` reflect the `k` that actually ran, so a
`--k-runs 10` figure says so. The median/confidence-interval figures'
text sidecars also report a projection to the published design (`k = 100`),
which stays fixed regardless of `--k-runs`: its purpose is to show what a
full-scale figure will look like from a smaller pass. `manifest.json` records
both values (`config.k_runs` and `config.paper_k_runs`) and whether `k` was
given on the command line (`config.k_runs_source`).

---

## Sweep options

Both `run_quick_test.py` and `run_full_simulation.py` accept these options.

| Option | Argument | Default | Effect |
|---|---|---|---|
| `--k-runs` | integer ≥ 1 | full sweep: none (required); quick test: 2 | Runs per (government, difficulty) cell. Runtime is linear in this value. |
| `--governments` | comma-separated names | all 8 | Run only these governments. Names are case-insensitive; valid names are `anarchy`, `democracy`, `republic`, `autocracy`, `autocracy_lookahead`, `oligarchy`, `federated`, `ads`. |
| `--difficulties` | comma-separated integers, 1–100 | the entry point's list | Run only these difficulty levels, e.g. `--difficulties 56,76`. |
| `--output-dir` | path | `benchmark_results/` or `quick_test_results/` | Write results here. |
| `--plots-only` | — | off | Redraw all figures from an existing output directory and exit (see above). |
| `--clean` | — | off | Delete the output directory before starting. Destructive, with no confirmation. Without it, results merge into the existing directory and a warning is logged if earlier results are found. |
| `--log-file` | path | `sim_full_<UTC>.log` or `sim_quick_<UTC>.log` in the current directory | Write the sweep log here. A directory is accepted; the generated file name is then placed inside it. A log is always written. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `WARNING` | Verbosity of the simulation engine's loggers. The sweep's own progress lines are always logged. `DEBUG` on a large sweep produces tens of gigabytes. |
| `--heartbeat-every` | integer | 25 (full) / 10 (quick) | Cycles between per-run progress lines in the log. |
| `--no-run-detail` | — | off | Do not write the per-run `run_detail.jsonl` trace. |
| `--no-decision-detail` | — | off | Omit the per-group `groupings` array from decision records in `run_detail.jsonl` (about 95% of its size). Evidence, winners, enacted laws and rejection counts are still written. |
| `--detailed-agents` | optional integer `N` | off; `N` = 10 if given without a value | Also write `agent_states.jsonl` every `N` cycles. Refused for sweeps of more than 20 runs in total; narrow the sweep with `--governments` and `--difficulties` first. |

Examples:

```bash
# Three governments at three difficulties, 10 runs each (90 runs)
python run_full_simulation.py --k-runs 10 \
  --governments ads,autocracy,autocracy_lookahead \
  --difficulties 50,75,100 \
  --output-dir results_subset

# Per-agent snapshots for one cell (ADS at difficulty 75, 5 runs)
python run_full_simulation.py --k-runs 5 --governments ads --difficulties 75 \
  --detailed-agents 5 --output-dir results_ads_d75
```

The files a sweep writes are documented in
[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md).

---

## Single simulations: run_simulation.py

`run_simulation.py` builds one environment and runs each requested government
on it once, in parallel threads, then prints a report per government and a
comparison table. It is meant for exploring the model; the paper's results come
from `run_full_simulation.py`.

```bash
python run_simulation.py                                   # anarchy, democracy, ads at difficulty 25
python run_simulation.py --governments all --difficulty 50
python run_simulation.py --government ads --scenario epidemic --verbose
python run_simulation.py --governments ads,democracy --save-viz viz/
python run_simulation.py --config config.json
```

Without `--scenario`, the run uses the same difficulty-driven environment as
the benchmark (resources, events and drain all scaled from `--difficulty`) on a
20×20 grid with 50 agents for 200 cycles, unless `--grid-size`, `--agents` or
`--cycles` say otherwise. Agents may move at most once per cycle unless
`--max-steps` is given.

| Option | Default | Effect |
|---|---|---|
| `--government`, `-g` NAME | — | Run a single government. |
| `--governments` LIST | `anarchy,democracy,ads` | Comma-separated governments, or `all` for all eight. |
| `--difficulty`, `-d` N | 25 | Difficulty, 1–100 (values outside the range are clamped). |
| `--scenario`, `-s` NAME | none | Run a named scenario instead of the difficulty-driven environment (see [Scenarios](#scenarios)). |
| `--grid-size` N | 20 | Use an N×N grid. |
| `--agents`, `-n` N | 50 | Number of agents. |
| `--cycles`, `-c` N | 200 | Number of cycles. |
| `--max-steps` K | 1 | Maximum moves per agent per cycle. |
| `--event-severity` X | from difficulty or scenario | Override the event severity multiplier. |
| `--seed` N | 42 | Random seed. |
| `--config` PATH | — | Read options from a JSON file (see below). |
| `--save-plan` PATH | — | Save the generated environment (grid, placements, events) as JSON, then run with it. |
| `--plan-file` PATH | — | Run with an environment saved by `--save-plan`. |
| `--generate-only` | off | Print the grid map and event plan without simulating. |
| `--verbose`, `-v` | off | Print progress every 10 cycles. |
| `--output-dir`, `-o` DIR | — | Write per-government metrics (`<government>.csv`) and a readable audit trail (`<government>_trail.log`). |
| `--log-level` | `NONE` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `NONE`. Writes `<government>_sim.log` to `--output-dir` (only when `--output-dir` is given). |
| `--visualize` | off | Show a dashboard for each government after its run. |
| `--save-viz` DIR | — | Save each government's dashboard, plus a comparison chart when several governments run. |
| `--save-video` DIR | — | Save an animation of each run (MP4 if `ffmpeg` is available, otherwise GIF). |
| `--video-fps` N | 5 | Frames per second for `--save-video`. |
| `--list-scenarios` | — | List the named scenarios and exit. |
| `--list-governments` | — | List the government names and exit. |

### Configuration files

`--config PATH` reads options from a JSON file. Keys may be command-line
option names (`governments`, `difficulty`, `seed`, `scenario`, `cycles`,
`grid_size`, `agents`, `max_steps`, `event_severity`, `output_dir`, …) or
simulation parameters (`grid_rows`, `grid_cols`, `resource_density`,
`event_frequency`, `event_warning_cycles`, `visibility_radius`, `max_cycles`,
…). Unknown keys are ignored. An option given explicitly on the command line
takes precedence over the same key in the file.

`config.json` in the repository root is a small example: three governments
(anarchy, democracy, ADS) at difficulty 25 on a 17×17 grid for 150 cycles. It is
not the benchmark configuration. It pins four difficulty-scaled fields
explicitly — `resource_density`, `event_frequency`, `event_severity`, and
`event_warning_cycles` — at the values shown in the file. Because an explicit
`--config` value always overrides the difficulty-derived default for the same
field, those four values apply exactly as written regardless of `difficulty`
in the file or any `--difficulty` flag on the command line; every other
difficulty-scaled field (drain, regeneration, metabolism, and the rest that
`SimulationConfig.from_difficulty` derives) still scales normally with
whatever difficulty is in effect. See the file's own header comment for the
full explanation.

```bash
python run_simulation.py --config config.json
python run_simulation.py --config config.json --difficulty 60
```

### `--config` file precedence

The rule is the same for every field, with no exceptions by flag or by
mode: **an explicit CLI flag always wins over the same key in the file, and
a file value always wins over the built-in default.** This holds even when
the explicit CLI value happens to equal the documented default — for
example `--config config.json --difficulty 25` runs at difficulty 25 even
if the file specifies a different difficulty, because `--difficulty` was
passed explicitly. Internally this works because `--difficulty`, `--seed`
and every other overridable flag default to a sentinel (not-passed) value
in argparse, so "the user didn't pass this flag" and "the user passed the
flag's own default value" are never confused with each other.

The auto-mode path (no `--scenario`) and `--scenario` runs apply overrides
in the same order: the difficulty-based (or scenario) baseline first, then
`--config` file values, then CLI-explicit flags last. `--generate-only`
(and `--save-plan` on its own) builds through the same code path as a real
run, so it always previews the exact environment — grid size, agent count,
every difficulty-scaled parameter, and the resulting plan — that the real
run would use for the same arguments.

```bash
# num_agents from the file is used (75, not the default 50)
python run_simulation.py --config config.json --government anarchy --cycles 5

# --agents on the command line still wins over the file's num_agents
python run_simulation.py --config config.json --government anarchy --agents 20

# --cycles on the command line wins over the file's max_cycles (150)
python run_simulation.py --config config.json --government anarchy --cycles 5 \
  --output-dir /tmp/check   # writes 5 rows per government's CSV, not 150

# --generate-only previews exactly what the equivalent real run would build
python run_simulation.py --config config.json --difficulty 60 --generate-only
python run_simulation.py --config config.json --difficulty 60 --government anarchy
```

---

## Scenarios

A scenario replaces the difficulty-driven environment with a fixed set of
parameters and a fixed list of scheduled events; random events still occur at
the scenario's event frequency. Scenarios are used only by
`run_simulation.py`; the benchmark sweeps do not use them.

| Scenario | Agents | Scheduled events (cycle: type) | Purpose |
|---|---|---|---|
| `basic_survival` | 80 | none | Steady resource depletion with light random events |
| `epidemic` | 100 | 15: epidemic | Quarantine and medicine coordination |
| `resource_scarcity` | 100 | 5: drought; 40: drought | Rationing and redistribution under scarcity |
| `climate_crisis` | 120 | 10: drought; 50: storm; 80: toxic spill; 110: epidemic | An escalating sequence of different threats |
| `ads_validation` | 100 | 20: epidemic; 80: toxic spill; 140: drought and epidemic; 220: storm | Exercises each stage of the ADS decision cycle |

Scenarios run on a 20×20 grid for 200 cycles unless `--grid-size` or `--cycles`
is given; `ads_validation`'s storm at cycle 220 occurs only with
`--cycles` above 220. The `--difficulty` value sets the health-drain multiplier
and related difficulty effects. Scenario definitions are in
`scenarios/scenario_base.py`.

```bash
python run_simulation.py --scenario climate_crisis --governments ads,anarchy --difficulty 75
python run_simulation.py --scenario ads_validation --government ads --cycles 300 --verbose
```

---

## Monitoring a sweep

The sweep log opens with a banner that records the configuration, the log file
path and the software versions, then prints progress for each difficulty level
and each run:

```
2026-09-27T02:13:58.319Z  INFO    [sweep             ]   [D=  1] Launching 8 worker process(es) — each runs 2 simulations on its own CPU core, deriving its own per-run scenario plan.
2026-09-27T02:14:17.012Z  INFO    [anarchy/D=1       ]     ⟶  anarchy      D=  1  starting 2 runs …
2026-09-27T02:14:17.203Z  INFO    [anarchy/D=1/k=1   ] cycle  11/30 alive=20/20 nhs=1.0000 health=1.000 events=[] laws=0 decisions=-
```

In a heartbeat line, `alive` is living agents over the initial population,
`nhs` is the normalized health score so far, `health` is the mean health of
living agents, `events` lists the active event types, and `laws` and
`decisions` are the government's active laws and decision rounds so far.

Useful commands while a sweep runs:

```bash
tail -f sim_full_*.log                                   # follow progress
find benchmark_results -name final_stats.json | wc -l    # completed runs
grep "RUN FAILED" sim_full_*.log                         # failed runs, if any
```

`manifest.json` in the output directory is written with `"status": "running"`
when the sweep starts and rewritten when it ends: `"complete"` if no runs were
lost, `"complete_with_losses"` otherwise. It also records the expected and
recorded run counts under `results` and the full configuration under `config`.

---

## Developer tools

**`run_benchmark.py`** is a deprecated forwarding shim: it takes the same
arguments and produces the same output as `run_full_simulation.py`, which is
where the sweep logic actually lives. Use `run_full_simulation.py` directly;
`run_benchmark.py` prints a deprecation notice and forwards to it.

**`benchmark_heuristics.py`** is a small, undocumented-by-design tuning tool
for exploring heuristic changes across a coarse grid of governments,
difficulties and seeds — not the published methodology, and not part of the
benchmark pipeline. Run it directly (`python benchmark_heuristics.py`);
`--save-baseline` writes the current results to `benchmark_baseline.json` in
the repository root, and a later run without `--no-compare` reads that file
back to print a delta column against it.

---

## Troubleshooting

**A run failed.** The log contains a `RUN FAILED` line naming the government,
difficulty, run and exception, followed by the path of that run's
`run_detail.jsonl`, whose final record holds the traceback. The rest of the
sweep continues, and the lost run is counted in `manifest.json`.

**The full sweep takes too long.** Start with `--k-runs 10`, or narrow it with
`--governments` and `--difficulties`. Run `run_quick_test.py` first to confirm
that the pipeline works.

**The output directory is too large.** `--no-decision-detail` removes most of
the size of `run_detail.jsonl`, and `--no-run-detail` omits it entirely. Do not
use `--log-level DEBUG` or `--detailed-agents` on a large sweep. Size estimates
are in [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md).

---

## See also

- [ENVIRONMENT.md](ENVIRONMENT.md) — what the simulation models
- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) — the files a sweep writes
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) — parameter and metric definitions
- [GOVERNMENTS.md](GOVERNMENTS.md) — the eight governments
