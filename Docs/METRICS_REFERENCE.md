# Parameters and Metrics Reference

Run parameters, the difficulty schedule, government constants, the outcome metrics, and the statistics computed over them.

---

## Contents

1. [Sweep Parameters](#sweep-parameters)
2. [Simulation Parameters](#simulation-parameters)
3. [Difficulty Schedule](#difficulty-schedule)
4. [Government Constants](#government-constants)
5. [Agent State](#agent-state)
6. [Outcome Metrics](#outcome-metrics)
7. [Per-Cycle Metrics](#per-cycle-metrics)
8. [Governance and Forecast Metrics](#governance-and-forecast-metrics)
9. [Statistics](#statistics)

---

## Sweep Parameters

Set by the entry points in `BenchmarkConfig` (`benchmark_core.py`).

| Parameter | `run_full_simulation.py` | `run_quick_test.py` | CLI override |
|---|---|---|---|
| governments | all 8 | all 8 | `--governments` |
| difficulties | 1, 5, 10, …, 100 (21 levels) | 1, 50, 100 | `--difficulties` |
| runs per cell (`k_runs`) | required; the published design is 100 | 2 | `--k-runs` |
| grid | 50 × 50 | 10 × 10 | none |
| agents | 500 | 20 | none |
| cycles | 150 | 30 | none |
| max moves per agent per cycle | 10 | 3 | none |
| base seed | 42 | 42 | none |
| heartbeat | every 25 cycles | every 10 cycles | `--heartbeat-every` |
| output directory | `benchmark_results/` | `quick_test_results/` | `--output-dir` |

At the published scale the sweep is 8 × 21 × 100 = 16,800 runs.

---

## Simulation Parameters

`SimulationConfig` (`engine/simulation.py`). In the benchmark every run is built with `SimulationConfig.from_difficulty(difficulty, ...)`: the grid size, agent count, cycle count, movement limit and seed come from the sweep; the difficulty-scaled fields come from the schedule below; the remaining fields are constant.

| Field | Benchmark value | Meaning |
|---|---|---|
| `grid_rows`, `grid_cols` | sweep grid size | grid dimensions |
| `num_agents` | sweep agent count | initial population |
| `max_cycles` | sweep cycle count | run length |
| `max_steps_per_cycle` | sweep movement limit | MOVE actions an agent may execute per cycle |
| `difficulty` | 1–100 | selects the schedule values |
| `seed` | derived per (difficulty, run) | environment seed |
| `terrain_variety` | 1.0 | full terrain mix at every difficulty |
| `initial_health` | 1.0 | starting health of every agent |
| `visibility_radius` | 5 | how far agents see (Euclidean radius, in cells) |
| `record_audit_trail` | false | text audit trail (used by `run_simulation.py`) |
| difficulty-scaled fields | see below | `resource_density`, `initial_food_stock`, `initial_water_stock`, `event_frequency`, `event_severity`, `event_warning_cycles`, `regen_mult`, `metabolic_rate`, `ambient_hazard` |

The dataclass defaults (20 × 20 grid, 50 agents, 200 cycles, one move per cycle) apply only when `SimulationConfig` is constructed directly.

---

## Difficulty Schedule

Difficulty (1–100) is the single environmental variable of the experiment: it
scales the drain multiplier, initial resources, regeneration, metabolic rate,
event frequency and severity, event warnings, scheduled event waves, infection
recovery, epidemic seeding and ambient hazard. `engine/difficulty.py` is the
single source of truth for the knee, the tail slope and every one of these
formulas; `SimulationConfig.from_difficulty` (`engine/simulation.py`) and
`scenarios/scenario_base.py` apply them when a run is built. The full set of
formulas, and their values at representative difficulties, are in
[ENVIRONMENT.md](ENVIRONMENT.md#difficulty-schedule). Event generation
mechanics (scheduled waves and random events) are also there, in
[ENVIRONMENT.md](ENVIRONMENT.md#events).

---

## Government Constants

| Government | Constant | Value |
|---|---|---|
| Democracy | ballot interval (`t_vote`) | 5 cycles, tallied the following cycle |
| Democracy | options per ballot | 3 |
| Democracy | duration of non-event laws (`t_law`) | 20 cycles |
| Republic | party election interval (`T_ELECTION`) | 5 cycles |
| Republic | parliament session interval (`T_SESSION`) | 2 cycles |
| Republic | parties / seats | 6 / 30 |
| Republic | duration of non-event laws (`T_LAW`) | 25 cycles |
| Autocracy | loyalists (`LOYALIST_FRACTION`) | 1% of the living population, at least 1 |
| Autocracy | loyalists' target share of food and water | 30% |
| Autocracy | leader's target stock | 12 units |
| Autocracy+Lookahead | evaluated-round interval | 2 cycles while a drought, storm or epidemic is active or warned |
| Autocracy+Lookahead | drought ration menu | 1.5, 2.5, 4.0, none |
| Oligarchy | elite / loyalists | 1% / 2% of the living population |
| Oligarchy | target share of food and water | 30% each |
| Federated | regions | `min(10, n_agents // 10 + 1)` horizontal strips |
| Federated | regional vote interval | 5 cycles |
| Federated | cross-region movement penalty | 0.01 health |
| ADS | decision interval (`t_decision`) | 5 cycles, 2 while an event is active |
| ADS | lookahead horizon (`LOOKAHEAD_CYCLES`) | 20 cycles |
| ADS | grouping schemes | 10, 5, 4, 3, 2, 1 groups |
| ADS | minimum ranked group size | 10% of the living population |
| ADS | laws per decision round | at most `max(3, alive // 20)` |

The ADS and Autocracy+Lookahead also share the evaluator and estimator constants listed in [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#key-parameters). Behaviour of each government is described in [GOVERNMENTS.md](GOVERNMENTS.md).

---

## Agent State

| Field | Initial value | Meaning |
|---|---|---|
| `health` | 1.0 | 0–1; the agent dies at 0 |
| `food_stock`, `water_stock` | `initial_food_stock` / `initial_water_stock` (difficulty-scaled) | carried resources; no cap |
| `medicine_stock` | 2.0 | carried medicine; curing costs 3 per epidemic carried |
| `hunger`, `thirst` | 0.0 | 0–1; rise while the stock is empty (≤ 0.1), fall otherwise; drain health above 0.4 |
| `epidemic_ids` | empty | epidemics currently carried; an agent is infected if non-empty |
| `age` | 0 | cycles survived |

Starting positions are clustered and come from the scenario plan. Grid, terrain and resource details are in [ENVIRONMENT.md](ENVIRONMENT.md).

---

## Outcome Metrics

Computed by `MetricsCollector.summary()` (`engine/metrics.py`) from the per-cycle snapshots and written to `final_stats.json`. The first six are `SUMMARY_FIELDS`: they appear in `combined_final_stats.csv`, `cell_stats.json` and the `run_detail.jsonl` footer.

| Metric | Definition | Range |
|---|---|---|
| `normalized_health_score` | sum of living agents' health at the last cycle / initial population. **The primary metric**: it combines survival and the condition of the survivors | 0–1, higher is better |
| `final_survival_rate` | living agents at the last cycle / initial population | 0–1 |
| `final_median_health` | median health of living agents at the last cycle (0 if none survive) | 0–1 |
| `final_health_gini` | Gini coefficient of living agents' health at the last cycle (0 with fewer than two survivors) | 0–1, lower is more equal |
| `min_survival_rate` | lowest survival rate over the run; deaths are permanent, so this equals `final_survival_rate` | 0–1 |
| `time_to_50pct_loss` | first cycle (0-based) at which survival is at or below 50%; `null` if it never happens | 0 to `max_cycles − 1` |
| `final_median_food`, `final_median_water` | median stock of living agents at the last cycle | ≥ 0 |
| `total_cycles` | index of the last cycle | |

Because the survivors' health enters `normalized_health_score` directly, two governments with the same survival rate can differ on it; and because dead agents stay in the denominator, it can never exceed the survival rate. `final_median_health` and `final_health_gini` describe survivors only, so they must be read together with survival: a government whose weakest agents die can show high median health and low inequality.

The four metrics plotted are `normalized_health_score`, `final_survival_rate`, `final_median_health` and `final_health_gini` (`PLOT_STATS` in `benchmark_core.py`).

---

## Per-Cycle Metrics

`health_stats.csv` (one row per cycle) records `normalized_health_score`, `survival_rate`, `median_health`, `health_gini`, `median_food`, `median_water`, `num_alive` and active event types, plus `mean_prediction_error` for the ADS. The `cycle` records of `run_detail.jsonl` carry a fuller picture: health distribution (mean, median, quartiles, min, max, standard deviation, Gini), stock distributions, counts of infected, hungry, thirsty, starving and dehydrated agents, grid totals, hazard coverage, active events, and law changes. See [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md#cycle).

---

## Governance and Forecast Metrics

**Run totals** (`run_detail.jsonl` footer, all governments):

| Metric | Meaning |
|---|---|
| `deaths` | initial population − survivors |
| `laws_enacted`, `laws_expired` | laws created and ended during the run |
| `decision_rounds` | cycles with a decision record (ADS and Autocracy+Lookahead) |
| `proposals_evaluated` | ADS candidates scored over the run |
| `events_triggered`, `events_expired` | events started and ended |

**Per-cycle government audit** (`government.audit` in each `cycle` record): vote tallies and winners (democracy, republic, federated), leader and loyalist resource shares (autocracy, Autocracy+Lookahead), elite and loyalist shares (oligarchy), and for the ADS the running decision and law counts, candidates scored in the last round, the top raw score, and `evidence_scores` (population mean food stock, mean water stock and mean health at the last round).

**Forecast accuracy** (ADS and Autocracy+Lookahead): the absolute difference between a forecast's projected mean health per agent for a group and that group's realized mean health 20 cycles later. Reported per cycle as a running mean, per run as an all-time mean, a first/second-half split and the full list of closures. Field definitions are in [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md#final_statsjson) and the mechanism in [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#forecast-grading).

**Parameter estimates** (ADS and Autocracy+Lookahead): the inferred `drain_mult`, `regen_mult` and movement cap each cycle. They estimate what the evaluator's model needs, not the configuration values; see [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#parameter-inference).

---

## Statistics

The unit of observation is one run. Each run within a cell faces a different environment, and all governments at the same (difficulty, run) face the same environment, so runs are paired across governments. Dispersion across runs therefore describes how a government's outcome varies across environments at that difficulty.

| Statistic | Where | Method |
|---|---|---|
| Per-cell mean and 95% CI | `cell_stats.json` | t-interval on the mean over non-missing values; Student-t critical value from a lookup table (linear interpolation between tabulated entries) for degrees of freedom below 30, and the normal approximation 1.96 for 30 or more degrees of freedom |
| Box plots | `<metric>.png` | Tukey: median, interquartile box, whiskers at 1.5 × IQR |
| Median and 95% CI | `<metric>_median_ci.png` | percentile bootstrap of the median, 10,000 resamples, deterministic seed per (government, difficulty); no interval for cells with fewer than 6 runs or no variation |
| Pooled government summary | `analysis/per_government_summary.csv` | mean ± 1.96 × SEM over all runs and difficulties |
| Difficulty trend | `analysis/regression_analysis.csv` | least-squares slope of the metric on difficulty per government |
| Pairwise differences | `analysis/pairwise_comparisons*.csv` | Welch-style t-test (normal-approximation p-value), pooled and per difficulty; per difficulty also a 10,000-resample percentile bootstrap CI of the mean difference |
| Forecast error trend | `calibration_trend_by_cycle.png` | per-cycle mean ± 1.96 × SEM, least-squares trend after an estimated burn-in breakpoint |

The pairwise tests are unpaired, which overstates the standard error of a between-government difference; a paired analysis can be built by joining runs on `difficulty`, `run` and `env_fingerprint` in `combined_final_stats.csv`. A comparison involving a cell with no run-to-run variation is reported as `Undefined`. Full descriptions are in [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md#analysis).

The per-cell mean CI's critical value is Student-t for degrees of freedom (run count minus one) below 30, taken from a lookup table with linear interpolation between tabulated entries (within about 0.15% of the exact value), and the normal value 1.96 once degrees of freedom reach 30 or more. At the paper's production sample size (`k_runs` = 100 runs per cell, 99 degrees of freedom), the exact Student-t critical value is approximately 1.984, so the reported bands are approximately 1.2% narrower than an exact Student-t interval would give at that sample size. The gap is largest just below the threshold (about 4% narrower at 30 degrees of freedom) and shrinks as degrees of freedom grow past 100.

---

## See Also

- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md): files, fields and figures
- [GOVERNMENTS.md](GOVERNMENTS.md): government behaviour
- [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md): ADS internals, parameter inference, forecast grading
- [ENVIRONMENT.md](ENVIRONMENT.md): grid, agents, events
