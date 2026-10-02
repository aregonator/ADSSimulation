# Output Structure Reference

What a sweep (`run_full_simulation.py` or `run_quick_test.py`) writes: directory layout, every file, its fields, and every figure. `run_simulation.py` has its own, simpler outputs, described in [RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md).

---

## Contents

1. [Directory Layout](#directory-layout)
2. [Sweep Log](#sweep-log)
3. [manifest.json](#manifestjson)
4. [combined_final_stats.csv](#combined_final_statscsv)
5. [Per-Run Files](#per-run-files)
6. [run_detail.jsonl](#run_detailjsonl)
7. [agent_states.jsonl](#agent_statesjsonl)
8. [cell_stats.json](#cell_statsjson)
9. [analysis/](#analysis)
10. [Figures](#figures)
11. [Serialization Rules](#serialization-rules)
12. [Sizes](#sizes)
13. [Troubleshooting](#troubleshooting)

---

## Directory Layout

```
<working directory>/
└── sim_full_<UTC>.log                  sweep log (sim_quick_<UTC>.log for the quick test)

<output_root>/                          benchmark_results/ or quick_test_results/ by default (--output-dir)
├── manifest.json                       provenance, configuration, outcome
├── sweep.log                           copy of the sweep log, made when the sweep ends
├── combined_final_stats.csv            one row per run, all governments and difficulties
├── plots/                              figures, each with a <name>_alt_text.txt sidecar
├── analysis/                           post-hoc statistical tables
│   ├── analysis_manifest.json
│   ├── per_government_summary.csv
│   ├── per_difficulty_summary.csv
│   ├── regression_analysis.csv
│   ├── pairwise_comparisons.csv
│   ├── pairwise_comparisons_by_difficulty.csv
│   ├── correlation_matrix.csv
│   └── per_metric_analysis/
│       └── <metric>_analysis.csv       one per plotted metric
│
└── <government>/                       anarchy, democracy, republic, autocracy,
    │                                   autocracy_lookahead, oligarchy, federated, ads
    └── <difficulty>/                   1, 5, 10, ..., 100
        ├── cell_stats.json             per-cell summary statistics
        ├── visualizations/             snapshots from run 1 only
        │   ├── cycle_001.png
        │   ├── cycle_050.png
        │   ├── cycle_100.png
        │   └── cycle_last.png
        └── run_01/ ... run_<k>/        one directory per run (two-digit, zero-padded)
            ├── health_stats.csv
            ├── final_stats.json
            ├── run_detail.jsonl        unless --no-run-detail
            └── agent_states.jsonl      only with --detailed-agents
```

Snapshot cycles: for runs longer than 100 cycles, cycles 1, 50, 100 and the last; for shorter runs, three evenly spaced cycles plus the last. The file stems are `cycle_NNN` (1-based) and `cycle_last`.

Results are merged into an existing `output_root`. A startup warning is logged if it already contains government directories; `--clean` deletes the directory first.

---

## Sweep Log

Written to the working directory at invocation (not to `output_root`), named `<prefix>_<UTC timestamp>.log`, where the prefix is `sim_full` or `sim_quick`. `--log-file PATH` overrides the location; if `PATH` is an existing directory the generated name is placed inside it. A copy is saved as `<output_root>/sweep.log` when the sweep ends, including when it aborts.

Plain text, one record per line:

```
2026-09-26T05:19:38.261Z  INFO    [anarchy/D=1       ]     ⟶  anarchy      D=  1  starting 10 runs …
2026-09-26T05:19:38.268Z  INFO    [anarchy/D=1/k=1   ] run 1/10 starting  env_seed=75004891545324820  gov_seed=4006368467652316088  dir=…/benchmark_results/anarchy/1/run_01
2026-09-26T08:34:19.552Z  INFO    [ads/D=75/k=1      ] cycle  26/150 alive=500/500 nhs=0.9842 health=0.984 events=[drought] laws=7 decisions=17
2026-09-26T08:45:29.206Z  INFO    [sweep             ]   D=75 mean NHS: ads 0.858 │ federated 0.553 │ anarchy 0.378 │ …
```

- The bracketed context is `sweep`, `<government>/D=<difficulty>` or `<government>/D=<difficulty>/k=<run>`, so `grep '\[ads/D=75'` extracts one cell.
- Heartbeat lines appear every `heartbeat_every` cycles (25 in the full sweep, 10 in the quick test) and on the last cycle.
- A failed run logs `RUN FAILED <gov> D=<d> run <k>/<K>: <exception>` followed by `forensic record -> <path to run_detail.jsonl>`; the traceback is in that file's footer.
- After each difficulty level the log lists the governments ranked by mean normalized health score.
- `grep -E ' (WARNING|ERROR) '` lists every incident in the sweep.

---

## manifest.json

Written at sweep start with `"status": "running"` and rewritten at the end. It is the authoritative record of what produced the directory; `--plots-only` reads the sweep's scale from it.

```json
{
  "schema_version": 2,
  "status": "complete",
  "entry_point": "run_full_simulation.py",
  "argv": ["run_full_simulation.py", "--k-runs", "100"],
  "started_at": "2026-09-26T05:19:34.468Z",
  "ended_at": "2026-09-28T04:41:10.002Z",
  "wall_seconds": 170496.1,
  "host": "<hostname>",
  "user": "<username>",
  "cwd": "<working directory>",
  "log_file": "<working directory>/sim_full_20260926T051934Z.log",
  "software": {"python": "3.12.3", "numpy": "2.5.3", "matplotlib": "3.11.2"},
  "config": {
    "governments": ["anarchy", "democracy", "republic", "autocracy",
                    "autocracy_lookahead", "oligarchy", "federated", "ads"],
    "difficulties": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    "k_runs": 100,
    "k_runs_source": "cli-override",
    "paper_k_runs": 100,
    "grid_size": 50,
    "n_agents": 500,
    "max_cycles": 150,
    "max_steps": 10,
    "base_seed": 42,
    "output_root": "<absolute path>",
    "heartbeat_every": 25,
    "engine_log_level": "WARNING",
    "write_run_detail": true,
    "decision_detail": true,
    "agent_states_every": null,
    "clean_output_root": false
  },
  "schedule": {
    "difficulty_knee_level": 90,
    "difficulty_tail_slope": 1.75,
    "difficulty_t_max": 1.25,
    "difficulty_t_at_100": 1.0757575757575757,
    "schedule_digest": "<sha256 hex>"
  },
  "results": {
    "total_cells": 168,
    "cells_completed": 168,
    "total_runs_expected": 16800,
    "total_runs_recorded": 16800,
    "failed_cells": [],
    "short_cells": [],
    "environment_audit": {
      "status": "pass",
      "rows_audited": 16800,
      "rows_missing_fingerprint": 0,
      "coordinates_checked": 2100,
      "difficulties_checked": 21,
      "k_runs_configured": 100,
      "fairness_violations": [],
      "variation_violations": []
    },
    "exit_code": 0
  }
}
```

| Field | Meaning |
|---|---|
| `status` | `running`, `complete`, `complete_with_losses` (some runs or cells lost), or `failed_environment_audit` (the fairness/variation check failed; `results` then holds `exit_code` and `environment_audit_error`) |
| `config.k_runs` | runs per cell actually executed |
| `config.k_runs_source` | `default`, `cli-override`, `archive` or `unknown` |
| `config.paper_k_runs` | the published design's runs per cell; used only to project figure properties to the published scale |
| `schedule` | the difficulty-schedule constants, plus `schedule_digest`, a SHA-256 over every difficulty-scaled value at every level 1–100. Two archives with different digests were produced by different schedules and must not be pooled |
| `results.failed_cells` | `{government, difficulty}` for cells whose worker raised |
| `results.short_cells` | `{government, difficulty, runs_recorded, runs_expected}` for cells that lost runs |
| `results.environment_audit` | FAIRNESS: every government shared one environment fingerprint at each (difficulty, run). VARIATION: runs within a difficulty had distinct fingerprints |
| `results.exit_code` | 0 only if every cell recorded all `k_runs` runs |

The manifest records the invoking user name and host name; remove them before publishing an archive if required.

---

## combined_final_stats.csv

One row per completed run, in completion order (sort before use).

| Column | Meaning |
|---|---|
| `government` | registry key |
| `difficulty` | 1–100 |
| `run` | 1-based run index |
| `env_fingerprint` | 16-hex-digit hash of the run's scenario plan; equal across governments at the same (difficulty, run) |
| `normalized_health_score` | primary metric |
| `final_survival_rate` | |
| `final_median_health` | |
| `final_health_gini` | |
| `min_survival_rate` | |
| `time_to_50pct_loss` | blank if survival never fell to 50% |

Metric definitions are in [METRICS_REFERENCE.md](METRICS_REFERENCE.md#outcome-metrics). This file drives `cell_stats.json`, `analysis/`, the eight cross-government figures, and the fingerprint audit (which can be re-run from the archive with `benchmark_core.verify_env_fingerprints_from_tree(output_root)`).

---

## Per-Run Files

### health_stats.csv

One row per cycle.

| Column | Meaning |
|---|---|
| `cycle` | 0-based |
| `normalized_health_score` | total health of living agents / initial population |
| `survival_rate` | alive / initial |
| `median_health`, `health_gini` | over living agents |
| `median_food`, `median_water` | stock over living agents |
| `num_alive` | |
| `events` | active event types, `|`-separated |
| `mean_prediction_error` | ADS only: running mean absolute forecast error of all closed forecasts; blank otherwise and before the first closure |
| `mean_calibration`, `proposals_enacted` | always blank |

### final_stats.json

The run summary. Every government writes:

| Field | Meaning |
|---|---|
| `total_cycles` | index of the last cycle (`max_cycles − 1`) |
| `normalized_health_score`, `final_survival_rate`, `final_median_health`, `final_health_gini` | values at the last cycle |
| `min_survival_rate`, `time_to_50pct_loss` | over the whole run |
| `final_median_food`, `final_median_water` | at the last cycle |
| `final_mean_prediction_error` | ADS: all-time mean absolute forecast error; `null` for every other government |
| `final_mean_calibration`, `final_proposals_enacted` | always `null` |
| `env_fingerprint` | as in the combined CSV |
| `run`, `difficulty`, `government` | identity |

**ADS and Autocracy+Lookahead** add their forecast-accuracy fields. Both governments open forecasts with the same ledger code (see [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#forecast-grading)): a forecast opened at cycle *t* predicts the mean health per agent of a named group of agents at *t* + 20 and is graded at *t* + 20 against that group's realized mean health (dead agents count as 0).

| Field | Meaning |
|---|---|
| `ads_forecast_schema_version` | `4`. Identifies the semantics of the fields below; emitted by both governments. Its absence from an ADS or Autocracy+Lookahead run means the reporting hook did not run |
| `calibration_mean_abs_err` | mean \|predicted − realized\| over every closed forecast |
| `calibration_mean_abs_delta_first_half`, `_second_half` | the same, split by the cycle on which the forecast closed, at `calibration_half_split_cycle`; `null` when a half closed nothing |
| `calibration_n_closed_first_half`, `_second_half` | observations behind each half |
| `calibration_half_split_cycle` | `max_cycles // 2` |
| `calibration_open_predictions_at_end` | forecasts whose review date fell after the last cycle |
| `calibration_n_degenerate` | closed forecasts whose predicted value was 0 (group projected extinct); graded normally |
| `calibration_n_closed_law_active` | reviews at which the originating law was still in force |
| `calibration_n_closed_law_lifted` | reviews at which it had already ended |
| `n_closed_no_law` | reviews of decisions that left no law in force (Autocracy+Lookahead `no_action` / `repealed`; always 0 for ADS) |
| `calibration_cycle_deltas` | every closure as `[closing_cycle, abs_error]`, cycle-ascending, errors rounded to 6 decimals |
| `calibration_cycle_deltas_by_outcome_then_scope` | the same pairs keyed by `outcome` then by `str(n_groups)` |
| `calibration_closures_by_outcome_then_scope` | `{outcome: {scope: {"n", "mean_abs_err"}}}` for the same cells |
| `parameter_estimates_final` | the government's end-of-run parameter estimates, same shape as the per-cycle `parameter_estimates` block |

`outcome` is always `enacted` for ADS, which opens a forecast only when it enacts a law. Autocracy+Lookahead opens one on every decision, with `outcome` in `enacted`, `replaced`, `retained`, `repealed`, `no_action`. The scope key `n_groups` is the ADS grouping scheme that produced the law (10, 5, 4, 3, 2 or 1); Autocracy+Lookahead always legislates for the whole population and records `1`. Keys are strings, so convert with `int()` before sorting.

`calibration_mean_abs_err` and the two `_half` means include every closure. For ADS that is every enacted law. For Autocracy+Lookahead it also includes `retained`, `repealed` and `no_action` decisions, which ADS never grades, so these scalars must not be compared across the two governments directly. The figures compare them through the partitioned fields restricted to `outcome ∈ {enacted, replaced}`.

These fields are deliberately absent from `combined_final_stats.csv` and `cell_stats.json`, which contain only cross-government metrics. Read them with `glob("<output_root>/ads/*/run_*/final_stats.json")`.

---

## run_detail.jsonl

Newline-delimited JSON, one record per line, identified by the `rec` field. Written by `engine/run_recorder.py`; flushed every 25 cycles. Disabled by `--no-run-detail`.

| `rec` | Count | Written by |
|---|---|---|
| `header` | exactly one, first line | every run |
| `cycle` | one per cycle, after the cycle completes | every run |
| `decision` | one per decision round, after that cycle's `cycle` record | ADS, Autocracy+Lookahead |
| `footer` | exactly one, last line | every run, including failed ones |

A missing footer means the process was killed.

### header

```json
{
  "rec": "header",
  "schema_version": 1,
  "run": {
    "government": "ads", "difficulty": 75, "run_index": 1,
    "run_dir": "ads/75/run_01", "run_id": "ads-d075-run01",
    "base_seed": 42,
    "env_seed": 8216165025444615928,
    "pop_seed": 6541488690590377084,
    "gov_seed": 4197804000465659359,
    "env_fingerprint": "d8716b1275d2bfbc"
  },
  "config": {
    "grid_rows": 50, "grid_cols": 50, "num_agents": 500, "max_cycles": 150,
    "max_steps_per_cycle": 10, "difficulty": 75, "drain_mult": 1.6212,
    "resource_density": 0.526, "terrain_variety": 1.0, "initial_health": 1.0,
    "initial_food_stock": 7.53, "initial_water_stock": 7.53,
    "event_frequency": 0.0, "event_severity": 0.998, "event_warning_cycles": 9,
    "regen_mult": 0.4394, "metabolic_rate": 0.1995, "ambient_hazard": 0.008,
    "visibility_radius": 5, "sim_config_seed": 8216165025444615928
  },
  "scheduled_events": [
    {"cycle": 17, "type": "drought", "severity": 1.299, "duration": 20, "region": null},
    {"cycle": 75, "type": "toxic_spill", "severity": 1.14, "duration": 20, "region": [4, 12, 6, 14]}
  ],
  "government_params": {"t_decision": 5, "lookahead_cycles": 20, "...": "..."},
  "started_at": "2026-09-26T08:34:09.429Z",
  "host": "<hostname>", "pid": 2977163,
  "software": {"python": "3.12.3", "numpy": "2.5.3", "matplotlib": "3.11.2"},
  "options": {"decision_detail": true, "agent_states_every": null, "heartbeat_every": 25}
}
```

- `env_seed` and `pop_seed` depend only on (difficulty, run) and are identical for all eight governments; `gov_seed` also depends on the government. See [ARCHITECTURE.md](ARCHITECTURE.md#seed-lattice).
- `scheduled_events` lists every event in the run's plan, including the ones drawn at random during plan generation. `config.event_frequency` is 0 because the simulation's own random-event generator is off when replaying a plan.
- `government_params` is `{}` for the six governments without a `get_params` method. ADS and Autocracy+Lookahead record their tunables, the estimator constants and bounds, the phantom-event settings, and `organization` (`"multinode"` for ADS, `"none"` for Autocracy+Lookahead).

### cycle

```json
{
  "rec": "cycle",
  "cycle": 20,
  "pop": {
    "alive": 500, "initial": 500, "deaths_this_cycle": 0,
    "normalized_health_score": 0.9825, "survival_rate": 1.0,
    "health": {"mean": 0.982, "median": 0.984, "min": 0.919, "max": 1.0,
               "p25": 0.984, "p75": 1.0, "std": 0.0196, "gini": 0.0091},
    "food":     {"mean": 9.4,  "median": 9.48, "min": 4.0,  "max": 22.83},
    "water":    {"mean": 6.03, "median": 3.12, "min": 3.12, "max": 11.61},
    "medicine": {"mean": 2.43, "median": 2.0,  "min": 2.0,  "max": 6.0},
    "infected": 0, "hungry": 0, "thirsty": 0, "starving": 0, "dehydrated": 0
  },
  "env": {
    "grid_mean_food": 21.2489, "grid_mean_water": 25.7365,
    "grid_total_food": 53122.14, "grid_total_water": 64341.29,
    "hazard_cell_count": 1432, "hazard_cell_fraction": 0.5728,
    "drain_mult": 1.6212, "drought_factor": 0.4546, "storm_damage": 0.0
  },
  "events": {
    "active": [{"uid": "E-0001", "type": "drought", "severity": 1.299,
                "cycles_remaining": 17, "start_cycle": 17, "duration": 20,
                "region": null, "epidemic_id": null,
                "effects": {"drought_factor": 0.4546}}],
    "triggered": [], "expired": [],
    "warnings": []
  },
  "laws": {
    "active_count": 6,
    "enacted": [{
      "law_id": "L-0010", "law_type": "REDISTRIBUTE_GENERIC", "source": "decision_round",
      "enacted_cycle": 20, "duration": 15, "expires_cycle": 35,
      "applies_to_all": false, "applies_to_count": 450,
      "event_type": null, "event_id": null,
      "description": "[ADS] Generic redistribution to group 7 (food+water both critical) [norm=0.9910, grouping=10g]",
      "params": {"recipient_ids_count": 50, "pct_food": 0.5, "pct_water": 0.5,
                 "pct_medicine": 0.3, "donor_amounts_count": 450}
    }],
    "expired": [{"law_id": "L-0002", "law_type": "REDISTRIBUTE_GENERIC",
                 "source": "decision_round", "enacted_cycle": 5, "duration": 15,
                 "reason": "duration_elapsed"}]
  },
  "government": {
    "name": "ADS",
    "decision_this_cycle": true,
    "audit": {"decisions_made_total": 15, "laws_enacted_total": 8,
              "proposals_evaluated_this_round": 32,
              "top_proposal": "REDISTRIBUTE_MEDICINE", "top_proposal_score": 436.87,
              "evidence_scores": "food:9.403 water:6.029 health:0.982"}
  },
  "calibration": {
    "open_predictions": 8, "closed_this_cycle": 0, "closed_total": 0,
    "n_degenerate": 0, "mean_abs_delta_alltime": null,
    "parameter_estimates": {
      "drain_mult": {"value": 1.3182, "n_cycles": 21, "sum_observed": 216.0897,
                     "sum_exposure": 133.29, "n_clamped": 0, "n_bound_clamps": 0},
      "regen_mult": {"value": 0.6903, "n_cycles": 21, "sum_observed": 6496.8028,
                     "sum_exposure": 16431.9971, "n_clamped": 0, "n_bound_clamps": 0},
      "max_steps":  {"value": 6, "n_cycles": 21, "running_max": 6,
                     "prior_ceiling": 1, "n_bound_clamps": 0}
    }
  }
}
```

`pop` and `env`:
- `hungry` / `thirsty`: hunger or thirst above 0.4, the level at which they start to drain health. `starving` / `dehydrated`: stock below 1.0.
- `drought_factor` is the largest active drought effect; `storm_damage` the per-cycle storm damage before `drain_mult`.
- `hazard_cell_count` / `hazard_cell_fraction`: cells whose hazard exceeds the cell's own ambient-hazard floor, i.e. terrain hazard or event/toxic hazard is present on top of the difficulty's uniform ambient hazard (every cell carries the ambient hazard from D2 upward, so counting raw `hazard > 0` would count every cell).

`events`:
- `uid` is assigned in order of first appearance within the run. `cycles_remaining` is −1 for an event without a fixed end.
- `triggered` / `expired` list uids that appeared or disappeared this cycle.
- `warnings` lists advance warnings for events not yet started (`cycles_until` > 0).

`laws`:
- `enacted` includes laws enacted anywhere in the cycle, including warning responses issued before the government's tick.
- `source` identifies the mechanism: for ADS `fast_response`, `decision_round`, `preempt_warning` or `eco_ration`; for Autocracy+Lookahead `leader_decision_round` for evaluated laws; `null` for laws without a recorded source.
- `expired[].reason`: `duration_elapsed`, `event_type_gone`, `event_id_cleared`, or `superseded_by_lookahead` (repealed by an Autocracy+Lookahead decision).
- `params`: lists and dicts are replaced by a `<name>_count` entry; numeric lists of up to 8 items (region boxes) are kept. Under `--detailed-agents` params are written in full.

`government.audit` is the government's `get_audit_info()`: vote tallies for the voting regimes, leader and loyalist resource shares for autocracy, elite shares for oligarchy, regional votes for federated, and for ADS the decision count, the proposals scored in the last round, the top raw score and `evidence_scores` (population mean food stock, mean water stock and mean health).

`calibration` appears only for ADS and Autocracy+Lookahead and is absent (not `null`) for the other governments:
- `open_predictions`: forecasts not yet graded. `closed_this_cycle`, `closed_total`: graded this cycle and so far. `mean_abs_delta_alltime`: running mean absolute error, `null` before the first closure. `n_degenerate`: closures whose prediction was 0.
- `parameter_estimates`: the environment parameters the government's evaluator is using, inferred from observations (see [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#parameter-inference)). `value` is the estimate used by the next decision round; `sum_observed` / `sum_exposure` are the accumulated realized effect and the effect the evaluator's model predicts at a multiplier of 1; `n_clamped` counts negative observations clamped to 0; `n_bound_clamps` counts estimates that left their sanity bounds and should be 0. `max_steps` is a running maximum of observed moves per agent per cycle, floored at `prior_ceiling`.

The estimates are not expected to equal the configuration's `drain_mult` or `regen_mult`: they estimate the multiplier that makes the evaluator's own simplified model reproduce what was observed.

### decision (ADS)

```json
{
  "rec": "decision",
  "cycle": 20,
  "mechanism": "ads_evidence_tree",
  "trigger": {"kind": "crisis_interval", "interval": 2,
              "pending_warnings": [], "t_decision": 5},
  "context": {"alive": 500, "min_group_fraction": 0.1, "max_laws": 25,
              "already_active_law_types": ["FOOD_RATION", "REDISTRIBUTE_FOOD",
                                           "REDISTRIBUTE_WATER", "SPREAD"]},
  "evidence": {"food": {"...": "..."}, "water": {"...": "..."},
               "health": {"...": "..."}, "terrain": {"...": "..."}},
  "summary": {"n_distributions": 6, "group_distribution_counts": [10, 5, 4, 3, 2, 1],
              "proposals_evaluated": 32, "top_proposal": "REDISTRIBUTE_MEDICINE",
              "top_raw_score": 436.87, "laws_enacted_this_round": 2},
  "groupings": [
    {"n_groups_target": 10, "n_groups_actual": 10,
     "groups": [
       {"group_id": 7, "agent_count": 50, "median_health": 0.984,
        "median_food": 9.476, "median_water": 3.118, "infected_count": 0,
        "law_scores": [
          {"law_type": "REDISTRIBUTE_GENERIC",
           "description": "[ADS] Generic redistribution to group 7 (food+water both critical)",
           "source_category": "food", "raw_score": 49.552,
           "norm_score": 0.991, "adjusted_score": 0.991},
          {"law_type": "REDISTRIBUTE_MEDICINE",
           "description": "[ADS] Redistribute medicine to group 7 (N=1.0)",
           "source_category": "health", "raw_score": 46.964,
           "norm_score": 0.9393, "adjusted_score": 0.9393}
        ],
        "best_law_type": "REDISTRIBUTE_GENERIC", "best_norm_score": 0.991}
     ]}
  ],
  "winner": {"n_groups": 10, "group_id": 7, "law_type": "REDISTRIBUTE_GENERIC",
             "description": "[ADS] Generic redistribution to group 7 (food+water both critical)",
             "norm_score": 0.991, "adjusted_score": 0.991, "base_norm_score": 0.991},
  "enacted": [
    {"law_id": "L-0010", "law_type": "REDISTRIBUTE_GENERIC", "n_groups": 10,
     "group_id": 7, "adjusted_score": 0.991, "applies_to_count": 450},
    {"law_id": "L-0011", "law_type": "REDISTRIBUTE_MEDICINE", "n_groups": 10,
     "group_id": 1, "adjusted_score": 0.9586, "applies_to_count": 450}
  ],
  "rejected_reasons": {"law_type_already_active": 30, "group_overlap": 0,
                       "max_laws_reached": 0, "below_min_group_fraction": 0,
                       "quarantine_bbox_unavailable": 0}
}
```

- `trigger.kind`: `scheduled_interval` (every 5 cycles), `crisis_interval` (every 2 cycles while an event is active) or `warning_preempt` (a crisis warning arrived; `interval` is `null`). Two rounds can run in one cycle; the record then describes the later one.
- `evidence` holds the four evidence dataclasses verbatim; fields are listed in [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#evidence-collection).
- `law_scores`: `raw_score` is the projected total surviving health of the group after 20 cycles, `norm_score = raw_score / agent_count` is the forecast, and `adjusted_score = norm_score × crisis_boost` is what candidates are ranked on (`crisis_boost` between 1.0 and 1.70). `adjusted_score` is `null` for a group smaller than 10% of the living population, which is never ranked.
- `winner` is the first law enacted in the round; its `norm_score` field carries the adjusted score and `base_norm_score` the unadjusted forecast. `enacted` lists every law enacted in the round (up to `max_laws`).
- `--no-decision-detail` drops `groupings` only (about 95% of the bytes).

### decision (Autocracy+Lookahead)

```json
{
  "rec": "decision",
  "cycle": 8,
  "mechanism": "autocracy_leader_lookahead",
  "trigger": {"kind": "crisis_interval", "t_decision": 5, "t_decision_crisis": 2,
              "event_types": ["drought"], "pending_warnings": ["drought"]},
  "context": {"alive": 500, "lookahead_cycles": 20, "target_groups": 1, "round": 2},
  "summary": {"candidates_evaluated": 4, "decisions": 1,
              "laws_enacted_total": 1, "laws_repealed_total": 0},
  "decisions": [{
    "event_type": "drought",
    "candidates": 4,
    "scored": [
      {"law_type": "FOOD_RATION", "params": {"max_per_cycle": 1.5}, "raw_score": 439.066, "norm_score": 0.8781},
      {"law_type": "FOOD_RATION", "params": {"max_per_cycle": 2.5}, "raw_score": 438.3206, "norm_score": 0.8766},
      {"law_type": "FOOD_RATION", "params": {"max_per_cycle": 4.0}, "raw_score": 437.9183, "norm_score": 0.8758},
      {"law_type": "NO_ACTION",   "params": {}, "raw_score": 438.0225, "norm_score": 0.876}
    ],
    "winner": {"law_type": "FOOD_RATION", "params": {"max_per_cycle": 1.5}, "norm_score": 0.8781},
    "outcome": "retained"
  }]
}
```

`trigger.kind` is `event_interval` (every 5 cycles), `crisis_interval` (every 2 cycles while an event is active or warned) or `warning_preempt`. `outcome` is one of `enacted`, `replaced`, `retained`, `repealed`, `no_action`.

### footer

Success:

```json
{
  "rec": "footer",
  "status": "ok",
  "cycles_completed": 150,
  "summary": {"total_cycles": 149, "normalized_health_score": 0.993,
              "final_survival_rate": 1.0, "...": "..."},
  "totals": {"deaths": 0, "laws_enacted": 156, "laws_expired": 154,
             "decision_rounds": 109, "proposals_evaluated": 1437,
             "events_triggered": 12, "events_expired": 11},
  "timing": {"started_at": "2026-09-26T08:34:09.429Z",
             "ended_at": "2026-09-26T08:35:27.041Z", "wall_seconds": 77.612},
  "recorder_errors": 0
}
```

`summary` is the complete `final_stats.json` content except the identity fields (floats rounded to 4 decimals, see [Serialization Rules](#serialization-rules)). `totals.proposals_evaluated` counts ADS candidates and is 0 for Autocracy+Lookahead, whose decision records report `candidates_evaluated` instead.

Failure replaces `summary` and `totals` with:

```json
"error": {"type": "ZeroDivisionError", "message": "float division by zero",
          "traceback": "Traceback (most recent call last): …"}
```

The traceback is truncated to its last 8,000 characters.

---

## agent_states.jsonl

Written only with `--detailed-agents [N]` (default cadence 10), which is refused when the sweep has more than 20 runs. One columnar record at cycle 0, every N cycles, and the last cycle:

```json
{
  "rec": "agents",
  "cycle": 40,
  "n": 500,
  "n_alive": 431,
  "fields": ["agent_id", "alive", "r", "c", "health", "food", "water", "medicine",
             "hunger", "thirst", "age", "n_epidemics", "epidemic_ids", "n_laws"],
  "rows": [
    ["C001", true, 12, 37, 0.964, 7.12, 6.8, 0.0, 0.0, 0.12, 40, 0, [], 3],
    ["C002", false, null, null, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 31, 1, ["epi0001"], 0]
  ],
  "active_laws": [{"law_id": "L-0042", "law_type": "REDISTRIBUTE_FOOD",
                   "expires_cycle": 57, "applies_to_count": 118}]
}
```

---

## cell_stats.json

One file per (government, difficulty), computed from `combined_final_stats.csv` rows.

```json
{
  "government": "ads",
  "difficulty": 75,
  "n": 10,
  "deterministic": false,
  "timestamp": "2026-09-26T09:29:10.239Z",
  "metrics": {
    "normalized_health_score": {"mean": 0.8578, "ci95_lower": 0.7407, "ci95_upper": 0.9749,
                                "n": 10, "n_distinct": 10, "deterministic": false},
    "time_to_50pct_loss": {"mean": 131.0, "ci95_lower": 131.0, "ci95_upper": 131.0,
                           "n": 1, "n_distinct": 1, "deterministic": true}
  }
}
```

- One entry per metric in `SUMMARY_FIELDS` (the six metric columns of the combined CSV).
- `mean ± t × SD/√n`, a t-interval on the mean, with `t` from a lookup table (1.96 for 30 or more degrees of freedom), rounded to 4 decimals. The interval is not clipped to the metric's range. With one value, the interval collapses to the mean.
- The metric-level `n` counts runs with a value; for `time_to_50pct_loss` that excludes runs that never lost half the population.
- The cell-level `n` is the number of runs recorded for the cell.
- `n_distinct` counts distinct values. `deterministic` is true when every run gave the same value; the cell-level flag is true when that holds for normalized health score, survival rate and median health together. A deterministic cell has one effective observation and a zero-width interval.

---

## analysis/

Plain CSV tables for the four plotted metrics (`normalized_health_score`, `final_survival_rate`, `final_median_health`, `final_health_gini`). Not regenerated by `--plots-only`.

| File | Contents |
|---|---|
| `per_government_summary.csv` | per government and metric, pooled over all difficulties: `n_obs`, `mean`, `std`, `median`, `min`, `max`, `ci_lower`, `ci_upper` (mean ± 1.96 × SEM) |
| `per_difficulty_summary.csv` | per difficulty and metric: governments ranked by mean (`rank`, `government`, `mean`) |
| `per_metric_analysis/<metric>_analysis.csv` | `government`, `difficulty`, `run`, value, one row per run |
| `regression_analysis.csv` | per government and metric: least-squares fit of the metric on difficulty over all runs (`n_obs`, `slope`, `intercept`, `r_squared`, `p_value`, `std_err`) |
| `pairwise_comparisons.csv` | every government pair and metric, pooled over difficulties: `n1`, `n2`, `mean_1`, `mean_2`, `mean_difference`, `t_statistic`, `p_value`, `significant_at_0_05` |
| `pairwise_comparisons_by_difficulty.csv` | the same per difficulty level, plus `ci_lower_diff`, `ci_upper_diff` (95% percentile bootstrap CI of the mean difference, 10,000 resamples, seeded per pair/difficulty/metric) and `ci_significant` (`Yes` when that CI excludes 0) |
| `correlation_matrix.csv` | Pearson correlations between the four metrics over all runs |
| `analysis_manifest.json` | file descriptions, governments, difficulties, and method notes |

Notes on the tests:
- The t-test is Welch-style (unequal variances). Its p-value, and the regression p-value, use the normal approximation to the t distribution, which understates p for small samples.
- Runs are paired by environment across governments (same `env_fingerprint` at the same difficulty and run), but these tests are unpaired. That is conservative: it overstates the standard error of a difference. A paired analysis can be built from `combined_final_stats.csv` by joining on `difficulty`, `run` and `env_fingerprint`.
- `significant_at_0_05` and `ci_significant` read `Undefined`, with the numeric cells blank, when either group has zero within-group variance; such a comparison has no sampling distribution.

---

## Figures

All figures are written to `<output_root>/plots/` at 150 dpi, 16 × 7 in, with difficulty (or cycle) on the x-axis. Titles name only the plotted quantity. Every PNG has a `<name>_alt_text.txt` sidecar containing the figure title, alt text of at most 100 words, the metric, governments, number of levels, the sweep scale, and the colour / line-style / marker legend. Colours come from a colour-blind-safe palette with ADS in black.

### Cross-government outcome figures (from combined_final_stats.csv)

| File | What is drawn |
|---|---|
| `normalized_health_score.png` | For each government at each difficulty, a Tukey box plot of the per-run values (box = interquartile range, whiskers = 1.5 × IQR, crosses = outliers) at the difficulty's x-position, and a line through the medians. A cell with one run is drawn as a single marker |
| `final_survival_rate.png` | as above |
| `final_median_health.png` | as above |
| `final_health_gini.png` | as above |
| `normalized_health_score_median_ci.png` | Same data summarised as the median with a 95% percentile-bootstrap confidence interval on the median (10,000 resamples, seed `derive_seed(base_seed, "plot.bootstrap", government, difficulty)`), governments fanned slightly around each tick, medians joined by a line |
| `final_survival_rate_median_ci.png` | as above |
| `final_median_health_median_ci.png` | as above |
| `final_health_gini_median_ci.png` | as above |

In the median/CI family, a cell that cannot support an interval is drawn as an open marker without a bar. The sidecar names the reason: `insufficient_n` (fewer than 6 runs, where the percentile interval is pinned to the sample range), `constant` (every run gave the same value), or `degenerate_zero_width` (the bootstrap distribution sits on one value). The sidecar also lists per-cell sample sizes and reports how visible the bands are: the median ratio of CI half-width to IQR half-width, the rendered band height in pixels, and the same height projected to `paper_k_runs` by 1/√n scaling. The box plots describe the spread of runs; the CI bars describe uncertainty in the median. They are different quantities.

### Forecast-accuracy figures (from per-run final_stats.json)

Drawn for ADS and Autocracy+Lookahead when their directories exist. The y-axis is mean \|predicted − realized\| health per agent; lower is a better forecast. ADS values use all of its closures. Autocracy+Lookahead values are restricted to closures with `outcome ∈ {enacted, replaced}` so that both series grade the same kind of decision.

| File | What is drawn |
|---|---|
| `mean_prediction_error.png` | Tukey box plot per government per difficulty of each run's mean absolute error over its closures |
| `calibration_half_split.png` | Each run's mean absolute error in an early and a late window, two box series per government per difficulty. The windows start at the burn-in breakpoint (below) and split the remaining cycles at their midpoint. When no breakpoint can be estimated, the figure falls back to the fixed split at `max_cycles // 2` |
| `calibration_half_split_median_ci.png` | The same per-run window means as median + 95% bootstrap CI |
| `calibration_trend_by_cycle.png` | Mean absolute error of all closures at each cycle, pooled over every run and difficulty, with a 95% band (mean ± 1.96 × SEM). The burn-in region is shaded; the headline least-squares trend is fitted to cycles at or after the breakpoint, and the full-range fit is drawn in grey for comparison |
| `mean_prediction_error_scope_matched.png` | ADS restricted to laws produced by its single-group scheme (`n_groups == 1`, i.e. legislating for the whole population, as Autocracy+Lookahead always does) against Autocracy+Lookahead. One pooled closure mean with 95% CI per difficulty; levels with fewer than 5 closures are omitted |
| `calibration_trend_by_cycle_scope_matched.png` | Per-cycle pooled mean with 95% CI for the same scope-matched subsets; cycles with fewer than 5 closures are omitted, no trend line is fitted, burn-in shaded with the same breakpoint |

Burn-in breakpoint: early in a run most agents are near full health, so forecasts and outcomes are both pinned near the ceiling and errors are small for reasons unrelated to forecast skill. The breakpoint is estimated once per archive from ADS's pooled `(cycle, error)` series by continuous two-segment least squares (`y = a + b·x + c·max(0, x − k)`, grid search over `k`, at least 200 observations per segment, 10-cycle margin at each end) and shared by every forecast-accuracy figure. The sidecar reports the fitted breakpoint, both slopes and the fraction of residual variance the segmented fit removes, which indicates how well the breakpoint is identified. Small sweeps usually have too few observations for an estimate; the figures then say so and use the unsegmented view.

No half-split figure is drawn for the scope-matched subset: ADS's single-group laws are too sparse to give most runs a per-run mean.

### Snapshot dashboards

`<government>/<difficulty>/visualizations/cycle_*.png` (run 1 only), 9.75 × 7.2 in at 300 dpi: the grid on the left (terrain, resource shading, agents coloured by health, infected agents marked, event overlays, legend) and the population's normalized health score, survival rate and health up to that cycle on the right. Frames are captured in memory during the run and cannot be re-rendered from the archive.

### Regenerating figures

```bash
python run_full_simulation.py --plots-only --output-dir <output_root>
```

reads `manifest.json`, `combined_final_stats.csv` and the per-run `final_stats.json` files and rewrites `plots/`. All bootstrap seeds are derived from the archived `base_seed`, so the output is reproducible.

---

## Serialization Rules

For `run_detail.jsonl` and `agent_states.jsonl`:

1. `json.dumps(..., allow_nan=False, separators=(",", ":"), ensure_ascii=False)`: compact, UTF-8, one record per line.
2. Before serialization every value passes through a sanitizer: non-finite floats become `null`, sets become sorted lists, tuples become lists, enums become their values, dataclasses become dicts, numpy scalars become Python numbers, and anything else becomes a truncated `repr`.
3. Rounding: health values 3 decimals (distribution std and Gini 4); resource stocks and grid totals 2 decimals; all other floats 4 decimals by default.
4. A record that fails to serialize is skipped and counted in `footer.recorder_errors`; the recorder never interrupts the simulation.

`final_stats.json`, `cell_stats.json` and `manifest.json` are indented JSON written with Python's standard `json` module.

---

## Sizes

Measured on a k = 10 sweep at publication scale (1,680 runs): about 1.1 GB in total, almost all of it `run_detail.jsonl`. A single `run_detail.jsonl` ranges from about 0.2 MB (voting governments) to about 1.3 MB (ADS), driven by the ADS `groupings` arrays. A k = 100 sweep is roughly ten times larger. `--no-decision-detail` removes most of the ADS volume; `--no-run-detail` removes the files entirely. JSONL compresses well with gzip.

---

## Troubleshooting

**`run_detail.jsonl` has no footer.** The process was killed (out of memory, signal, power loss). Records up to the last flush (every 25 cycles) are intact.

**`footer.recorder_errors > 0`.** Some records failed to serialize; gaps in the `cycle` sequence show which. The run's results are otherwise complete.

**`manifest.json` status is `running`.** The sweep did not finish. Check the log in the working directory.

**`manifest.json` status is `complete_with_losses`.** See `results.failed_cells` and `results.short_cells`, then `grep -E ' (WARNING|ERROR) '` in the log.

**`manifest.json` status is `failed_environment_audit`.** Governments did not share environments, or runs repeated an environment. The log lists every offending coordinate as `FAIRNESS VIOLATION` or `VARIATION VIOLATION`; no statistics or figures were produced.
