# ADS Deep Dive

How the Adaptive Decision System (`governments/ads.py`) decides, how it infers the environment it projects forward (`governments/parameter_inference.py`), and how its forecasts are graded (`governments/forecast_ledger.py`).

The ADS is a `Government` subclass. It governs the same `CitizenAgent` population as every other regime; there are no special agent types. What distinguishes it is a two-tier structure: reflexes that act every cycle without evaluation, and decision rounds that build candidate laws from evidence and score each one with a 20-cycle forward projection.

---

## Contents

1. [Components](#components)
2. [Per-Cycle Tick](#per-cycle-tick)
3. [Fast Response](#fast-response)
4. [Decision Round](#decision-round)
5. [Evidence Collection](#evidence-collection)
6. [Risk Bands and Grouping](#risk-bands-and-grouping)
7. [Candidate Laws](#candidate-laws)
8. [The Evaluator](#the-evaluator)
9. [Ranking and Enactment](#ranking-and-enactment)
10. [Warnings](#warnings)
11. [Parameter Inference](#parameter-inference)
12. [Forecast Grading](#forecast-grading)
13. [Autocracy+Lookahead Comparison](#autocracylookahead-comparison)
14. [Key Parameters](#key-parameters)
15. [Modelling Limitations](#modelling-limitations)

---

## Components

```
AdsGovernment
├── ParameterEstimator          infers drain_mult, regen_mult, max_steps from observations (every cycle)
├── ForecastLedger              opens a forecast per enacted law, grades it 20 cycles later
├── fast response               reflexes, every cycle, no evaluation
└── decision round
    ├── FoodEvidenceNode ─┐
    ├── WaterEvidenceNode │     situational summaries of the population and grid
    ├── HealthEvidenceNode│
    ├── TerrainEvidenceNode┘
    ├── HypothesisGeneratorNode tags agents with risk bands, forms groups, proposes laws
    ├── EvaluatorNode           20-cycle projection per (group, candidate law)
    └── enactment               global ranking, conflict filters, law creation
```

`EvaluatorNode`, `ParameterEstimator` and `ForecastLedger` are also used, unmodified, by Autocracy+Lookahead.

---

## Per-Cycle Tick

`AdsGovernment.tick(cycle)` runs after agents have acted and health dynamics have been applied:

1. `ParameterEstimator.observe(cycle, sim)`: fold this cycle's evidence into the three estimators. It must run every cycle, in order; a repeated or skipped cycle raises.
2. `_expire_laws(cycle)`.
3. `ForecastLedger.close(...)`: grade every forecast whose review date is this cycle.
4. `_fast_response(cycle)`.
5. If `cycle > 0` and `cycle % interval == 0`, run a decision round, where `interval` is 2 while any epidemic, storm, drought or toxic spill is active and `t_decision` (5) otherwise.

Warnings arrive earlier in the cycle (before agents act) through `receive_event_warnings`, which can trigger an additional round; see [Warnings](#warnings).

---

## Fast Response

`_fast_response` runs every cycle. Each law is enacted only if no active law of the same type exists.

| Trigger | Action |
|---|---|
| active storm | `MANDATORY_SHELTER` for everyone, lasting the storm's remaining cycles + 3 and tied to the storm |
| active epidemic | `QUARANTINE_EPIDEMIC` over the tightest bounding box of infected agents plus one cell, and `EPIDEMIC_RESPONSE` (treatment mandate), both tied to the epidemic |
| active drought | `FOOD_RATION` with cap `max(2, min(4, grid food / (20 × alive)))`, tied to the drought |
| active toxic spill | `FLEE_CURRENT_LOCATION`, 8 cycles |
| a zone in distress | The grid is divided into `max(2, min(6, ⌊√alive / 3⌋))²` zones. In a zone with at least 3 agents where more than 25% hold under 2 units of food (or water) or the median is below 3, the poorest 30% become recipients; zones with a median above 6 supply their richest 30% as donors. If no zone qualifies but more than 20% of the population is below 2 or the population median is below 3, the poorest 30% receive from the richest 30%. Enacted as a 6-cycle `REDISTRIBUTE_FOOD` / `_WATER` of 3 units per recipient |
| any agent below 1.5 food or water | Direct rescue, not a law: first from the agent's own cell (up to 3 units), then from agents holding more than 6 units (up to 1.5 per transfer, donors keep 6) |
| every 5 cycles | Equalisation, not a law: agents below 65% of the population median and below 4 units receive from agents above 130% of the median, for food, water (donor floor 4) and medicine (donor floor 3) |
| epidemic with more than 15% infected | Medicine rescue, not a law: infected agents short of `3 × active epidemics` medicine receive from agents holding more than 5 |
| ecological depletion | See below |

**Rationing ahead of depletion.** The grid tracks cumulative extraction of food and water as a fraction of four times the initial grid stock. Past a threshold of 0.28 of that total, every collection triggers an accelerating global loss of stock (see [ENVIRONMENT.md](ENVIRONMENT.md)). When the food (or water) fraction reaches 72% of the threshold, the ADS enacts a 12-cycle `FOOD_RATION` (or `LIMIT_WATER_CONSUMPTION`) whose cap falls from 5 × (1 − 0.20) at the warning level to 5 × (1 − 0.35) at the threshold and to at most 5 × (1 − 0.50) beyond it, never below a survival floor computed from the estimated drain multiplier (`survival_floor`, clamped to 3.0–5.0). These laws carry `source = "eco_ration"`.

**Emergency collection.** `AdsGovernment.filter_actions` adds a collect action at the front of the list for any agent holding under 2 units of food or water on a cell that has some, so a shelter order cannot stop a critically short agent from collecting.

---

## Decision Round

`_decision_round(cycle)`:

```
1. Evidence     collect the four evidence summaries, folding in pending warnings
2. Tagging      assign each living agent a 0-9 risk band per category
3. Environment  env = estimator.environment_model(sim, cycle)   (frozen for the round)
4. For each grouping scheme n in [10, 5, 4, 3, 2, 1]:
      form n groups by total risk
      for each group:
          propose candidate laws (types already in force are dropped)
          for each candidate:
              raw  = evaluator.evaluate(candidate, group agents, env)   projected total health
              norm = raw / group size                                   the forecast
              if group size >= 10% of alive: adjusted = norm x crisis_boost, add to ranking
5. Rank all candidates globally by adjusted score; enact the best non-conflicting ones
6. Open a forecast for every enacted law; store the round for the decision record
```

---

## Evidence Collection

Each node returns a dataclass. Warned events are included alongside active ones where a field refers to them. All four are written verbatim to the `evidence` block of the decision record.

**FoodEvidence**

| Field | Meaning |
|---|---|
| `mean_stock` | mean food stock of living agents |
| `pct_critical` | fraction with food < 3 |
| `pct_starving` | fraction with food < 1 |
| `grid_mean`, `grid_total` | food per cell and total on the grid |
| `cycles_until_empty` | `grid_total / (2 × alive)` |
| `drought_active`, `warning_drought` | a drought is active / warned |

**WaterEvidence**

| Field | Meaning |
|---|---|
| `mean_stock`, `pct_critical` (< 3), `pct_dehydrated` (< 1) | as for food |
| `grid_mean`, `grid_total`, `cycles_until_empty` | as for food |
| `warning_toxic`, `toxic_active` | a toxic spill is warned / active |
| `toxic_region` | bounding box `(r_min, c_min, r_max, c_max)` of cells with event hazard, or `null` |

**HealthEvidence**

| Field | Meaning |
|---|---|
| `mean_health` | mean health of living agents |
| `pct_critical` | fraction with health < 0.4 |
| `pct_low_medicine` | fraction with medicine < 3 (not enough to cure one infection) |
| `infected_fraction` | fraction carrying any epidemic |
| `epidemic_ids` | epidemics carried by living agents |
| `warning_epidemic` | an epidemic is warned |
| `infected_centroid` | mean `(row, col)` of infected agents |
| `infected_spread_radius` | mean Manhattan distance of infected agents from the centroid |
| `quarantine_feasible` | the infected agents' bounding box covers less than 30% of the grid |

**TerrainEvidence**

| Field | Meaning |
|---|---|
| `mean_hazard` | mean hazard (terrain + ambient + event) under living agents |
| `pct_on_shelter` | fraction standing on shelter terrain |
| `pct_near_shelter` | fraction within Manhattan distance 3 of a shelter cell |
| `avg_dist_to_shelter` | mean distance to the nearest shelter cell |
| `shelter_cell_count` | shelter cells on the grid |
| `storm_active`, `storm_severity` | a storm is active; its per-cycle storm damage |
| `toxic_active`, `warning_storm` | a toxic spill is active; a storm is warned |

`pct_starving`, `pct_dehydrated`, `cycles_until_empty`, `mean_hazard`, `avg_dist_to_shelter` and `shelter_cell_count` are recorded but not used by any proposal rule.

---

## Risk Bands and Grouping

**Bands** (0 = safest, 9 = highest risk), per agent:

| Category | Band |
|---|---|
| food | `9 − min(9, ⌊food / 3⌋)`; divisor 2 while a drought is active or warned |
| water | `9 − min(9, ⌊water / 3⌋)` |
| health | 9 if infected, else `⌊(1 − health) × 10⌋` |
| terrain | 9 if a storm is active and the agent is not on shelter; 8 if a toxic spill is active and the cell's hazard exceeds 0.1; else `min(9, ⌊hazard × 40⌋)` |

**Groups.** For each scheme in `GROUP_DISTRIBUTION_COUNTS = [10, 5, 4, 3, 2, 1]`, living agents are sorted by the sum of their four bands (ascending, so group 0 is the lowest-risk group) and cut into that many consecutive, equal-sized slices, the last slice taking the remainder. If fewer agents are alive than the scheme's target, the number of groups is reduced to the number of agents (`n_groups_actual`). A group's representative bands are those of its median member. The six schemes overlap: every agent appears in one group of each scheme, and the one-group scheme is the whole population.

---

## Candidate Laws

`HypothesisGeneratorNode.propose_laws` produces a few candidates per group, roughly one per category, from the group's representative bands and the evidence. "Group mean" below means the mean over the group's members; durations are in cycles.

**Food** (at most one):

| Condition | Candidate |
|---|---|
| food band < 6 and grid food depletion ≥ 72% of the threshold | `FOOD_RATION` with the depletion-scaled cap described under Fast Response, 12 |
| food band ≥ 6 and water band ≥ 6 | `REDISTRIBUTE_GENERIC`: food and water `max(2, 5 − group mean)`, medicine `max(1, 3 − group mean)`, 15 (replaces the water candidate) |
| food band ≥ 6, grid food < 3 × alive, a food-rich region exists, no storm or toxic spill, ≤ 5% infected | `MOVE_TO_REGION` to the richest 4 × 4 food window, duration 4–10 depending on distance |
| food band ≥ 6 otherwise | `REDISTRIBUTE_FOOD`, `max(2, 5 − group mean)` per recipient, 15 |
| drought active or warned | `FOOD_RATION`, cap `min(4, max(1.5, grid food / (20 × alive)))`, 20 |
| otherwise | `FOOD_RATION` cap 4.0 (grid mean food > 30) or `max(2.5, grid mean / 12)`, 8; or, if any member holds under 2 food, `REDISTRIBUTE_FOOD` top-up `max(1.5, 5 − group mean)`, 8 |

**Water** (skipped when the generic redistribution was proposed):

| Condition | Candidate |
|---|---|
| water band ≥ 6 | `REDISTRIBUTE_WATER`, `max(2, 5 − group mean)`, 15 |
| otherwise | `LIMIT_WATER_CONSUMPTION`, cap `min(6, max(2, grid mean water / 15))`, 8, if that cap is below 4.5 and no member holds under 2 water; else `REDISTRIBUTE_WATER` top-up `max(1.5, 5 − group mean)`, 8 |

**Health:**

| Condition | Candidate(s) |
|---|---|
| epidemic threat: health band 9, > 10% of the group infected, an epidemic warned, or > 15% of the population infected | `EPIDEMIC_RESPONSE` (treatment mandate), 20, if > 40% of the group is infected; else `QUARANTINE_EPIDEMIC`, 20, if quarantine is feasible; else `EPIDEMIC_RESPONSE`, 15. Plus `REDISTRIBUTE_MEDICINE`, 15, when the group's mean medicine falls more than 1 unit short of `3 × epidemics` |
| health band ≥ 5, or > 30% of the population holds < 3 medicine | `REDISTRIBUTE_MEDICINE`, `max(1, 3 − group mean)`, 15 |
| otherwise | `EPIDEMIC_RESPONSE` without mandate, 5 |

**Terrain:**

| Condition | Candidate |
|---|---|
| storm active or warned, and > 50% of agents within 3 cells of shelter or > 5% infected | `MANDATORY_SHELTER`, `max(8, round(15 × storm damage))` |
| storm active or warned otherwise | `MOVE_TO_REGION` to the richest 4 × 4 food + water window (flagged as storm-motivated), duration 4 to `max(6, round(12 × storm damage))` |
| toxic spill active | `FLEE_CURRENT_LOCATION`, 8 |
| > 5% infected | `QUARANTINE_EPIDEMIC`, 20, if feasible and < 30% infected; else `SPREAD` for the group's non-infected members, 10, if < 25% infected and the spread radius exceeds 3; else `EPIDEMIC_RESPONSE`, 15 |
| terrain band ≥ 7 | `MOVE_TO_REGION` to the richest food + water window, duration 3–8; `FLEE_CURRENT_LOCATION` if none |
| otherwise | `SPREAD`, 8 |

**Resource-rich regions.** `_find_resource_rich_regions` slides a 4 × 4 window over the grid in steps of 2, scores each position by total food, water or both, and returns up to three windows whose centres are at least 4 cells apart (Manhattan), best first. The first is used. If the grid is smaller than the window, the whole grid is the target.

**Movement duration.** `_movement_duration` uses the median Manhattan distance of the group's agents to the nearest point of the target region, plus 2, clamped between the minimum and base durations above.

---

## The Evaluator

`EvaluatorNode.evaluate(proposal, group_agents, sim, cycle, env)` projects the group forward `LOOKAHEAD_CYCLES = 20` cycles under the proposal and returns the total health of the group members alive at the end. Every candidate is projected over the same horizon so the scores are comparable. The normalized score `raw / group size` is the projected mean health per agent (dead agents count as 0) and is the ADS's forecast.

The projection works on copies of the group members' state and of the cells they stand on. Each projected cycle:

1. **Events.** Real events already in progress apply until their true end. Hypothetical future droughts, storms and epidemics are injected at random with the per-cycle rates, effects and durations in `env.event_stats`, which come from `EventSystem.observed_event_stats`: the rate of a category is `n / (cycle + 20)` over events already seen, capped at 0.5. Nothing about the scheduled future is visible.
2. **Regeneration** of the snapshotted cells, scaled by the drought factor and by `env.regen_mult`.
3. **Per agent:**
   - relocation toward shelter, only while a storm is doing damage and only under `MANDATORY_SHELTER` (toward the nearest shelter cell) or a storm-motivated `MOVE_TO_REGION` (toward the region's centre), up to `env.max_steps_per_cycle` cells per cycle;
   - eat and drink up to 2 units each;
   - collect up to 5 food and 5 water (less under a ration or water limit) and 2 medicine from the agent's starting cell;
   - redistribution candidates add their per-recipient amount;
   - a treatment mandate cures an agent that holds 3 medicine per epidemic;
   - infection by each active epidemic with probability 0.04 per cycle (0.01 inside a quarantine region);
   - health drain with the evaluator's coefficients × `env.drain_mult`: hunger 0.02 and thirst 0.015 (scaled by hunger/thirst above 0.4), 0.03 per epidemic carried, terrain hazard, and storm damage when not sheltered;
   - death at health 0.

**Common random numbers.** The evaluator has two random streams, both derived from the run's evaluator seed. The infection stream is reseeded on every call from `eval_seed XOR cycle`, so identical candidates score identically regardless of evaluation order. The event-injection stream is reseeded from `derive_seed(eval_seed, "eval.round", cycle)` and consumes a fixed number of draws per call, so every candidate in a round faces exactly the same hypothetical future.

**The environment model is inferred, not read.** `env` is built once per round by the government's `ParameterEstimator` and frozen. The evaluator never reads the simulation's true `drain_mult`, `regen_mult`, movement cap or difficulty. See [Parameter Inference](#parameter-inference).

---

## Ranking and Enactment

**Crisis boost.** A candidate for a group of at least 10% of the living population is ranked on `adjusted_score = norm_score × crisis_boost`, where the boost starts at 1.0 and adds 0.35 if the group's median food or water is below 2.5, 0.25 if its median health is below 0.5, and 0.10 if any member is infected (maximum 1.70). Smaller groups are scored and recorded but never ranked.

**Ordering.** All ranked candidates from all six schemes are sorted by adjusted score, highest first. Exact ties are broken by a uniform draw from the government's own seeded RNG.

**Filters.** Walking down the list, a candidate is enacted unless:

| Rejection reason | Rule |
|---|---|
| `max_laws_reached` | `max(3, alive // 20)` laws have already been enacted this round (25 at 500 agents) |
| `law_type_already_active` | a law of that type is in force or was enacted earlier in the round |
| `group_overlap` | more than half the group's members are covered by a law already enacted this round |
| `quarantine_bbox_unavailable` | a quarantine was chosen but no infected agent has a position |
| `below_min_group_fraction` | (counted at scoring time) the group is below 10% of the living population |

Counts of each reason are written to the decision record.

**Enactment details.**

- `QUARANTINE_EPIDEMIC` receives the tight bounding box of all infected agents plus a 1-cell buffer, clipped to the grid. The other governments use a centroid-and-radius square instead.
- For `REDISTRIBUTE_FOOD` / `_WATER` / `_MEDICINE` the group members are the recipients and every other living agent is a potential donor. `Government._compute_redistribution_amounts` spreads the total `recipients × amount` evenly over donors, shifts any donor's shortfall to donors with slack, and scales the per-recipient amount down if donors hold too little. The law applies to the donors. `REDISTRIBUTE_GENERIC` does this for all three resources.
- Other laws apply to the group's members (for the epidemic-motivated `SPREAD`, only its non-infected members).
- Laws are linked to the relevant active event (epidemic, drought, storm) so they lapse when it ends.
- A forecast is opened for each enacted law (see [Forecast Grading](#forecast-grading)).

---

## Warnings

`receive_event_warnings` is called at the start of every cycle with that cycle's warnings. Scheduled events are warned every cycle from `event_warning_cycles` before they start, and every event produces an "active" warning when it starts. When any warning concerns an epidemic, drought, storm or toxic spill:

1. A storm due within 3 cycles triggers an immediate `MANDATORY_SHELTER` (`source = "preempt_warning"`) if none is in force.
2. A decision round runs immediately (`trigger.kind = "warning_preempt"`). The evidence records which events are warned (`warning_drought`, `warning_epidemic`, `warning_storm`, `warning_toxic`), and the proposal rules treat a warned drought, storm or epidemic like an active one, so the round can propose rationing, shelter or epidemic measures before the event starts. The projection itself models an event only once it has started.

A regular tick-driven round can run later in the same cycle.

---

## Parameter Inference

`ParameterEstimator` gives the evaluator three environment parameters that depend on difficulty and that no government may read directly. Each estimate is updated once per cycle from quantities a government can observe:

| Parameter | Observation | Estimator |
|---|---|---|
| `drain_mult` | Realized health loss per agent from the health-dynamics step (`Simulation.last_health_drain`), summed over living agents, against the drain the evaluator's own coefficients predict for each agent's hunger, thirst, infections, hazard and storm exposure at a multiplier of 1 | shrunk ratio, prior mean 1.0, bounds 0–5 |
| `regen_mult` | Realized food + water regrowth on cells below half their terrain cap that nobody collected from this cycle, against the regrowth the evaluator's model predicts at a multiplier of 1 (drought-adjusted) | shrunk ratio, prior mean 1.0, bounds 0–2 |
| `max_steps` | Largest number of completed moves by any agent this cycle (`Simulation.last_move_counts`) | running maximum, floored at 1, cap 50 |

**Shrunk ratio.** With accumulated observed effect `O`, accumulated predicted exposure `E` over `n` cycles and prior weight `K = 20 × E / n`:

```
estimate = (1.0 × K + O) / (K + E)  =  w × 1.0 + (1 − w) × (O / E),   w = 20 / (20 + n)
```

so the prior's weight depends only on the number of cycles observed (0.12 after 150 cycles). With no evidence the estimate equals the prior, which is the evaluator's neutral baseline (drain and regeneration at 1.0, one step per cycle, no hypothetical events).

**What the estimates mean.** The target is the multiplier that makes the evaluator's own simplified model reproduce what was observed, not the configuration constant. Two consequences:

- The evaluator's drain coefficients differ from the engine's (hunger 0.02 vs 0.025, thirst 0.015 vs 0.020, disease 0.03 vs 0.015), so the drain estimate converges to the engine's multiplier scaled by a channel-mix factor between about 0.5 (disease-dominated) and 1.33 (thirst-dominated).
- The engine applies a global depletion loss the evaluator does not model, so the raw regrowth ratio is below the configured `regen_mult`; but the prior mean 1.0 is above it at every difficulty above 1, and at run lengths of 60–150 cycles the shrinkage usually leaves the reported estimate above the configured value. Estimates from runs of different lengths are therefore not directly comparable.

`n_bound_clamps` in the per-cycle record counts estimates that left their bounds; it should be 0. `n_clamped` counts negative observations clamped to 0.

Event statistics for the evaluator come from `EventSystem.observed_event_stats` rather than from this estimator, but follow the same prior-weighted form.

---

## Forecast Grading

The ADS measures how accurate its projections are. This is a measurement only; nothing it computes feeds back into scoring, ranking or the forecast.

- **Open.** When a law is enacted, the ledger records `predicted = norm_score` (the unboosted projection, not the adjusted ranking score) for the group's members (`evaluated_agent_ids`; for redistribution these are the recipients, not the donors who form `applies_to`), with tags `outcome = "enacted"`, `n_groups` (the scheme that produced it), `category`, `law_type` and `law_id`.
- **Close.** Exactly 20 cycles later, whatever happened to the law, `realized` is the mean health of the same agents with dead agents counted as 0, the same functional form as the projection. The error is `|predicted − realized|`.
- **Classify.** Each closure records whether the originating law was still in force (`law_active`) or had ended (`law_lifted`); a closure whose prediction was 0 is counted as degenerate but graded normally.

Outputs: the per-cycle `calibration` block in `run_detail.jsonl` (open, closed, running mean error, parameter estimates), the `mean_prediction_error` column of `health_stats.csv`, and the per-run fields in `final_stats.json`, including every closure as `[cycle, error]` and the same closures partitioned by outcome and grouping scheme. Field definitions are in [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md#final_statsjson); the figures built from them are described under [Figures](OUTPUT_STRUCTURE.md#figures).

Two properties matter when reading the error series. Early in a run most agents are near full health, so predictions and outcomes are both close to 1.0 and errors are small regardless of skill; the figures estimate and exclude this burn-in. And forecasts made for fine-grained groups (the ten-group scheme produces most enactments) are a different prediction problem from forecasts for the whole population; the scope-matched figures restrict the ADS to its one-group laws for comparison with Autocracy+Lookahead.

---

## Autocracy+Lookahead Comparison

Autocracy+Lookahead (`governments/autocracy_lookahead.py`) is plain autocracy plus the same evaluator, estimator and ledger classes (its own instances). The table summarises what it shares with the ADS and what it lacks:

| Aspect | ADS | Autocracy+Lookahead |
|---|---|---|
| Projection | `EvaluatorNode`, 20 cycles | same |
| Environment parameters | inferred by `ParameterEstimator` | same |
| Forecast grading | `ForecastLedger` | same |
| Evidence nodes, risk bands, groups | yes | no |
| Candidates | generated per group from evidence | fixed menu per event type |
| Target population | a risk-banded group or everyone | everyone |
| Rounds | every 5 cycles (2 during an event) and on warnings | every 2 cycles while a drought, storm or epidemic is active or warned, and on warnings |
| Warned events | proposal rules treat warned droughts, storms and epidemics as active | menus cover warned events, but the projection sees them only once they start (same for both) |
| Forecasts opened | on every enactment (`outcome = enacted`) | on every decision (`enacted`, `replaced`, `retained`, `repealed`, `no_action`) |
| Scope tag `n_groups` | 10, 5, 4, 3, 2 or 1 | always 1 |

A like-for-like forecast comparison restricts Autocracy+Lookahead to `enacted` and `replaced` decisions, and for scope, restricts the ADS to `n_groups == 1`.

---

## Key Parameters

| Parameter | Value | Where |
|---|---|---|
| `t_decision` | 5 cycles (2 while an event is active) | `AdsGovernment.__init__`, `tick` |
| `LOOKAHEAD_CYCLES` | 20 | `ads.py`; also the forecast review delay |
| `GROUP_DISTRIBUTION_COUNTS` | `[10, 5, 4, 3, 2, 1]` | `ads.py` |
| `MIN_GROUP_FRACTION` | 0.10 | `ads.py` |
| laws per round | `max(3, alive // 20)` | `_decision_round` |
| crisis boost | +0.35 food/water, +0.25 health, +0.10 infection | `_decision_round` |
| region scan | 4 × 4 window, step 2, top 3 | `RESOURCE_REGION_ROWS`, `TOP_K_REGIONS` |
| estimator prior | 20 cycles; prior means 1.0 / 1.0 / 1 step | `parameter_inference.py` |
| estimator bounds | drain 0–5, regen 0–2, steps ≤ 50 | `parameter_inference.py` |
| event-rate prior, cap | 20 cycles, 0.5 per cycle | `engine/events.py` |

These values, together with the estimator constants, are written to `government_params` in each run's `run_detail.jsonl` header.

---

## Modelling Limitations

- The projection covers only the group being legislated for; interactions with the rest of the population (donors, neighbours, shared cells) are not modelled.
- Resource collection and terrain hazard stay pinned to each agent's starting cell for the whole projection, so the benefit of moving to a resource-rich or low-hazard region is invisible to it. Only storm-shelter relocation is modelled.
- Redistribution is projected as a per-cycle top-up to recipients without drawing down donors.
- The evaluator's drain coefficients differ from the engine's; parameter inference absorbs the difference into `drain_mult` rather than correcting it.
- Toxic spills and resource rushes are never injected as hypothetical events, because the projection has no channel through which they could act.
- Agents have no voice: laws are chosen entirely by the algorithm.
