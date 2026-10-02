# Government Types Reference

The eight governments compared in the study: how each one decides, what laws it enacts, and how the laws are enforced. All eight govern the same `CitizenAgent` population in the same environments; only the government differs. Registry keys (used on the command line and in output paths) are given in parentheses.

---

## Comparative Summary

| | Anarchy | Democracy | Republic | Autocracy | Autocracy+Lookahead | Oligarchy | Federated | ADS |
|---|---|---|---|---|---|---|---|---|
| **Who decides** | nobody | all agents, majority vote | 30-seat parliament allocated by party vote | one leader | one leader, choices scored by lookahead | elite (1%) | each region's occupants | algorithm |
| **When** | never | ballot every 5 cycles, tallied the next cycle | party election every 5 cycles, parliament session every 2 | every cycle | every cycle, plus an evaluated round every 2 cycles while a drought, storm or epidemic is active or warned | every cycle | regional vote every 5 cycles | reflexes every cycle; decision round every 5 cycles (every 2 during an active event) and on every crisis warning |
| **Response to warnings** | none | shelter order; quarantine if the warning carries a region (≤ 3 cycles ahead) | as democracy | shelter order; quarantine if the warning carries a region | as autocracy, plus an evaluated round | shelter order on a storm warning | none | shelter order (storm ≤ 3 cycles ahead) and an immediate decision round whose proposals treat warned events as active |
| **Planning horizon** | none | none | none | none | 20-cycle projection | none | none | 20-cycle projection |
| **Law scope** | none | whole population | whole population | whole population | whole population | whole population | region | risk-banded group or whole population |
| **Drought ration cap** | none | 3.0 | 4.0 | 2.5 | 1.5, 2.5, 4.0 or none, chosen by projection | 3.0 | not in the regional ballot | computed from grid food and inferred drain |
| **Quarantine region** | none | centroid ± radius | centroid ± radius | centroid ± radius | centroid ± radius (only whether to quarantine is decided) | centroid ± radius | centroid ± radius, clipped to the region | tight bounding box of infected agents + 1 cell |
| **Resource extraction** | none | none | none | leader topped up to 12 units; loyalists (1%) held at 30% of all food and water | as autocracy | elite (1%) and loyalists (2%) each held at 30% | none | none |

"Centroid ± radius" is `Government._compute_quarantine_region`: a square centred on the mean position of infected agents, with radius `max(2, half the infected spread + 1, infected fraction × grid side)`, clipped to the grid.

---

## 1. Anarchy (`anarchy`)

No laws, no votes, no collective action. Each agent follows its own survival heuristics (see [ENVIRONMENT.md](ENVIRONMENT.md)). Movement is unrestricted and collection is uncapped.

**Question it answers:** how far does individual self-interest alone get the population?

---

## 2. Direct Democracy (`democracy`)

**Ballot.** Every 5 cycles the government builds a ballot of up to three options from the current situation, in this order: `QUARANTINE_EPIDEMIC`, `EPIDEMIC_RESPONSE` and (if fewer than 25% are infected) `SPREAD` during an epidemic; `MANDATORY_SHELTER` during a storm; `FOOD_RATION` during a drought; `FLEE_TOXIC` during a toxic spill; `REDISTRIBUTE_FOOD` + `FOOD_RATION` if median food < 3; `REDISTRIBUTE_WATER` + `WATER_RATION` if median water < 3; `REDISTRIBUTE_MEDICINE` if median medicine < 2; then `NO_ACTION`. The list is de-duplicated and cut to three; short lists are padded with random options.

**Vote.** During the next cycle every living agent votes for the option that best serves its own needs (`CitizenAgent._choose_vote`: redistribution scores by the agent's shortage, quarantine by infection status, shelter during storms; ties broken at random). The plurality winner is enacted unless a law of that type is already active or the winner is `NO_ACTION`.

**Laws.**

| Winner | Law enacted |
|---|---|
| `FOOD_RATION` | `FOOD_RATION` cap 3.0, lasts while a drought is active |
| `WATER_RATION` | `LIMIT_WATER_CONSUMPTION` cap 3.0, 20 cycles |
| `QUARANTINE_EPIDEMIC` | centroid ± radius region, tied to the epidemic |
| `EPIDEMIC_RESPONSE` | infected agents must treat themselves, tied to the epidemic |
| `MANDATORY_SHELTER` | shelter order, lasts while a storm is active |
| `REDISTRIBUTE_*` | the poorest 30% receive from the richest 30%, 20 cycles |
| `SPREAD` | non-infected agents move away from their nearest neighbour, 20 cycles |
| `FLEE_TOXIC` | `FLEE_CURRENT_LOCATION`, 8 cycles |
| `RESOURCE_RESERVE` | a random 3 × 3 reserve, 20 cycles (no enforcement mechanism in the engine) |

**Warnings.** A storm warning 1–3 cycles ahead triggers an emergency shelter order; an epidemic warning with a region triggers a quarantine of that region.

**Question it answers:** how does periodic majority rule by self-interested voters cope with fast-moving threats?

---

## 3. Representative Republic (`republic`)

**Parties.** Six fixed parties with platform weights over law types:

| Party | Platform emphasis |
|---|---|
| HealthParty | quarantine, epidemic response, medicine redistribution, shelter |
| ResourceParty | food/water redistribution, rationing, resource reserve |
| FreedomParty | no action, spreading out; low weight on restrictions |
| SolidarityParty | redistribution of all three resources |
| GreenParty | spreading out, fleeing toxic areas, resource reserve |
| SecurityParty | shelter, quarantine, fleeing toxic areas, movement limits |

**Election** (every 5 cycles, starting at cycle 0). Each living agent votes for the party whose platform, weighted by the agent's current needs, scores highest (`CitizenAgent._choose_party`). The 30 seats are allocated in proportion to vote share. (A set of 20 "representative" agents is also drawn at random; it has no role in law-making.)

**Parliament session** (every 2 cycles). Each seat contributes its party's platform weights, boosted for active events (+0.4 to quarantine and epidemic response during an epidemic, +0.5 to shelter during a storm, +0.4 to rationing and +0.3 to food redistribution during a drought, +0.5 to fleeing during a toxic spill). The law type with the highest total is passed unless one of that type is already active or it is `NO_ACTION`.

**Laws.** As democracy, with a food ration cap of 4.0, a water cap of 4.0 for 25 cycles, redistribution and reserves lasting 25 cycles, and `SPREAD` applying to all agents for 25 cycles. `LIMIT_STEPS` appears in a platform but is never enacted.

**Warnings.** As democracy.

**Question it answers:** does filtering preferences through parties and a parliament improve on direct voting?

---

## 4. Autocracy (`autocracy`)

**Leader.** The healthiest living agent at appointment. A new leader is appointed only when the current one dies.

**Loyalists.** The richest 1% of the population by food + water (at least one). Vacancies are filled immediately and the newcomer is topped up from the commons.

**Each cycle**, in order:
1. **Leader extraction.** If the leader's food or water is below 12, a short `REDISTRIBUTE_*` law transfers the shortfall from the commons.
2. **Loyalist extraction.** Food and water are moved directly from the commons to the loyalists until the loyalists hold 30% of the population's total (per-loyalist cap 300; at most 80% of the commons' stock per cycle).
3. **Event response.** During an epidemic: centroid ± radius quarantine. During a storm: shelter order. During a drought: `FOOD_RATION` cap 2.5. Each lasts while its event is active.

**Warnings.** Any storm warning triggers a shelter order; an epidemic warning with a region triggers a quarantine.

**Question it answers:** does immediate, centralised response offset self-serving priorities and a single decision-maker?

---

## 5. Autocracy + Lookahead (`autocracy_lookahead`)

Plain autocracy with the ADS's forward projection added, and nothing else. It exists to separate the two ways the ADS differs from every other government: it plans ahead, and it coordinates through role-separated structure (evidence nodes, risk-banded groups, per-domain proposals). This government has the first without the second, so the contrast `autocracy` → `autocracy_lookahead` → `ads` isolates the contribution of coordination from that of foresight.

**Inherited unchanged:** leader selection and succession, loyalists, resource extraction, and the reflex event responses and warning responses above.

**Added: evaluated decision rounds.** Whenever a drought, storm or epidemic is active or warned, a round runs every 2 cycles, and immediately when a crisis warning arrives. For each such event type the leader scores a small menu over the whole living population with the ADS's `EvaluatorNode`:

| Event | Menu |
|---|---|
| drought | `FOOD_RATION` at 1.5, 2.5 or 4.0 per cycle, or no ration |
| storm | `MANDATORY_SHELTER`, or no shelter order |
| epidemic | `QUARANTINE_EPIDEMIC` over the same centroid ± radius region autocracy uses, or no quarantine |

The highest projected total health wins; "do nothing" is listed last so that ties favour acting. The winner is installed as one law for everyone:

| Outcome | Meaning |
|---|---|
| `enacted` | no law of that type was standing; the winner is enacted |
| `replaced` | a different variant was standing; it is repealed and the winner enacted |
| `retained` | the winner is already in force |
| `repealed` | "do nothing" won while a law stood; the law is repealed and the reflex is suppressed for that event |
| `no_action` | "do nothing" won and nothing stood |

**Shared components.** It uses its own instance of the same `ParameterEstimator` as the ADS, so both project with inferred environment parameters, and its own instance of the same `ForecastLedger`, so their forecast accuracy is measured by the same code. Unlike the ADS it opens a forecast on every decision, not only on enactments. See [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md#autocracylookahead-comparison).

**What it deliberately lacks:** evidence nodes, risk banding, agent grouping, per-domain proposal generation, and any correction of the projected score.

**Question it answers:** how much of the ADS's advantage comes from the forward projection alone?

---

## 6. Oligarchy (`oligarchy`)

**Elite and loyalists.** The richest 1% (elite) and next richest 2% (loyalists) by food + water, each at least one agent. Members who die are replaced immediately and the newcomer is topped up from the commons.

**Each cycle:**
1. **Extraction.** For food and water separately, the combined shortfall of the elite and the loyalists against a target of 30% of the population's total each (per-member caps 500 and 300) is taken proportionally from the commons (at most 80% of their stock per cycle) and divided between the two tiers in proportion to their shortfalls.
2. **Event response.** As autocracy, with a drought ration cap of 3.0.

**Warnings.** A storm warning triggers a shelter order.

**Question it answers:** what happens to population health when a small group's accumulation takes priority?

---

## 7. Federated (`federated`)

**Regions.** At the first tick the grid is cut into up to 10 horizontal strips (`min(10, n_agents // 10 + 1)`) holding roughly equal numbers of the initial population. Each agent's home region is the one it started in. Regions do not change during the run.

**Regional votes** (every 5 cycles). Each region's current occupants vote by a fixed heuristic: infected agents vote `QUARANTINE_EPIDEMIC`; otherwise `MANDATORY_SHELTER` during a storm, `QUARANTINE_EPIDEMIC` during an epidemic, `REDISTRIBUTE_FOOD` during a drought or when the agent's food is below 3, `REDISTRIBUTE_WATER` when water is below 3, else `NO_ACTION`. The plurality winner is enacted for that region's occupants:

| Winner | Law enacted |
|---|---|
| `QUARANTINE_EPIDEMIC` | centroid ± radius around the region's infected agents, clipped to the region |
| `MANDATORY_SHELTER` | `MANDATORY_SHELTER_REGION` for the region |
| `REDISTRIBUTE_FOOD` / `_WATER` | the region's poorest 30% receive from its richest 30%, 20 cycles |

**Cross-region penalty.** An agent that steps into a region other than its home region loses 0.01 health if that region already contains other agents.

There is no inter-region trade and no warning response.

**Question it answers:** how does local self-government handle threats that ignore regional boundaries?

---

## 8. ADS: Adaptive Decision System (`ads`)

Algorithmic governance built from evidence nodes, a hypothesis generator and a lookahead evaluator. Summary; full details in [ADS_DEEP_DIVE.md](ADS_DEEP_DIVE.md).

- **Every cycle:** update the parameter estimator, grade due forecasts, and run fast-response reflexes (shelter during storms, bounding-box quarantine and treatment mandate during epidemics, drought rationing, fleeing toxic spills, zone-targeted redistribution, direct rescue of agents with critically low stocks, periodic equalisation, medicine rescue during epidemics, and rationing ahead of ecological depletion).
- **Decision rounds** (every 5 cycles, every 2 during an active event, and on each crisis warning):
  1. Four evidence nodes summarise food, water, health and terrain, including which events are warned.
  2. Every agent receives a risk band (0–9) per category; agents are sorted by total risk and partitioned under six grouping schemes (10, 5, 4, 3, 2 and 1 groups).
  3. Each group receives a few candidate laws, roughly one per category.
  4. Each candidate is scored by a 20-cycle projection of that group under the law, using inferred environment parameters; the score is projected surviving health per agent.
  5. Candidates are ranked across all schemes by score × crisis boost; the best non-conflicting ones are enacted, up to `max(3, alive // 20)` per round.
- **Forecast grading:** each enacted law's projected score is compared with the group's actual mean health 20 cycles later.

**Question it answers:** can evidence-driven, group-targeted, forward-looking governance outperform voting and command structures?

---

## Law Types and Enforcement

Laws act through three hooks in `governments/base.py`. `Law.applies_to` limits a law to listed agents; `None` means everyone.

| Law type | Enforcement | Enacted by |
|---|---|---|
| `FOOD_RATION` | caps food collected per cycle at `max_per_cycle` (`food_collection_limit`) | democracy, republic, autocracy, A+L, oligarchy, ADS |
| `LIMIT_WATER_CONSUMPTION` | caps water collected per cycle | democracy, republic, ADS |
| `QUARANTINE_EPIDEMIC` | `can_move`: agents inside the region who are infected (or listed) cannot leave; others cannot enter. `filter_actions` also spreads agents out inside the region and pushes nearby outsiders away | all except anarchy |
| `MANDATORY_SHELTER` | `can_move` forbids leaving a shelter cell; agents who see the law head for the nearest visible shelter | all except anarchy and federated |
| `MANDATORY_SHELTER_REGION` | as `MANDATORY_SHELTER`, within the region | federated |
| `REDISTRIBUTE_FOOD` / `_WATER` / `_MEDICINE` | each donor in `donor_amounts` is given a law-directed share action splitting its amount across `recipient_ids` | democracy, republic, autocracy, A+L, federated (food/water), ADS |
| `REDISTRIBUTE_GENERIC` | as above for all three resources in one law | ADS |
| `EPIDEMIC_RESPONSE` | infected agents are made to treat themselves (3 medicine per epidemic carried) | democracy, republic, ADS |
| `MOVE_TO_REGION` | each agent steps toward a random point inside the target region (redrawn each cycle) | ADS |
| `FLEE_CURRENT_LOCATION` | agents step toward the lowest-hazard unvisited neighbouring cell | democracy, republic, ADS |
| `SPREAD` | agents step away from their nearest neighbour | democracy, republic, ADS |
| `RESOURCE_RESERVE` | recorded, but no engine mechanism enforces it | democracy, republic |

Agents never share voluntarily; every transfer between agents is law-directed or a direct government transfer (autocracy loyalist and oligarchy extraction, ADS rescue and equalisation). Autocracy and oligarchy log their direct transfers as `REDISTRIBUTE_*` laws without a donor list, so those law records have no further effect; the ADS transfers are not logged as laws.

**Lifetime.** A law ends when its duration elapses, when its linked event type is no longer active (`event_type_gone`), or when no living agent carries its linked epidemic (`event_id_cleared`). Laws enacted "while the event lasts" receive a duration from `_heuristic_duration` (for example, a storm's remaining cycles + 3 for a shelter order) and are also linked to the event.

---

## Metrics for Comparison

The primary comparison metric is `normalized_health_score`: total health of the survivors divided by the initial population. Survival rate, median health, the health Gini coefficient, minimum survival rate and time to 50% loss are reported alongside it. See [METRICS_REFERENCE.md](METRICS_REFERENCE.md). Forecast accuracy (ADS and Autocracy+Lookahead only) is described in [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md#final_statsjson).
