# Environment

This page describes the simulated world: the grid, resources, agents, events
and the difficulty schedule. Governments are described in
[GOVERNMENTS.md](GOVERNMENTS.md). Values below are the ones in the code; the
source file is named in each section.

---

## The simulation cycle

Each cycle runs these steps in order (`engine/simulation.py`, `Simulation._step`):

1. **Events.** Scheduled events due this cycle start, advance warnings are
   issued, and active events apply their effects. Warnings are passed to the
   government.
2. **Regeneration.** Every cell regenerates food, water and medicine.
3. **Observation.** Each living agent observes its surroundings.
4. **Action.** Each living agent chooses actions; the government may filter or
   modify them according to its active laws; the actions are executed.
5. **Health.** Metabolism, hunger, thirst, disease, hazards and storms change
   each agent's health; agents at health 0 die.
6. **Government.** The government takes its turn (votes, decisions, new laws).
7. **Metrics.** Population and grid statistics are recorded.

---

## Grid and terrain

The world is a square grid (50×50 in the benchmark, 20×20 by default in
`run_simulation.py`). Each cell has a terrain type, stocks of food, water and
medicine, a hazard level, and a contamination flag used by epidemics. Several
agents may occupy the same cell. Source: `engine/grid.py`.

| Terrain | Food regen / cycle | Water regen / cycle | Terrain hazard / cycle | Shelter | Initial food | Initial water | Initial medicine | Caps (food / water / medicine) |
|---|---|---|---|---|---|---|---|---|
| Plains | 0.5 | 0.2 | 0.00 | no | 30–70 | 20–50 | 5–15 | 70 / 50 / 15 |
| Forest | 1.0 | 0.4 | 0.01 | yes | 60–100 | 25–55 | 8–20 | 100 / 55 / 20 |
| Water | 0.0 | 5.0 | 0.02 | no | 0 | 80–100 | 5–15 | 0 / 100 / 15 |
| Mountain | 0.1 | 0.1 | 0.04 | yes | 10–35 | 10–30 | 5–15 | 35 / 30 / 15 |
| Wasteland | 0.0 | 0.0 | 0.05 | no | 0 | 0 | 0 | 0 / 0 / 0 |

- **Layout.** Forest, water, mountain and wasteland are grown as up to four
  irregular patches each, targeting 20%, 15%, 15% and 10% of the cells; the
  rest is plains. Patch growth can stop slightly short of its target.
- **Initial stocks** are drawn uniformly from the ranges above and multiplied
  by the difficulty's `resource_density`.
- **Hazard.** A cell's hazard is its terrain hazard from the table above,
  plus the difficulty's ambient hazard (the same for every cell; see the
  [difficulty schedule](#difficulty-schedule)), plus any toxic-spill hazard on
  the cell, capped at 1. Every consumer reads this one value: health drain,
  the laws that move agents away from hazard, citizen movement scoring, the
  ADS lookahead, and the Autocracy+Lookahead evaluator.
  At difficulty 20 and above, for example, the hazard of a cell with no
  toxic spill is 0.008 on plains, 0.018 on forest, 0.028 on water, 0.048 on
  mountain and 0.058 on wasteland.
- **Shelter** (forest and mountain) protects occupants from storm damage.

---

## Resources

**Regeneration** (every cell except wasteland, every cycle):

```
multiplier = (1 − drought_factor) × regen_mult
food       = min(food cap,     food     + food regen  × multiplier)
water      = min(water cap,    water    + water regen × multiplier)
medicine   = min(medicine cap, medicine + 0.05        × multiplier)
```

`regen_mult` falls with difficulty (see the
[difficulty schedule](#difficulty-schedule)); `drought_factor` is the strongest
active drought's factor, or 0.

**Harvesting.** An agent collects from the cell it stands on, up to the amount
it requests (a citizen requests at most 5 food, 5 water or 2 medicine per
action), the cell's stock, and any limit the government's laws impose.

**Global depletion.** Harvesting also shrinks the same resource in every cell of
the grid by a small fraction, applied at the next regeneration step. The
fraction per unit harvested is

```
4 × 10⁻⁶ × (0.20 + 1.60 × d_eff / 100) × sqrt(100 / initial population)
```

and it grows exponentially once cumulative extraction of that resource passes
28% of four times the grid's initial stock. `d_eff` is the knee-adjusted
difficulty (see the [difficulty schedule](#difficulty-schedule)). This makes
unrestrained harvesting progressively costly for everyone.

---

## Agents

Every government governs the same kind of agent, `CitizenAgent`
(`agents/citizen.py`), a self-interested survival agent.

**Initial state.** Health 1.0, hunger and thirst 0, medicine 2.0, and food and
water stocks set by difficulty (15 each at difficulty 1, falling to 4.24 at
difficulty 100). Agents start on distinct cells forming one connected cluster
on non-water terrain. Stocks have no upper limit.

**Behaviour.** Each cycle an agent eats and drinks from its stocks when it can,
then acts on the first applicable priority: treat its own infection; seek
shelter from a storm (or when a law requires it); collect or move toward water
or food when critically short; give medicine to an infected neighbour who has
none; stock up on medicine during or ahead of an epidemic; otherwise top up its
food and water, moving toward the best food it has seen. It also votes and
makes proposals where the government holds votes.

**Observation.** An agent sees every cell and agent within a Euclidean radius
of 5 cells (81 cells away from the grid edge), its own state, the active events
and warnings, and the government's active laws.

**Actions** (`engine/agent.py`, executed in `engine/simulation.py`). An agent
may move up to `max_steps_per_cycle` times per cycle (10 in the benchmark, 1 by
default in `run_simulation.py`); every other action type at most once per cycle.

| Action | Effect |
|---|---|
| Move | One step to any of the 8 neighbouring cells; costs 0.002 health. Moves off the grid or forbidden by a law are ignored. |
| Collect food / water / medicine | Take from the current cell (see [Harvesting](#resources)). |
| Eat | Consume just enough food to restore health to 1.0, at 0.05 health per unit; each unit also lowers hunger by 0.3. |
| Drink | As eating, at 0.04 health per unit; each unit lowers thirst by 0.3. |
| Treat self | Spend 3 medicine per active infection to cure it. |
| Share food / water / medicine | Transfer stock to other agents. Executed only when directed by a law, except that an agent may give medicine to an adjacent infected agent on its own. |
| Rest | +0.005 health when on a shelter cell. |
| Vote / propose | Passed to the government. |

**Health dynamics** (after actions, every cycle). With `m` the difficulty's
drain multiplier:

```
food stock, water stock  −= metabolic_rate
hunger += 0.05 if food stock ≤ 0.1, else −0.02      (kept within 0–1)
thirst += 0.06 if water stock ≤ 0.1, else −0.02     (kept within 0–1)

Δhealth = − 0.025 × m × hunger              if hunger > 0.4
          − 0.020 × m × thirst              if thirst > 0.4
          − 0.015 × m × (active infections)
          − cell hazard × m
          − storm damage × m                if a storm is active and the cell has no shelter
health  = clamp(health + Δhealth, 0, 1)
```

An agent on a contaminated cell catches each active epidemic it does not
already carry with probability 0.05 per cycle. Infections clear on their own
after a difficulty-dependent number of cycles (see "Infection recovery" in the
[difficulty schedule](#difficulty-schedule)) unless treated sooner. An agent
whose health reaches 0 dies and leaves the grid.

---

## Events

Source: `engine/events.py` and `engine/scenario_plan.py`.

| Event | Effect while active | Region |
|---|---|---|
| Drought | Reduces regeneration of food, water and medicine by `drought_factor` = 0.35 × severity | whole grid |
| Storm | Agents not on shelter lose 0.04 × severity × `m` health per cycle | whole grid |
| Epidemic | Infects a cluster of agents at onset; each carrier then infects each uninfected agent in the 8 surrounding cells with probability 0.15 × severity per cycle, and marks its cell contaminated | starts near a random cell |
| Toxic spill | Adds 0.08 × severity to the hazard of each cell in the region | 3×3 cells |
| Resource rush | Adds 5 × severity food and 2.5 × severity water to each cell per cycle, up to 100 | 4×4 cells |

At the onset of an epidemic, the agent nearest the seed cell and the agents
nearest to it are infected, a difficulty-dependent fraction of the living
population (see "Epidemic seeding" in the
[difficulty schedule](#difficulty-schedule)). Infections outlast the event and
end only through treatment or natural recovery.

**Where events come from.** In the benchmark (and in `run_simulation.py`
without a scenario) each run's events are generated before it starts, from two
sources:

- **Scheduled waves.** `round(10 × t)` waves, where `t` is the difficulty
  position described below: none at difficulty 1, 5 at 50, 9 at 90, 11 at 100.
  Wave types cycle through drought, storm, epidemic and toxic spill; waves are
  spaced evenly over the run with a random jitter of up to 15% of the spacing.
  Each lasts 20 cycles (epidemics 12). Wave severity is the difficulty's base
  severity (0.60 at difficulty 1, 1.40 at 90, 1.57 at 100) times a random
  factor between 0.85 and 1.15. Source: `scenarios/scenario_base.py`,
  `_auto_event_schedule`.
- **Random events.** In each cycle an event starts with probability
  `event_frequency`. Its type is drawn with weights drought 0.25, storm 0.20,
  epidemic 0.25, toxic spill 0.15 and resource rush 0.15; its severity is
  `event_severity` times a random factor between 0.7 and 1.3; it lasts 8–14
  cycles (drought, epidemic), 3–6 (storm), 5–14 (toxic spill) or 5–9 (resource
  rush).

**Warnings.** Each scheduled event is announced to the government every cycle
from `event_warning_cycles` cycles before it starts; random events are
announced only when they start. How a government uses warnings depends on the
government.

The five named scenarios used by `run_simulation.py --scenario` replace the
scheduled waves with their own event lists; see
[RUNNING_SIMULATIONS.md](RUNNING_SIMULATIONS.md#scenarios).

---

## Difficulty schedule

Difficulty is an integer from 1 to 100. Source: `engine/difficulty.py` and
`SimulationConfig.from_difficulty` in `engine/simulation.py`.

Each difficulty maps to a position `t` on the schedule. Up to difficulty 90,
`t = (d − 1) / 99`, so the schedule is linear. Above 90 it advances 1.75 times
as fast per level (`DIFFICULTY_KNEE_LEVEL = 90`,
`DIFFICULTY_TAIL_SLOPE = 1.75`), which takes difficulties 95 and 100 beyond the
endpoints of the linear range:

```
d_eff = d                            for d ≤ 90
d_eff = 90 + 1.75 × (d − 90)         for d > 90
t     = (d_eff − 1) / 99             (0 at d = 1, 0.899 at d = 90, 1.076 at d = 100)
```

Every difficulty-dependent quantity except the ambient hazard is a linear
function of `t`; the ambient hazard's exception is described after the table.
This is the complete set of formulas — every other document that needs a
difficulty-scaled value links here rather than restating it:

| Quantity | Formula | Used by |
|---|---|---|
| `t` | `(d_eff − 1) / 99` | every row below |
| Health-drain multiplier `m` | `0.50 + 1.50 t` (4 dp) | all health drains |
| `resource_density` | `0.90 − 0.50 t` (3 dp) | initial grid resources |
| Initial food and water stock | `15 − 10 t` (2 dp) | agents' starting stocks |
| `regen_mult` | `max(0, 1 − 0.75 t)` (4 dp) | per-cycle resource regrowth |
| `metabolic_rate` | `0.05 + 0.20 t` (4 dp) | food and water consumed per agent per cycle |
| `event_frequency` | `0.01 + 0.05 t` (4 dp) | probability of a random event per cycle |
| `event_severity` | `0.40 + 0.80 t` (3 dp) | severity of random events |
| `event_warning_cycles` | `max(1, round(20 − 15 t))` | advance warning for scheduled events |
| Scheduled event waves | `round(10 t)` | number of scheduled events |
| Wave base severity | `0.60 + 0.90 t`, not above 1.40 at or below D90 (2 dp) | severity of scheduled events (each × a random factor between 0.85 and 1.15) |
| Infection recovery | `int(15 + 35 t)` cycles | natural recovery from an epidemic |
| Epidemic seeding | `0.01 + 0.09 t` | fraction of the living population infected when an epidemic starts |
| Ambient hazard | `0.008 × min(1, (d − 1) / 19)` — not a function of `t`; see below | uniform hazard added to every cell's terrain hazard |

**Values at representative difficulties** (`engine/difficulty.py`,
`SimulationConfig.from_difficulty`, `scenarios/scenario_base.py`):

| Parameter | D1 | D5 | D25 | D50 | D75 | D90 | D95 | D100 |
|---|---|---|---|---|---|---|---|---|
| `t` | 0.000 | 0.040 | 0.242 | 0.495 | 0.747 | 0.899 | 0.987 | 1.076 |
| Health-drain multiplier `m` | 0.5000 | 0.5606 | 0.8636 | 1.2424 | 1.6212 | 1.8485 | 1.9811 | 2.1136 |
| `resource_density` | 0.900 | 0.880 | 0.779 | 0.653 | 0.526 | 0.451 | 0.406 | 0.362 |
| Initial food and water stock | 15.00 | 14.60 | 12.58 | 10.05 | 7.53 | 6.01 | 5.13 | 4.24 |
| `regen_mult` | 1.0000 | 0.9697 | 0.8182 | 0.6288 | 0.4394 | 0.3258 | 0.2595 | 0.1932 |
| `metabolic_rate` | 0.0500 | 0.0581 | 0.0985 | 0.1490 | 0.1995 | 0.2298 | 0.2475 | 0.2652 |
| `event_frequency` (random events per cycle) | 0.0100 | 0.0120 | 0.0221 | 0.0347 | 0.0474 | 0.0549 | 0.0594 | 0.0638 |
| `event_severity` (random events) | 0.400 | 0.432 | 0.594 | 0.796 | 0.998 | 1.119 | 1.190 | 1.261 |
| `event_warning_cycles` | 20 | 19 | 16 | 13 | 9 | 7 | 5 | 4 |
| Scheduled event waves | 0 | 0 | 2 | 5 | 7 | 9 | 10 | 11 |
| Wave base severity | 0.60 | 0.64 | 0.82 | 1.05 | 1.27 | 1.40 | 1.49 | 1.57 |
| Infection recovery (cycles) | 15 | 16 | 23 | 32 | 41 | 46 | 49 | 52 |
| Epidemic seeding (fraction infected) | 0.0100 | 0.0136 | 0.0318 | 0.0545 | 0.0773 | 0.0909 | 0.0989 | 0.1068 |
| Ambient hazard | 0.0000 | 0.0017 | 0.0080 | 0.0080 | 0.0080 | 0.0080 | 0.0080 | 0.0080 |

**Ambient hazard** is the one channel that does not follow `t` over the whole
range. It is a uniform hazard added to every cell's terrain hazard, rising
linearly from 0 at difficulty 1 to 0.008 at difficulty 20 and constant from
there to 100 (`ambient_hazard`, with the constants `AMBIENT_HAZARD_FULL = 0.008`
and `AMBIENT_HAZARD_RAMP_END_LEVEL = 20`):

```
0 at d = 1, 0.0017 at 5, 0.0038 at 10, 0.0059 at 15, 0.008 at d ≥ 20
```

Because the ramp ends below the knee, the ambient hazard does not depend on
`DIFFICULTY_TAIL_SLOPE`. It is carried in each run's configuration and
recorded in the scenario plan (`config_params.ambient_hazard`), so it is
covered by the run's environment fingerprint and by the manifest's
`schedule_digest`.

Grid size, number of agents, number of cycles and moves per cycle do not depend
on difficulty; each entry point sets them explicitly. Terrain variety is at its
maximum at every difficulty. The sweep records the schedule's constants in each
`manifest.json` (`schedule` block).
