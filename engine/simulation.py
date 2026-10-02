"""
Main simulation loop. Executes one cycle at a time:
  1. Event system updates
  2. Agents observe
  3. Agents act (filtered through government laws)
  4. Health dynamics applied
  5. Government system ticks
  6. Metrics recorded
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .grid import Grid, cluster_positions
from .agent import Agent, Action, ActionType, DIRECTION_DELTAS
from .events import EventSystem, EventType
from .metrics import MetricsCollector
from .audit import AuditTrail

if TYPE_CHECKING:
    from governments.base import Government


# The difficulty schedule lives in its own leaf module so that grid.py and
# events.py can consume it too.  They cannot import from this module: this
# module already imports THEM (see the .grid / .events imports above), so the
# dependency has to point the other way.  Re-exported here because
# `from engine.simulation import difficulty_multiplier` is an import path
# several comments in governments/ads.py refer to.
from .difficulty import (  # noqa: F401  (re-exported for backward compatibility)
    DIFFICULTY_KNEE_LEVEL,
    DIFFICULTY_TAIL_SLOPE,
    effective_difficulty,
    difficulty_t,
    difficulty_multiplier,
    ambient_hazard,
)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class SimulationConfig:
    """
    All simulation parameters.

    Grid size (grid_rows, grid_cols), agent count (num_agents), and
    max_cycles are set explicitly by the user — they are NOT scaled by
    difficulty.  Difficulty (1–100) controls only:
      • resource_density  — how rich the initial grid is
      • event_frequency   — how often random events occur
      • event_severity    — how intense events are
      • drain_mult        — the health/hunger/thirst drain multiplier
      • event_warning_cycles — how many cycles of advance warning agents get
                               (20 at difficulty 1 → 7 at difficulty 90,
                               extrapolating to 4 by difficulty 100)
      • ambient_hazard    — uniform hazard added to every cell's terrain
                            hazard (0 at D1 → 0.008 at D20, constant above;
                            :func:`engine.difficulty.ambient_hazard`)

    Difficulty scale (actual emitted values, verified against
    ``from_difficulty`` at the shipped ``DIFFICULTY_TAIL_SLOPE = 1.75``;
    every row below is what the code produces, not an aspiration):
        1   very easy  — 0.50× drain, density 0.900, 20-cycle warnings
        25  normal     — 0.86× drain, density 0.779, 16-cycle warnings
        50  hard       — 1.24× drain, density 0.653, 13-cycle warnings
        90  brutal     — 1.85× drain, density 0.451,  7-cycle warnings
        95  extreme    — 1.98× drain, density 0.406,  5-cycle warnings
        100 extreme    — 2.11× drain, density 0.362,  4-cycle warnings

    D95 and D100 extrapolate past the slope-1.0 lerp endpoints (D95 is
    1.92×/0.425/6-cycle at slope 1.0, D100 2.00×/0.400/5-cycle -- see
    :data:`engine.difficulty.DIFFICULTY_TAIL_SLOPE`).  D1-D90 are
    bit-identical at every slope.

    The schedule is piecewise-linear: one step per difficulty level below
    D90, then ``DIFFICULTY_TAIL_SLOPE`` steps per level from D90 to D100.  At
    the shipped 1.75 that is steeper than 1-for-1, so D91-D100 extrapolate
    past the original 1-100 lerp domain (bounded by
    :data:`engine.difficulty.DIFFICULTY_T_MAX`).  (At slope 1.0, the
    schedule is linear across the whole 1-100 domain with no extrapolation.
    At slope 0.0, scaling stops at :data:`DIFFICULTY_KNEE_LEVEL` and
    D90/D95/D100 are one condition.)
    """
    # Grid — set by user, not scaled by difficulty
    grid_rows: int = 20
    grid_cols: int = 20
    resource_density: float = 0.85
    terrain_variety: float = 1.0   # always maximum; terrain variety is not difficulty-gated
    # Population — set by user, not scaled by difficulty
    num_agents: int = 50
    initial_health: float = 1.0
    initial_food_stock: float = 10.0   # agents' starting food (scaled by difficulty)
    initial_water_stock: float = 10.0  # agents' starting water (scaled by difficulty)
    # Difficulty: integer 1–100 (controls drain/resource/event only)
    difficulty: int = 1
    event_frequency: float = 0.02
    event_severity: float = 0.70
    event_warning_cycles: int = 10  # unscaled default; from_difficulty() scales 20 (D1) -> 4 (D100 at shipped slope 1.75; 5 at slope 1.0), physical floor 1
    regen_mult: float = 1.0         # base grid regen multiplier (1.0 = normal, scaled by difficulty)
    metabolic_rate: float = 0.08    # food/water consumed per agent per cycle (baseline survival cost)
    # Uniform hazard added to every cell's terrain hazard.  None (the default)
    # means "take it from the difficulty schedule": __post_init__ replaces it
    # with engine.difficulty.ambient_hazard(difficulty), so a config built
    # directly (scenario presets, run_simulation.py, plan reconstruction)
    # carries the same value as one built by from_difficulty.  After
    # construction it is always a float.  An explicit value is kept as given
    # and validated when the Grid is built.
    ambient_hazard: Optional[float] = None
    # Observation
    visibility_radius: int = 5
    # Movement
    max_steps_per_cycle: int = 1   # agents may MOVE up to this many times per cycle
    # Misc — set by user, not scaled by difficulty
    seed: Optional[int] = 42
    max_cycles: int = 200
    record_every: int = 1
    verbose: bool = False
    # When False, no AuditTrail is built and _step() skips recording into it.
    # The benchmark harness sets this: AuditTrail._population_section /
    # _health_drains_section iterate every living agent, sort four lists and
    # format strings on every cycle of every run, to build a text log the
    # benchmark never saves.  run_simulation.py keeps the default and is
    # unaffected — it is the tool that actually writes the trail out.
    record_audit_trail: bool = True

    def __post_init__(self) -> None:
        if self.ambient_hazard is None:
            self.ambient_hazard = ambient_hazard(self.difficulty)

    @classmethod
    def from_difficulty(cls, difficulty: int, **overrides) -> "SimulationConfig":
        """
        Build a SimulationConfig where only resource/event/drain parameters
        scale with *difficulty* (integer 1–100).

        Grid size, agent count, and max_cycles are NOT scaled — they keep their
        defaults (20×20, 50 agents, 200 cycles) unless the caller passes
        overrides.  Pass keyword overrides to fix any individual field.

        Difficulty-scaled fields, D1 -> D90 -> D100 at the shipped
        ``DIFFICULTY_TAIL_SLOPE = 1.75`` (D1-D90 are bit-identical at every
        slope, D91-D100 extrapolate past the slope-1.0 lerp endpoints shown
        in parentheses):
          resource_density:     0.900 → 0.451  → 0.362  (0.400 at slope 1.0)
          initial_food_stock:   15.00 → 6.01   → 4.24   (5.00 at slope 1.0)
          initial_water_stock:  15.00 → 6.01   → 4.24   (5.00 at slope 1.0)
          event_frequency:      0.010 → 0.0549 → 0.0638 (0.060 at slope 1.0)
          event_severity:       0.400 → 1.119  → 1.261  (1.200 at slope 1.0)
          event_warning_cycles: 20    → 7      → 4      (5 at slope 1.0)
          regen_mult:           1.000 → 0.3258 → 0.1932 (0.250 at slope 1.0)
          metabolic_rate:       0.050 → 0.2298 → 0.2652 (0.250 at slope 1.0)
          ambient_hazard:       0.0   → 0.008  → 0.008  (full from D20; same at every slope)
          terrain_variety:      always 1.0 (maximum variety at all difficulties)

        At slope 0.0 every row stops at the D90 column, so D90, D95 and D100
        receive byte-identical configs.  At slope 1.0, D100 lands exactly on
        the lerp endpoint with no extrapolation (see the slope-1.0 values
        above).
        """
        d = max(1, min(100, int(difficulty)))

        # Single source of truth for the difficulty schedule.  Two knobs
        # implementing one knee via two different expressions can silently
        # disagree: a hardcoded `min(t, 0.899)` at one site versus
        # `min(90, d)` at another, with 0.899 sitting 1.01e-05 ABOVE D90's
        # true t of 89/99 = 0.89898989..., leaves D90 unclamped while
        # D95/D100 clamp — a 0.03% step that ADS's global argmax over
        # candidate laws can amplify into a large, spurious swing in
        # normalized_health_score at higher difficulty.  Routing both knobs
        # through difficulty_t() fixes the structure, so this class of bug
        # cannot recur: there is exactly one place the knee is expressed.
        #
        # `t` can exceed 1.0 once DIFFICULTY_TAIL_SLOPE > 1;
        # engine.difficulty.validate_schedule() guarantees it stays within
        # DIFFICULTY_T_MAX, the domain in which every lerp below is still
        # physically meaningful.
        t = difficulty_t(d)

        # metabolic_rate scales with difficulty (0.05 at D1, 0.2652 at D100
        # at the shipped slope 1.75; 0.25 at D100 at slope 1.0 -- uncapped
        # lerp, extrapolates cleanly above the knee).
        metabolic = round(_lerp(0.05, 0.25, t), 4)

        fields = dict(
            resource_density     = round(_lerp(0.90, 0.40, t), 3),   # rich → scarce (extrapolates past D100)
            terrain_variety      = 1.0,
            initial_health       = 1.0,
            initial_food_stock   = round(_lerp(15.0,  5.0, t), 2),   # generous → limited (extrapolates)
            initial_water_stock  = round(_lerp(15.0,  5.0, t), 2),
            difficulty           = d,
            event_frequency      = round(_lerp(0.01, 0.06, t), 4),   # rare → moderate (extrapolates)
            event_severity       = round(_lerp(0.40, 1.20, t), 3),   # mild → hard (extrapolates)
            # Physical bound: 1 is the minimum number of warning cycles at
            # which a warning is still issued at all.  A t<=1 output clamp
            # (e.g. a floor of 5) would masquerade as this physical limit but
            # would freeze the channel once the tail slope pushes past 1.0.
            # At the shipped slope 1.75, D100 rounds to 4, below any such
            # output clamp -- this relaxation is exactly what lets the tail
            # extrapolate past the slope-1.0 endpoint.
            event_warning_cycles = max(1, round(_lerp(20, 5, t))),
            # Physical bound: 0.0 is the minimum regen multiplier (no net
            # regeneration), unreachable inside DIFFICULTY_T_MAX (minimum
            # 0.0625).  An output clamp at 0.25 (this channel's slope-1.0
            # endpoint) would have the same t<=1 masquerade problem as above:
            # at the shipped slope 1.75, D100 is 0.1932, below that value --
            # by design.
            regen_mult           = round(max(0.0, _lerp(1.0, 0.25, t)), 4),
            metabolic_rate       = metabolic,
            ambient_hazard       = ambient_hazard(d),
            visibility_radius    = 5,
            max_steps_per_cycle  = 1,
            seed                 = 42,
            record_every         = 1,
            verbose              = False,
        )
        fields.update(overrides)
        return cls(**fields)


class Simulation:
    """
    Core simulation engine. The government parameter implements the decision
    framework (anarchy, democracy, republic, autocracy, federated, or ADS).
    """

    def __init__(
        self,
        config: SimulationConfig,
        government: "Government",
        agents: List[Agent],
    ):
        self.config = config
        self.government = government
        self.rng = random.Random(config.seed)

        self.grid = Grid(
            rows=config.grid_rows,
            cols=config.grid_cols,
            resource_density=config.resource_density,
            terrain_variety=config.terrain_variety,
            seed=config.seed,
            ambient_hazard=config.ambient_hazard,
        )

        self.agents = agents
        self.agent_map: Dict[str, Agent] = {a.agent_id: a for a in agents}

        self.event_system = EventSystem(
            event_frequency=config.event_frequency,
            event_severity=config.event_severity,
            event_warning_cycles=config.event_warning_cycles,
            seed=config.seed,
            difficulty=config.difficulty,
        )

        self.metrics = MetricsCollector(initial_population=len(agents))
        # None when audit recording is disabled — callers that save the trail
        # (run_simulation.py) leave config.record_audit_trail at its default.
        self.audit = (
            AuditTrail(government_name=government.name)
            if config.record_audit_trail else None
        )
        #: Event warnings emitted during the most recent _step(), exposed so an
        #: external observer (the benchmark's RunRecorder) can record them
        #: without re-deriving them from event-system internals.
        self.last_warnings: List["EventWarning"] = []
        #: OBSERVATION CHANNEL — realized health drain per agent from the most
        #: recent ``_apply_health_dynamics`` pass, as ``agent_id -> delta``,
        #: where ``delta = prev_health - agent.health >= 0``.
        #:
        #: Same rationale as ``last_warnings`` above: exposed so an external
        #: observer can record a realized outcome without re-deriving it from
        #: internals.  Here the re-derivation is not merely inconvenient, it is
        #: IMPOSSIBLE from outside: an observer diffing ``agent.health`` between
        #: its own per-cycle hooks measures ``drain - eat/drink/rest restores +
        #: move cost``, because ``_execute_actions`` (step 4) writes health four
        #: separate times before the drain at step 5 ever runs.  Separating the
        #: drain from the restores is the whole reason this exists.
        #:
        #: Records the CLIPPED delta — what an outside observer could in
        #: principle measure — not the uncensored arithmetic one.  An agent
        #: whose drain took it below zero shows the truncated value, which is
        #: why ``ParameterEstimator`` drops agents that died this cycle rather
        #: than trusting their final entry.
        #:
        #: Reports the past only; never a parameter and never a forecast.
        #: Available to every government; only ADS chooses to read it.
        self.last_health_drain: Dict[str, float] = {}
        #: OBSERVATION CHANNEL — completed relocations per agent during the most
        #: recent step-4 action loop, as ``agent_id -> count``.
        #:
        #: Counts *successful* moves, incremented in ``_do_move`` after
        #: ``place_agent`` returns, not the ``moves_done`` budget counter: a
        #: completed relocation is externally observable and a rejected one is
        #: not.  The count is a valid (merely slightly loose) lower bound on
        #: ``config.max_steps_per_cycle``, which is what a government inferring
        #: travel speed needs and is not allowed to read directly.
        #:
        #: Cleared once per cycle, before the loop, so its contents describe a
        #: CYCLE rather than whichever agent happened to act last.
        self.last_move_counts: Dict[str, int] = {}
        self.cycle = 0
        self.drain_mult = difficulty_multiplier(config.difficulty)
        self.logger = logging.getLogger(f"sim.{government.name}")

        # Drain rates (base, before difficulty multiplier)
        self.hunger_build_rate = 0.05    # hunger increases per cycle without food
        self.thirst_build_rate = 0.06    # thirst increases per cycle without water
        self.hunger_drain = 0.025        # health lost per cycle when hungry
        self.thirst_drain = 0.020        # health lost per cycle when thirsty
        self.disease_drain = 0.015       # health lost per cycle when infected
        self.eat_restore = 0.05          # health restored per food unit eaten
        self.drink_restore = 0.04        # health restored per water unit drunk
        self.medicine_restore = 0.15     # health restored per medicine unit

        # Place agents on grid
        self._place_agents()

        # Initialise global resource depletion tracking (must be after grid generation
        # so initial totals reflect the actual starting cell values)
        self.grid.initialize_depletion_params(len(agents), config.difficulty)

        # Give government a reference to simulation
        self.government.bind(self)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _place_agents(self) -> None:
        positions = cluster_positions(self.grid, len(self.agents), self.rng)
        for agent, (r, c) in zip(self.agents, positions):
            self.grid.place_agent(agent, r, c)
            agent.health = self.config.initial_health
            agent.food_stock = self.config.initial_food_stock
            agent.water_stock = self.config.initial_water_stock

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> MetricsCollector:
        for cycle in range(self.config.max_cycles):
            self.cycle = cycle
            self._step(cycle)
        return self.metrics

    def _step(self, cycle: int) -> None:
        # 1. Events
        warnings, drought_factor = self.event_system.update(cycle, self.grid, self.agents)
        self.last_warnings = warnings
        self.government.receive_event_warnings(warnings, cycle)
        if warnings:
            self.logger.debug(
                "cycle=%d events_triggered=%d drought_factor=%.3f warnings=%s",
                cycle, len(warnings), drought_factor,
                [w.event_type.value for w in warnings],
            )

        # 2. Grid regeneration (drought reduces it; difficulty also reduces base rate)
        self.grid.regenerate(drought_factor=drought_factor,
                             base_regen_mult=self.config.regen_mult)

        # 3. Agents observe → act
        observations = {}
        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                obs = agent.observe(self.grid, self.government, cycle)
                # Inject event warnings
                obs["event_warnings"] = [w.__dict__ for w in warnings]
                obs["active_events"] = self.event_system.active_summary(cycle)
                obs["cycle"] = cycle
                observations[agent.agent_id] = obs
            except Exception as e:
                if self.config.verbose:
                    print(f"  [WARN] Agent {agent.agent_id} observe error: {e}")
                observations[agent.agent_id] = {}

        # 4. Execute actions (government mediates)
        # One clear per cycle, before the loop — NOT per agent inside
        # _execute_actions — so last_move_counts describes this cycle's whole
        # population rather than only the agent that acted last.
        self.last_move_counts.clear()
        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                obs = observations.get(agent.agent_id, {})
                actions = agent.act(obs, self.government, cycle)
                # Government may filter/override actions based on active laws
                actions = self.government.filter_actions(agent, actions, cycle)
                self._execute_actions(agent, actions, cycle)
            except Exception as e:
                if self.config.verbose:
                    print(f"  [WARN] Agent {agent.agent_id} act error: {e}")

        # 5. Health dynamics
        self._apply_health_dynamics(cycle)

        # 6. Government tick (voting, ADS cycle, etc.)
        self.government.tick(cycle)

        # 7. Metrics
        if cycle % self.config.record_every == 0:
            event_names = [e["type"] for e in self.event_system.active_summary(cycle)]
            self.metrics.record(cycle, self.agents, self.grid, self.government, event_names)
            if self.audit is not None:
                self.audit.record(cycle, self, warnings)

        if self.config.verbose and cycle % 10 == 0:
            alive = sum(1 for a in self.agents if a.alive)
            mean_h = sum(a.health for a in self.agents if a.alive) / max(1, alive)
            print(f"  Cycle {cycle:4d} | alive={alive:3d} | mean_health={mean_h:.3f} "
                  f"| events={event_names}")

        # INFO-level cycle summary (logged regardless of verbose flag)
        if cycle % self.config.record_every == 0:
            alive = sum(1 for a in self.agents if a.alive)
            mean_h = (sum(a.health for a in self.agents if a.alive) / alive) if alive else 0.0
            active_events = [e["type"] for e in self.event_system.active_summary(cycle)]
            self.logger.info(
                "cycle=%d alive=%d mean_health=%.3f active_events=%s",
                cycle, alive, mean_h, active_events,
            )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_actions(self, agent: Agent, actions: List[Action], cycle: int) -> None:
        """
        Execute agent actions for one cycle.

        Rules:
        - MOVE: up to max_steps_per_cycle times.
        - All other action types: at most ONCE per cycle (tracked by done_types set).
          This means an agent can eat AND drink AND share AND collect AND treat AND vote
          in the same cycle, but cannot eat twice.
        - Sharing (SHARE_FOOD/WATER/MEDICINE/RESOURCE): only executed when the action
          was placed by a government law (flag set on action params via 'law_directed').
          Agents do not voluntarily share on their own.
        """
        moves_done = 0
        max_moves = self.config.max_steps_per_cycle
        done_types: set = set()

        for action in actions:
            atype = action.type
            params = action.params

            # ---- MOVE -------------------------------------------------------
            if atype == ActionType.MOVE:
                if moves_done < max_moves:
                    self._do_move(agent, params.get("direction", "north"))
                    moves_done += 1
                continue

            # ---- All other types: once only ---------------------------------
            if atype in done_types:
                continue
            done_types.add(atype)

            # ---- COLLECT ----------------------------------------------------
            if atype == ActionType.COLLECT_FOOD:
                if agent.position:
                    r, c = agent.position
                    amount = params.get("amount", 5.0)
                    max_food = self.government.food_collection_limit(agent, cycle)
                    amount = min(amount, max_food)
                    taken = self.grid.collect_food(r, c, amount)
                    agent.food_stock += taken
                    self.logger.debug(
                        "cycle=%d agent=%s COLLECT_FOOD taken=%.3f food_stock=%.3f",
                        cycle, agent.agent_id, taken, agent.food_stock,
                    )

            elif atype == ActionType.COLLECT_WATER:
                if agent.position:
                    r, c = agent.position
                    amount = params.get("amount", 5.0)
                    max_water = self.government.water_collection_limit(agent, cycle)
                    amount = min(amount, max_water)
                    taken = self.grid.collect_water(r, c, amount)
                    agent.water_stock += taken
                    self.logger.debug(
                        "cycle=%d agent=%s COLLECT_WATER taken=%.3f water_stock=%.3f",
                        cycle, agent.agent_id, taken, agent.water_stock,
                    )

            elif atype == ActionType.COLLECT_MEDICINE:
                if agent.position:
                    r, c = agent.position
                    amount = params.get("amount", 2.0)
                    max_med = self.government.medicine_collection_limit(agent, cycle)
                    amount = min(amount, max_med)
                    taken = self.grid.collect_medicine(r, c, amount)
                    agent.medicine_stock += taken
                    self.logger.debug(
                        "cycle=%d agent=%s COLLECT_MEDICINE taken=%.3f medicine_stock=%.3f",
                        cycle, agent.agent_id, taken, agent.medicine_stock,
                    )

            # ---- EAT / DRINK ------------------------------------------------
            elif atype == ActionType.EAT:
                health_deficit = max(0.0, 1.0 - agent.health)
                needed = (health_deficit / self.eat_restore) if self.eat_restore > 0 else 0.0
                amount = min(needed, agent.food_stock)
                if amount > 0:
                    agent.food_stock -= amount
                    agent.health = min(1.0, agent.health + amount * self.eat_restore)
                    agent.hunger = max(0.0, agent.hunger - amount * 0.3)

            elif atype == ActionType.DRINK:
                health_deficit = max(0.0, 1.0 - agent.health)
                needed = (health_deficit / self.drink_restore) if self.drink_restore > 0 else 0.0
                amount = min(needed, agent.water_stock)
                if amount > 0:
                    agent.water_stock -= amount
                    agent.health = min(1.0, agent.health + amount * self.drink_restore)
                    agent.thirst = max(0.0, agent.thirst - amount * 0.3)

            # ---- TREAT SELF -------------------------------------------------
            elif atype == ActionType.TREAT_SELF:
                needed = agent.medicine_needed_to_cure()
                if needed > 0 and agent.medicine_stock >= needed:
                    agent.medicine_stock -= needed
                    agent.cure_all()
                elif needed > 0 and agent.medicine_stock > 0:
                    for eid in list(agent.epidemic_ids):
                        if agent.medicine_stock >= 3.0:
                            agent.medicine_stock -= 3.0
                            agent.epidemic_ids.discard(eid)
                        else:
                            break

            # ---- SHARE (law-directed only) -----------------------------------
            elif atype == ActionType.SHARE_FOOD:
                if not params.get("law_directed"):
                    continue
                self._do_share_resource(agent, params, "food")

            elif atype == ActionType.SHARE_WATER:
                if not params.get("law_directed"):
                    continue
                self._do_share_resource(agent, params, "water")

            elif atype == ActionType.SHARE_MEDICINE:
                if not params.get("law_directed"):
                    target = self.agent_map.get(params.get("target_id", ""))
                    if not target or not target.alive or not target.infected:
                        continue
                    if not self._adjacent(agent, target):
                        continue
                self._do_share_resource(agent, params, "medicine")

            elif atype == ActionType.SHARE_RESOURCE:
                if not params.get("law_directed"):
                    continue
                resource = params.get("resource", "food")
                self._do_share_resource(agent, params, resource)

            # ---- REST -------------------------------------------------------
            elif atype == ActionType.REST:
                if agent.position:
                    r, c = agent.position
                    if self.grid.cell(r, c).shelter:
                        agent.health = min(1.0, agent.health + 0.005)

            # ---- VOTE / PROPOSE ---------------------------------------------
            elif atype == ActionType.VOTE:
                self.government.receive_vote(agent, params.get("option"), cycle)

            elif atype == ActionType.PROPOSE:
                self.government.receive_proposal(agent, params.get("proposal"), cycle)

    def _do_share_resource(self, agent: Agent, params: dict, resource: str) -> None:
        """Transfer a resource from agent to one or more targets (law-directed)."""
        # Support single target_id or list of target_ids
        target_ids = params.get("target_ids") or []
        if not target_ids:
            single = params.get("target_id")
            if single:
                target_ids = [single]
        if not target_ids:
            return

        amount = params.get("amount", 0.0)
        fraction = params.get("fraction", 0.0)  # if > 0, share this fraction of stock

        stock_attr = f"{resource}_stock"
        total_stock = getattr(agent, stock_attr, 0.0)

        if fraction > 0:
            to_give = total_stock * fraction
        else:
            to_give = min(amount * len(target_ids), total_stock)

        if to_give <= 0:
            return

        per_target = to_give / len(target_ids)
        actually_given = 0.0
        for tid in target_ids:
            target = self.agent_map.get(tid)
            if target and target.alive:
                give = min(per_target, getattr(agent, stock_attr, 0.0) - actually_given)
                if give > 0:
                    setattr(target, stock_attr, getattr(target, stock_attr, 0.0) + give)
                    actually_given += give

        current = getattr(agent, stock_attr, 0.0)
        setattr(agent, stock_attr, max(0.0, current - actually_given))

    def _do_move(self, agent: Agent, direction: str) -> None:
        if not agent.position:
            return
        delta = DIRECTION_DELTAS.get(direction, (0, 0))
        r, c = agent.position
        nr, nc = r + delta[0], c + delta[1]
        if not self.grid.in_bounds(nr, nc):
            return
        if not self.government.can_move(agent, nr, nc, self.cycle):
            return
        agent.health -= 0.002  # movement cost
        self.grid.place_agent(agent, nr, nc)
        # Observation channel — see Simulation.last_move_counts.
        #
        # Counted HERE, after the relocation actually succeeded, and this
        # placement is load-bearing rather than incidental.  Two failure modes
        # are made IMPOSSIBLE rather than guarded:
        # a future mechanic that teleports an agent by calling
        # `grid.place_agent` directly never passes through this function and so
        # can never inflate the count; and a cycle-0 agent with `position is
        # None` returns at the top of this function and so generates no
        # observation at all.  Counting per cycle rather than diffing positions
        # across cycles is what buys both.
        self.last_move_counts[agent.agent_id] = (
            self.last_move_counts.get(agent.agent_id, 0) + 1
        )
        # Federation cross-region penalty
        from governments.federated import FederatedGovernment
        if isinstance(self.government, FederatedGovernment):
            self.government.apply_cross_region_penalty(agent, nr, nc)

    def _adjacent(self, a: Agent, b: Agent) -> bool:
        if not a.position or not b.position:
            return False
        return abs(a.position[0] - b.position[0]) <= 1 and abs(a.position[1] - b.position[1]) <= 1

    # ------------------------------------------------------------------
    # Health dynamics
    # ------------------------------------------------------------------

    def _apply_health_dynamics(self, cycle: int) -> None:
        # Opens this cycle's drain observation window.  Cleared at entry so the
        # dict never mixes two cycles' deltas; an agent that died on an earlier
        # cycle simply stops appearing.  See Simulation.last_health_drain.
        self.last_health_drain.clear()
        dm = self.drain_mult
        storm_damage = self.event_system.storm_damage_per_cycle()
        # Knee-folded difficulty; see effective_difficulty().  Uses the
        # single source of truth rather than an inline
        # `min(self.config.difficulty, 90)`, which would be another
        # open-coded copy of the knee.
        d_eff = effective_difficulty(self.config.difficulty)
        recovery_cycles = int(15 + (d_eff - 1) / 99 * 35)

        # Metabolic consumption rate per cycle (agents need food/water to survive).
        # Scales with difficulty so harder settings drain stocks faster, forcing
        # agents to collect from an increasingly scarce grid.
        metabolic = self.config.metabolic_rate

        for agent in self.agents:
            if not agent.alive:
                continue

            delta = 0.0

            # Metabolic drain: food and water consumed each cycle regardless of health
            agent.food_stock  = max(0.0, agent.food_stock  - metabolic)
            agent.water_stock = max(0.0, agent.water_stock - metabolic)

            # Hunger/thirst build-up
            if agent.food_stock <= 0.1:
                agent.hunger = min(1.0, agent.hunger + self.hunger_build_rate)
            else:
                agent.hunger = max(0.0, agent.hunger - 0.02)

            if agent.water_stock <= 0.1:
                agent.thirst = min(1.0, agent.thirst + self.thirst_build_rate)
            else:
                agent.thirst = max(0.0, agent.thirst - 0.02)

            # Health drains
            if agent.hunger > 0.4:
                delta -= self.hunger_drain * dm * agent.hunger

            if agent.thirst > 0.4:
                delta -= self.thirst_drain * dm * agent.thirst

            # Each active epidemic drains health independently
            if agent.epidemic_ids:
                delta -= self.disease_drain * dm * len(agent.epidemic_ids)

            # Terrain hazard
            if agent.position:
                r, c = agent.position
                cell = self.grid.cell(r, c)
                delta -= cell.hazard * dm

                # Storm damage (sheltered agents are protected)
                if storm_damage > 0 and not cell.shelter:
                    delta -= storm_damage * dm

                # Disease spread from contaminated cells
                if cell.contaminated:
                    for ev in self.event_system.active_events:
                        from .events import EventType
                        if ev.event_type == EventType.EPIDEMIC and ev.epidemic_id:
                            if ev.epidemic_id not in agent.epidemic_ids:
                                if self.rng.random() < 0.05:
                                    agent.infect(ev.epidemic_id)

            prev_health = agent.health
            agent.health = max(0.0, min(1.0, agent.health + delta))
            agent.age += 1

            # Observation channel — see Simulation.last_health_drain.
            #
            # Recorded AFTER the clip, so this is the realized loss rather than
            # `-delta`.  The two differ for an agent the drain would have taken
            # below 0.0 (the loss is truncated) or, in principle, above 1.0.
            # Recording the realized figure is deliberate: it is the quantity an
            # outside observer could measure, and publishing the uncensored
            # arithmetic value here would leak more than an observer can see.
            # The censoring is handled on the reader's side instead, by dropping
            # agents that died this cycle.
            self.last_health_drain[agent.agent_id] = prev_health - agent.health

            if abs(agent.health - prev_health) > 0.001:
                self.logger.debug(
                    "cycle=%d agent=%s health %.3f -> %.3f delta=%.3f "
                    "hunger=%.2f thirst=%.2f infected=%d",
                    cycle, agent.agent_id, prev_health, agent.health, delta,
                    agent.hunger, agent.thirst, len(agent.epidemic_ids),
                )

            # Natural recovery from infections over time
            if agent.epidemic_ids:
                agent.tick_infections(recovery_cycles)

            if agent.health <= 0.0:
                agent.alive = False
                agent.health = 0.0
                self.grid.remove_agent(agent)
                self.logger.debug(
                    "cycle=%d agent=%s DIED age=%d", cycle, agent.agent_id, agent.age
                )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _all_dead(self) -> bool:
        return not any(a.alive for a in self.agents)

    def living_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agent_map.get(agent_id)

    def mean_health(self) -> float:
        alive = self.living_agents()
        if not alive:
            return 0.0
        return sum(a.health for a in alive) / len(alive)

    def normalized_health_score(self) -> float:
        """Total health of remaining agents / initial population. Range [0, 1].
        Primary comparison metric across government types."""
        return sum(a.health for a in self.agents if a.alive) / max(1, len(self.agents))

    def objective_score(self) -> float:
        """Global objective O = 0.6*O_pop + 0.3*O_env + 0.1*O_sys"""
        alive = self.living_agents()
        o_pop = sum(a.health for a in alive) / max(1, len(self.agents))
        o_env = (self.grid.mean_food() + self.grid.mean_water()) / 200.0
        o_sys = 1.0  # placeholder; ADS government overrides
        return 0.6 * o_pop + 0.3 * o_env + 0.1 * o_sys
