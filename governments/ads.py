"""
ADS — Adaptive Decision Science Governance.

Architecture: flat tree under root:

  EvidenceNode (4 nodes)
    FoodEvidenceNode    — food distribution across agents and grid
    WaterEvidenceNode   — water distribution
    HealthEvidenceNode  — health / epidemic state
    TerrainEvidenceNode — terrain hazards and shelter availability

  HypothesisGeneratorNode (4 category HGs + 1 combiner + 1 final)
    Per-category HG divides agents into 10 risk bands (0 = safest, 9 = highest risk)
      and attaches a tag from that category.
    Combiner HG produces a single list of 10 groups, each group tagged by all categories.
    Final HG generates up to K candidate laws per group (one per category tag).

  EvaluatorNode (4 nodes, one per evidence category)
    Each evaluates the candidate laws for the groups its category produced.
    Uses a 20-cycle simplified lookahead (limited movement; see EvaluatorNode).

  Law enactment (inline in AdsGovernment._decision_round)
    The best-evaluated law per group is enacted, subject to the
    already-active-law-type and group-overlap filters.

If an event WARNING is active, evidence is gathered as if the event is already
taking place (i.e. the event's effects are pre-simulated into the evidence).

Multiple Group Distributions:
  The decision round evaluates every grouping scheme listed in
  GROUP_DISTRIBUTION_COUNTS (see Constants below — that list is the single
  source of truth; the schemes descend to 1 = the whole population).
  Scores are normalized per-agent so groupings are comparable apples-to-apples.
  The globally best (grouping, group, law) triple wins and is enacted.

Terrain-Aware Movement Heuristics:
  _find_resource_rich_regions() returns top-3 subregions by resource density.
  MOVE_TO_REGION proposals are weighted against active-event dangers and agent
  proximity to the target — closer agents get shorter movement durations.

Enhanced Trail Logs:
  get_log_info() reports per-group stats and ALL law scores for every grouping
  evaluated during the last decision round, stored in _last_decision_details.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Mapping, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING,
)

from .base import DEFAULT_GOVERNMENT_SEED, Government, Law
from .forecast_ledger import ForecastLedger
from .parameter_inference import (
    CRUDE_DRAIN_MULT,
    CRUDE_ENVIRONMENT,
    CRUDE_EVENT_RATE,
    CRUDE_MAX_STEPS_PER_CYCLE,
    CRUDE_REGEN_MULT,
    DRAIN_COEF_DISEASE,
    DRAIN_COEF_HAZARD,
    DRAIN_COEF_HUNGER,
    DRAIN_COEF_STORM,
    DRAIN_COEF_THIRST,
    DRAIN_MULT_MAX,
    DRAIN_MULT_MIN,
    DRAIN_NEED_THRESHOLD,
    ESTIMATOR_PRIOR_CYCLES,
    REGEN_MULT_MAX,
    REGEN_MULT_MIN,
    REGEN_OBS_MAX_FILL,
    EnvironmentModel,
    ParameterEstimator,
)
from engine.events import (
    EVENT_RATE_CAP,
    EVENT_RATE_PRIOR_CYCLES,
    OBSERVED_EVENT_CATEGORIES,
    EventType,
)
from engine.grid import FOOD_CAP, MED_CAP, WATER_CAP
from engine.scenario_plan import derive_seed

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning
    from engine.grid import Grid
    from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Length of the evaluator's fake forward rollout, in cycles.  FIXED, and the
#: same number for every candidate and for both regimes that use the evaluator.
#:
#: The value is load-bearing in three places at once and they must not be
#: allowed to drift apart:
#:   1. the RANKING horizon — every candidate is rolled exactly this far, which
#:      is what makes their scores commensurable.  A per-candidate horizon (for
#:      instance ``min(law.duration, 30)``) would score a short law over fewer
#:      cycles of monotonic health decay and hand it an automatic win;
#:      ``autocracy_lookahead``'s ``NO_ACTION`` candidate carries ``duration=0``
#:      and would beat every acting candidate in every menu, silently collapsing
#:      that arm into plain autocracy.
#:   2. the PREDICTION TARGET — the projected value at this horizon is what
#:      ``_close_predictions`` grades, which is why
#:   3. the REVIEW DELAY — a prediction opened at cycle ``t`` closes at
#:      ``t + LOOKAHEAD_CYCLES``, not when the originating law happens to expire.
#: Changing this constant changes every number in the corpus.
LOOKAHEAD_CYCLES = 20

MIN_GROUP_FRACTION = 0.10  # a law must apply to at least 10% of surviving agents

#: Event categories the rollout stochastically injects, in draw order.
#:
#: ORDER IS LOAD-BEARING.  Exactly one uniform is drawn per category per fake
#: cycle, unconditionally and in this order, so a stream position is a function
#: of ``(fake_cycle, category_index)`` and of nothing else — not of the rate,
#: the law type, the group size or the horizon.  That is what lets two
#: candidates scored in the same decision round face the bit-identical phantom
#: future, which is the whole point of the dedicated injection stream.
#: Reordering this tuple changes every number in the corpus.
PHANTOM_CATEGORIES: Tuple[str, ...] = OBSERVED_EVENT_CATEGORIES

#: Prefix for synthetic epidemic identifiers minted during a rollout.  Chosen so
#: that ``sorted()`` places them after the engine's real ``epi%04d`` ids: ``_step``
#: iterates the id tuple while drawing from ``_eval_rng``, so iteration order
#: decides which epidemic gets which draw, and the order must be a function of
#: the identifiers rather than of a set's hash layout.
PHANTOM_EPIDEMIC_PREFIX = "phi"

# --- Forecast accuracy (the feedback pass described in ADS_JPART_main.tex §2.6) ---
#
# The loop: ``_record_prediction`` (at enactment) -> ``_close_predictions`` (at
# enactment + LOOKAHEAD_CYCLES) -> the reported accuracy series.  That is the
# WHOLE loop.  Note what is missing from it: there is no step that corrects the
# forecast.
#
# NOTHING DOWNSTREAM OF ``evaluate()`` MAY SCALE, CLIP OR OTHERWISE POST-PROCESS
# ITS RETURN VALUE.  This is a central invariant and the first thing a reader
# should check.  There is exactly ONE number and two consumers read it
# unchanged:
#
#     adjusted_score = norm_score * crisis_boost      # what candidates rank on
#     predicted-for-grading = norm_score              # what _close_predictions grades
#     reported error = abs(norm_score - realized)     # what the figures plot
#
# Two mechanisms are deliberately absent from this design; understanding why
# matters before a third is proposed.  An ADDITIVE ranking nudge
# (``norm_score + 0.3 * bias``) would never touch the graded prediction, so
# the archived figures would measure the fidelity of a forecast the feedback
# loop could not affect.  A MULTIPLICATIVE EWMA per ``(category, law_type)``
# that feeds ranking directly is measurably WORSE than the null in 12 of 12
# run-cells, for a structural reason: a multiplicative correction cannot serve a
# quantity pinned near its own ceiling.  Within one bucket the projection is
# badly pessimistic for small sick groups and exactly right for healthy ones,
# and one scalar cannot represent both; it learns the large upward correction
# the first case demands and then applies it to the second, turning a perfect
# forecast of 1.000 into a prediction of 1.63.
#
# THE REPLACEMENT IS TO CORRECT THE MODEL, NEVER THE ANSWER.  Accuracy now comes
# from feeding the rollout better INPUTS — the environment-dynamics parameters
# in ``governments/parameter_inference.py`` — and the output is reported as the
# evaluator produced it.  If a future change wants the forecast to be more
# accurate, the lever is the parameter set going in.

#: Identifies the SEMANTICS of the forecast-accuracy fields in
#: ``final_stats.json`` and ``run_detail.jsonl``.
#:
#: This REPLACES ``calibration_schema_version`` rather than bumping it, because
#: the field's meaning changed entirely: there is no calibration table, no
#: multiplier and no correction left for that name to describe, and a key called
#: ``calibration_schema_version`` in every future archive would mislead every
#: future reader indefinitely.  The integer continues the old sequence rather
#: than restarting, so no value is ever reused across the two names and a
#: consumer can order all generations on the number alone:
#:
#:   neither key present            earliest: additive ranking nudge; the
#:                                  graded forecast was untouched by the loop
#:   calibration_schema_version: 3  multiplicative (category, law_type) EWMA;
#:                                  calibration_cycle_deltas =
#:                                  abs(c * raw - realized)
#:   ads_forecast_schema_version: 4 no output correction of any kind; the
#:                                  forecast is the evaluator's own projection
#:                                  under INFERRED input parameters;
#:                                  calibration_cycle_deltas =
#:                                  abs(predicted - realized)
#:
#: A v3 archive stays unambiguously readable because it carries the old key and
#: not this one.  Nothing branches on either value; both are markers for human
#: and future-tool consumption.
#:
#: THE ``ads_`` PREFIX IS VESTIGIAL: the field is not ADS-specific.
#: ``autocracy_lookahead`` emits the same key with the same value, because both
#: arms compute ``abs(predicted - realized)`` through literally the same code
#: (``governments/forecast_ledger.py``) — provenance is a property of the RUN,
#: not of the regime.  It is NOT renamed: the generations above are a
#: human-readable signature table, and renaming the token invalidates the table
#: for no runtime gain, since nothing branches on it.
#:
#: This does NOT bump the version to 5 for ``n_closed_no_law`` and the two
#: ``*_by_outcome_then_scope`` partitions: those are partitions of the SAME
#: measurement, not a new one; the forecast is still the evaluator's own
#: projection under inferred inputs and ``calibration_cycle_deltas`` is still
#: ``abs(predicted - realized)``.
ADS_FORECAST_SCHEMA_VERSION = 4

# Grouping schemes evaluated each decision round (descending, 1=whole population).
# This list is the single source of truth for both the decision round and the trail log —
# add or remove a scheme here and every count/label follows automatically.
GROUP_DISTRIBUTION_COUNTS = [10, 5, 4, 3, 2, 1]

# Subregion scan window for resource-rich region detection.
# The scan window is square, so this value is used for both rows and columns.
RESOURCE_REGION_ROWS = 4
TOP_K_REGIONS = 3          # number of top subregions returned by _find_resource_rich_regions


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

# NOTE: several evidence fields below (pct_starving, pct_dehydrated, cycles_until_empty,
# mean_hazard, avg_dist_to_shelter, shelter_cell_count) are computed but not yet consumed by
# any heuristic — they are intentionally retained as inputs to the structured decision log.

@dataclass
class FoodEvidence:
    mean_stock: float
    pct_critical: float       # fraction of agents with food_stock < 3
    pct_starving: float       # fraction with food_stock < 1 (acute starvation)
    grid_mean: float
    grid_total: float         # total food on grid right now
    cycles_until_empty: float # estimated cycles until grid food exhausted at current consumption
    drought_active: bool
    warning_drought: bool

@dataclass
class WaterEvidence:
    mean_stock: float
    pct_critical: float       # fraction with water_stock < 3
    pct_dehydrated: float     # fraction with water_stock < 1
    grid_mean: float
    grid_total: float
    cycles_until_empty: float
    warning_toxic: bool
    toxic_active: bool
    toxic_region: Optional[Tuple[int, int, int, int]]  # bounding box of toxic spill

@dataclass
class HealthEvidence:
    mean_health: float
    pct_critical: float        # fraction with health < 0.4
    pct_low_medicine: float    # fraction with medicine_stock < 3.0 (can't cure one epidemic)
    infected_fraction: float
    epidemic_ids: Set[str]
    warning_epidemic: bool
    infected_centroid: Optional[Tuple[float, float]]  # (row, col) centroid of infected agents
    infected_spread_radius: float                      # typical spatial spread of infected
    quarantine_feasible: bool  # infected agents can be bounded in a region < 30% of grid

@dataclass
class TerrainEvidence:
    mean_hazard: float
    pct_on_shelter: float
    pct_near_shelter: float    # fraction of agents within 3 cells of any shelter
    avg_dist_to_shelter: float # average Manhattan distance to nearest shelter cell
    shelter_cell_count: int    # total shelter cells on grid
    storm_active: bool
    storm_severity: float      # storm_damage from active storm event
    toxic_active: bool
    warning_storm: bool


def _find_resource_rich_regions(
    sim: "Simulation",
    resource_type: str = "combined",
    min_size: int = 4,
    top_k: int = TOP_K_REGIONS,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Scan the grid for the top-k rows×cols sub-regions richest in `resource_type`.

    Returns a list of up to `top_k` tuples (r_min, c_min, r_max, c_max, total)
    sorted descending by total resource in the subregion.

    Args:
        sim:           The live Simulation instance.
        resource_type: One of "food", "water", "medicine", or "combined"
                       (food + water, the default).
        min_size:      Side length of the square subregion to scan (default 4).
        top_k:         Maximum number of distinct regions to return.

    Returns:
        List of (r_min, c_min, r_max, c_max, total_resource) tuples, highest first.
        The returned regions are non-overlapping by majority: each new region must
        not share its centroid cell with any already-selected region (greedy exclusion).
    """
    grid = sim.grid
    rows = min_size
    cols = min_size
    step = max(1, min_size // 2)

    # First pass: score every candidate subregion
    candidates: List[Tuple[float, int, int]] = []
    for r in range(0, grid.rows - rows + 1, step):
        for c in range(0, grid.cols - cols + 1, step):
            score = 0.0
            for dr in range(rows):
                for dc in range(cols):
                    cell = grid.cell(r + dr, c + dc)
                    if resource_type == "food":
                        score += cell.food
                    elif resource_type == "water":
                        score += cell.water
                    elif resource_type == "medicine":
                        score += getattr(cell, "medicine", 0.0)
                    else:  # "combined" — food + water
                        score += cell.food + cell.water
            candidates.append((score, r, c))

    # Sort descending by score
    candidates.sort(key=lambda t: t[0], reverse=True)

    # Greedy non-overlapping selection: skip candidates whose centroid is within
    # min_size cells of an already-selected centroid
    selected: List[Tuple[int, int, int, int, float]] = []
    selected_centroids: List[Tuple[float, float]] = []
    for score, r, c in candidates:
        if len(selected) >= top_k:
            break
        cr = r + rows / 2.0
        cc = c + cols / 2.0
        # Check Manhattan distance to already-selected centroids
        too_close = any(
            abs(cr - sr) + abs(cc - sc) < min_size
            for sr, sc in selected_centroids
        )
        if not too_close:
            selected.append((r, c, r + rows - 1, c + cols - 1, score))
            selected_centroids.append((cr, cc))

    return selected


def _movement_duration(
    agents: List["Agent"],
    target_region: Tuple[int, int, int, int],
    base: int = 10,
    min_dur: int = 4,
) -> int:
    """
    Compute a movement law duration that accounts for how far the agent
    group is from the target region.

    If agents are already close to the target (median Manhattan distance to the
    nearest corner of the region is small relative to `base`), the duration is
    shortened proportionally.  The floor is `min_dur` cycles — enough for agents
    to finish moving even if they started right at the border.

    Args:
        agents:        The agents in the group (only those with a position count).
        target_region: (r_min, c_min, r_max, c_max) bounding box of the target.
        base:          Default duration when agents are far from the target.
        min_dur:       Minimum duration — never go below this.

    Returns:
        An integer duration in cycles.
    """
    r_min, c_min, r_max, c_max = target_region
    positioned = [a for a in agents if a.position]
    if not positioned:
        return base

    # Manhattan distance from each agent to the nearest point inside the region
    distances = []
    for a in positioned:
        ar, ac = a.position
        # Clamp agent position to region boundary to get nearest-point distance
        nearest_r = max(r_min, min(r_max, ar))
        nearest_c = max(c_min, min(c_max, ac))
        distances.append(abs(ar - nearest_r) + abs(ac - nearest_c))

    # Use median distance as the representative measure (robust to outliers)
    distances.sort()
    median_dist = distances[len(distances) // 2]

    # Scale duration: assume 1 step per cycle, add 2 buffer cycles
    scaled = median_dist + 2
    return max(min_dur, min(base, scaled))


def _nearest_shelter_stats(
    alive: List["Agent"], sim: "Simulation"
) -> Tuple[float, float, int]:
    """Return (pct_near_shelter, avg_dist_to_shelter, shelter_cell_count)."""
    # Build list of shelter cells once
    shelter_cells = []
    for r in range(sim.grid.rows):
        for c in range(sim.grid.cols):
            if sim.grid.cell(r, c).shelter:
                shelter_cells.append((r, c))
    shelter_count = len(shelter_cells)
    if not shelter_cells or not alive:
        return 0.0, float("inf"), shelter_count

    near_count = 0
    total_dist = 0.0
    near_threshold = 3
    for a in alive:
        if not a.position:
            continue
        ar, ac = a.position
        min_dist = min(abs(ar - sr) + abs(ac - sc) for sr, sc in shelter_cells)
        total_dist += min_dist
        if min_dist <= near_threshold:
            near_count += 1
    n = len([a for a in alive if a.position]) or 1
    return near_count / n, total_dist / n, shelter_count


class FoodEvidenceNode:
    def collect(self, sim: "Simulation", pending_warnings: List["EventWarning"]) -> FoodEvidence:
        alive = [a for a in sim.agents if a.alive]
        n = len(alive) or 1
        stocks = [a.food_stock for a in alive]
        mean_stock = sum(stocks) / n
        pct_critical = sum(1 for s in stocks if s < 3.0) / n
        pct_starving  = sum(1 for s in stocks if s < 1.0) / n
        grid_mean = sim.grid.mean_food()
        grid_total = grid_mean * sim.grid.rows * sim.grid.cols
        # Estimate cycles until empty: total grid / (agents * mean_consumption_per_cycle)
        mean_consumption = 2.0  # approximate eat + stock depletion per cycle
        cycles_est = grid_total / max(0.1, n * mean_consumption)
        drought = any(
            e.event_type.value == "drought"
            for e in sim.event_system.active_events if not e.cancelled
        )
        warn_drought = any(w.event_type.value == "drought" for w in pending_warnings)
        return FoodEvidence(
            mean_stock=mean_stock,
            pct_critical=pct_critical,
            pct_starving=pct_starving,
            grid_mean=grid_mean,
            grid_total=grid_total,
            cycles_until_empty=cycles_est,
            drought_active=drought,
            warning_drought=warn_drought,
        )


class WaterEvidenceNode:
    def collect(self, sim: "Simulation", pending_warnings: List["EventWarning"]) -> WaterEvidence:
        alive = [a for a in sim.agents if a.alive]
        n = len(alive) or 1
        stocks = [a.water_stock for a in alive]
        mean_stock = sum(stocks) / n
        pct_critical = sum(1 for s in stocks if s < 3.0) / n
        pct_dehydrated = sum(1 for s in stocks if s < 1.0) / n
        grid_mean = sim.grid.mean_water()
        grid_total = grid_mean * sim.grid.rows * sim.grid.cols
        mean_consumption = 2.0
        cycles_est = grid_total / max(0.1, n * mean_consumption)
        warn_toxic = any(w.event_type.value == "toxic_spill" for w in pending_warnings)
        toxic_active = any(
            e.event_type.value == "toxic_spill"
            for e in sim.event_system.active_events if not e.cancelled
        )
        # Estimate bounding box of toxic spill from hazard_extra on cells
        toxic_region = None
        if toxic_active:
            toxic_cells = [
                (r, c)
                for r in range(sim.grid.rows)
                for c in range(sim.grid.cols)
                if getattr(sim.grid.cell(r, c), "hazard_extra", 0.0) > 0
            ]
            if toxic_cells:
                rs = [p[0] for p in toxic_cells]
                cs = [p[1] for p in toxic_cells]
                toxic_region = (min(rs), min(cs), max(rs), max(cs))
        return WaterEvidence(
            mean_stock=mean_stock,
            pct_critical=pct_critical,
            pct_dehydrated=pct_dehydrated,
            grid_mean=grid_mean,
            grid_total=grid_total,
            cycles_until_empty=cycles_est,
            warning_toxic=warn_toxic,
            toxic_active=toxic_active,
            toxic_region=toxic_region,
        )


class HealthEvidenceNode:
    def collect(self, sim: "Simulation", pending_warnings: List["EventWarning"]) -> HealthEvidence:
        alive = [a for a in sim.agents if a.alive]
        n = len(alive) or 1
        healths = [a.health for a in alive]
        mean_health = sum(healths) / n
        pct_critical = sum(1 for h in healths if h < 0.4) / n
        pct_low_med = sum(1 for a in alive if a.medicine_stock < 3.0) / n
        infected_agents = [a for a in alive if a.infected]
        infected_frac = len(infected_agents) / n
        epidemic_ids: Set[str] = set()
        for a in infected_agents:
            epidemic_ids |= a.epidemic_ids
        warn_epidemic = any(w.event_type.value == "epidemic" for w in pending_warnings)

        # Spatial statistics of infected agents
        infected_centroid = None
        infected_spread_radius = 0.0
        quarantine_feasible = False
        if infected_agents:
            pos = [a.position for a in infected_agents if a.position]
            if pos:
                cr = sum(p[0] for p in pos) / len(pos)
                cc = sum(p[1] for p in pos) / len(pos)
                infected_centroid = (cr, cc)
                # Spread radius = average Manhattan distance from centroid
                infected_spread_radius = sum(
                    abs(p[0] - cr) + abs(p[1] - cc) for p in pos
                ) / len(pos)
                # Quarantine is feasible if bounding box < 30% of grid area
                r_span = max(p[0] for p in pos) - min(p[0] for p in pos) + 1
                c_span = max(p[1] for p in pos) - min(p[1] for p in pos) + 1
                grid_area = sim.grid.rows * sim.grid.cols
                quarantine_feasible = (r_span * c_span) < (0.30 * grid_area)

        return HealthEvidence(
            mean_health=mean_health,
            pct_critical=pct_critical,
            pct_low_medicine=pct_low_med,
            infected_fraction=infected_frac,
            epidemic_ids=epidemic_ids,
            warning_epidemic=warn_epidemic,
            infected_centroid=infected_centroid,
            infected_spread_radius=infected_spread_radius,
            quarantine_feasible=quarantine_feasible,
        )


class TerrainEvidenceNode:
    def collect(self, sim: "Simulation", pending_warnings: List["EventWarning"]) -> TerrainEvidence:
        alive = [a for a in sim.agents if a.alive]
        n = len(alive) or 1
        hazards = []
        on_shelter = 0
        for a in alive:
            if a.position and sim.grid.in_bounds(*a.position):
                cell = sim.grid.cell(*a.position)
                hazards.append(cell.hazard + getattr(cell, "hazard_extra", 0.0))
                if cell.shelter:
                    on_shelter += 1
        mean_hazard = sum(hazards) / n if hazards else 0.0
        pct_shelter = on_shelter / n

        pct_near, avg_dist, shelter_count = _nearest_shelter_stats(alive, sim)

        storm_active = any(
            e.event_type.value == "storm"
            for e in sim.event_system.active_events if not e.cancelled
        )
        storm_severity = 0.0
        if storm_active:
            for e in sim.event_system.active_events:
                if e.event_type.value == "storm" and not e.cancelled:
                    storm_severity = max(storm_severity, getattr(e, "storm_damage", 0.0))

        toxic_active = any(
            e.event_type.value == "toxic_spill"
            for e in sim.event_system.active_events if not e.cancelled
        )
        warn_storm = any(w.event_type.value == "storm" for w in pending_warnings)

        return TerrainEvidence(
            mean_hazard=mean_hazard,
            pct_on_shelter=pct_shelter,
            pct_near_shelter=pct_near,
            avg_dist_to_shelter=avg_dist,
            shelter_cell_count=shelter_count,
            storm_active=storm_active,
            storm_severity=storm_severity,
            toxic_active=toxic_active,
            warning_storm=warn_storm,
        )


# ---------------------------------------------------------------------------
# Risk-band tagging per category
# ---------------------------------------------------------------------------

@dataclass
class AgentTag:
    """One tag from one evidence category describing an agent's risk band."""
    category: str      # "food" | "water" | "health" | "terrain"
    band: int          # 0 = safest, 9 = highest risk


def _food_tag(agent: "Agent", ev: FoodEvidence) -> AgentTag:
    s = agent.food_stock
    if ev.drought_active or ev.warning_drought:
        # During/before drought, low food is more dangerous
        band = max(0, 9 - min(9, int(s / 2)))
    else:
        band = max(0, 9 - min(9, int(s / 3)))
    return AgentTag("food", band)


def _water_tag(agent: "Agent", ev: WaterEvidence) -> AgentTag:
    s = agent.water_stock
    band = max(0, 9 - min(9, int(s / 3)))
    return AgentTag("water", band)


def _health_tag(agent: "Agent", ev: HealthEvidence) -> AgentTag:
    if agent.infected:
        band = 9
    else:
        band = max(0, int((1.0 - agent.health) * 10))
    return AgentTag("health", band)


def _terrain_tag(agent: "Agent", ev: TerrainEvidence, sim: "Simulation") -> AgentTag:
    if not agent.position or not sim.grid.in_bounds(*agent.position):
        return AgentTag("terrain", 0)
    cell = sim.grid.cell(*agent.position)
    hazard = cell.hazard
    if ev.storm_active and not cell.shelter:
        band = 9
    elif ev.toxic_active and hazard > 0.1:
        band = 8
    else:
        band = min(9, int(hazard * 40))
    return AgentTag("terrain", band)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------

@dataclass
class AgentGroup:
    """A subset of agents sharing similar risk profiles — the unit a law applies to."""
    group_id: int
    agent_ids: List[str]
    tags: List[AgentTag]            # one tag per evidence category


@dataclass
class ProposedLaw:
    law_type: str
    params: Dict[str, Any]
    duration: int
    description: str
    source_category: str    # which evidence category proposed this law
    evaluation_score: float = 0.0
    applies_to: Optional[List[str]] = None  # override for specific agent subsets


# ---------------------------------------------------------------------------
# Shared rationing arithmetic
# ---------------------------------------------------------------------------

def survival_floor(drain_mult: float) -> float:
    """
    Minimum FOOD_RATION cap that still lets an agent meet its hunger drain.

    ``DRAIN_COEF_HUNGER`` health/cycle is lost to hunger and eating restores
    0.05 health per food unit, so an agent needs
    ``DRAIN_COEF_HUNGER * drain_mult / 0.05`` food units per cycle.  Agents
    reach a food cell roughly every 3 cycles, so one collection event must
    carry three cycles' worth.  The result is clamped to ``[3.0, 5.0]``: 5.0 is
    the unrationed cap, so a floor above it would make the "ration" a licence
    to eat *more*, and below 3.0 the ration starves the population it exists to
    protect.

    WHY THIS IS A FUNCTION AND NOT TWO COPIES OF THREE LINES.  This quantity was
    once computed independently in two places — the ecological ration in
    :meth:`HypothesisGeneratorNode.propose_laws` and the depletion-aware
    rationing in :class:`AdsGovernment` — and the two drifted: one used the
    inferred ``drain_mult`` while the other went on reconstructing ground truth
    from ``sim.config.difficulty``.  ADS then computed the same rule two ways,
    disagreeing by up to ~14% at
    difficulty 100, and the epistemic firewall was false on the production
    sweep shape.  A single definition makes that class of drift impossible
    rather than merely unlikely, and gives the firewall one line to guard
    instead of a family of look-alike call sites.

    ``drain_mult`` is always the government's ESTIMATE, never ``sim.drain_mult``
    — see ``parameter_inference`` on why the estimand is the multiplier that
    reproduces the evaluator's own model rather than the config constant.
    """
    min_food_rate = DRAIN_COEF_HUNGER * float(drain_mult) / 0.05
    return max(3.0, min(5.0, min_food_rate * 3.0))


# ---------------------------------------------------------------------------
# HypothesisGeneratorNode
# ---------------------------------------------------------------------------

class HypothesisGeneratorNode:
    """
    Given evidence from one category, assign each agent a risk band (0–9),
    return a sorted list of (agent, tag) pairs.
    """

    def generate_tags(
        self,
        agents: List["Agent"],
        food_ev: FoodEvidence,
        water_ev: WaterEvidence,
        health_ev: HealthEvidence,
        terrain_ev: TerrainEvidence,
        sim: "Simulation",
    ) -> Dict[str, List[AgentTag]]:
        """Return {agent_id: [food_tag, water_tag, health_tag, terrain_tag]}."""
        result: Dict[str, List[AgentTag]] = {}
        for agent in agents:
            result[agent.agent_id] = [
                _food_tag(agent, food_ev),
                _water_tag(agent, water_ev),
                _health_tag(agent, health_ev),
                _terrain_tag(agent, terrain_ev, sim),
            ]
        return result

    def form_groups(
        self,
        agents: List["Agent"],
        tags: Dict[str, List[AgentTag]],
        n_groups: int,
    ) -> List[AgentGroup]:
        """
        Sort agents by combined risk score (sum of band values),
        split into n_groups groups.  Each group's tags are the modal tags.
        """
        if not agents:
            return []
        scored = sorted(
            agents,
            key=lambda a: sum(t.band for t in tags.get(a.agent_id, [])),
        )
        n = len(scored)
        group_size = max(1, n // n_groups)
        groups: List[AgentGroup] = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else n
            members = scored[start:end]
            if not members:
                continue
            member_ids = [a.agent_id for a in members]
            # Representative tags: use the median agent's tags
            mid = members[len(members) // 2]
            group_tags = tags.get(mid.agent_id, [])
            groups.append(AgentGroup(group_id=i, agent_ids=member_ids, tags=group_tags))
        return groups

    def propose_laws(
        self,
        group: AgentGroup,
        food_ev: FoodEvidence,
        water_ev: WaterEvidence,
        health_ev: HealthEvidence,
        terrain_ev: TerrainEvidence,
        sim: "Simulation",
        *,
        env: EnvironmentModel,
    ) -> List[ProposedLaw]:
        """
        Generate candidate laws for this group using evidence-driven heuristics.

        Each category (food, water, health, terrain) contributes one proposal.
        Law type selection and parameter computation are both driven by evidence
        statistics — not hand-coded thresholds alone.

        ``env`` is the round's frozen :class:`EnvironmentModel` — the same object
        every candidate is scored against — and it is the ONLY route by which
        this node may learn anything about the environment's hidden dynamics.
        It is REQUIRED and keyword-only on purpose.  A default would let a
        future call site silently fall back to the crude model, which is the
        quiet-wrong failure: proposals would still be generated, the suite would
        still pass, and the arm would have reverted to uninformed rationing with
        nothing in the output to say so.  Missing it is a TypeError instead.

        Movement proposals (MOVE_TO_REGION) are terrain-aware:
          - _find_resource_rich_regions() identifies the top-3 subregions per resource.
          - Movement is only proposed when it outweighs staying put (i.e. no active
            events make movement unusually dangerous, or the benefit is large enough).
          - Duration is shortened when agents are already close to the target region.
        """
        proposals: List[ProposedLaw] = []
        all_alive = [a for a in sim.agents if a.alive]
        n_total = len(all_alive)
        g_ids = set(group.agent_ids)
        g_agents = [a for a in all_alive if a.agent_id in g_ids]
        K = len(g_agents)

        # Danger multiplier: active events that make movement costly
        # Storm and toxic spill create significant hazard during transit
        movement_dangerous = terrain_ev.storm_active or terrain_ev.toxic_active
        # Epidemic active makes group movement a spread vector
        epidemic_active = health_ev.infected_fraction > 0.05

        # ---- Food category --------------------------------------------------
        food_band = next((t.band for t in group.tags if t.category == "food"), 0)

        # Ecological depletion check: propose a proactive ration before exponential
        # threshold is crossed.  Only fires when agents are not in immediate crisis
        # (food_band < 6) so it doesn't conflict with emergency redistribution.
        _eco_ration_proposed = False
        if food_band < 6:
            grid = sim.grid
            if grid.initial_grid_food > 0:
                from engine.grid import _DEPLETION_THRESHOLD
                food_fraction = grid.food_depletion_fraction
                eco_warning   = _DEPLETION_THRESHOLD * 0.72
                if food_fraction >= eco_warning:
                    # Graduated percentage reduction — same formula as _apply_depletion_aware_rationing
                    if food_fraction < _DEPLETION_THRESHOLD:
                        t       = (food_fraction - eco_warning) / (_DEPLETION_THRESHOLD - eco_warning)
                        red     = 0.20 + t * 0.15
                    else:
                        excess  = food_fraction - _DEPLETION_THRESHOLD
                        red     = 0.35 + min(0.15, excess * 2.0)
                    # Drain-scaled floor to prevent starvation at higher drain.
                    #
                    # NOT `difficulty_multiplier(sim.config.difficulty)`, which
                    # would reconstruct `sim.drain_mult` EXACTLY (simulation.py
                    # sets it from the same call on the same argument) and
                    # smuggle ground truth into a proposal heuristic instead of
                    # going through the evaluator's own estimate.
                    #
                    # This branch is not latent.  It is gated on
                    # `food_depletion_fraction >= _DEPLETION_THRESHOLD * 0.72`,
                    # which the test suite never reaches (peak 0.0489) but the
                    # production sweep does — at D100, 500 agents, 50x50, the
                    # gate opens at cycle 128 and this line executes.  A
                    # ground-truth leak here would be invisible to every test
                    # while corrupting the published corpus at the
                    # difficulties where it matters most.  A dynamic tripwire
                    # cannot catch what it does not execute; the static AST
                    # scan in `test_ads_calibration.py` is the guard that can.
                    #
                    # `env.drain_mult` is the round's inferred estimate, shared
                    # by reference with every candidate scored this round.
                    _floor = survival_floor(env.drain_mult)
                    eco_cap = max(_floor, 5.0 * (1.0 - red))
                    proposals.append(ProposedLaw(
                        law_type="FOOD_RATION",
                        params={"max_per_cycle": round(eco_cap, 2)},
                        duration=12,
                        description=(
                            f"[ADS-ECO] Ecological ration proposal for group {group.group_id}: "
                            f"depletion={food_fraction:.2f}, cap={eco_cap:.1f}"
                        ),
                        source_category="food",
                    ))
                    _eco_ration_proposed = True

        # Choose GENERIC vs FOOD-specific: use generic when food AND water are
        # both critical (avoids two separate laws on same group)
        water_band_grp = next((t.band for t in group.tags if t.category == "water"), 0)
        both_critical = food_band >= 6 and water_band_grp >= 6

        if _eco_ration_proposed:
            pass  # ecological ration already proposed; skip normal food logic for this group
        elif both_critical:
            # REDISTRIBUTE_GENERIC — covers food + water (and medicine if needed)
            g_avg_food  = sum(a.food_stock  for a in g_agents) / max(1, K)
            g_avg_water = sum(a.water_stock for a in g_agents) / max(1, K)
            g_avg_med   = sum(a.medicine_stock for a in g_agents) / max(1, K)
            pct_food_target  = 0.5   # % of initial stock (10) to deliver
            pct_water_target = 0.5
            pct_med_target   = 0.3
            proposals.append(ProposedLaw(
                law_type="REDISTRIBUTE_GENERIC",
                params={
                    "recipient_ids": group.agent_ids,
                    "pct_food":    pct_food_target,
                    "pct_water":   pct_water_target,
                    "pct_medicine": pct_med_target,
                    # amount_per_recipient per resource resolved at enactment
                    "amount_per_recipient_food":    max(2.0, 10.0 * pct_food_target  - g_avg_food),
                    "amount_per_recipient_water":   max(2.0, 10.0 * pct_water_target - g_avg_water),
                    "amount_per_recipient_medicine":max(1.0, 10.0 * pct_med_target   - g_avg_med),
                },
                duration=15,
                description=(f"[ADS] Generic redistribution to group {group.group_id} "
                             f"(food+water both critical)"),
                source_category="food",
            ))
            # NOTE: No additional REDISTRIBUTE_FOOD/WATER fallback here — stacking
            # multiple redistribution laws rapidly drains donors.  The continuous
            # equalization in _fast_response handles the residual need.
        elif food_band >= 6:
            # REDISTRIBUTE_FOOD: N = amount needed to bring group above 5-unit threshold
            g_avg_food = sum(a.food_stock for a in g_agents) / max(1, K)
            n_food = max(2.0, 5.0 - g_avg_food)
            # If grid has very little food left, prefer terrain-aware MOVE_TO_REGION
            # — but only when movement is not actively dangerous.
            food_regions = _find_resource_rich_regions(sim, resource_type="food",
                                                       min_size=RESOURCE_REGION_ROWS)
            grid_food_scarce = food_ev.grid_total < (n_total * 3.0)
            if grid_food_scarce and food_regions and not movement_dangerous and not epidemic_active:
                target_region_data = food_regions[0]
                target_region = target_region_data[:4]  # (r_min,c_min,r_max,c_max)
                region_total = target_region_data[4]
                # Duration: shorter if agents are already close to the target region
                duration = _movement_duration(g_agents, target_region, base=10, min_dur=4)
                proposals.append(ProposedLaw(
                    law_type="MOVE_TO_REGION",
                    params={"region": target_region},
                    duration=duration,
                    description=(
                        f"[ADS] Move food-critical group {group.group_id} to "
                        f"food-rich region {target_region} (total={region_total:.1f})"
                    ),
                    source_category="food",
                ))
            elif grid_food_scarce and food_regions and (movement_dangerous or epidemic_active):
                # Movement is risky — fall back to redistribution even if food is scarce
                proposals.append(ProposedLaw(
                    law_type="REDISTRIBUTE_FOOD",
                    params={"resource": "food", "recipient_ids": group.agent_ids,
                            "amount_per_recipient": n_food},
                    duration=15,
                    description=(f"[ADS] Redistribute food to group {group.group_id} "
                                 f"(move suppressed: dangerous conditions, band={food_band})"),
                    source_category="food",
                ))
            else:
                proposals.append(ProposedLaw(
                    law_type="REDISTRIBUTE_FOOD",
                    params={"resource": "food", "recipient_ids": group.agent_ids,
                            "amount_per_recipient": n_food},
                    duration=15,
                    description=(f"[ADS] Redistribute food to group {group.group_id} "
                                 f"(band={food_band}, N={n_food:.1f})"),
                    source_category="food",
                ))
        elif food_ev.drought_active or food_ev.warning_drought:
            # Ration amount calibrated to expected depletion: cap at amount that keeps
            # grid food lasting at least 20 more cycles at current population size
            safe_consumption = max(1.5, food_ev.grid_total / max(1, n_total * 20))
            proposals.append(ProposedLaw(
                law_type="FOOD_RATION",
                params={"max_per_cycle": round(min(4.0, safe_consumption), 2)},
                duration=20,
                description=f"[ADS] Food ration (drought, cap={safe_consumption:.1f}) group {group.group_id}",
                source_category="food",
            ))
        else:
            # Light conservation only when the cap is genuinely restrictive.
            # Natural collection limit is 5.0 units; a ration cap above 4.5 has no effect.
            # When no drought and no starvation, redistribute to the bottom quartile instead.
            cap = 4.0 if food_ev.grid_mean > 30 else max(2.5, food_ev.grid_mean / 12)
            any_group_starving = min(a.food_stock for a in g_agents) < 2.0
            if cap < 4.5 and not any_group_starving:
                proposals.append(ProposedLaw(
                    law_type="FOOD_RATION",
                    params={"max_per_cycle": round(cap, 2)},
                    duration=8,
                    description=f"[ADS] Food conservation group {group.group_id} (cap={cap:.1f})",
                    source_category="food",
                ))
            else:
                # Prefer bottom-quartile redistribution over a useless ration
                g_avg_food = sum(a.food_stock for a in g_agents) / max(1, K)
                n_food = max(1.5, 5.0 - g_avg_food)
                proposals.append(ProposedLaw(
                    law_type="REDISTRIBUTE_FOOD",
                    params={"resource": "food", "recipient_ids": group.agent_ids,
                            "amount_per_recipient": n_food},
                    duration=8,
                    description=f"[ADS] Proactive food top-up group {group.group_id}",
                    source_category="food",
                ))

        # ---- Water category -------------------------------------------------
        water_band = next((t.band for t in group.tags if t.category == "water"), 0)
        if not both_critical:   # skip water if already handled by GENERIC above
            if water_band >= 6:
                g_avg_water = sum(a.water_stock for a in g_agents) / max(1, K)
                n_water = max(2.0, 5.0 - g_avg_water)
                proposals.append(ProposedLaw(
                    law_type="REDISTRIBUTE_WATER",
                    params={"resource": "water", "recipient_ids": group.agent_ids,
                            "amount_per_recipient": n_water},
                    duration=15,
                    description=f"[ADS] Redistribute water to group {group.group_id} (N={n_water:.1f})",
                    source_category="water",
                ))
            else:
                # Conservation: cap proportional to grid water availability.
                # Only restrict if cap is genuinely restrictive (< 4.5) AND no agent
                # in this group is critically dehydrated — otherwise top-up instead.
                cap_w = max(2.0, min(6.0, water_ev.grid_mean / 15))
                any_group_dehydrated = min(a.water_stock for a in g_agents) < 2.0
                if cap_w < 4.5 and not any_group_dehydrated:
                    proposals.append(ProposedLaw(
                        law_type="LIMIT_WATER_CONSUMPTION",
                        params={"max_per_cycle": round(cap_w, 2)},
                        duration=8,
                        description=f"[ADS] Water conservation group {group.group_id} (cap={cap_w:.1f})",
                        source_category="water",
                    ))
                else:
                    # Redistribute to bottom quartile — more useful than restricting
                    g_avg_w = sum(a.water_stock for a in g_agents) / max(1, K)
                    n_w = max(1.5, 5.0 - g_avg_w)
                    proposals.append(ProposedLaw(
                        law_type="REDISTRIBUTE_WATER",
                        params={"resource": "water", "recipient_ids": group.agent_ids,
                                "amount_per_recipient": n_w},
                        duration=8,
                        description=f"[ADS] Proactive water top-up group {group.group_id}",
                        source_category="water",
                    ))

        # ---- Health / epidemic category -------------------------------------
        health_band = next((t.band for t in group.tags if t.category == "health"), 0)
        group_infected = [a for a in g_agents if a.infected]
        group_infected_frac = len(group_infected) / max(1, K)

        epidemic_threat = (
            health_band == 9
            or group_infected_frac > 0.1
            or health_ev.warning_epidemic
            or health_ev.infected_fraction > 0.15
        )
        if epidemic_threat:
            # EPIDEMIC_RESPONSE (treatment mandate) vs QUARANTINE_EPIDEMIC:
            # Use EPIDEMIC_RESPONSE when many in the group are infected (cure them).
            # Use QUARANTINE when infected fraction is low but group is spatial cluster
            # and quarantine is feasible (bounding box is small).
            if group_infected_frac > 0.4:
                proposals.append(ProposedLaw(
                    law_type="EPIDEMIC_RESPONSE",
                    params={"medicine_mandate": True},
                    duration=20,
                    description=f"[ADS] Epidemic treatment mandate group {group.group_id} "
                                f"({group_infected_frac:.0%} infected)",
                    source_category="health",
                ))
            elif health_ev.quarantine_feasible:
                proposals.append(ProposedLaw(
                    law_type="QUARANTINE_EPIDEMIC",
                    params={},  # region resolved at enactment
                    duration=20,
                    description=f"[ADS] Quarantine feasible — contain infected group {group.group_id}",
                    source_category="health",
                ))
            else:
                # Too spread out for quarantine; use epidemic response for infected members
                proposals.append(ProposedLaw(
                    law_type="EPIDEMIC_RESPONSE",
                    params={"medicine_mandate": True},
                    duration=15,
                    description=f"[ADS] Epidemic response (no quarantine feasible) group {group.group_id}",
                    source_category="health",
                ))
            # Also propose REDISTRIBUTE_MEDICINE when the group genuinely lacks enough
            # to self-treat.  Amount is the actual deficit (cure cost minus current avg).
            # Only fire when there is a real deficit to avoid draining donors needlessly.
            g_avg_med = sum(a.medicine_stock for a in g_agents) / max(1, K)
            n_epidemic_ids = max(1, len(health_ev.epidemic_ids))
            cure_cost_per_agent = 3.0 * n_epidemic_ids  # 3 medicine per infection
            n_med_epi = max(0.0, cure_cost_per_agent - g_avg_med)  # real deficit only
            if n_med_epi > 1.0:  # only propose if group is genuinely medicine-poor
                proposals.append(ProposedLaw(
                    law_type="REDISTRIBUTE_MEDICINE",
                    params={"resource": "medicine", "recipient_ids": group.agent_ids,
                            "amount_per_recipient": n_med_epi},
                    duration=15,
                    description=(f"[ADS] Medicine for epidemic ({n_epidemic_ids} strains, "
                                 f"deficit={n_med_epi:.1f}) group {group.group_id}"),
                    source_category="health",
                ))
        elif health_band >= 5 or health_ev.pct_low_medicine > 0.3:
            # Redistribute medicine: N = amount needed to ensure each recipient can
            # cure at least one epidemic infection (3 medicine units)
            g_avg_med = sum(a.medicine_stock for a in g_agents) / max(1, K)
            n_med = max(1.0, 3.0 - g_avg_med)
            proposals.append(ProposedLaw(
                law_type="REDISTRIBUTE_MEDICINE",
                params={"resource": "medicine", "recipient_ids": group.agent_ids,
                        "amount_per_recipient": n_med},
                duration=15,
                description=f"[ADS] Redistribute medicine to group {group.group_id} (N={n_med:.1f})",
                source_category="health",
            ))
        else:
            proposals.append(ProposedLaw(
                law_type="EPIDEMIC_RESPONSE",
                params={"medicine_mandate": False},
                duration=5,
                description=f"[ADS] Health monitoring group {group.group_id}",
                source_category="health",
            ))

        # ---- Terrain category (terrain-aware movement) -----------------------
        terrain_band = next((t.band for t in group.tags if t.category == "terrain"), 0)

        if terrain_ev.storm_active or terrain_ev.warning_storm:
            # MANDATORY_SHELTER vs MOVE_TO_REGION:
            # If most agents are already near shelters, mandate them to move there.
            # If agents are far from shelters, moving them as a group (MOVE_TO_REGION)
            # is more effective than individual wandering.
            # Also: if epidemic is active, avoid group movement to prevent spread.
            if terrain_ev.pct_near_shelter > 0.5 or epidemic_active:
                # Many already near shelters (or epidemic risk) → mandate individually
                proposals.append(ProposedLaw(
                    law_type="MANDATORY_SHELTER",
                    params={},
                    duration=max(8, round(terrain_ev.storm_severity * 15)),
                    description=f"[ADS] Shelter mandate group {group.group_id} "
                                f"(severity={terrain_ev.storm_severity:.2f})",
                    source_category="terrain",
                ))
            else:
                # Agents are far from shelters; direct them to the resource-rich/shelter area.
                # Prefer top combined-resource region as movement target.
                combined_regions = _find_resource_rich_regions(
                    sim, resource_type="combined", min_size=RESOURCE_REGION_ROWS
                )
                if combined_regions:
                    best_region_data = combined_regions[0]
                    region = best_region_data[:4]
                    region_score = best_region_data[4]
                else:
                    # No subregion could be scored — only happens when the grid is
                    # smaller than the RESOURCE_REGION_ROWS scan window.  Target the
                    # whole grid so the movement law still has a valid bounding box.
                    region = (0, 0, sim.grid.rows - 1, sim.grid.cols - 1)
                    region_score = 0.0

                duration = _movement_duration(
                    g_agents, region,
                    base=max(6, round(terrain_ev.storm_severity * 12)),
                    min_dur=4,
                )
                proposals.append(ProposedLaw(
                    law_type="MOVE_TO_REGION",
                    # ``trigger`` is read by EvaluatorNode._step and by nothing
                    # else; the engine ignores unknown law params.  It marks
                    # this as the ONE MOVE_TO_REGION whose motivation is storm
                    # shelter, and therefore the only one whose relocation the
                    # projection may model.  The other two sites (food-critical
                    # and hazard-escape) are motivated by destination resources
                    # and hazard respectively, neither of which the projection
                    # models — charging them a relocation cost with no modelled
                    # benefit made them strictly worse off than being ignored.
                    params={"region": region, "trigger": "storm"},
                    duration=duration,
                    description=(
                        f"[ADS] Move group {group.group_id} to shelter region {region} "
                        f"(resource_score={region_score:.1f})"
                    ),
                    source_category="terrain",
                ))

        elif terrain_ev.toxic_active:
            # FLEE_CURRENT_LOCATION for agents on or near the toxic region
            if water_ev.toxic_region:
                proposals.append(ProposedLaw(
                    law_type="FLEE_CURRENT_LOCATION",
                    params={"source_region": water_ev.toxic_region},
                    duration=8,
                    description=f"[ADS] Flee toxic area {water_ev.toxic_region} group {group.group_id}",
                    source_category="terrain",
                ))
            else:
                proposals.append(ProposedLaw(
                    law_type="FLEE_CURRENT_LOCATION",
                    params={},
                    duration=8,
                    description=f"[ADS] Flee toxic area group {group.group_id}",
                    source_category="terrain",
                ))

        elif epidemic_active:
            # Epidemic present, no storm/toxic: decide quarantine vs spread.
            # Spread heuristic: only effective if infected fraction is LOW and agents
            # are clustered (spread can physically separate infected from healthy).
            # Quarantine heuristic: effective when infected agents are already localized.
            if health_ev.quarantine_feasible and health_ev.infected_fraction < 0.3:
                proposals.append(ProposedLaw(
                    law_type="QUARANTINE_EPIDEMIC",
                    params={},
                    duration=20,
                    description=f"[ADS] Epidemic quarantine group {group.group_id} "
                                f"(spread_radius={health_ev.infected_spread_radius:.1f})",
                    source_category="terrain",
                ))
            elif health_ev.infected_fraction < 0.25 and health_ev.infected_spread_radius > 3:
                # Many non-infected agents close to infected → tell non-infected to spread out
                non_infected_ids = [
                    a.agent_id for a in g_agents if not a.infected
                ]
                proposals.append(ProposedLaw(
                    law_type="SPREAD",
                    params={},
                    duration=10,
                    applies_to=non_infected_ids or None,
                    description=f"[ADS] Spread non-infected group {group.group_id}",
                    source_category="terrain",
                ))
            else:
                # Widespread or contained — fallback to epidemic response
                proposals.append(ProposedLaw(
                    law_type="EPIDEMIC_RESPONSE",
                    params={"medicine_mandate": True},
                    duration=15,
                    description=f"[ADS] Epidemic fallback group {group.group_id}",
                    source_category="terrain",
                ))

        elif terrain_band >= 7:
            # High-hazard terrain: check if there is a better region to move to
            # (Propose resource-rich movement instead of blind flee when safe.)
            combined_regions = _find_resource_rich_regions(
                sim, resource_type="combined", min_size=RESOURCE_REGION_ROWS
            )
            if combined_regions:
                best_region_data = combined_regions[0]
                target_region = best_region_data[:4]
                duration = _movement_duration(g_agents, target_region, base=8, min_dur=3)
                proposals.append(ProposedLaw(
                    law_type="MOVE_TO_REGION",
                    params={"region": target_region},
                    duration=duration,
                    description=(
                        f"[ADS] Move high-hazard group {group.group_id} "
                        f"to resource-rich region {target_region} (band={terrain_band})"
                    ),
                    source_category="terrain",
                ))
            else:
                proposals.append(ProposedLaw(
                    law_type="FLEE_CURRENT_LOCATION",
                    params={},
                    duration=8,
                    description=f"[ADS] High-hazard flee group {group.group_id} (band {terrain_band})",
                    source_category="terrain",
                ))
        else:
            proposals.append(ProposedLaw(
                law_type="SPREAD",
                params={},
                duration=8,
                description=f"[ADS] Spread out (low risk) group {group.group_id}",
                source_category="terrain",
            ))

        return proposals


# ---------------------------------------------------------------------------
# EvaluatorNode — LOOKAHEAD_CYCLES-cycle lookahead per group per proposal
# ---------------------------------------------------------------------------

class Candidate(NamedTuple):
    """
    One scored (grouping scheme, group, proposal) triple, awaiting enactment.

    Named fields guard against a naming trap at two separate unpack sites:
    the crisis-boosted ranking score and the un-boosted normalised score are
    both floats in the same rough range, and reading the wrong one is the
    difference between grading a health forecast and grading a priority
    multiplier.  Named fields make that class of mistake a typo rather than
    a silent statistical error.
    """
    #: What the candidate is RANKED on: ``norm_score * crisis_boost``.  A
    #: priority, not a health forecast — never grade a prediction against it.
    adjusted_score: float
    n_groups_target: int
    group: "AgentGroup"
    proposal: "ProposedLaw"
    group_agents: List["Agent"]
    #: The evaluator's per-agent projection.  Now THE ONE forecast: what is
    #: recorded at enactment and what is later graded.  There is no second,
    #: corrected number — accuracy comes from the INPUTS (see
    #: ``governments/parameter_inference.ParameterEstimator``), not from a
    #: post-hoc scalar applied to this one.
    norm_score: float


def _phantom_span(start: int, duration: Any, horizon: int) -> int:
    """
    Last fake-cycle index (1-based, inclusive) on which a phantom event that
    arrived at fake cycle *start* is still in force.

    ``duration is None`` means indefinite, the engine's convention for an
    epidemic whose spread window never closes (``ActiveEvent.duration``); such
    an event runs to the end of the horizon.  A duration of ``d`` covers cycles
    ``start .. start + d - 1``, so an event that arrives and lasts one cycle
    acts on exactly the cycle it arrived on — the same constant-then-cliff
    shape the real engine has, applied symmetrically to injected events.

    ``max(1, ...)`` floors the span: a mean observed duration that rounds down
    to zero would otherwise mint an event that is injected and then never felt,
    which is indistinguishable from a bug in the injection test above.
    """
    if duration is None:
        return horizon
    return min(start + max(1, int(duration)) - 1, horizon)


class EvaluatorNode:
    """
    Runs a simplified lookahead for each proposed law over its target
    group and returns the total surviving health score.

    Models: consumption, collection, regeneration, redistribution,
    epidemic response/spread, hunger/thirst/infection drain,
    storm damage, terrain hazard, shelter protection, and — only while a storm
    is actually doing damage — one-cell-per-cycle relocation toward a
    ``MANDATORY_SHELTER`` / ``MOVE_TO_REGION`` target (see ``_step``).

    Movement is modelled for **storm-shelter exposure only**, and is gated on
    ``storm_damage > 0``.  Two consequences worth knowing before reading a
    score:

    * Resource collection stays pinned to the agent's original cell for the
      whole horizon (see the ``home_position`` comment in ``_step``), so the
      "resource-rich region" rationale behind a ``MOVE_TO_REGION`` proposal is
      invisible to the projection.
    * Terrain ``hazard`` likewise stays pinned to the origin cell, so the
      hazard-escape rationale is invisible too.  Because of the gate, a
      hazard-motivated ``MOVE_TO_REGION`` is **completely inert** — scored
      exactly as it would be if movement were not modelled at all — rather
      than being charged a relocation cost it has no modelled way to recoup.

    The relocation model diverges from the engine in three known ways, pulling
    in **opposite directions**, so the sign of the net bias is unknown: travel
    is one cell per cycle where the engine allows up to ``max_steps_per_cycle``
    (understates the mandate's benefit); the engine's shelter reflex fires on an
    active storm whether or not a mandate exists, while the projection moves
    agents only under the mandate (over-states it); and ``_shelter_target``
    scans the whole grid where a real agent sees only ``visibility_radius``
    (over-states it).  Do not treat a projected
    ``MANDATORY_SHELTER - NO_ACTION`` delta as a lower bound.  See the long
    comment in ``_step``.
    """

    #: Sentinel for the shelter memo, so a legitimately cached ``None`` (a grid
    #: with no shelter terrain at all) is not mistaken for a cache miss.
    _MEMO_MISS = object()

    def __init__(self, eval_seed: int = 0):
        #: Root of the rollout streams.  ``evaluate()`` re-seeds BOTH generators
        #: below on every call — see the long note there for why the per-call
        #: reseed is load-bearing and must be preserved.
        self._eval_seed = int(eval_seed)
        #: Stream 1 — epidemic spread inside ``_step``.  Pre-existing, and its
        #: sole draw site is the ``spread_rate`` test at the bottom of ``_step``.
        #: Consumed a number of times that VARIES BY CANDIDATE (once per agent
        #: per not-yet-carried epidemic id per fake cycle, and group size and
        #: infection count both vary), which is exactly why the injection draws
        #: cannot share it.
        self._eval_rng = random.Random(self._eval_seed)
        #: Stream 2 — phantom event injection, and nothing else.  Consumed
        #: exactly ``len(PHANTOM_CATEGORIES) * LOOKAHEAD_CYCLES`` times per
        #: ``evaluate()``, unconditionally, in a fixed order.  Because that
        #: count is candidate-independent and the stream is re-seeded from a
        #: per-decision-round value, every candidate in a round faces the
        #: bit-identical phantom future — common random numbers, the standard
        #: variance-reduction device for exactly this comparison.  Sharing
        #: ``_eval_rng`` instead would let a candidate win on a luckier draw.
        self._event_rng = random.Random(self._eval_seed)
        #: Diagnostic snapshot of the most recent rollout's phantom schedule, so
        #: a government can log the injected future once per round instead of
        #: once per candidate.  Purely descriptive; nothing reads it back into
        #: the projection.  Identical across every candidate of a round, which
        #: is itself the property worth being able to check from a log.
        self.last_phantom_summary: Dict[str, Any] = {}
        # Memo for ``nearest_shelter``, which is an unindexed O(n_cells) scan
        # that also rebuilds the ``shelter_cells()`` list on every call
        # (``engine/grid.py``).  Safe to cache because shelter is a *static
        # terrain* property: ``Cell.shelter`` is read-only and backed by
        # ``_terrain_shelter``, which the ``Cell.terrain`` setter re-derives
        # from ``TERRAIN_STATS`` on every assignment, so it always agrees with
        # the cell's terrain.  Terrain is assigned only while the grid is built
        # (and, for a plan-driven run, restored from the plan), before the
        # first cycle; nothing changes it during a run.
        # ``test_terrain_consistency.py`` checks the agreement on the
        # production path.
        #
        # The grid *object* is held, not its ``id()``: a collected grid could
        # have its address reused by a later one and silently validate a stale
        # cache.  Keyed by position, so the memo is bounded by the cell count
        # (2,500 on the production grid).
        #
        # Pure performance.  It changes no projected value and draws no
        # randomness.  Without it the shelter lookup dominates the lookahead.
        self._shelter_memo_grid: Optional["Grid"] = None
        self._shelter_memo: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

    def set_seed(self, eval_seed: int) -> None:
        """
        Re-root both rollout streams.  See ``AdsGovernment.set_eval_seed``.

        The explicit ``.seed()`` calls here are belt-and-braces: ``evaluate()``
        re-seeds both generators from ``_eval_seed`` on entry, so the only state
        that actually matters is the root.  They are kept so that an
        ``EvaluatorNode`` inspected between calls is in a defined state rather
        than in whatever position the previous root left it.
        """
        self._eval_seed = int(eval_seed)
        self._eval_rng.seed(self._eval_seed)
        self._event_rng.seed(self._eval_seed)

    def _shelter_target(
        self, grid: "Grid", position: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Memoised ``grid.nearest_shelter``; ``None`` if the grid has none."""
        if position is None or not grid.in_bounds(*position):
            return None
        if self._shelter_memo_grid is not grid:
            self._shelter_memo_grid = grid
            self._shelter_memo = {}
        cached = self._shelter_memo.get(position, self._MEMO_MISS)
        if cached is self._MEMO_MISS:
            cached = grid.nearest_shelter(*position)
            self._shelter_memo[position] = cached
        return cached

    @staticmethod
    def _region_center(
        grid: "Grid", region: Optional[Tuple]
    ) -> Optional[Tuple[int, int]]:
        """
        Geometric centre of a ``MOVE_TO_REGION`` bounding box, or ``None``.

        Note the tuple order: ``(r_min, c_min, r_max, c_max)``, *not* the
        conventional ``(r_min, r_max, c_min, c_max)``.  Verified against
        ``base.py``'s ``MOVE_TO_REGION`` handler, which unpacks it this way and
        draws ``randint(r_min, r_max)`` / ``randint(c_min, c_max)``.

        Deliberately the centre and nothing else.  The real engine re-draws a
        fresh random point inside the region for each unassigned agent; that is
        a known quirk preserved on purpose, and reproducing it here would put a
        random draw back into a component that is currently zero-draw and must
        stay that way.

        Floor division, not ``round()``: both are deterministic, but ``round()``
        is banker's rounding on exact halves, which is a surprise waiting for
        the next reader.  ``//`` on ints is unambiguous and always in range,
        since ``r_min <= (r_min + r_max) // 2 <= r_max``.
        """
        if not region or len(region) != 4:
            return None
        r_min, c_min, r_max, c_max = region
        center = (int(r_min + r_max) // 2, int(c_min + c_max) // 2)
        return center if grid.in_bounds(*center) else None

    def evaluate(
        self,
        proposal: ProposedLaw,
        group_agents: List["Agent"],
        sim: "Simulation",
        *,
        cycle: Optional[int] = None,
        env: EnvironmentModel = CRUDE_ENVIRONMENT,
    ) -> float:
        """
        Project *group_agents* forward ``LOOKAHEAD_CYCLES`` fake cycles under
        *proposal* and return their total surviving health.

        Parameters
        ----------
        cycle  The real simulation cycle this evaluation is being made on.  Both
               RNG streams are keyed on it, so it is ranking-semantic rather
               than incidental and the governments pass it explicitly.  Defaults
               to ``sim.cycle`` so a unit test can construct a bare
               ``Simulation`` and still get a defined rollout.
        env    Every environment-dynamics parameter the rollout consumes, in one
               frozen object: event statistics, ``drain_mult``, ``regen_mult``
               and ``max_steps_per_cycle``.  Passed in rather than read here so
               that a decision round fixes it ONCE for every candidate it
               scores, which is what makes "every candidate faced the same model
               of the future" structural rather than incidental — and so this
               method is table-testable with no ``EventSystem`` at all.

        **CRUDE IS THE DEFAULT, AND THAT DIRECTION IS DELIBERATE.**  A caller
        that forgets to say which epistemic regime it is in gets the LESS
        informed one.  Defaulting instead to reading
        ``sim.event_system.observed_event_stats()`` automatically would let a
        regime silently acquire evidence-based foresight; ``autocracy_lookahead``'s
        role as a control requires it to be denied that foresight unless it
        explicitly opts in.  A regime must opt IN to inference, visibly, at its
        call site.

        The only way to pass event statistics in is ``env=``, an
        ``EnvironmentModel``; there is no separate ``event_stats=`` keyword
        that could be silently accepted and read as live event statistics
        instead of an explicit, self-describing environment.

        Keyword-only: these are model inputs, not positional detail, and a
        caller that gets their order wrong should fail at the call site.
        """
        eval_cycle = (
            int(getattr(sim, "cycle", 0) or 0) if cycle is None else int(cycle)
        )
        if not group_agents:
            # Stamp the diagnostic even on the early return.  It is read by the
            # government's once-per-round EVAL_PHANTOM log, and leaving the
            # previous call's value in place would make that log silently report
            # ANOTHER cycle's phantom schedule whenever a round scored nothing.
            # That is a diagnostic good enough to be believed and wrong often
            # enough to mislead if left stale.
            self.last_phantom_summary = {
                "cycle": eval_cycle,
                "horizon": LOOKAHEAD_CYCLES,
                "injected": [],
                "n_injected": 0,
                "real_storm_until": 0,
                "real_drought_until": 0,
                "skipped": "empty_group",
            }
            return 0.0
        event_stats = env.event_stats

        states = [self._snap(a, sim) for a in group_agents]
        cells = self._snap_cells(group_agents, sim)

        horizon = LOOKAHEAD_CYCLES

        # --- Pre-existing events, bounded by when they would REALLY stop -----
        #
        # Real storms last 3-7 cycles and droughts 6-15, against a 20-cycle
        # projection, so treating an active event as "on" for the whole horizon
        # would systematically over-state event load and make the projection
        # uniformly pessimistic.
        #
        # There is NO gradual decay to model, and inventing one would be wrong:
        # `ActiveEvent.storm_damage` / `drought_factor` are flat constants for
        # the event's whole lifetime and then stop dead at
        # `cycle >= start_cycle + duration`.  Constant-then-cliff is the real
        # shape, so `cycles_remaining` is the entire fix.
        #
        # WINDOW CONVENTION, used identically for real and phantom events:
        # `*_until` is the LAST fake-cycle index (1-based, inclusive) on which
        # the effect still applies, and every test is `j <= *_until`.  One
        # convention and one operator throughout — a mix of `<=` for real
        # events and `<` for phantom ones would be exactly the kind of
        # off-by-one that would never show up in a test.
        def _clamp_window(remaining: int) -> int:
            """`cycles_remaining` -> fake-cycle bound.  -1 means indefinite."""
            return horizon if remaining < 0 else min(remaining, horizon)

        event_system = sim.event_system
        real_storm_damage = event_system.storm_damage_per_cycle()
        real_storm_until = (
            _clamp_window(event_system.active_event_window(eval_cycle, EventType.STORM))
            if real_storm_damage > 0 else 0
        )
        real_drought_factor = max(
            (e.drought_factor for e in event_system.active_events
             if e.event_type.value == "drought" and not e.cancelled),
            default=0.0,
        )
        real_drought_until = (
            _clamp_window(event_system.active_event_window(eval_cycle, EventType.DROUGHT))
            if real_drought_factor > 0 else 0
        )

        # Epidemics get a per-id window rather than a single category window,
        # because `_step` consumes them individually.  Note what the window
        # bounds and what it does not: an epidemic's `duration` limits how long
        # it keeps SPREADING, exactly as in the engine, where `_apply_epidemic`
        # stops running at expiry but `_end_event` never clears anyone's
        # `epidemic_ids`.  An agent who caught it keeps carrying it — and keeps
        # taking the infection drain — for the rest of the projection.  That
        # asymmetry is the engine's actual behaviour, not a simplification.
        epi_until: Dict[str, int] = {}
        for _ev in event_system.active_events:
            if (_ev.event_type.value != "epidemic" or _ev.cancelled
                    or not _ev.epidemic_id):
                continue
            _w = _clamp_window(_ev.cycles_remaining(eval_cycle))
            if _w > epi_until.get(_ev.epidemic_id, -1):
                epi_until[_ev.epidemic_id] = _w

        # From the environment MODEL, never from the simulation.  Reading
        # `sim.drain_mult` / `sim.config.regen_mult` here would hand the
        # evaluator ground truth about a difficulty-scaled quantity that no
        # government is allowed to see.  The values arrive instead from either
        # ParameterEstimator (inferred from observations) or the crude baseline.
        drain_mult = env.drain_mult
        regen_mult = env.regen_mult
        max_steps = env.max_steps_per_cycle

        lt = proposal.law_type
        lp = proposal.params
        redist_amount = lp.get("amount_per_recipient", 2.0)

        quarantine_region = None
        if lt == "QUARANTINE_EPIDEMIC":
            quarantine_region = lp.get("region")

        # --- Movement targets, resolved ONCE per evaluate() call -------------
        # Both are hoisted out of the O(agents x LOOKAHEAD_CYCLES) loop below.
        # ``region_center`` is identical for every agent under a given
        # proposal, so computing it per agent per cycle would be 30x-per-agent
        # wasted work for the same answer.  ``shelter_target`` genuinely varies
        # per agent, but is constant across the horizon, so it is resolved once
        # per agent here rather than once per agent per cycle.
        #
        # Gated on the law type on purpose: ``_shelter_target`` is memoised but
        # still a dict lookup per agent, and neither value is ever read by
        # ``_step`` under any other law.
        region_center: Optional[Tuple[int, int]] = None
        if lt == "MOVE_TO_REGION":
            # Only a storm-motivated relocation gets its movement modelled.
            # Leaving ``region_center`` None for the other two proposal sites
            # makes them bit-identical to unmodelled movement — see the
            # ``trigger`` comment at the proposal site and the gate comment in
            # ``_step``.  Checked here rather than in ``_step`` so the cost is
            # paid once per evaluate() instead of once per agent per cycle.
            if lp.get("trigger") == "storm":
                region_center = self._region_center(sim.grid, lp.get("region"))
        elif lt == "MANDATORY_SHELTER":
            for state in states:
                state["shelter_target"] = self._shelter_target(
                    sim.grid, state["position"]
                )

        # Re-seed per call, from (run-specific root XOR current cycle).
        #
        # The per-call reseed is deliberate and must be preserved: it makes
        # evaluate() a pure function of its arguments, so two identical
        # candidates score identically within a round and the candidate ranking
        # cannot depend on the order in which candidates happened to be
        # evaluated.  Removing it — "seed once per run" — would reintroduce
        # exactly that coupling.
        #
        # A CONSTANT root would be the wrong value here: it would mean the
        # stochastic component of the lookahead (the epidemic-spread draw
        # further down in `_step`) replays one frozen pattern on every call, of
        # every cycle, of every run, of every difficulty, for the entire
        # published corpus.  `_eval_seed` is derived per (government,
        # difficulty, run), and XOR-ing the cycle varies it within a run while
        # keeping within-cycle purity intact.  `sim.cycle` needs no new
        # plumbing — `evaluate` already receives `sim`.
        #
        # `eval_cycle` rather than `sim.cycle` directly: a caller constructing a
        # bare Simulation for a unit test should get a defined rollout, not an
        # AttributeError from the RNG seeding line.
        self._eval_rng.seed(self._eval_seed ^ eval_cycle)

        # The INJECTION stream, re-seeded once per decision round.
        #
        # Keyed on `cycle` and on nothing finer, deliberately.  ADS ranks
        # candidates GLOBALLY across every group and every grouping scheme, so
        # every `evaluate()` call in a round feeds one comparison and they must
        # all face the identical phantom future; keying on the group or the
        # proposal would let a candidate win because its private phantom draw
        # happened to be kinder, which is ranking noise dressed as evidence.
        #
        # Keyed on `cycle` AT ALL, equally deliberately.  A constant reseed
        # would replay one frozen uniform vector for every round of every run,
        # so a phantom storm would fire at fake-cycle j in every evaluation
        # ever performed — a fixed schedule wearing a stochastic costume, with
        # whatever bias that one vector happens to carry baked into the entire
        # corpus.
        #
        # `derive_seed` rather than another XOR: XOR of a large root with a
        # small cycle index only perturbs the low bits, so consecutive rounds
        # get highly correlated seeds.  That is survivable for `_eval_rng`
        # (changing it now would re-key the corpus for no benefit) but not for
        # a stream whose whole job is to sample a fresh future each round.
        self._event_rng.seed(derive_seed(self._eval_seed, "eval.round", eval_cycle))

        # --- Phantom (stochastically injected) event state -------------------
        phantom_storm_until = 0
        phantom_storm_effect = 0.0
        phantom_drought_until = 0
        phantom_drought_effect = 0.0
        phantom_counter = 0
        phantom_log: List[Tuple[int, str]] = []

        # Rebuilt lazily inside the loop; `epi_dirty` forces the first build so
        # that the construction lives at exactly one site.
        active_epi_ids: Tuple[str, ...] = ()
        epi_min_until = horizon
        epi_dirty = True

        for j in range(1, horizon + 1):
            # --- Injection: exactly one draw per category, always ------------
            #
            # The uniform is drawn even when the rate is 0.0.  Stream position
            # must be a function of (fake cycle, category index) and of nothing
            # else — that is what guarantees two candidates see the same
            # phantom future, and skipping the draw for a zero-rate category
            # would couple stream position to the statistics.
            for category in PHANTOM_CATEGORIES:
                u = self._event_rng.random()
                cat_stats = event_stats.get(category) or {}
                if u >= float(cat_stats.get("rate") or 0.0):
                    continue
                span = _phantom_span(j, cat_stats.get("duration"), horizon)
                if category == "storm":
                    phantom_storm_effect = float(cat_stats.get("effect") or 0.0)
                    phantom_storm_until = max(phantom_storm_until, span)
                elif category == "drought":
                    phantom_drought_effect = float(cat_stats.get("effect") or 0.0)
                    phantom_drought_until = max(phantom_drought_until, span)
                else:
                    phantom_counter += 1
                    epi_until[
                        f"{PHANTOM_EPIDEMIC_PREFIX}{phantom_counter:02d}"
                    ] = span
                    epi_dirty = True
                phantom_log.append((j, category))

            # --- Effective environment for this fake cycle -------------------
            # `max`, not a sum: concurrent storms do not stack in the engine
            # either (`storm_damage_per_cycle` takes the max), and the same
            # holds for the drought factor.
            storm_damage_j = real_storm_damage if j <= real_storm_until else 0.0
            if j <= phantom_storm_until and phantom_storm_effect > storm_damage_j:
                storm_damage_j = phantom_storm_effect

            drought_j = real_drought_factor if j <= real_drought_until else 0.0
            if j <= phantom_drought_until and phantom_drought_effect > drought_j:
                drought_j = phantom_drought_effect

            eff_regen_j = max(0.0, 1.0 - drought_j) * max(0.0, regen_mult)

            # Sorted tuple, NOT a set — reproducibility-critical, do not
            # "simplify".  ``_step`` iterates this while drawing from
            # ``self._eval_rng`` inside the loop, so iteration order decides
            # which epidemic receives which draw.  A ``set`` orders by string
            # hash, which makes that assignment depend on the identifier values
            # rather than on the seed.  Sorting pins the order to the
            # identifiers themselves, which ``EventSystem._next_epidemic_id``
            # derives deterministically and ``PHANTOM_EPIDEMIC_PREFIX`` is
            # chosen to sort stably against; the two are complementary and
            # neither is sufficient alone.
            #
            # Rebuilt only when the membership can actually have changed — a
            # phantom arrived, or the earliest expiry has now passed — because
            # this sits directly above an O(agents) inner loop that is the hot
            # path.
            if epi_dirty or j > epi_min_until:
                active_epi_ids = tuple(sorted(
                    eid for eid, until in epi_until.items() if j <= until
                ))
                epi_min_until = min(
                    (until for until in epi_until.values() if until >= j),
                    default=horizon,
                )
                epi_dirty = False

            for pos, cell in cells.items():
                cell["food"] = min(cell.get("food_cap", 70), cell["food"] + cell.get("food_regen", 0.5) * eff_regen_j)
                cell["water"] = min(cell.get("water_cap", 50), cell["water"] + cell.get("water_regen", 0.2) * eff_regen_j)
                cell["medicine"] = min(cell.get("med_cap", 15), cell["medicine"] + 0.05 * eff_regen_j)
            for state in states:
                if not state["alive"]:
                    continue
                self._step(state, cells, lt, lp, active_epi_ids,
                           storm_damage_j, drain_mult, redist_amount, quarantine_region,
                           sim.grid, region_center, max_steps)

        # Descriptive only — see the field's declaration in __init__.  Written
        # unconditionally rather than under a log-level check so that a test can
        # assert two candidates of one round saw the identical phantom future
        # without having to turn logging on.
        self.last_phantom_summary = {
            "cycle": eval_cycle,
            "horizon": horizon,
            "injected": phantom_log,
            "n_injected": len(phantom_log),
            "real_storm_until": real_storm_until,
            "real_drought_until": real_drought_until,
        }

        return sum(s["health"] for s in states if s["alive"])

    @staticmethod
    def _snap(agent: "Agent", sim: "Simulation") -> dict:
        on_shelter = False
        hazard = 0.0
        if agent.position and sim.grid.in_bounds(*agent.position):
            cell = sim.grid.cell(*agent.position)
            on_shelter = cell.shelter
            hazard = cell.hazard
        return {
            "alive": agent.alive,
            "health": agent.health,
            "hunger": agent.hunger,
            "thirst": agent.thirst,
            "epidemic_ids": set(agent.epidemic_ids),
            "food_stock": agent.food_stock,
            "water_stock": agent.water_stock,
            "medicine_stock": agent.medicine_stock,
            # ``position`` MOVES during the projection under the two movement
            # law types; ``home_position`` never does.  Resource collection
            # reads ``home_position``, shelter exposure reads ``position``.
            # See ``_step`` for why they were split.
            "position": agent.position,
            "home_position": agent.position,
            "on_shelter": on_shelter,
            "hazard": hazard,
            # Filled in by ``evaluate`` for MANDATORY_SHELTER only — the sole
            # law type that reads it.  Resolving it here for every proposal
            # would pay an O(n_cells) grid scan per agent to answer a question
            # nothing asks.
            "shelter_target": None,
        }

    @staticmethod
    def _snap_cells(agents: List["Agent"], sim: "Simulation") -> Dict[Tuple, dict]:
        # Cap tables come from engine.grid, which is the site that actually
        # enforces them, rather than a local copy; see the comment on FOOD_CAP
        # there for the three sites that must agree.
        cells: Dict[Tuple, dict] = {}
        for a in agents:
            if a.position and a.position not in cells and sim.grid.in_bounds(*a.position):
                cell = sim.grid.cell(*a.position)
                cells[a.position] = {
                    "food": cell.food, "water": cell.water, "medicine": cell.medicine,
                    "food_regen": cell.food_regen, "water_regen": cell.water_regen,
                    "food_cap": FOOD_CAP.get(cell.terrain, 70),
                    "water_cap": WATER_CAP.get(cell.terrain, 50),
                    "med_cap": MED_CAP.get(cell.terrain, 15),
                }
        return cells

    def _step(
        self,
        state: dict,
        cells: Dict[Tuple, dict],
        law_type: str,
        law_params: dict,
        # Ordered on purpose — see the construction site in ``evaluate``.
        active_epi_ids: Tuple[str, ...],
        storm_damage: float,
        drain_mult: float,
        redist_amount: float,
        quarantine_region: Optional[Tuple],
        # The *live* grid, for static terrain lookups only.  ``cells`` is a
        # mutable resource snapshot and cannot answer "is this cell shelter?"
        # for any cell an agent moves into, because it only holds cells agents
        # started on.
        grid: "Grid",
        # Precomputed once per evaluate() call; only meaningful for
        # MOVE_TO_REGION.  See ``_region_center``.
        region_center: Optional[Tuple[int, int]],
        # Chebyshev cells per projected cycle, from EnvironmentModel.  1 in the
        # crude regime (the evaluator's historical hardcoded value); an inferred
        # lower bound on the engine's real cap for ADS.
        max_steps: int,
    ) -> None:
        # --- Movement, for STORM-SHELTER EXPOSURE ONLY -----------------------
        # Without this, MANDATORY_SHELTER and MOVE_TO_REGION would be exact
        # no-ops on projected storm damage: the storm block below would read an
        # ``on_shelter`` snapshotted from wherever the agent already happened to
        # be standing, so the projection would score "order everyone to
        # shelter" identically to "do nothing" — a permanent forced tie on the
        # ablation's storm menu that would silently penalise ADS's
        # terrain-category proposals in its global cross-category ranking.
        #
        # ``storm_damage > 0`` GATE — read this before removing it.
        # Movement models the *cost* of relocating (leaving shelter mid-storm
        # hurts) but none of its *benefits*: hazard stays pinned to the origin
        # cell, and so does resource collection.  Charging that one-sided cost
        # to a proposal whose rationale the evaluator cannot see would make the
        # projection worse than the blind version it replaced.  Measured at a
        # synthetic operating point, the asymmetry is +0.32 for
        # MANDATORY_SHELTER against -23.44 for a hazard-motivated
        # MOVE_TO_REGION.  The gate confines the whole mechanism to the one
        # case where the modelled cost and the modelled benefit are the same
        # channel.
        #
        # The gate is also *self-justifying*, which is why it is phrased on
        # storm damage rather than on a trigger tag threaded through
        # ``ProposedLaw.params``: ``on_shelter`` is read in exactly one place,
        # the ``if storm_damage > 0`` block below.  When there is no storm
        # damage, moving the agent cannot change any projected quantity, so
        # skipping it is behaviour-preserving by construction rather than by
        # assumption about who proposed the law.
        #
        # The storm-damage gate ALONE would very nearly have sufficed, and it is
        # worth recording why it was not relied on.  All three MOVE_TO_REGION
        # proposal sites happen to be mutually exclusive with an active storm:
        # the food-category one (``propose_laws``, "food-critical group") is
        # guarded by ``not movement_dangerous`` = ``storm_active or
        # toxic_active``; the hazard-escape one (``elif terrain_band >= 7``)
        # sits behind the storm branch of the same if/elif chain; and
        # ``storm_damage > 0`` implies ``storm_active``, since both scan the
        # same non-cancelled storm events.  Measured over four production-scale
        # runs, MOVE_TO_REGION was evaluated with ``storm_damage > 0`` exactly
        # 0 times.
        #
        # But that is an invariant of a *different* class, three branches apart,
        # with no test enforcing it — and if a future MOVE_TO_REGION site can
        # fire during a storm, the one-sided cost silently returns.  So the
        # storm-motivated proposal is tagged explicitly at its source
        # (``params["trigger"] == "storm"``) and ``region_center`` is left None
        # otherwise, which makes the other two rationales inert unconditionally
        # rather than contingently.  Belt and braces, deliberately.
        #
        # ``max_steps`` cells per cycle, Chebyshev — up to +/-``max_steps`` on
        # each axis independently, so a diagonal is one step, matching the
        # *direction* rule in ``Agent.direction_toward`` (``engine/agent.py``).
        #
        # SIMPLIFICATION, not a faithful model of what the engine does.  THREE
        # known divergences, and they do NOT all push the same way — do not
        # summarise this block as "conservative": the net direction is
        # unquantified, as detailed below.
        #
        #  (1) Travel speed.  RESOLVED FOR ADS, STILL OPEN FOR
        #      ``autocracy_lookahead``.  The engine takes up to
        #      ``max_steps_per_cycle`` (10 in production, 3 in the quick test)
        #      via ``Agent.steps_toward``, while this block moves one cell per
        #      cycle by default.  Real agent-movement-cycles cover more than
        #      one cell 27.1% of the time at production scale, so the gap is
        #      large.
        #
        #      It is closed by INFERENCE, not by reading the value:
        #      ``max_steps`` arrives via ``EnvironmentModel``, carrying either a
        #      ``FlooredMaxEstimator`` lower bound learned from observed
        #      relocations (ADS) or the crude constant 1
        #      (``autocracy_lookahead``).  The divergence is DELIBERATELY
        #      RETAINED for the control arm, which is the asymmetry the design
        #      asks for.  Direction, where it remains: over-states travel time,
        #      so over-states en-route exposure — **understates** the mandate's
        #      benefit.
        #
        #  (2) The engine's shelter reflex is NOT conditional on the law.
        #      ``agents/citizen.py`` sets ``need_shelter`` from
        #      ``storm_active or shelter_mandated or (storm_warned and ...)`` —
        #      an active storm alone sends agents to shelter.  This block only
        #      runs when ``storm_damage > 0``, which implies an active storm, so
        #      in precisely the situation being projected the engine relocates
        #      agents under NO_ACTION too.  The projection moves them only under
        #      the mandate.  Direction: **over**-states the mandate's benefit,
        #      by crediting it with movement that would have happened anyway.
        #
        #  (3) The projection knows more than the agent.  ``_shelter_target``
        #      uses ``grid.nearest_shelter``, a global scan; the real agent
        #      searches only ``visible_cells`` within ``visibility_radius`` (5)
        #      and falls back to ``last_known_shelter``, so it can fail to find
        #      shelter the projection always finds.  Direction: **over**-states
        #      the mandate's benefit.
        #
        # Net direction is UNQUANTIFIED.  (2) in particular is structural rather
        # than marginal, so do not assume the projected
        # ``MANDATORY_SHELTER - NO_ACTION`` delta is a lower bound on the real
        # effect; it is as likely to be an upper bound.
        #
        # (2) and (3) are deliberately left as-is: replicating the
        # law-independent reflex or bounding the shelter search by visibility is
        # a model change needing its own design pass, not a comment fix.  (1) is
        # handled for ADS, as described above.
        #
        # Pure integer arithmetic: no RNG, by construction.  ``EvaluatorNode``
        # must stay zero-draw apart from the pre-existing epidemic-spread draw
        # below — reproducibility depends on it staying that way.
        if storm_damage > 0:
            if law_type == "MANDATORY_SHELTER":
                target = state["shelter_target"]
            elif law_type == "MOVE_TO_REGION":
                target = region_center
            else:
                target = None

            pos = state["position"]
            if target is not None and pos is not None and pos != target:
                r, c = pos
                tr, tc = target
                # Chebyshev step of up to ``max_steps`` on each axis, clamped so
                # the agent never overshoots its target.  At ``max_steps == 1``
                # this is arithmetically identical to the unit sign-step it
                # replaced, which is what keeps ``autocracy_lookahead``
                # bit-identical to pre-Phase-7 on this axis while letting ADS's
                # inferred cap actually bite.
                nr = r + max(-max_steps, min(max_steps, tr - r))
                nc = c + max(-max_steps, min(max_steps, tc - c))
                # Both targets are in-grid and we step from an in-grid cell, so
                # this should always hold; if it somehow does not, stall rather
                # than let ``grid.cell`` raise IndexError inside a scoring loop.
                if grid.in_bounds(nr, nc):
                    state["position"] = (nr, nc)
                    # Recomputed from the live grid rather than from ``cells``:
                    # shelter is static terrain (never mutated during a run),
                    # and ``cells`` does not contain cells the agent moved into.
                    state["on_shelter"] = grid.cell(nr, nc).shelter

        EAT = 2.0
        DRINK = 2.0
        COLLECT_FOOD = 5.0
        COLLECT_WATER = 5.0
        COLLECT_MED = 2.0

        eat = min(EAT, state["food_stock"])
        drink = min(DRINK, state["water_stock"])
        state["food_stock"] -= eat
        state["water_stock"] -= drink

        if eat > 0:
            state["health"] = min(1.0, state["health"] + eat * 0.05)
            state["hunger"] = max(0.0, state["hunger"] - eat * 0.3)
        else:
            state["hunger"] = min(1.0, state["hunger"] + 0.04)

        if drink > 0:
            state["health"] = min(1.0, state["health"] + drink * 0.04)
            state["thirst"] = max(0.0, state["thirst"] - drink * 0.3)
        else:
            state["thirst"] = min(1.0, state["thirst"] + 0.05)

        # Resource collection stays pinned to the agent's ORIGINAL cell for the
        # whole horizon, even when the agent has moved above.  This is a
        # deliberate simplification, not an oversight:
        # ``cells`` only ever contains cells that agents *started* on, so
        # following the agent would make ``pos not in cells`` the moment it
        # moved and would silently switch collection off — strictly worse than
        # the approximation, and it would have made MANDATORY_SHELTER look good
        # for the wrong reason (no depletion) while looking bad for another (no
        # collection).  Cost: a sheltering agent is credited with its origin
        # cell's food/water/medicine rather than its destination's.
        pos = state["position"]
        home = state["home_position"]
        if home and home in cells:
            cell = cells[home]
            food_cap = (
                law_params.get("max_per_cycle", COLLECT_FOOD)
                if law_type == "FOOD_RATION" else COLLECT_FOOD
            )
            water_cap = (
                law_params.get("max_per_cycle", COLLECT_WATER)
                if law_type == "LIMIT_WATER_CONSUMPTION" else COLLECT_WATER
            )
            cf = min(food_cap, cell["food"])
            cw = min(water_cap, cell["water"])
            cm = min(COLLECT_MED, cell["medicine"])
            state["food_stock"] += cf
            state["water_stock"] += cw
            state["medicine_stock"] += cm
            cell["food"] = max(0.0, cell["food"] - cf)
            cell["water"] = max(0.0, cell["water"] - cw)
            cell["medicine"] = max(0.0, cell["medicine"] - cm)

        if law_type in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER", "REDISTRIBUTE_MEDICINE"):
            resource = law_type.split("_", 1)[1].lower()
            stock_key = f"{resource}_stock"
            state[stock_key] = state.get(stock_key, 0.0) + redist_amount

        if law_type == "REDISTRIBUTE_GENERIC":
            state["food_stock"] += redist_amount
            state["water_stock"] += redist_amount
            state["medicine_stock"] += min(1.0, redist_amount * 0.5)

        if law_type == "EPIDEMIC_RESPONSE" and law_params.get("medicine_mandate"):
            epi_ids = state["epidemic_ids"]
            needed = len(epi_ids) * 3.0
            if epi_ids and state["medicine_stock"] >= needed:
                state["medicine_stock"] -= needed
                state["epidemic_ids"] = set()

        spread_rate = 0.04
        if law_type == "QUARANTINE_EPIDEMIC" and quarantine_region and pos:
            r, c = pos
            r_min, c_min, r_max, c_max = quarantine_region
            if r_min <= r <= r_max and c_min <= c <= c_max:
                spread_rate = 0.01

        if active_epi_ids:
            for eid in active_epi_ids:
                if eid not in state["epidemic_ids"]:
                    if self._eval_rng.random() < spread_rate:
                        state["epidemic_ids"].add(eid)

        # DRAIN COEFFICIENTS COME FROM parameter_inference, NOT FROM LITERALS.
        # ParameterEstimator's exposure term is the same expression evaluated at
        # drain_mult = 1.0 on real observed agent state, and reading the same
        # named constants from the same module is what makes ``drain_mult_hat``
        # mean "the multiplier that fixes THIS model" rather than "some number
        # near the engine's constant".  Two copies of these five numbers could
        # drift; one cannot.
        drain = 0.0
        if state["hunger"] > DRAIN_NEED_THRESHOLD:
            drain += DRAIN_COEF_HUNGER * drain_mult * state["hunger"]
        if state["thirst"] > DRAIN_NEED_THRESHOLD:
            drain += DRAIN_COEF_THIRST * drain_mult * state["thirst"]
        if state["epidemic_ids"]:
            drain += DRAIN_COEF_DISEASE * drain_mult * len(state["epidemic_ids"])

        drain += DRAIN_COEF_HAZARD * state.get("hazard", 0.0) * drain_mult

        if storm_damage > 0:
            sheltered = state.get("on_shelter", False)
            if law_type == "MANDATORY_SHELTER" and sheltered:
                pass
            elif not sheltered:
                drain += DRAIN_COEF_STORM * storm_damage * drain_mult

        state["health"] = max(0.0, state["health"] - drain)
        if state["health"] <= 0.0:
            state["alive"] = False


# ---------------------------------------------------------------------------
# AdsGovernment
# ---------------------------------------------------------------------------

class AdsGovernment(Government):
    """
    Science-based governance via a flat tree of evidence → HG → evaluator →
    law-generator nodes.

    Every T_DECISION cycles:
      1. Collect evidence from 4 EvidenceNodes.
      2. HypothesisGeneratorNode forms up to 10 agent groups and proposes
         candidate laws per group (one per evidence category).
      3. EvaluatorNode scores each candidate with a 20-cycle lookahead.
      4. LawGeneratorNode (one per group) enacts the best-scoring candidate.

    The minimum group size is 10% of surviving agents; if fewer than 10 agents
    remain, all are put in one group.

    Event warnings trigger an immediate pre-emptive decision round.
    """

    name = "ADS"

    def __init__(
        self,
        t_decision: int = 5,
        seed: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        t_decision  Cycles between scheduled decision rounds.
        seed        The government's own decision RNG seed.  In the benchmark
                    this is ``derive_seed(base, "gov", gov_name, difficulty,
                    run_idx)``.
        eval_seed   Root of the evaluator's rollout RNG, kept SEPARATE from
                    ``seed`` so a change in how often ADS draws for a decision
                    cannot shift the lookahead's random realisations, and vice
                    versa.  Optional, and deliberately so: ``benchmark_core``
                    constructs all eight regimes uniformly as
                    ``gov_cls(seed=...)``, and a required ADS-only kwarg would
                    force a per-government branch there.  It is normally
                    injected after construction via :meth:`set_eval_seed`, which
                    ``_run_one`` calls by duck-typing; the constructor path
                    exists for direct users such as ``run_simulation.py`` and
                    for tests.
        """
        super().__init__()
        self.t_decision = t_decision
        # Fall back to the base class's fixed default rather than to OS entropy.
        # `random.Random(None)` seeds from entropy; the tie-break in
        # `_decision_round` draws from `self.rng` on every round, so an
        # unseeded AdsGovernment would otherwise be irreproducible.  base.py
        # already documents this exact intent for the fallback generator it
        # installs.
        self.rng = random.Random(
            DEFAULT_GOVERNMENT_SEED if seed is None else seed
        )

        self._food_node     = FoodEvidenceNode()
        self._water_node    = WaterEvidenceNode()
        self._health_node   = HealthEvidenceNode()
        self._terrain_node  = TerrainEvidenceNode()
        self._hg_node       = HypothesisGeneratorNode()
        #: ADS's evidence-based model of the environment's hidden dynamics
        #: parameters.  Updated once per cycle from ``tick``, read once per
        #: decision round.
        #:
        #: THIS OBJECT DOES NOT MARK AN EPISTEMIC ASYMMETRY BETWEEN THE ARMS.
        #: ``autocracy_lookahead`` also constructs its own ``ParameterEstimator``
        #: from the same neutral module, so both arms forecast on inferred
        #: inputs and the ablation isolates ORGANISATION instead of information
        #: quality.  The two arms share the estimator CLASS; they never share an
        #: INSTANCE.
        self._estimator     = ParameterEstimator(self._logger)
        self._evaluator     = EvaluatorNode(
            eval_seed=(
                eval_seed if eval_seed is not None
                else derive_seed(
                    DEFAULT_GOVERNMENT_SEED if seed is None else seed,
                    "gov.eval",
                )
            )
        )

        self._pending_warnings: List["EventWarning"] = []
        self._decisions_made = 0
        self._laws_enacted = 0
        # Audit tracking
        self._last_evidence_scores: dict = {}
        self._last_proposals_evaluated: int = 0
        self._last_top_proposal: str = ""
        self._last_top_score: float = 0.0
        # Full detail store for the last decision round
        # Structure: {
        #   "groupings": {
        #     <n_groups>: [
        #       {
        #         "group_id": int,
        #         "agent_count": int,
        #         "median_health": float,
        #         "median_food": float,
        #         "median_water": float,
        #         "infected_count": int,
        #         "law_scores": [{"law_type": str, "description": str, "raw_score": float, "norm_score": float}],
        #         "best_law_type": str,
        #         "best_norm_score": float,
        #       },
        #       ...
        #     ],
        #     ...
        #   },
        #   "winner": {
        #     "n_groups": int,
        #     "group_id": int,
        #     "law_type": str,
        #     "description": str,
        #     "norm_score": float,
        #   }
        # }
        self._last_decision_details: Dict[str, Any] = {}

        # --- Structured decision-log capture ------------------------------
        # _last_decision_details persists indefinitely between rounds, so a
        # consumer needs to know WHICH cycle produced it; without that, a stale
        # record would be duplicated onto every intervening cycle.
        self._last_decision_cycle: Optional[int] = None
        #: dataclasses.asdict() of the four evidence dataclasses from the most
        #: recent round.  Dumped whole rather than field by field, so adding a
        #: field to (say) FoodEvidence surfaces it in the log with no further
        #: change here — that property is what stops these fields rotting back
        #: into unconsumed state.
        self._last_evidence: Dict[str, Any] = {}
        self._last_decision_trigger: Dict[str, Any] = {}

        # --- Forecast ledger ---------------------------------------------
        #
        # This machinery lives in ``governments/forecast_ledger.py`` rather
        # than inline here, so that ``autocracy_lookahead`` can share the same
        # measurement loop WITHOUT a second copy of the arithmetic.  The
        # manuscript's claim is "both arms forecast equally well"; two copies
        # would make that unverifiable the first time they drifted.  Same
        # argument that keeps ``EvaluatorNode`` unforked, and the same reason
        # ``ParameterEstimator`` lives in its own module.
        #
        # ADS's persisted output is held to an additive-only +
        # projection-identical gate, so sharing the ledger changed nothing
        # about what ADS already emitted.
        #
        # The horizon is passed in rather than imported by the ledger, because
        # ``LOOKAHEAD_CYCLES`` lives in this module and the ledger importing it
        # would close a cycle.  The two arms share the ledger CLASS; they never
        # share an INSTANCE.
        self._ledger = ForecastLedger(LOOKAHEAD_CYCLES, self._logger)

    # ------------------------------------------------------------------
    # Seeding hooks
    # ------------------------------------------------------------------

    def set_eval_seed(self, eval_seed: int) -> None:
        """
        Set the root seed of the evaluator's rollout RNG.

        Called by ``benchmark_core._run_one`` via ``getattr(gov,
        "set_eval_seed", None)``.  ADS is the only regime with a second random
        stream, and the harness constructs every regime through the same
        ``gov_cls(seed=...)`` call; duck-typing the injection keeps that call
        uniform, exactly as ``get_calibration_summary`` already does for the
        ADS-only calibration metrics.  A regime without this method is
        unaffected.

        Safe to call at any point before the first ``evaluate()``: the evaluator
        re-seeds from this root on every call, so there is no "already started"
        state to corrupt.
        """
        self._evaluator.set_seed(eval_seed)

    # ------------------------------------------------------------------
    # Per-cycle tick
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        if not self._sim:
            return
        # EVIDENCE FIRST, before anything in this tick can perturb what it
        # measures — the same ordering rationale the _close_predictions comment
        # below already states, but here it is a HARD REQUIREMENT rather than a
        # preference.  The movement and regen accumulators are per-cycle and
        # un-diffable, so a cycle on which this does not run is a cycle of
        # evidence silently lost; the regen diff would additionally attribute
        # two cycles of regrowth to one cycle of exposure.  `observe` raises on
        # a gap rather than letting that happen quietly.
        #
        # MUST NEVER be moved inside the decision-interval gate at the bottom of
        # this method.  It runs on every cycle; a decision round runs on a few.
        #
        # Placing it before `_decision_round` also means a round on cycle t
        # consumes evidence through cycle t INCLUSIVE — one rule ("the estimator
        # is always current when read") instead of an off-by-one to reason about.
        self._estimator.observe(cycle, self._sim)
        self._expire_laws(cycle)
        # Closure fires on a fixed T+20 review date and is NOT coupled to
        # `_expire_laws` or to `_laws_expired_this_cycle`.
        # It runs here, before anything that can enact, for two reasons: a
        # prediction opened by a round later in this
        # same tick must not be graded against a cycle it has not lived through
        # (it cannot be, since its close cycle is 20 ahead, but the ordering
        # makes that independent of arithmetic); and the realized outcome should
        # be read before this cycle's interventions perturb the agents it is
        # measured over.
        #
        # This must run on EVERY cycle, not only on decision cycles: the review
        # date is enactment + 20, which need not be a decision cycle, and a
        # missed cycle would silently strand its predictions in the ledger.
        self._ledger.close(
            cycle, self._sim, {law.law_id for law in self.active_laws},
        )
        self._fast_response(cycle)
        has_crisis = any(
            e.event_type.value in ("epidemic", "storm", "drought", "toxic_spill")
            and not e.cancelled
            for e in self._sim.event_system.active_events
        )
        interval = 2 if has_crisis else self.t_decision
        if cycle > 0 and cycle % interval == 0:
            self._last_decision_trigger = {
                "kind": "crisis_interval" if has_crisis else "scheduled_interval",
                "interval": interval,
                "pending_warnings": [
                    w.event_type.value for w in self._pending_warnings
                ],
                "t_decision": self.t_decision,
            }
            self._decision_round(cycle)
        self._pending_warnings = []

    def _fast_response(self, cycle: int) -> None:
        """Immediate response to clear threats — no evaluator needed."""
        if not self._sim:
            return
        active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
        alive = self._sim.living_agents()
        n = len(alive)
        if n == 0:
            return

        for ev in self._sim.event_system.active_events:
            if ev.cancelled:
                continue

            if ev.event_type.value == "storm" and "MANDATORY_SHELTER" not in active_types:
                self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                event_type="storm",
                                description="[ADS-FAST] Mandatory shelter during storm.",
                                source="fast_response")
                active_types.add("MANDATORY_SHELTER")

            elif ev.event_type.value == "epidemic" and "QUARANTINE_EPIDEMIC" not in active_types:
                qregion = AdsGovernment._compute_quarantine_bbox(self._sim)
                if qregion:
                    self._enact_law("QUARANTINE_EPIDEMIC", {"region": qregion}, cycle,
                                    duration=None, event_id=ev.epidemic_id,
                                    event_type="epidemic",
                                    description="[ADS-FAST] Quarantine during epidemic.",
                                    source="fast_response")
                    active_types.add("QUARANTINE_EPIDEMIC")

            elif ev.event_type.value == "drought" and "FOOD_RATION" not in active_types:
                grid_total = self._sim.grid.mean_food() * self._sim.grid.rows * self._sim.grid.cols
                ration_cap = max(2.0, min(4.0, grid_total / (n * 20)))
                self._enact_law("FOOD_RATION", {"max_per_cycle": ration_cap}, cycle,
                                duration=None, event_type="drought",
                                description=f"[ADS-FAST] Food ration {ration_cap:.1f} during drought.",
                                source="fast_response")
                active_types.add("FOOD_RATION")

            elif ev.event_type.value == "toxic_spill" and "FLEE_CURRENT_LOCATION" not in active_types:
                self._enact_law("FLEE_CURRENT_LOCATION", {}, cycle, duration=8,
                                description="[ADS-FAST] Flee toxic area.",
                                source="fast_response")
                active_types.add("FLEE_CURRENT_LOCATION")

        if "epidemic" in {ev.event_type.value for ev in self._sim.event_system.active_events if not ev.cancelled}:
            if "EPIDEMIC_RESPONSE" not in active_types:
                epi_id = None
                for ev in self._sim.event_system.active_events:
                    if ev.event_type.value == "epidemic" and not ev.cancelled:
                        epi_id = ev.epidemic_id
                        break
                self._enact_law("EPIDEMIC_RESPONSE", {"medicine_mandate": True}, cycle,
                                duration=None, event_id=epi_id, event_type="epidemic",
                                description="[ADS-FAST] Mandate medicine use for infected.",
                                source="fast_response")

        import statistics
        alive_by_id = {a.agent_id: a for a in alive}
        grid = self._sim.grid

        n_zones = max(2, min(6, int(n ** 0.5 / 3)))
        zone_rows = max(1, grid.rows // n_zones)
        zone_cols = max(1, grid.cols // n_zones)

        for resource, attr, law_type in [
            ("food", "food_stock", "REDISTRIBUTE_FOOD"),
            ("water", "water_stock", "REDISTRIBUTE_WATER"),
        ]:
            if law_type in active_types or n < 4:
                continue

            all_needy: List[str] = []
            all_donors: List[str] = []

            for zr in range(n_zones):
                for zc in range(n_zones):
                    r_min = zr * zone_rows
                    r_max = min(grid.rows - 1, (zr + 1) * zone_rows - 1)
                    c_min = zc * zone_cols
                    c_max = min(grid.cols - 1, (zc + 1) * zone_cols - 1)
                    zone_agents = [
                        a for a in alive
                        if a.position and r_min <= a.position[0] <= r_max
                        and c_min <= a.position[1] <= c_max
                    ]
                    if len(zone_agents) < 3:
                        continue
                    zone_stocks = [getattr(a, attr) for a in zone_agents]
                    zone_median = statistics.median(zone_stocks)
                    zone_critical = sum(1 for s in zone_stocks if s < 2.0)
                    if zone_critical / len(zone_agents) > 0.25 or zone_median < 3.0:
                        sorted_zone = sorted(zone_agents, key=lambda a: getattr(a, attr))
                        k = max(1, len(sorted_zone) * 30 // 100)
                        all_needy.extend(a.agent_id for a in sorted_zone[:k])
                    elif zone_median > 6.0:
                        sorted_zone = sorted(zone_agents, key=lambda a: getattr(a, attr), reverse=True)
                        d = max(1, len(sorted_zone) * 30 // 100)
                        all_donors.extend(a.agent_id for a in sorted_zone[:d])

            if not all_needy:
                global_stocks = [getattr(a, attr) for a in alive]
                global_median = statistics.median(global_stocks)
                pct_crit = sum(1 for s in global_stocks if s < 2.0) / n
                if pct_crit > 0.20 or global_median < 3.0:
                    sorted_all = sorted(alive, key=lambda a: getattr(a, attr))
                    k = max(2, n * 30 // 100)
                    all_needy = [a.agent_id for a in sorted_all[:k]]
                    all_donors = [a.agent_id for a in sorted_all[n * 70 // 100:]]

            if all_needy and all_donors:
                donor_amounts, actual_n = self._compute_redistribution_amounts(
                    alive_by_id, all_needy, all_donors, resource, 3.0
                )
                if donor_amounts:
                    self._enact_law(law_type, {
                        "resource": resource, "recipient_ids": all_needy,
                        "donor_amounts": donor_amounts, "amount_per_recipient": actual_n,
                    }, cycle, duration=6, applies_to=list(donor_amounts.keys()),
                        description=f"[ADS-FAST] Zone-targeted {resource} redistribution.",
                        source="fast_response")
                    active_types.add(law_type)

        # -----------------------------------------------------------------------
        # EMERGENCY DIRECT-STOCK RESCUE (runs every cycle, bypasses law system)
        #
        # Agents can deplete their food/water stocks while spending many cycles
        # seeking shelter (CitizenAgent early-returns without collecting during
        # shelter-seeking).  The zone-based redistribution above only fires when
        # an entire zone is in distress — individual outlier agents on depleted
        # cells can slip through.  This block directly transfers stock from the
        # grid cell or from nearby rich agents, guaranteeing no agent starves when
        # resources are globally available.
        # -----------------------------------------------------------------------
        # Rescue thresholds are intentionally conservative to avoid draining donors.
        # Only intervene for truly critical agents (< 1.5 stock), not as a general
        # top-up, to prevent cascading depletion across the population.
        RESCUE_FOOD = 1.5   # below this → actively rescue (conservative)
        RESCUE_WATER = 1.5
        DONOR_FLOOR = 6.0   # donors must keep at least this much after giving

        if self._sim:
            g = self._sim.grid
            # Pass 1: top up from the cell the agent is standing on
            for agent in alive:
                if not agent.position:
                    continue
                cell = g.cell(*agent.position)

                if agent.food_stock < RESCUE_FOOD and cell.food > 0.5:
                    give = min(cell.food, RESCUE_FOOD - agent.food_stock + 1.0, 3.0)
                    agent.food_stock += give
                    cell.food = max(0.0, cell.food - give)

                if agent.water_stock < RESCUE_WATER and cell.water > 0.5:
                    give = min(cell.water, RESCUE_WATER - agent.water_stock + 1.0, 3.0)
                    agent.water_stock += give
                    cell.water = max(0.0, cell.water - give)

            # Pass 2: peer-to-peer rescue for agents still critical after cell top-up
            for res_attr in ("food_stock", "water_stock"):
                rescue_thresh = RESCUE_FOOD if res_attr == "food_stock" else RESCUE_WATER
                critical = sorted(
                    [a for a in alive if getattr(a, res_attr, 0.0) < rescue_thresh],
                    key=lambda a: getattr(a, res_attr, 0.0),
                )
                if not critical:
                    continue
                donors = sorted(
                    [a for a in alive if getattr(a, res_attr, 0.0) > DONOR_FLOOR],
                    key=lambda a: -getattr(a, res_attr, 0.0),
                )
                if not donors:
                    continue
                di = 0
                for c_agent in critical:
                    need = rescue_thresh - getattr(c_agent, res_attr, 0.0)
                    while need > 0.01 and di < len(donors):
                        donor = donors[di]
                        avail = max(0.0, getattr(donor, res_attr, 0.0) - DONOR_FLOOR)
                        if avail < 0.01:
                            di += 1
                            continue
                        give = min(need, avail, 1.5)
                        setattr(donor, res_attr, getattr(donor, res_attr, 0.0) - give)
                        setattr(c_agent, res_attr, getattr(c_agent, res_attr, 0.0) + give)
                        need -= give

        # -----------------------------------------------------------------------
        # CONTINUOUS RESOURCE EQUALIZATION (federated-style, every 5 cycles)
        #
        # Like federated's _evaluate_trade_opportunities, this runs on a fixed
        # cadence and directly transfers food/water/medicine from surplus agents to
        # deficit agents.  It is dynamic — recomputed each firing with current
        # positions and stocks — so it is immune to the staleness of law-based
        # redistribution whose donor_amounts are frozen at enactment time.
        # -----------------------------------------------------------------------
        # Pre-compute epidemic state for use in both equalization and rescue blocks.
        n_active_epidemics = sum(
            1 for ev in self._sim.event_system.active_events
            if ev.event_type.value == "epidemic" and not ev.cancelled
        ) if self._sim else 0
        _infected_count = sum(1 for a in alive if a.infected)
        infection_rate = _infected_count / max(1, n)
        active_epidemic = n_active_epidemics > 0
        # During high epidemic, reduce equalization donor floors so surplus agents
        # can give more to infected agents who need extra resources to fight drain.
        high_epidemic = active_epidemic and infection_rate > 0.20

        if cycle % 5 == 0 and n >= 4:
            for res_attr, floor, cap_give in [
                ("food_stock",     4.0, 3.0),
                ("water_stock",    4.0, 3.0),
                ("medicine_stock", 3.0, 2.0),
            ]:
                all_stocks = [getattr(a, res_attr, 0.0) for a in alive]
                if not all_stocks:
                    continue
                import statistics as _stats
                g_median = _stats.median(all_stocks)
                if g_median <= 0:
                    continue

                deficit_agents = sorted(
                    [a for a in alive
                     if getattr(a, res_attr, 0.0) < g_median * 0.65
                     and getattr(a, res_attr, 0.0) < 4.0],
                    key=lambda a: getattr(a, res_attr, 0.0),
                )
                if not deficit_agents:
                    continue

                surplus_agents = sorted(
                    [a for a in alive
                     if getattr(a, res_attr, 0.0) > g_median * 1.3
                     and getattr(a, res_attr, 0.0) > floor],
                    key=lambda a: -getattr(a, res_attr, 0.0),
                )
                if not surplus_agents:
                    continue

                si = 0
                for d_agent in deficit_agents:
                    need = g_median - getattr(d_agent, res_attr, 0.0)
                    while need > 0.01 and si < len(surplus_agents):
                        s_agent = surplus_agents[si]
                        avail = max(0.0, getattr(s_agent, res_attr, 0.0) - floor)
                        if avail < 0.01:
                            si += 1
                            continue
                        give = min(need, avail, cap_give)
                        setattr(s_agent, res_attr,
                                getattr(s_agent, res_attr, 0.0) - give)
                        setattr(d_agent, res_attr,
                                getattr(d_agent, res_attr, 0.0) + give)
                        need -= give
                    if need <= 0.01:
                        pass  # fully topped up, keep iterating other deficit agents

        # -----------------------------------------------------------------------
        # EPIDEMIC MEDICINE RESCUE (runs every cycle during active infection)
        #
        # Proactively redistribute medicine from agents with large stocks to
        # infected agents who lack enough to cure all their infections.
        # Cure cost scales with the number of simultaneous active epidemics.
        # -----------------------------------------------------------------------
        if n_active_epidemics > 0 and infection_rate > 0.15:
            cure_cost = 3.0 * n_active_epidemics
            MEDICINE_DONOR_FLOOR = 5.0
            infected_alive = [a for a in alive if a.infected]

            medicine_poor_infected = sorted(
                [a for a in infected_alive if a.medicine_stock < cure_cost],
                key=lambda a: a.medicine_stock,
            )
            medicine_rich = sorted(
                [a for a in alive if a.medicine_stock > MEDICINE_DONOR_FLOOR],
                key=lambda a: -a.medicine_stock,
            )
            mi = 0
            for ill in medicine_poor_infected:
                need = cure_cost - ill.medicine_stock
                while need > 0.01 and mi < len(medicine_rich):
                    donor = medicine_rich[mi]
                    avail = max(0.0, donor.medicine_stock - MEDICINE_DONOR_FLOOR)
                    if avail < 0.01:
                        mi += 1
                        continue
                    give = min(need, avail, 2.0)
                    donor.medicine_stock -= give
                    ill.medicine_stock += give
                    need -= give

        # -----------------------------------------------------------------------
        # ECOLOGICAL DEPLETION-AWARE RATIONING (ADS-exclusive, every cycle)
        #
        # Reads cumulative grid extraction data to proactively enact food/water rations
        # before the exponential depletion threshold is crossed.  No other government
        # has access to this information.
        # -----------------------------------------------------------------------
        self._apply_depletion_aware_rationing(alive, cycle, active_types)

    # ------------------------------------------------------------------
    # Ecological depletion-aware rationing (ADS-exclusive)
    # ------------------------------------------------------------------

    def _apply_depletion_aware_rationing(
        self,
        alive: list,
        cycle: int,
        active_types: set,
    ) -> None:
        """
        Proactive resource rationing based on long-term ecological depletion trajectory.

        This heuristic is unique to ADS: it has awareness of the global
        resource depletion mechanic (cumulative extraction drives exponential ecological
        consequences) and acts to keep cumulative consumption below the threshold that
        triggers the exponential phase.

        Other governments react only to immediate agent stock levels; this government
        also observes cumulative grid pressure and rations preemptively.

        Rationing starts once the grid-wide food or water depletion fraction
        reaches 72% of the depletion threshold.  The per-cycle collection cap
        is cut by 20% at that point, rising linearly to 35% at the threshold
        and up to 50% beyond it, never below the drain-scaled survival floor.
        Each ration law lasts 12 cycles.
        """
        if not self._sim:
            return
        grid = self._sim.grid
        if grid.initial_grid_food <= 0:
            return

        n_alive = len(alive)
        if n_alive == 0:
            return

        from engine.grid import _DEPLETION_THRESHOLD

        # warning zone: start rationing at 72% of threshold to delay the exponential
        eco_warning = _DEPLETION_THRESHOLD * 0.72

        # Survival floor: minimum cap that ensures agents can meet their hunger drain.
        # hunger_drain = 0.02 health/cycle; eat_restore = 0.05 health/food unit.
        # Agents collect approximately every 3 cycles → need min_rate * 3 per event.
        #
        # NOT `difficulty_multiplier(self._sim.config.difficulty)` — that would
        # push the hidden difficulty knob through a public helper to
        # reconstruct `sim.drain_mult` EXACTLY.  It sits in a rationing
        # heuristic rather than in the evaluator, which is an easy site to miss
        # when auditing for reads of `sim.drain_mult` / `sim.config.regen_mult`
        # directly, and reading it here would make "ADS infers the environment
        # rather than reading it" false by a second, less obvious route.
        #
        # The estimator is not merely a legal substitute here, it is the more
        # appropriate input: the comment above reasons in EVALUATOR units
        # ("hunger_drain = 0.02 health/cycle", which is DRAIN_COEF_HUNGER, not
        # the engine's 0.025), and evaluator units are exactly what
        # `drain_mult_hat` is denominated in.
        #
        # Two consequences, stated rather than discovered: this widens the
        # behavioural diff slightly beyond the look-ahead path; and at cycle 0
        # the estimator returns 1.0 where `difficulty_multiplier` would return
        # 0.50-1.85, shifting `survival_floor` at run start.  The latter is
        # bounded by the enclosing max(3.0, min(5.0, ...)) clamp, so it cannot
        # run away.
        #
        # The arithmetic lives in the module-level `survival_floor` helper,
        # shared with the ecological-ration proposal in `propose_laws`, so the
        # two sites cannot independently drift.  See `survival_floor`'s
        # docstring.
        survival_floor_value = survival_floor(self._estimator.drain_mult)

        def _eco_cap(fraction: float, normal_max: float = 5.0) -> float:
            """
            Compute a graduated collection cap based on depletion progress.
            Uses percentage reduction (20–35% from eco_warning to threshold).
            The floor is drain-scaled so agents never starve from the ration.

            Reduction schedule:
              fraction = eco_warning: 20% reduction (cap ≈ 4.0 at d=25)
              fraction = threshold:   35% reduction (cap ≈ 3.25 at d=25)
              fraction > threshold:   35-50%, capped by survival_floor_value
            """
            if fraction < eco_warning:
                return normal_max
            if fraction < _DEPLETION_THRESHOLD:
                t = (fraction - eco_warning) / (_DEPLETION_THRESHOLD - eco_warning)
                reduction = 0.20 + t * 0.15          # 20% → 35%
            else:
                excess    = fraction - _DEPLETION_THRESHOLD
                reduction = 0.35 + min(0.15, excess * 2.0)   # 35% → 50%
            return max(survival_floor_value, normal_max * (1.0 - reduction))

        # --- Food ecological ration ---
        if "FOOD_RATION" not in active_types:
            food_fraction = grid.food_depletion_fraction
            if food_fraction >= eco_warning:
                eco_cap = _eco_cap(food_fraction)
                self._enact_law(
                    "FOOD_RATION",
                    {"max_per_cycle": round(eco_cap, 2)},
                    cycle,
                    duration=12,
                    description=(
                        f"[ADS-ECO] Food ecological ration: "
                        f"depletion={food_fraction:.2f}/{_DEPLETION_THRESHOLD:.2f}, "
                        f"cap={eco_cap:.1f}"
                    ),
                    source="eco_ration",
                )
                active_types.add("FOOD_RATION")

        # --- Water ecological ration ---
        if "LIMIT_WATER_CONSUMPTION" not in active_types:
            water_fraction = grid.water_depletion_fraction
            if water_fraction >= eco_warning:
                eco_cap_w = _eco_cap(water_fraction)
                self._enact_law(
                    "LIMIT_WATER_CONSUMPTION",
                    {"max_per_cycle": round(eco_cap_w, 2)},
                    cycle,
                    duration=12,
                    description=(
                        f"[ADS-ECO] Water ecological ration: "
                        f"depletion={water_fraction:.2f}/{_DEPLETION_THRESHOLD:.2f}, "
                        f"cap={eco_cap_w:.1f}"
                    ),
                    source="eco_ration",
                )
                active_types.add("LIMIT_WATER_CONSUMPTION")

    # ------------------------------------------------------------------
    # filter_actions override — inject emergency collection for critical agents
    # even when the base-class shelter law would cause an early return that
    # skips food/water collection in CitizenAgent.act().
    # ------------------------------------------------------------------

    def filter_actions(
        self, agent: "Agent", actions: List["Action"], cycle: int
    ) -> List["Action"]:
        from engine.agent import Action, ActionType

        modified = super().filter_actions(agent, actions, cycle)

        if not self._sim or not agent.position:
            return modified
        if not self._sim.grid.in_bounds(*agent.position):
            return modified

        # When an agent is critically low on food or water AND the shelter-seeking
        # early-return in CitizenAgent.act() would skip collection, inject COLLECT
        # actions at the FRONT so they execute before any movement.
        # Only fire for genuinely critical agents to avoid distorting normal foraging.
        if agent.food_stock < 2.0 or agent.water_stock < 2.0:
            cell = self._sim.grid.cell(*agent.position)
            food_limit = self.food_collection_limit(agent, cycle)
            water_limit = self.water_collection_limit(agent, cycle)

            if agent.food_stock < 2.0 and cell.food > 1.0:
                amt = min(cell.food, food_limit, max(2.0, 4.0 - agent.food_stock))
                if amt > 0.1:
                    modified.insert(0, Action(ActionType.COLLECT_FOOD, {"amount": amt}))

            if agent.water_stock < 2.0 and cell.water > 1.0:
                amt = min(cell.water, water_limit, max(2.0, 4.0 - agent.water_stock))
                if amt > 0.1:
                    modified.insert(0, Action(ActionType.COLLECT_WATER, {"amount": amt}))

        return modified

    # ------------------------------------------------------------------
    # Decision round
    # ------------------------------------------------------------------

    def _decision_round(self, cycle: int) -> None:
        """
        Multiple group distributions:

        For each grouping scheme in GROUP_DISTRIBUTION_COUNTS (that list is the
        single source of truth for which schemes are evaluated; it descends to
        1 = the whole population treated as one group):
          - Divide all alive agents into n_groups groups sorted by combined risk score.
          - For each group: propose candidate laws, evaluate each with a 20-cycle
            lookahead, and record scores normalized per-agent (total_health / group_size).
        Find the globally best (n_groups, group, law) triple by normalized score.
        Enact that single winning law for the corresponding agent subset.
        Store ALL results in self._last_decision_details for trail-log reporting.
        """
        sim = self._sim
        alive = [a for a in sim.agents if a.alive]
        n = len(alive)
        if n == 0:
            return

        # Step 1: Gather evidence (treat pending warnings as if events are live)
        warnings = list(self._pending_warnings)
        food_ev    = self._food_node.collect(sim, warnings)
        water_ev   = self._water_node.collect(sim, warnings)
        health_ev  = self._health_node.collect(sim, warnings)
        terrain_ev = self._terrain_node.collect(sim, warnings)

        # Step 2: Tag all alive agents once — reused across all grouping schemes
        tags = self._hg_node.generate_tags(
            alive, food_ev, water_ev, health_ev, terrain_ev, sim
        )

        self._decisions_made += 1
        # Already-active law types block re-enactment to prevent donor-draining stacks.
        # Dynamic equalization in _fast_response (running every 5 cycles) handles the
        # continuous need for redistribution; laws handle one-time targeted interventions.
        enacted_types: Set[str] = {l.law_type for l in self.active_laws}
        # Snapshot before the enactment loop mutates `enacted_types`, so the
        # record shows what was already in force when the round *started*.
        law_types_active_at_start: List[str] = sorted(enacted_types)

        self._last_evidence_scores = {
            "food": round(float(food_ev.mean_stock), 3),
            "water": round(float(water_ev.mean_stock), 3),
            "health": round(float(health_ev.mean_health), 3),
        }
        # Full evidence capture for the structured decision log.  asdict() means
        # every field of the four dataclasses is preserved verbatim — including
        # pct_starving / pct_dehydrated / cycles_until_empty / mean_hazard /
        # avg_dist_to_shelter / shelter_cell_count, which are computed on every
        # round but consumed by no heuristic.
        self._last_evidence = {
            "food": dataclasses.asdict(food_ev),
            "water": dataclasses.asdict(water_ev),
            "health": dataclasses.asdict(health_ev),
            "terrain": dataclasses.asdict(terrain_ev),
        }
        # Why a candidate (group, law) pair did not become a law.  The enactment
        # loop already branches on exactly these conditions; counting them turns
        # "ADS enacted nothing this round" from a mystery into a fact.
        rejected_reasons: Dict[str, int] = {
            "law_type_already_active": 0,
            "group_overlap": 0,
            "max_laws_reached": 0,
            "below_min_group_fraction": 0,
            "quarantine_bbox_unavailable": 0,
        }
        self._last_proposals_evaluated = 0
        self._last_top_proposal = ""
        self._last_top_score = 0.0

        # Build agent lookup dict once
        alive_by_id: Dict[str, "Agent"] = {a.agent_id: a for a in alive}

        # -----------------------------------------------------------------------
        # Step 3: Evaluate all grouping schemes, record ALL scores
        # -----------------------------------------------------------------------
        groupings_detail: Dict[int, List[Dict[str, Any]]] = {}
        # This computes `actual_n_groups = max(1, min(target, n))` and then
        # discards it.  Once the population collapses below 10, the "10-group"
        # scheme is not a 10-group scheme, and any analysis that assumes
        # otherwise is wrong.  Recording it costs one dict entry.
        # Kept alongside `groupings_detail` rather than nested inside it so
        # get_log_info()'s existing `groupings[key] -> rows` shape is untouched.
        n_groups_actual: Dict[int, int] = {}

        global_best_norm: float = -1.0
        global_winner_n_groups: int = 0
        global_winner_group: Optional[AgentGroup] = None
        global_winner_proposal: Optional[ProposedLaw] = None
        global_winner_raw: float = 0.0
        global_winner_agents: List["Agent"] = []

        all_candidates: List[Candidate] = []

        # The environment model, fixed ONCE for the whole round.
        #
        # Every candidate in this round faces the same model of the future,
        # which is half of the fair-comparison guarantee (the other half being
        # the shared injection stream, keyed on this same cycle).  This
        # property is enforced by IMMUTABILITY: EnvironmentModel is frozen,
        # so a candidate cannot perturb it for the ones scored after it.
        #
        # `source="inferred"` — this is the call that makes ADS the treatment
        # arm.  `observed_event_stats` is still invoked exactly once per round,
        # inside `environment_model`, so there is no performance change and no
        # arithmetic change to the event-statistics half of the model.
        env = self._estimator.environment_model(sim, cycle)

        for n_groups_target in GROUP_DISTRIBUTION_COUNTS:
            # Cap at actual alive population (can't have more groups than agents)
            actual_n_groups = max(1, min(n_groups_target, n))
            n_groups_actual[n_groups_target] = actual_n_groups
            groups = self._hg_node.form_groups(alive, tags, actual_n_groups)

            grouping_rows: List[Dict[str, Any]] = []

            for group in groups:
                if not group.agent_ids:
                    continue

                group_agents = [alive_by_id[aid] for aid in group.agent_ids
                                if aid in alive_by_id]
                k = len(group_agents)
                if k == 0:
                    continue

                # Compute group stats for the trail log
                healths = sorted(a.health for a in group_agents)
                foods   = sorted(a.food_stock for a in group_agents)
                waters  = sorted(a.water_stock for a in group_agents)
                mid = k // 2
                median_h = healths[mid]
                median_f = foods[mid]
                median_w = waters[mid]
                infected_count = sum(1 for a in group_agents if a.infected)

                # Generate candidate laws and score each
                proposals = self._hg_node.propose_laws(
                    group, food_ev, water_ev, health_ev, terrain_ev, sim,
                    env=env,
                )
                # Filter already-enacted law types (no double-enacting same type)
                proposals = [p for p in proposals if p.law_type not in enacted_types]

                law_scores: List[Dict[str, Any]] = []
                best_proposal_this_group: Optional[ProposedLaw] = None
                best_raw_this_group: float = -1.0
                best_norm_this_group: float = -1.0

                for proposal in proposals:
                    raw_score = self._evaluator.evaluate(
                        proposal, group_agents, sim,
                        cycle=cycle, env=env,
                    )
                    norm_score = raw_score / max(1, k)
                    self._last_proposals_evaluated += 1

                    # `norm_score` IS the forecast, and nothing between here and
                    # the enactment site may alter it.  It is what
                    # `_record_prediction` records, what `_close_predictions`
                    # grades, and (times the crisis boost) what candidates rank
                    # on.  There is deliberately no second, corrected number:
                    # accuracy is bought by the inferred parameters that went
                    # INTO `evaluate`, not by scaling what came out.
                    adjusted_score: Optional[float] = None
                    if k < max(1, int(n * MIN_GROUP_FRACTION)):
                        rejected_reasons["below_min_group_fraction"] += 1
                    else:
                        # Crisis-aware score boost: groups with critical food/water/health
                        # get a multiplier so the decision round prioritises the most
                        # vulnerable agents, not just the group with the highest baseline health.
                        crisis_boost = 1.0
                        if median_f < 2.5 or median_w < 2.5:
                            crisis_boost += 0.35   # resource crisis — urgent
                        if median_h < 0.5:
                            crisis_boost += 0.25   # health crisis — also urgent
                        if infected_count > 0:
                            crisis_boost += 0.10   # some infection
                        adjusted_score = norm_score * crisis_boost
                        all_candidates.append(Candidate(
                            adjusted_score=adjusted_score,
                            n_groups_target=n_groups_target,
                            group=group,
                            proposal=proposal,
                            group_agents=group_agents,
                            norm_score=norm_score,
                        ))

                    # One append per proposal, in proposal order.
                    # `calibration_multiplier` and `calibrated_norm_score` are
                    # GONE, not zeroed: there is no multiplier to report, and
                    # emitting a constant 1.0 would imply a mechanism that does
                    # not exist.  A reader pooling archives distinguishes the
                    # generations by `ads_forecast_schema_version`.
                    law_scores.append({
                        "law_type": proposal.law_type,
                        "description": proposal.description,
                        "source_category": proposal.source_category,
                        "raw_score": round(raw_score, 3),
                        "norm_score": round(norm_score, 4),
                        "adjusted_score": (
                            None if adjusted_score is None else round(adjusted_score, 4)
                        ),
                    })

                    # `best_law_type` / `best_norm_score` are
                    # descriptive statistics of the raw evaluation for one group
                    # (they drive the `*` marker in the text trail); the actual
                    # ranking is global, across every grouping scheme, and lives
                    # in `all_candidates`.  So `best_law_type` can legitimately
                    # differ from what was enacted, independent of any
                    # correction mechanism, and is not an oversight.
                    # Re-pointing it at the calibrated score would silently
                    # change the meaning of an already-published field.
                    if norm_score > best_norm_this_group:
                        best_norm_this_group = norm_score
                        best_raw_this_group = raw_score
                        best_proposal_this_group = proposal

                    if raw_score > self._last_top_score:
                        self._last_top_score = round(raw_score, 3)
                        self._last_top_proposal = proposal.law_type

                group_detail: Dict[str, Any] = {
                    "group_id": group.group_id,
                    "agent_count": k,
                    "median_health": round(median_h, 3),
                    "median_food": round(median_f, 3),
                    "median_water": round(median_w, 3),
                    "infected_count": infected_count,
                    "law_scores": law_scores,
                    "best_law_type": (
                        best_proposal_this_group.law_type
                        if best_proposal_this_group else ""
                    ),
                    "best_norm_score": round(best_norm_this_group, 4),
                }
                grouping_rows.append(group_detail)

                # Check if this is the global best across all grouping schemes
                if (best_norm_this_group > global_best_norm
                        and best_proposal_this_group is not None):
                    global_best_norm = best_norm_this_group
                    global_winner_n_groups = n_groups_target
                    global_winner_group = group
                    global_winner_proposal = best_proposal_this_group
                    global_winner_raw = best_raw_this_group
                    global_winner_agents = group_agents

            groupings_detail[n_groups_target] = grouping_rows

        # -----------------------------------------------------------------------
        # Step 4: Enact winning laws — multiple laws allowed if different types
        #         and each applies to >= 10% of the population.
        # -----------------------------------------------------------------------
        winner_info: Dict[str, Any] = {}
        enacted_records: List[Dict[str, Any]] = []

        # Global ranking across every grouping scheme, with a UNIFORM RANDOM
        # tie-break rather than enumeration order.
        #
        # This sort decides which law ADS actually enacts.  A plain descending
        # sort on the adjusted score would resolve exact ties to whichever
        # candidate was appended first, since Python's sort is stable — i.e. to
        # whichever grouping scheme appears earliest in
        # GROUP_DISTRIBUTION_COUNTS, then to group order, then to proposal
        # order.  None of those orderings carries any meaning about which
        # intervention is better; that would make the tie-break an enumeration
        # artifact sitting on the single most consequential decision in the
        # regime — an argmax that has been measured to amplify a 0.03%
        # configuration difference into a 44% outcome swing.
        #
        # Ties are exact-equality only, deliberately.  An epsilon band would
        # randomise among candidates that are merely close, which is a different
        # and much larger claim: it would let a measurably worse law win.
        # Exact ties are common here because identical proposals over
        # identically-composed groups produce bit-identical lookahead scores.
        #
        # Drawn from `self.rng`, the government's own decision stream, keyed per
        # (government, difficulty, run): ADS's choice among options it cannot
        # distinguish is a property of ADS, not of the environment, so it must
        # NOT come from a stream shared with the other regimes.  The draw is
        # unconditional — one per candidate per round, whether or not anything
        # ties — so RNG consumption does not depend on the tie pattern.
        # Sorting INDICES, not the candidate tuples: a candidate tuple holds an
        # AgentGroup and a ProposedLaw, neither of which is orderable, so any
        # key collision that fell through to comparing the tuples themselves
        # would raise TypeError mid-round.  With an index key the fallback is
        # the stable original order and there is nothing to compare.
        tie_breaks = [self.rng.random() for _ in all_candidates]
        order = sorted(
            range(len(all_candidates)),
            key=lambda i: (-all_candidates[i].adjusted_score, tie_breaks[i]),
        )
        all_candidates = [all_candidates[i] for i in order]
        enacted_agent_sets: List[Set[str]] = []
        max_laws = max(3, n // 20)

        for cand_idx, candidate in enumerate(all_candidates):
            norm_sc = candidate.adjusted_score
            n_g = candidate.n_groups_target
            group = candidate.group
            best = candidate.proposal
            group_agents = candidate.group_agents
            if len(enacted_agent_sets) >= max_laws:
                # Every remaining candidate is rejected for the same reason;
                # count them all rather than only the one that hit the break.
                rejected_reasons["max_laws_reached"] += len(all_candidates) - cand_idx
                break
            if best.law_type in enacted_types:
                rejected_reasons["law_type_already_active"] += 1
                continue
            agent_set = set(group.agent_ids)
            overlap = any(len(agent_set & prev) > len(agent_set) * 0.5 for prev in enacted_agent_sets)
            if overlap:
                rejected_reasons["group_overlap"] += 1
                continue

            params = dict(best.params)

            if best.law_type == "QUARANTINE_EPIDEMIC" and "region" not in params:
                qregion = AdsGovernment._compute_quarantine_bbox(sim)
                if qregion:
                    params["region"] = qregion
                else:
                    rejected_reasons["quarantine_bbox_unavailable"] += 1
                    continue

            if best.law_type in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER",
                                 "REDISTRIBUTE_MEDICINE"):
                resource = best.law_type.split("_", 1)[1].lower()
                recipient_ids = params.get("recipient_ids", group.agent_ids)
                recip_set = set(recipient_ids)
                donor_candidates = [a for a in alive if a.agent_id not in recip_set]
                n_amount = params.get("amount_per_recipient", 3.0)
                donor_amounts, actual_n = self._compute_redistribution_amounts(
                    alive_by_id, list(recipient_ids),
                    [a.agent_id for a in donor_candidates], resource, n_amount,
                )
                params["donor_amounts"] = donor_amounts
                params["amount_per_recipient"] = actual_n
                params["recipient_ids"] = list(recipient_ids)
                applies_to: List[str] = list(donor_amounts.keys()) or list(group.agent_ids)

            elif best.law_type == "REDISTRIBUTE_GENERIC":
                recipient_ids = params.get("recipient_ids", group.agent_ids)
                recip_set = set(recipient_ids)
                donor_candidates = [a for a in alive if a.agent_id not in recip_set]
                donor_ids_all = [a.agent_id for a in donor_candidates]
                combined_amounts: Dict[str, Dict[str, float]] = {}
                for res in ("food", "water", "medicine"):
                    n_res = params.get(f"amount_per_recipient_{res}", 2.0)
                    da, _ = self._compute_redistribution_amounts(
                        alive_by_id, list(recipient_ids), donor_ids_all, res, n_res,
                    )
                    for did, amt in da.items():
                        combined_amounts.setdefault(did, {})[res] = amt
                params["donor_amounts"] = combined_amounts
                params["recipient_ids"] = list(recipient_ids)
                applies_to = list(combined_amounts.keys()) or list(group.agent_ids)

            else:
                applies_to = (
                    list(best.applies_to) if best.applies_to else list(group.agent_ids)
                )

            enact_event_id: Optional[str] = None
            enact_event_type: Optional[str] = None
            if best.law_type in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE"):
                enact_event_type = "epidemic"
                for _ev in sim.event_system.active_events:
                    if _ev.event_type.value == "epidemic" and not _ev.cancelled:
                        enact_event_id = _ev.epidemic_id
                        break
            elif best.law_type == "FOOD_RATION" and any(
                _ev.event_type.value == "drought" and not _ev.cancelled
                for _ev in sim.event_system.active_events
            ):
                enact_event_type = "drought"
            elif best.law_type in ("MANDATORY_SHELTER", "MOVE_TO_REGION") and any(
                _ev.event_type.value == "storm" and not _ev.cancelled
                for _ev in sim.event_system.active_events
            ):
                enact_event_type = "storm"

            law = self._enact_law(
                best.law_type,
                params,
                cycle,
                duration=best.duration,
                description=(
                    best.description
                    + f" [norm={norm_sc:.4f}, grouping={n_g}g]"
                ),
                applies_to=applies_to,
                event_id=enact_event_id,
                event_type=enact_event_type,
                source="decision_round",
            )
            # Open the prediction.  NOT `candidate.adjusted_score`: that is the
            # crisis-boosted ranking score, a priority in [1.0, 1.70] x the
            # forecast, not a health projection — differencing it against
            # realized health would measure how often crises occurred.
            # `group.agent_ids` — NOT `applies_to` — is the population the
            # forecast was made over; for REDISTRIBUTE_* the two are disjoint.
            # Both traps are spelled out in full in forecast_ledger.py's module
            # docstring.
            #
            # `n_groups=n_g` lets an analyst select ADS's single-group rounds
            # (`GROUP_DISTRIBUTION_COUNTS` includes 1, so ADS already grades
            # over the whole living population on those rounds) and compare
            # them to A+L at exactly matched scope.  Without the tag that
            # subset is unrecoverable from the archive and the paper's
            # scope-matched secondary series cannot be built at all — the
            # alternative, aggregating the risk-banded groups into a
            # "population" statistic, was rejected as methodologically weaker.
            # `n_g` is already in scope in this loop iteration: the same
            # variable that `enacted_records.append` writes below and that the
            # law description interpolates as `grouping={n_g}g` a few lines up.
            # Those two write paths are disjoint from this one, which is what
            # makes the archive cross-validation able to catch a transposition
            # here.
            #
            # `outcome="enacted"` is constant on this arm: ADS only ever opens a
            # forecast on an enactment, because `ads.py`'s candidate filter
            # removes standing law types before `evaluate()` is reached, so there
            # is no "keep what is already there" candidate to score.  Making that
            # evaluator-driven was considered and rejected; do not design it.
            # A+L, which does have a null candidate, emits five values here.
            # The two tags are duals — each arm's informative axis is the
            # other's degenerate one.
            self._ledger.open(
                cycle=cycle,
                predicted=candidate.norm_score,
                evaluated_agent_ids=list(group.agent_ids),
                tags={
                    "law_id": law.law_id,
                    "outcome": "enacted",
                    "n_groups": n_g,
                    "category": best.source_category,
                    "law_type": best.law_type,
                    "applies_to": list(applies_to or ()),
                    "adjusted_at_enactment": float(norm_sc),
                },
            )

            enacted_types.add(best.law_type)
            enacted_agent_sets.append(agent_set)
            self._laws_enacted += 1

            # The loop enacts up to max(3, n // 20) laws per round, but
            # winner_info only ever captured the FIRST.  Recording the complete
            # list is the difference between seeing ADS's output and seeing a
            # sample of it.
            enacted_records.append({
                "law_id": law.law_id,
                "law_type": best.law_type,
                "n_groups": n_g,
                "group_id": group.group_id,
                "adjusted_score": round(norm_sc, 4),
                "applies_to_count": (None if applies_to is None else len(applies_to)),
            })

            if not winner_info:
                winner_info = {
                    "n_groups": n_g,
                    "group_id": group.group_id,
                    "law_type": best.law_type,
                    "description": best.description,
                    # `norm_score` is retained verbatim (it is what the text
                    # trail prints) even though it carries the crisis-boosted
                    # value; `adjusted_score` names it honestly and
                    # `base_norm_score` is the un-boosted figure.
                    "norm_score": round(norm_sc, 4),
                    "adjusted_score": round(norm_sc, 4),
                    "base_norm_score": round(candidate.norm_score, 4),
                }

        # -----------------------------------------------------------------------
        # Step 5: Persist full detail for trail log and the structured
        #         decision record.
        # -----------------------------------------------------------------------
        self._last_decision_details = {
            "groupings": groupings_detail,
            "n_groups_actual": n_groups_actual,
            "winner": winner_info,
            "enacted": enacted_records,
            "rejected_reasons": rejected_reasons,
            "n_distributions": len(GROUP_DISTRIBUTION_COUNTS),
            "context": {
                "alive": n,
                "min_group_fraction": MIN_GROUP_FRACTION,
                "max_laws": max_laws,
                "already_active_law_types": law_types_active_at_start,
            },
        }
        # Marks this round as belonging to THIS cycle.  Without it the recorder
        # cannot distinguish "a round ran now" from "a round ran five cycles
        # ago and its details are still sitting here".
        self._last_decision_cycle = cycle

        # One INFO line per round summarising the forecast pass.  `closed=` is
        # the count from this cycle's _close_predictions, which for a
        # tick-driven round ran moments ago; for an out-of-band round triggered
        # by receive_event_warnings it is the most recent tick's figure, since
        # no closure sweep happens on the warning path.
        #
        # The inferred parameters are logged alongside, because "what did ADS
        # BELIEVE about the world when it made this decision" is otherwise
        # unanswerable from a stored run — and those beliefs are the entire
        # mechanism by which the forecast is supposed to improve.
        mean_abs = self._ledger.mean_abs_delta()
        self._logger.info(
            "cycle=%d FORECAST_ROUND open=%d closed_this_cycle=%d closed_total=%d "
            "degenerate=%d mean_abs_err=%s env=%s",
            cycle, self._ledger.open_predictions,
            self._ledger.closed_this_cycle,
            self._ledger.closed_total, self._ledger.n_degenerate,
            "n/a" if mean_abs is None else f"{mean_abs:.6f}",
            env.as_dict(),
        )
        # The phantom future every candidate in this round was scored against.
        # Logged once here rather than once per `evaluate()` — the schedule is
        # identical across candidates by construction (shared injection stream,
        # shared statistics), and that identity is the fair-ranking guarantee,
        # so one line per round is both cheaper and the more honest record.
        #
        # The cycle stamp is CHECKED, not assumed.  A round that scores no
        # candidates never calls `evaluate`, so without this guard the line
        # would report whatever schedule the previous round left behind —
        # which reads exactly like a seeding bug (two different cycles
        # "sharing" a phantom future) and was mistaken for one once already.
        if self._logger.isEnabledFor(10):      # logging.DEBUG
            summary = self._evaluator.last_phantom_summary
            fresh = summary.get("cycle") == cycle
            self._logger.debug(
                "cycle=%d EVAL_PHANTOM stats=%s schedule=%s",
                cycle,
                {
                    cat: {
                        "n": s.get("n"),
                        "rate": round(float(s.get("rate") or 0.0), 5),
                        "effect": round(float(s.get("effect") or 0.0), 5),
                        "duration": s.get("duration"),
                    }
                    for cat, s in sorted(env.event_stats.items())
                },
                summary.get("injected", []) if fresh
                else f"<no candidate evaluated this round; last was cycle "
                     f"{summary.get('cycle')}>",
            )

    @staticmethod
    def _compute_quarantine_bbox(
        sim: "Simulation",
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the smallest bounding box that contains all infected agents,
        expanded by a 1-cell buffer so agents on the boundary can still move
        slightly without immediately escaping containment.

        Named `_bbox` rather than `_region` deliberately: Government (base.py)
        defines an instance method `_compute_quarantine_region(infected, n_alive,
        grid_rows, grid_cols)` with a different signature and a radius-based
        (not tight-bounding-box) region.  Keeping the names distinct prevents this
        static method from shadowing the inherited one.
        """
        infected = [a for a in sim.agents if a.alive and a.infected and a.position]
        if not infected:
            return None
        rows = [a.position[0] for a in infected]
        cols = [a.position[1] for a in infected]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        buffer = 1  # 1-cell buffer around the tight bounding box
        return (
            max(0, r_min - buffer),
            max(0, c_min - buffer),
            min(sim.grid.rows - 1, r_max + buffer),
            min(sim.grid.cols - 1, c_max + buffer),
        )

    # ------------------------------------------------------------------
    # Event warnings — trigger immediate pre-emptive decision round
    # ------------------------------------------------------------------

    def receive_event_warnings(
        self, warnings: List["EventWarning"], cycle: int
    ) -> None:
        self._pending_warnings.extend(warnings)
        critical = {"epidemic", "drought", "storm", "toxic_spill"}
        has_critical = any(w.event_type.value in critical for w in warnings)
        if has_critical:
            active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
            for w in warnings:
                if w.event_type.value == "storm" and w.cycles_until <= 3:
                    if "MANDATORY_SHELTER" not in active_types:
                        self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                        event_type="storm",
                                        description="[ADS-PREEMPT] Shelter from storm warning.",
                                        source="preempt_warning")
                        active_types.add("MANDATORY_SHELTER")
            # A warning-driven round is neither of tick()'s two interval kinds:
            # it is out-of-band and can fire on any cycle, including one that
            # tick() will run a second round on.
            self._last_decision_trigger = {
                "kind": "warning_preempt",
                "interval": None,
                "pending_warnings": [w.event_type.value for w in warnings],
                "t_decision": self.t_decision,
            }
            self._decision_round(cycle)

    # ------------------------------------------------------------------
    # The predicted/realized accuracy pass — lives in forecast_ledger.py
    # ------------------------------------------------------------------
    #
    # ``_record_prediction``, ``_close_predictions`` and ``_mean_abs_delta``
    # live in ``governments/forecast_ledger.py``, shared with
    # ``autocracy_lookahead`` rather than duplicated per arm.
    #
    # THE MECHANISM IS DOCUMENTED THERE, DELIBERATELY, NOT HERE.  It explains
    # why the forecast is the evaluator's own projection and not the
    # crisis-boosted ranking score, why the population is ``evaluated_agent_ids``
    # and not ``applies_to``, why the review is a DATE and not a WINDOW — none of
    # which is ADS-specific.  Documenting it here instead would read as though
    # the ledger belongs to the treatment arm.  Read it there before changing
    # anything about grading.
    #
    # Two call sites live in this file:
    #
    #   self._ledger.open(...)    from _decision_round, right after _enact_law
    #   self._ledger.close(...)   from tick, on EVERY cycle
    #
    # This is a MEASUREMENT loop, not a control loop.  Nothing it computes is
    # fed back into ranking, scoring or the forecast.  The mechanism by which
    # accuracy is expected to improve over a run lives entirely upstream, in
    # ParameterEstimator's inference on the evaluator's INPUTS.

    def get_calibration_record(self, cycle: int) -> Dict[str, Any]:
        """
        Per-cycle forecast block for ``run_detail.jsonl``.

        Pure read of state already computed during ``tick``; no side effects, so
        it is safe to call on every cycle.  Unlike ``get_decision_record`` this
        never returns None — the point of the series is that it is dense, so a
        consumer can plot mean_abs_delta against cycle without reindexing.

        KEEPS ITS NAME despite now reporting parameter estimates rather than a
        calibration table, and the reason is sharper than inertia: this method
        is reached by ``getattr`` duck-typing in
        ``engine/run_recorder.py:_calibration_payload``.  A missed rename there
        would not raise — the ``getattr`` returns ``None`` and the entire block
        silently vanishes from every ``run_detail.jsonl``.  That is the worst
        available failure mode, so the name stays and a canary field
        (``ads_forecast_schema_version``, in the summary below) is the cheaper
        and safer way to prove the chain is intact.
        """
        return {
            # The five ledger fields, in their original emission order.  Held
            # byte-identical by a no-op gate against ADS's historical output:
            # `run_detail.jsonl` gains nothing new here, and neither
            # `n_closed_no_law` nor either partition appears here — those are
            # per-RUN summaries.
            **self._ledger.as_record(),
            # REPLACES `calibration_table`.  What ADS currently believes about
            # the environment's hidden dynamics — the entire mechanism by which
            # its forecast is meant to improve, so this is the series a reader
            # should plot against the error series.
            "parameter_estimates": self._estimator.as_record(),
        }

    def get_calibration_summary(
        self, total_cycles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Per-run forecast-accuracy summary, merged into ``final_stats.json``.

        The arithmetic lives in :meth:`ForecastLedger.as_summary` — read that
        docstring for the half-split semantics, the ``n_`` counts, and
        (importantly) for WHY three of the scalars it returns are unfiltered
        all-time means that no A+L figure may read directly.

        This method assembles: the ledger's block, plus the two keys that belong
        to the government rather than to the measurement.  See the comment below.
        """
        # The half-split arithmetic, the eleven scalars and the flat
        # `calibration_cycle_deltas` series all live in
        # `ForecastLedger.as_summary`.  Three of those keys -- `n_closed_no_law`
        # and the two `*_by_outcome_then_scope` partitions -- are the no-op
        # gate's entire allowlist; a fourth key fails it.
        #
        # Two keys are added HERE rather than in the ledger, and the split is
        # deliberate.  Neither is a property of the measurement:
        #
        #   * `ads_forecast_schema_version` is a PROVENANCE TOKEN for humans and
        #     ad-hoc scripts, whose meaning is defined by the generation table
        #     at the top of this file.  The three additive keys partition an
        #     existing measurement rather than introducing a new one, so this
        #     is not bumped for them — a bump would invalidate that table for
        #     no semantic change.  The `ads_` prefix is VESTIGIAL: the field is
        #     not ADS-specific.  `autocracy_lookahead` also emits the same key
        #     with the same value, because provenance is a property of the RUN,
        #     not of the regime, and both governments compute it with the same
        #     code.
        #   * `parameter_estimates_final` is the ESTIMATOR's state, not the
        #     ledger's.  Inference is upstream of and independent from the
        #     measurement loop; keeping the two objects' outputs assembled here
        #     is what keeps that visible.
        summary = self._ledger.as_summary(total_cycles)
        summary["ads_forecast_schema_version"] = ADS_FORECAST_SCHEMA_VERSION
        # What ADS ended the run believing about the environment.  Pairs with the
        # per-cycle `parameter_estimates` series in run_detail.jsonl and makes a
        # completed run self-describing on the one mechanism that is supposed to
        # drive its accuracy.
        summary["parameter_estimates_final"] = self._estimator.as_record()
        return summary

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_ads_metrics(self) -> Dict[str, Any]:
        return {
            "decisions_made": self._decisions_made,
            "laws_enacted":   self._laws_enacted,
            "laws_active":    len(self.active_laws),
            # MetricsCollector.record reads `mean_prediction_error` off this
            # dict; leaving it unset would silently report None every cycle
            # even though the hook exists specifically for this feature.
            # Filling it costs no schema change and gives health_stats.csv a
            # per-cycle error series for free.
            # `mean_calibration` and `proposals_submitted` are
            # left None deliberately: their intended semantics are not recorded
            # anywhere and inventing one would be worse than an empty column.
            "mean_prediction_error": self._ledger.mean_abs_delta(),
        }

    def get_audit_info(self, cycle: int) -> dict:
        ev = self._last_evidence_scores
        ev_str = " ".join(f"{k}:{v}" for k, v in ev.items() if v != "")
        return {
            "decisions_made_total": self._decisions_made,
            "laws_enacted_total": self._laws_enacted,
            "proposals_evaluated_this_round": self._last_proposals_evaluated,
            "top_proposal": self._last_top_proposal,
            "top_proposal_score": self._last_top_score,
            "evidence_scores": ev_str,
        }

    def get_params(self) -> Dict[str, Any]:
        """Static ADS tunables, recorded once in each run's detail header."""
        return {
            "t_decision": self.t_decision,
            "lookahead_cycles": LOOKAHEAD_CYCLES,
            "group_distribution_counts": list(GROUP_DISTRIBUTION_COUNTS),
            "min_group_fraction": MIN_GROUP_FRACTION,
            # --- The three-arm ablation's self-description --------------------
            #
            # PRESENT ON BOTH ARMS, and that symmetry is the point.  A+L's
            # header describes its own epistemics ("environment_model") and so
            # does ADS's, so the two arms of a comparison are comparably
            # self-describing and a reader never has to infer one arm's
            # position from the absence of a key.  Inferring a regime from which
            # keys are MISSING is the anti-pattern; a field that says so is the
            # fix.  Never ship one of these without the other.
            #
            # `environment_model` is NOT ARM-DISCRIMINATING.  Both arms say
            # "inferred", which is the correct report rather than information
            # loss: the field answers "which construction path built the
            # evaluator's environment", and that question has exactly two
            # answers.  Do not add a third value to make it regime-identifying;
            # that would create an incentive for rollout logic to branch on a
            # tag whose contract is DIAGNOSTIC-ONLY.
            "environment_model": "inferred",
            "parameter_inference_enabled": True,
            # `organization` IS the axis the ablation isolates:
            #     autocracy (no foresight)
            #       -> autocracy_lookahead (good foresight, no organisation)
            #       -> ads (good foresight + organisation)
            #
            # Do not prune this as redundant.  It IS mechanically redundant —
            # `group_distribution_counts` and `min_group_fraction` here, and
            # `target_groups` / `base_regime` on A+L, let a determined reader
            # work the arm out.  But they do so BY IMPLICATION and by key
            # presence, which is the same anti-pattern one level down: a reader
            # must already know that `group_distribution_counts` implies an
            # organisational tier.  This is the single explicit, positive
            # statement of the axis the whole comparison now turns on.  The
            # other six regimes do not emit it, and their absence reads
            # correctly as "this regime has no organisational tier to describe".
            "organization": "multinode",
            # Recorded in the header so a stored run is self-describing: the
            # parameter estimates in its cycle records cannot be interpreted
            # without knowing which constants produced them, and these are
            # expected to be sensitivity-swept.
            #
            # The five `calibration_*` constants are not emitted: they would
            # name a correction mechanism this file does not implement, and
            # emitting them with any value would imply it is still operating.
            #
            # The crude baseline is recorded even though ADS does not use it,
            # because it is the reference point the inferred values are read
            # against — "drain_mult 1.42" is only interpretable next to "crude
            # would have been 1.0".
            "crude_drain_mult": CRUDE_DRAIN_MULT,
            "crude_regen_mult": CRUDE_REGEN_MULT,
            "crude_max_steps_per_cycle": CRUDE_MAX_STEPS_PER_CYCLE,
            "crude_event_rate": CRUDE_EVENT_RATE,
            "estimator_prior_cycles": ESTIMATOR_PRIOR_CYCLES,
            "estimator_drain_bounds": [DRAIN_MULT_MIN, DRAIN_MULT_MAX],
            "estimator_regen_bounds": [REGEN_MULT_MIN, REGEN_MULT_MAX],
            "regen_obs_max_fill": REGEN_OBS_MAX_FILL,
            "ads_forecast_schema_version": ADS_FORECAST_SCHEMA_VERSION,
            # The phantom-event model.  Without these the projected scores in a
            # stored run are not reproducible from the record alone.
            "phantom_categories": list(PHANTOM_CATEGORIES),
            "event_rate_prior_cycles": EVENT_RATE_PRIOR_CYCLES,
            "event_rate_cap": EVENT_RATE_CAP,
        }

    def get_decision_record(self, cycle: int) -> Optional[Dict[str, Any]]:
        """
        Structured record of the decision round that ran on THIS cycle, or None.

        Assembled from state already captured during ``_decision_round``; this
        method computes nothing new and has no side effects, so it is safe to
        call on every cycle.  Returns ``None`` unless a round actually ran on
        *this* cycle — ``_last_decision_details`` persists between rounds, and
        emitting it unconditionally would duplicate a stale record onto every
        intervening cycle.
        """
        if self._last_decision_cycle != cycle:
            return None

        details = self._last_decision_details or {}
        actual = details.get("n_groups_actual", {}) or {}
        groupings_by_target = details.get("groupings", {}) or {}

        # An array, not the internal dict: JSON object keys must be strings, so
        # a dict keyed by n_groups would serialise to {"10": …} — lossy in type
        # and hostile to numeric ordering in jq.  The array preserves both the
        # integer key and the evaluation order.
        groupings = [
            {
                "n_groups_target": target,
                "n_groups_actual": actual.get(target, target),
                "groups": groupings_by_target.get(target, []),
            }
            for target in GROUP_DISTRIBUTION_COUNTS
            if target in groupings_by_target
        ]

        evidence = self._last_evidence or {}
        health_evidence = dict(evidence.get("health", {}))
        if isinstance(health_evidence.get("epidemic_ids"), (set, frozenset)):
            # Emitted sorted so two identical runs produce identical records.
            health_evidence["epidemic_ids"] = sorted(health_evidence["epidemic_ids"])

        return {
            "mechanism": "ads_evidence_tree",
            "trigger": dict(self._last_decision_trigger or {}),
            "context": details.get("context", {}),
            "evidence": {
                "food": evidence.get("food", {}),
                "water": evidence.get("water", {}),
                "health": health_evidence,
                "terrain": evidence.get("terrain", {}),
            },
            "summary": {
                "n_distributions": details.get(
                    "n_distributions", len(GROUP_DISTRIBUTION_COUNTS)
                ),
                "group_distribution_counts": list(GROUP_DISTRIBUTION_COUNTS),
                "proposals_evaluated": self._last_proposals_evaluated,
                "top_proposal": self._last_top_proposal,
                "top_raw_score": self._last_top_score,
                "laws_enacted_this_round": len(details.get("enacted", [])),
            },
            "groupings": groupings,
            "winner": details.get("winner", {}),
            "enacted": details.get("enacted", []),
            "rejected_reasons": details.get("rejected_reasons", {}),
        }

    def get_log_info(self, cycle: int) -> dict:
        """
        Enhanced trail logs.

        When a decision was made this round (_last_decision_details is populated),
        the log includes:
          - Number of group distributions evaluated — one per entry in
            GROUP_DISTRIBUTION_COUNTS; the counts and labels below are rendered
            from that list, so they follow it automatically if it changes.
          - For each grouping scheme:
            - Number of groups in that distribution
            - For each group: agent count, median health/food/water, infected count
            - For each group: ALL proposed laws with raw and normalized scores
          - The overall winning combination: grouping, group, law, norm score
        """
        ev = self._last_evidence_scores
        details = self._last_decision_details

        # ---- Decision summary ------------------------------------------------
        summary_parts = []
        if self._last_proposals_evaluated > 0:
            summary_parts.append(
                f"Evidence-based decision round: evaluated {self._last_proposals_evaluated} proposals "
                f"across {details.get('n_distributions', len(GROUP_DISTRIBUTION_COUNTS))} "
                f"group distributions ({', '.join(str(g) for g in GROUP_DISTRIBUTION_COUNTS)} groups)."
            )
            winner = details.get("winner", {})
            if winner:
                summary_parts.append(
                    f"    Winner: grouping={winner['n_groups']}g "
                    f"group={winner['group_id']} "
                    f"law={winner['law_type']} "
                    f"norm_score={winner['norm_score']:.4f}"
                )

        # ---- Trigger ---------------------------------------------------------
        trigger_parts = []
        if self._pending_warnings:
            warning_types = [w.event_type.value for w in self._pending_warnings]
            trigger_parts.append(f"event warnings: {', '.join(warning_types)}")
        if not trigger_parts:
            trigger_parts.append(f"scheduled decision round (every {self.t_decision} cycles)")

        # ---- Evidence scores ------------------------------------------------
        evidence_lines = []
        for domain, score in ev.items():
            if score != "":
                evidence_lines.append(f"    {domain}: {score}")
        evidence_str = (
            "\n".join(evidence_lines) if evidence_lines
            else "    (no evidence collected this cycle)"
        )

        # ---- Per-grouping detail -----------------------------------------
        grouping_lines: List[str] = []
        groupings = details.get("groupings", {})
        for n_groups_key in GROUP_DISTRIBUTION_COUNTS:
            rows = groupings.get(n_groups_key, [])
            if not rows:
                grouping_lines.append(f"  [{n_groups_key}g] (no groups formed)")
                continue
            grouping_lines.append(
                f"  [{n_groups_key}g] {len(rows)} group(s):"
            )
            for gd in rows:
                grouping_lines.append(
                    f"    group={gd['group_id']} "
                    f"n={gd['agent_count']} "
                    f"health={gd['median_health']:.3f} "
                    f"food={gd['median_food']:.3f} "
                    f"water={gd['median_water']:.3f} "
                    f"infected={gd['infected_count']}"
                )
                for ls in gd.get("law_scores", []):
                    marker = "*" if ls["law_type"] == gd["best_law_type"] else " "
                    grouping_lines.append(
                        f"      {marker} {ls['law_type']} "
                        f"raw={ls['raw_score']:.3f} norm={ls['norm_score']:.4f} "
                        f"| {ls['description'][:70]}"
                    )
                if not gd.get("law_scores"):
                    grouping_lines.append("      (no proposals — law types already active)")

        grouping_detail_str = "\n".join(grouping_lines) if grouping_lines else ""

        # ---- Winner summary --------------------------------------------------
        winner = details.get("winner", {})
        if winner:
            winner_line = (
                f"  WINNER: {winner['law_type']} "
                f"from {winner['n_groups']}-group distribution, "
                f"group {winner['group_id']}, "
                f"norm_score={winner['norm_score']:.4f}"
            )
        else:
            winner_line = "  WINNER: none (all law types already active)"

        narrative = (
            f"  Total decisions made: {self._decisions_made} | "
            f"Total laws enacted: {self._laws_enacted}\n"
            f"  Evidence scores:\n{evidence_str}\n"
            f"  Decision method: 4-node evidence tree -> "
            f"{len(GROUP_DISTRIBUTION_COUNTS)} grouping distributions -> "
            f"{LOOKAHEAD_CYCLES}-cycle lookahead evaluation (normalized per-agent)\n"
            + (f"\n{grouping_detail_str}\n{winner_line}" if grouping_detail_str else "")
        )

        return {
            "decision_summary": "\n".join(summary_parts),
            "decision_trigger": "; ".join(trigger_parts),
            "vote_details": {},
            "state_narrative": narrative,
        }


# ---------------------------------------------------------------------------
# Implementation notes
# ---------------------------------------------------------------------------
# Event statistics come from ``EventSystem.observed_event_stats(cycle)`` in
# engine/events.py, derived from the existing ``history`` list rather than
# from separate counters that could desync from it.  It reports rate +
# severity + effect + duration + n per category, since simulation start, and
# lives on ``EventSystem`` rather than on either government, so ``ads`` and
# ``autocracy_lookahead`` read provably identical numbers.  The magnitude
# injected during a rollout is the observed ``effect`` (mean drought_factor /
# storm_damage / disease_spread_rate) rather than ``severity * CONVERSION``,
# because the benchmark builds its events in ``engine/scenario_plan.py``,
# whose epidemic conversion differs from this engine's (0.15 vs 0.20); reading
# the realised field cannot drift.
#
# This file implements no multiplicative forecast-calibration mechanism (a
# per-(category, law_type) EWMA correction feeding ranking directly) — see the
# module docstring above for why: it measures worse than no correction at all
# in every run-cell tested.  There is no calibration table, multiplier or
# correction anywhere in this file, by design, not by omission.
#
# ``AdsGovernment.tick`` calls ``self._estimator.observe(cycle, self._sim)``
# as the first statement after the ``_sim`` guard, outside the decision gate,
# so the estimator updates on every cycle regardless of whether a decision
# round runs that cycle.
#
# INVARIANT TO CHECK FIRST: nothing downstream of ``evaluate()`` scales, clips
# or post-processes its return value.  ``norm_score`` reaches
# ``_record_prediction`` unmodified, and ``adjusted_score`` multiplies it only
# by ``crisis_boost``, which is a ranking priority and is never graded.
#
# Deliberately out of scope, flagged rather than silently skipped:
# reconciling the evaluator's drain coefficients with the engine's (the
# estimator absorbs the mismatch instead — see parameter_inference's
# ``DRAIN_COEF_*`` table); the evaluator's movement-model divergences from the
# engine (see ``EvaluatorNode._step``); and refactoring
# ``observed_event_stats`` onto ``ShrunkRatioEstimator``.
# ---------------------------------------------------------------------------
