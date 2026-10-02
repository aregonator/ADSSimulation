"""
Scenario definitions and simulation factory.

Each scenario specifies environment config, events, government type, and agent
composition. Grid size, agent count, and max_cycles are NOT stored here — they
come from CLI arguments with difficulty-based defaults.

When no --scenario is given, build_auto_simulation() runs a comprehensive
difficulty-driven mode that schedules all event types.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Type, Tuple

from engine.simulation import Simulation, SimulationConfig
from engine.difficulty import DIFFICULTY_KNEE_LEVEL, difficulty_t, past_knee
from engine.events import ActiveEvent, EventType
from engine.agent import Agent
from governments.base import Government
from governments import GOVERNMENT_REGISTRY
from agents.citizen import CitizenAgent


# ---------------------------------------------------------------------------
# Population factories
# ---------------------------------------------------------------------------

def _make_citizen_population(n: int, seed: int = 42) -> List[Agent]:
    """Build *n* CitizenAgents, each with its own generator derived from *seed*.

    In the benchmark, *seed* is ``derive_seed(base_seed, "pop", difficulty,
    run_idx)``.  Government identity is deliberately NOT a key: "all governments
    face identical starting conditions" includes the agents' initial state, so
    all eight regimes get the same population, agent for agent and generator for
    generator.  They diverge because behaviour consumes those generators in a
    different order, which is what "same start, different outcome" means.

    Population HETEROGENEITY is still not created here.  Every agent gets full
    ``initial_health`` and the config's ``metabolic_rate``; only the RNG differs.
    Drawing per-agent health / metabolism / risk tolerance from *seed* is the
    obvious extension and this function is exactly where it would attach, but it
    is a modelling decision for the authors, deliberately deferred: it would
    invent variation the model does not currently claim and move every
    published number.  Do not add it as a "fix".

    *seed* is not inert: ``CitizenAgent`` draws from ``self.rng`` in
    ``_move_smart`` and ``_choose_vote`` (agents/citizen.py) to break
    iteration-order ties, so the stream is live and *seed* changes outcomes.
    Two consequences worth knowing:

    * there is no northward drift imposed on tied moves, and so no
      population-scale artifact of ``DIRECTION_DELTAS`` insertion order;
    * a cell's ``n_distinct`` can differ from 1 through this path as well as
      through the per-run environment.
    """
    rng = random.Random(seed)
    return [CitizenAgent(agent_id=f"C{i:03d}", seed=rng.randint(0, 9999)) for i in range(n)]


# ADS population uses the same CitizenAgent — alias kept for import compatibility.
_make_ads_population = _make_citizen_population


# ---------------------------------------------------------------------------
# Scenario definitions
# Note: grid_rows, grid_cols, max_cycles, and initial_health are intentionally
# absent — these come from CLI args with difficulty-based defaults.
# ---------------------------------------------------------------------------

ScenarioDef = Dict[str, Any]


SCENARIOS: Dict[str, ScenarioDef] = {

    "basic_survival": {
        "description": "Steady resource depletion with light random events. Tests baseline governance.",
        "config": {
            "num_agents": 80,
            "resource_density": 0.8,
            "terrain_variety": 0.4,
            "difficulty": 25,
            "event_frequency": 0.04,
            "event_severity": 0.8,
            "event_warning_cycles": 2,
            "visibility_radius": 6,
        },
        "scheduled_events": [],
    },

    "epidemic": {
        "description": "A fast-spreading disease. Tests quarantine and medicine coordination.",
        "config": {
            "num_agents": 100,
            "resource_density": 0.9,
            "terrain_variety": 0.3,
            "difficulty": 25,
            "event_frequency": 0.02,
            "event_severity": 1.2,
            "event_warning_cycles": 1,
            "visibility_radius": 5,
        },
        "scheduled_events": [
            (15, EventType.EPIDEMIC, 0, 1.5),
        ],
    },

    "resource_scarcity": {
        "description": "Severe drought tests rationing and redistribution under scarcity.",
        "config": {
            "num_agents": 100,
            "resource_density": 0.6,
            "terrain_variety": 0.5,
            "difficulty": 50,
            "event_frequency": 0.02,
            "event_severity": 1.5,
            "event_warning_cycles": 3,
            "visibility_radius": 5,
        },
        "scheduled_events": [
            (5,  EventType.DROUGHT, 0, 2.0),
            (40, EventType.DROUGHT, 0, 1.5),
        ],
    },

    "climate_crisis": {
        "description": "Escalating sequence: drought → storm → toxic event → epidemic. Tests adaptive governance.",
        "config": {
            "num_agents": 120,
            "resource_density": 0.75,
            "terrain_variety": 0.5,
            "difficulty": 50,
            "event_frequency": 0.02,
            "event_severity": 1.3,
            "event_warning_cycles": 1,
            "visibility_radius": 5,
        },
        "scheduled_events": [
            (10,  EventType.DROUGHT,     0, 1.5),
            (50,  EventType.STORM,       0, 1.2),
            (80,  EventType.TOXIC_SPILL, 0, 1.4),
            (110, EventType.EPIDEMIC,    0, 1.6),
        ],
    },

    "ads_validation": {
        "description": (
            "Designed to validate ADS components: "
            "Phase 1 = normal operation, Phase 2 = novel event, "
            "Phase 3 = conflicting proposals, Phase 4 = calibration convergence test."
        ),
        "config": {
            "num_agents": 100,
            "resource_density": 0.8,
            "terrain_variety": 0.4,
            "difficulty": 25,
            "event_frequency": 0.01,
            "event_severity": 1.0,
            "event_warning_cycles": 2,
            "visibility_radius": 5,
        },
        "scheduled_events": [
            (20,  EventType.EPIDEMIC,    0, 1.0),
            (80,  EventType.TOXIC_SPILL, 0, 1.5),
            (140, EventType.DROUGHT,     0, 1.0),
            (140, EventType.EPIDEMIC,    0, 0.8),
            (220, EventType.STORM,       0, 1.2),
        ],
    },
}


# ---------------------------------------------------------------------------
# Auto scenario (difficulty-driven, no --scenario flag)
# ---------------------------------------------------------------------------

def _raw_base_severity(t: float) -> float:
    """The event-severity lerp: 0.60 at t=0 (D1), 1.50 at t=1 (D100 at slope
    1.0).  At the shipped ``DIFFICULTY_TAIL_SLOPE = 1.75``, ``t`` reaches
    1.0758 at D100, so this returns 1.57 there (rounded by
    :func:`_auto_base_severity`).  ONE expression, referenced twice in
    :func:`_auto_event_schedule`.  Uncapped above t=1: raises no floor or
    ceiling of its own, so it extrapolates cleanly if ``t`` exceeds 1.0 above
    the knee -- this channel needs no physical bound."""
    return 0.6 + t * 0.9


def _auto_n_waves(difficulty: int) -> int:
    """Event-wave count: 0 at D1, 9 at D90, 11 at D100 (shipped slope 1.75;
    10 at slope 1.0).  The 11th wave (i=10) is a scheduled EPIDEMIC -- see
    :func:`_auto_event_schedule`'s ``all_types[i % 4]`` cycling: D100
    schedules 3 epidemics against 2 at D90-D95, a real change in hazard
    composition, not just count.

    There is deliberately no ceiling on the count beyond
    ``engine.difficulty.DIFFICULTY_T_MAX`` (``round(10 * 1.25) == 12``),
    enforced package-wide by ``validate_schedule()``.  A literal ``min(10,
    ...)`` sitting at the old t=1 endpoint would freeze the wave count from
    the first tuned tail slope onward, at precisely the level where the
    discrete channels carry the most signal.  At slope 1.0 an unenforced
    ``min(10, ...)`` would be a no-op (``min(10, round(t*10)) ==
    round(t*10)`` for every d in 1..100); at the shipped slope 1.75, D100
    rounds to 11, one above that value -- exactly the relaxation that lets
    the 11th (epidemic) wave schedule.

    Extracted into its own function so ``benchmark_core.py``'s manifest
    ``schedule_digest`` can hash the real wave-count formula instead of
    re-transcribing it.
    """
    t = difficulty_t(difficulty)
    return max(0, round(t * 10))


def _auto_base_severity(difficulty: int) -> float:
    """The clamped, rounded event severity :func:`_auto_event_schedule` uses:
    the raw lerp (:func:`_raw_base_severity`) evaluated at *difficulty*, held
    at the pre-knee (D90) ceiling below the knee via :func:`past_knee`
    scoping, then rounded to 2 decimals.  0.60 at D1, 1.40 at D90, 1.57 at
    D100 (shipped slope 1.75; 1.50 at slope 1.0); flat at 1.40 for D >= 90 at
    slope 0.0.  The ceiling is not a literal: it is the same lerp evaluated
    at the last pre-knee level, so it follows any change to the lerp or the
    knee automatically.

    Extracted into its own function, rather than re-implemented inline by
    each caller, so a test can assert on the REAL severity computation: a
    local re-implementation of this clamp can silently diverge from it (for
    example if the D90 severity-ceiling clamp above were dropped, reverting
    to the unconditional ``round(min(0.6 + t * 0.9, 1.40), 2)``) without a
    test noticing, since the wave count alone would still be read from real
    code.
    """
    t = difficulty_t(difficulty)
    raw_severity = _raw_base_severity(t)
    if not past_knee(difficulty):
        raw_severity = min(
            raw_severity,
            _raw_base_severity(difficulty_t(DIFFICULTY_KNEE_LEVEL - 1)))
    return round(raw_severity, 2)


def _auto_event_schedule(
    max_cycles: int,
    difficulty: int,
    seed: int,
) -> List[Tuple[int, EventType, int, float]]:
    """
    Generate a comprehensive event schedule covering all event types.
    Spacing and severity scale with difficulty.

    Returns list of (start_cycle, event_type, _, severity).
    """
    rng = random.Random(seed + 77)

    # Canonical difficulty schedule.  Wave count and base severity both share
    # one interpolation position, via `difficulty_t` inside each helper below,
    # rather than each re-deriving its own -- a second independent expression
    # of "where difficulty stops scaling" than the engine's knee is exactly
    # what would let this schedule and the engine's silently drift apart.
    #
    # Wave count and base severity are both extracted into their own functions
    # so a test -- and the manifest's `schedule_digest` in benchmark_core.py --
    # can read the REAL formula instead of a re-implementation.  See each
    # helper's docstring for what it computes and why.
    n_waves = _auto_n_waves(difficulty)
    base_severity = _auto_base_severity(difficulty)

    # Spread events evenly across the timeline, with jitter
    segment = max_cycles / (n_waves + 1) if n_waves > 0 else max_cycles

    # Only harmful events on this scheduled path.  (RESOURCE_RUSH can still
    # arrive through the engine's random-event path.)
    all_types = [
        EventType.DROUGHT,
        EventType.STORM,
        EventType.EPIDEMIC,
        EventType.TOXIC_SPILL,
    ]

    schedule = []
    for i in range(n_waves):
        center = segment * (i + 1)
        jitter = rng.uniform(-segment * 0.15, segment * 0.15)
        cycle = max(5, min(max_cycles - 10, round(center + jitter)))
        event_type = all_types[i % len(all_types)]
        sev = round(base_severity * rng.uniform(0.85, 1.15), 3)
        schedule.append((cycle, event_type, 0, sev))

    schedule.sort(key=lambda x: x[0])
    return schedule


def build_auto_simulation(
    government_name: str,
    difficulty: int = 25,
    seed: int = 42,
    verbose: bool = False,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    max_cycles: Optional[int] = None,
    num_agents: Optional[int] = None,
    **config_overrides: Any,
) -> Simulation:
    """
    Build a comprehensive difficulty-driven simulation with no fixed scenario.
    All parameters scale from difficulty unless explicitly overridden.
    Schedules all event types across the timeline.
    """
    if government_name not in GOVERNMENT_REGISTRY:
        raise ValueError(f"Unknown government '{government_name}'. Choose from: {list(GOVERNMENT_REGISTRY)}")

    config = SimulationConfig.from_difficulty(difficulty, seed=seed, verbose=verbose)

    # Apply CLI overrides if provided
    if grid_rows is not None:
        config.grid_rows = grid_rows
    if grid_cols is not None:
        config.grid_cols = grid_cols
    if max_cycles is not None:
        config.max_cycles = max_cycles
    if num_agents is not None:
        config.num_agents = num_agents
    for key, val in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)

    gov_cls: Type[Government] = GOVERNMENT_REGISTRY[government_name]
    government = gov_cls(seed=seed) if government_name != "anarchy" else gov_cls()

    agents = _make_citizen_population(config.num_agents, seed=seed)

    sim = Simulation(config=config, government=government, agents=agents)

    # Schedule comprehensive auto events
    auto_events: ScenarioDef = {
        "scheduled_events": _auto_event_schedule(config.max_cycles, difficulty, seed),
    }
    _schedule_scenario_events(sim, auto_events, seed)

    return sim


# ---------------------------------------------------------------------------
# Simulation factory
# ---------------------------------------------------------------------------

def build_simulation(
    scenario_name: str,
    government_name: str,
    seed: int = 42,
    verbose: bool = False,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    max_cycles: Optional[int] = None,
    **config_overrides: Any,
) -> Simulation:
    """
    Build a fully configured simulation from a scenario name and government type.
    grid_rows, grid_cols, max_cycles come from CLI args; if None, scenario uses
    difficulty-based defaults via SimulationConfig.from_difficulty().
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from: {list(SCENARIOS)}")
    if government_name not in GOVERNMENT_REGISTRY:
        raise ValueError(f"Unknown government '{government_name}'. Choose from: {list(GOVERNMENT_REGISTRY)}")

    scenario = SCENARIOS[scenario_name]
    cfg_dict = dict(scenario["config"])
    cfg_dict.update(config_overrides)
    cfg_dict["seed"] = seed
    cfg_dict["verbose"] = verbose
    cfg_dict["initial_health"] = 1.0  # always full health

    # Ensure difficulty is an integer in 1-100
    d = cfg_dict.get("difficulty", 25)
    if isinstance(d, str):
        mapping = {"easy": 12, "normal": 25, "hard": 50, "extreme": 100}
        d = mapping.get(d.lower(), 25)
    cfg_dict["difficulty"] = max(1, min(100, int(d)))

    # Grid size and max_cycles: use CLI args if provided, else fall through to
    # SimulationConfig's own field defaults (20x20 grid, 200 cycles — see
    # SimulationConfig.from_difficulty: these fields are not difficulty-scaled).
    if grid_rows is not None:
        cfg_dict["grid_rows"] = grid_rows
    if grid_cols is not None:
        cfg_dict["grid_cols"] = grid_cols
    if max_cycles is not None:
        cfg_dict["max_cycles"] = max_cycles

    config = SimulationConfig(**cfg_dict)

    gov_cls: Type[Government] = GOVERNMENT_REGISTRY[government_name]
    if government_name == "anarchy":
        government = gov_cls()
    else:
        government = gov_cls(seed=seed)

    agents = _make_citizen_population(config.num_agents, seed=seed)

    sim = Simulation(config=config, government=government, agents=agents)
    _schedule_scenario_events(sim, scenario, seed)

    return sim


def _schedule_scenario_events(sim: Simulation, scenario: ScenarioDef, seed: int) -> None:
    rng = random.Random(seed)

    for entry in scenario.get("scheduled_events", []):
        start_cycle, event_type, _, severity = entry

        if event_type == EventType.EPIDEMIC:
            r0 = rng.randint(0, sim.grid.rows - 1)
            c0 = rng.randint(0, sim.grid.cols - 1)
            ev = ActiveEvent(
                event_type=EventType.EPIDEMIC,
                start_cycle=start_cycle,
                duration=rng.randint(10, 20),
                severity=severity,
                disease_spread_rate=0.15 * severity,
                epidemic_seed_row=r0,
                epidemic_seed_col=c0,
                description=f"Scheduled epidemic (severity={severity}) seed=({r0},{c0}).",
            )
        elif event_type == EventType.DROUGHT:
            ev = ActiveEvent(
                event_type=EventType.DROUGHT,
                start_cycle=start_cycle,
                duration=rng.randint(8, 15),
                severity=severity,
                drought_factor=0.35 * severity,
                description=f"Scheduled drought (severity={severity}).",
            )
        elif event_type == EventType.STORM:
            ev = ActiveEvent(
                event_type=EventType.STORM,
                start_cycle=start_cycle,
                duration=rng.randint(5, 10),
                severity=severity,
                storm_damage=0.04 * severity,
                description=f"Scheduled storm (severity={severity}).",
            )
        elif event_type == EventType.TOXIC_SPILL:
            r0 = rng.randint(2, max(2, sim.grid.rows - 4))
            c0 = rng.randint(2, max(2, sim.grid.cols - 4))
            region = (r0, c0, min(r0 + 3, sim.grid.rows - 1), min(c0 + 3, sim.grid.cols - 1))
            ev = ActiveEvent(
                event_type=EventType.TOXIC_SPILL,
                start_cycle=start_cycle,
                duration=rng.randint(10, 20),
                severity=severity,
                region=region,
                toxic_drain=0.06 * severity,
                description=f"Scheduled toxic spill at {region} (severity={severity}).",
            )
        elif event_type == EventType.RESOURCE_RUSH:
            r0 = rng.randint(0, max(0, sim.grid.rows - 4))
            c0 = rng.randint(0, max(0, sim.grid.cols - 4))
            region = (r0, c0, min(r0 + 3, sim.grid.rows - 1), min(c0 + 3, sim.grid.cols - 1))
            ev = ActiveEvent(
                event_type=EventType.RESOURCE_RUSH,
                start_cycle=start_cycle,
                duration=rng.randint(5, 10),
                severity=severity,
                region=region,
                resource_boost=5.0 * severity,
                description=f"Scheduled resource rush at {region} (severity={severity}).",
            )
        else:
            continue

        sim.event_system.schedule(start_cycle, ev)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY = list(SCENARIOS.keys())
