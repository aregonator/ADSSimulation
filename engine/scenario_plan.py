"""
ScenarioPlan: pre-generated grid + events plan for fair government comparison.

Use generate_scenario_plan() to create a plan once, save/load it with
ScenarioPlan.save() / ScenarioPlan.load(), then build each government's
simulation with build_simulation_from_plan() so every run uses the same
grid layout, agent starting positions, and event schedule.

Fairness across governments is an equality of *derivation*, not of object
identity: a plan is a pure function of ``(base_seed, difficulty, run_idx)``, so
any process that evaluates that function gets the identical plan without a plan
object being shipped anywhere.  :func:`derive_seed` is the function that makes
that claim true, and :meth:`ScenarioPlan.fingerprint` is how the claim is
checked after the fact.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .events import EventType, ActiveEvent

if TYPE_CHECKING:
    from .grid import Grid
    from .agent import Agent
    from governments.base import Government
    from .simulation import Simulation, SimulationConfig


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

#: Width of a derived seed.  63 rather than 64 so the result is always a
#: non-negative signed 64-bit integer: it round-trips through JSON, SQLite and
#: every consumer that might narrow it, without ever going negative.
_SEED_BITS = 63
_SEED_MASK = (1 << _SEED_BITS) - 1


def derive_seed(base_seed: int, domain: str, *parts: Any) -> int:
    """
    Derive a reproducible sub-seed from ``base_seed`` for a named domain.

    The whole seed lattice of the benchmark is built from this one function.
    ``domain`` names the random stream (``"env"``, ``"env.grid"``, ``"pop"``,
    ``"gov"``, ``"gov.eval"``, …) and ``parts`` are the lattice keys that stream
    depends on — which keys are passed *is the design*.  Two streams that must
    never alias differ in their domain label; two runs that must differ pass a
    different ``run_idx``.

    Why blake2b and not something simpler
    -------------------------------------
    * ``hash()`` is disqualified outright.  ``PYTHONHASHSEED`` is randomised per
      process and the sweep starts its workers with ``spawn``, so two workers
      would derive *different* "identical" environments and the fairness
      property would fail silently.
    * Arithmetic offsets (``base_seed + run_idx``) cannot express a three-key
      lattice without aliasing: ``(d=50, run=3)`` and ``(d=53, run=0)`` collide
      the moment difficulty enters the key.
    * ``numpy.random.SeedSequence`` would work but couples every
      ``random.Random`` consumer in the tree to numpy, and its output is less
      legible when printed into run metadata.

    blake2b is in the stdlib, is stable across Python versions and platforms,
    and the digest does not depend on any process-local state.  Golden-value
    tests in ``test_seed_derivation.py`` pin specific inputs to specific
    outputs, so a future refactor cannot silently re-key the entire corpus and
    invalidate every archived run without a test failing.

    Parameters
    ----------
    base_seed  The sweep's root seed (``BenchmarkConfig.base_seed``).
    domain     Non-empty stream label.  Dots denote sub-streams by convention;
               the function treats the whole string as opaque.
    parts      Lattice keys, stringified in the order given.  Order is
               significant: ``("ads", 50, 3)`` and ``(50, 3, "ads")`` derive
               different seeds.

    Returns
    -------
    int in ``[0, 2**63)``.

    Notes
    -----
    Parts are joined with ``"|"`` and are not escaped, so a part containing a
    literal ``"|"`` could in principle collide with a different part tuple
    (``("a|b",)`` vs ``("a", "b")``).  Every real key is a government name, an
    int difficulty, an int run index or an int event index, none of which can
    contain the separator; adding escaping would change every derived seed for
    no gain, so it is deliberately not done.  If a future key can contain
    ``"|"``, escape it *and* bump a domain label so the change is visible.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError(
            f"base_seed must be an int, got {type(base_seed).__name__}"
        )
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")

    payload = f"{base_seed}|{domain}|" + "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioPlan:
    """
    Immutable snapshot of a simulation environment that can be reused across
    government types for fair comparison.

    Fields
    ------
    rows, cols          Grid dimensions.
    grid_cells          [r][c] dicts with keys terrain, food, water, medicine.
    agent_placements    (row, col) starting position for each agent by index.
    events_plan         All events (scheduled + random) as serialisable dicts.
    n_cycles            Number of cycles the plan covers.
    seed                RNG seed used to generate the plan.
    config_params       Key SimulationConfig values used for reconstruction.
    """

    rows: int
    cols: int
    grid_cells: List[List[dict]]
    agent_placements: List[Tuple[int, int]]
    events_plan: List[dict]
    n_cycles: int
    seed: int
    config_params: Dict[str, Any]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "grid_cells": self.grid_cells,
            "agent_placements": [list(p) for p in self.agent_placements],
            "events_plan": self.events_plan,
            "n_cycles": self.n_cycles,
            "seed": self.seed,
            "config_params": self.config_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioPlan":
        return cls(
            rows=d["rows"],
            cols=d["cols"],
            grid_cells=d["grid_cells"],
            agent_placements=[tuple(p) for p in d["agent_placements"]],
            events_plan=d["events_plan"],
            n_cycles=d["n_cycles"],
            seed=d["seed"],
            config_params=d["config_params"],
        )

    def save(self, path: str) -> None:
        """Export plan to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScenarioPlan":
        """Import plan from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Fairness witness
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """
        Return a short content hash of the entire plan.

        This is the *witness* for the two claims the published results rest on:

        * FAIRNESS — every government at a given ``(difficulty, run_idx)`` faced
          the identical environment.  All their fingerprints are equal.
        * VARIATION — the ``k_runs`` runs at a given difficulty faced *different*
          environments.  Their fingerprints are pairwise distinct.

        Because plans are derived independently in each worker rather than
        shipped as one shared object, those claims are guaranteed by purity
        rather than by object identity.  Purity
        is only an acceptable guarantee if it is checked, so the fingerprint is
        recorded per run and the sweep asserts both properties over the recorded
        values (``benchmark_core.verify_env_fingerprints``).  Someone who did not
        run the sweep can re-check both from the archive alone.

        Canonical JSON (``sort_keys=True``, no whitespace) is hashed rather than
        the object, so the value is stable across processes, across
        ``PYTHONHASHSEED`` settings, and across a ``save()``/``load()``
        round-trip.  Truncated to 16 hex chars (64 bits): ample against the
        ~16,800 plans a sweep produces, and short enough to sit in a CSV column
        and a log line.  Costs ~0.3 ms on a production-scale plan.
        """
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_events = len(self.events_plan)
        cp = self.config_params
        return (
            f"ScenarioPlan  seed={self.seed}  grid={self.rows}×{self.cols}  "
            f"agents={cp.get('num_agents', '?')}  cycles={self.n_cycles}  "
            f"events={n_events}  difficulty={cp.get('difficulty', '?')}"
        )


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_scenario_plan(
    config: "SimulationConfig",
    n_cycles: Optional[int] = None,
    scheduled_events: Optional[List[dict]] = None,
) -> ScenarioPlan:
    """
    Pre-generate a reproducible ScenarioPlan for *n_cycles* cycles.

    Parameters
    ----------
    config            SimulationConfig providing grid/population/event params.
    n_cycles          Cycles to generate events for. Defaults to config.max_cycles.
    scheduled_events  Deterministic events as dicts (use _make_scheduled_event_dict()).

    Returns
    -------
    ScenarioPlan  Ready for export with .save() or direct use with
                  build_simulation_from_plan().
    """
    from .grid import Grid

    n_cycles = n_cycles or config.max_cycles

    # Three independent sub-streams, domain-separated off the plan's own seed,
    # each derived by name via `derive_seed` rather than by an arithmetic
    # offset off `config.seed` (e.g. the same integer for placements and grid,
    # `config.seed + 1` for random events).  Two `random.Random`s one integer
    # apart are not correlated in any way anyone has demonstrated, but "seed"
    # and "seed + 1" is the kind of adjacency that becomes an aliasing bug the
    # first time somebody adds a fourth stream at `+ 1` too.  Deriving each by
    # name removes the last place in the plan path where two streams are
    # related by arithmetic, and makes the set of sub-streams
    # self-documenting.
    placement_rng = random.Random(derive_seed(config.seed, "env.placement"))
    grid_seed = derive_seed(config.seed, "env.grid")
    event_rng = random.Random(derive_seed(config.seed, "env.events.random"))

    # --- Build grid ---
    grid = Grid(
        rows=config.grid_rows,
        cols=config.grid_cols,
        resource_density=config.resource_density,
        terrain_variety=config.terrain_variety,
        seed=grid_seed,
        ambient_hazard=config.ambient_hazard,
    )

    grid_cells: List[List[dict]] = [
        [
            {
                "terrain": cell.terrain.value,
                "food": round(cell.food, 3),
                "water": round(cell.water, 3),
                "medicine": round(cell.medicine, 3),
            }
            for cell in row
        ]
        for row in grid._cells
    ]

    # --- Agent placements (clustered, non-water) ---
    from .grid import cluster_positions as _cluster_positions
    agent_placements = _cluster_positions(grid, config.num_agents, placement_rng)

    # --- Events plan ---
    events_plan: List[dict] = list(scheduled_events or [])

    for cycle in range(n_cycles):
        if event_rng.random() < config.event_frequency:
            ev = _random_event_dict(
                event_rng, cycle,
                config.grid_rows, config.grid_cols,
                config.event_severity,
            )
            events_plan.append(ev)

    events_plan.sort(key=lambda e: e["cycle"])

    config_params: Dict[str, Any] = {
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "resource_density": config.resource_density,
        "terrain_variety": config.terrain_variety,
        "num_agents": config.num_agents,
        "initial_health": config.initial_health,
        "initial_food_stock": getattr(config, "initial_food_stock", 10.0),
        "initial_water_stock": getattr(config, "initial_water_stock", 10.0),
        "difficulty": config.difficulty,
        "event_frequency": config.event_frequency,
        "event_severity": config.event_severity,
        "event_warning_cycles": config.event_warning_cycles,
        "regen_mult": getattr(config, "regen_mult", 1.0),
        "metabolic_rate": getattr(config, "metabolic_rate", 0.08),
        # An environment parameter that varies with difficulty, recorded so the
        # plan is self-describing and so ScenarioPlan.fingerprint covers it.
        "ambient_hazard": config.ambient_hazard,
        "visibility_radius": config.visibility_radius,
        "max_steps_per_cycle": getattr(config, "max_steps_per_cycle", 1),
        "seed": config.seed,
        "max_cycles": config.max_cycles,
    }

    return ScenarioPlan(
        rows=config.grid_rows,
        cols=config.grid_cols,
        grid_cells=grid_cells,
        agent_placements=agent_placements,
        events_plan=events_plan,
        n_cycles=n_cycles,
        seed=config.seed,
        config_params=config_params,
    )


def _random_event_dict(
    rng: random.Random,
    cycle: int,
    grid_rows: int,
    grid_cols: int,
    base_severity: float,
) -> dict:
    """Return a single random event as a plain dict (no side effects)."""
    weights = [0.25, 0.20, 0.25, 0.15, 0.15]
    event_type = rng.choices(list(EventType), weights=weights, k=1)[0]
    severity = round(base_severity * rng.uniform(0.7, 1.3), 4)

    entry: dict = {
        "cycle": cycle,
        "event_type": event_type.value,
        "severity": severity,
        "duration": 0,
        "drought_factor": 0.0,
        "storm_damage": 0.0,
        "disease_spread_rate": 0.0,
        "toxic_drain": 0.0,
        "resource_boost": 0.0,
        "region": None,
        "epidemic_seed_row": None,
        "epidemic_seed_col": None,
        "description": "",
        "source": "random",
    }

    if event_type == EventType.DROUGHT:
        entry["duration"] = int(rng.uniform(8, 15))
        entry["drought_factor"] = round(0.35 * severity, 4)
        entry["description"] = f"Drought (severity={severity:.2f})"

    elif event_type == EventType.STORM:
        entry["duration"] = int(rng.uniform(3, 7))
        entry["storm_damage"] = round(0.04 * severity, 5)
        entry["description"] = f"Storm (severity={severity:.2f})"

    elif event_type == EventType.EPIDEMIC:
        entry["duration"] = int(rng.uniform(8, 15))  # finite spread window
        entry["disease_spread_rate"] = round(0.15 * severity, 4)
        r0 = rng.randint(0, grid_rows - 1)
        c0 = rng.randint(0, grid_cols - 1)
        entry["epidemic_seed_row"] = r0
        entry["epidemic_seed_col"] = c0
        entry["description"] = f"Epidemic (severity={severity:.2f}) seed=({r0},{c0})"

    elif event_type == EventType.TOXIC_SPILL:
        entry["duration"] = int(rng.uniform(5, 15))
        entry["toxic_drain"] = round(0.08 * severity, 5)
        r_min = rng.randint(0, max(0, grid_rows - 3))
        c_min = rng.randint(0, max(0, grid_cols - 3))
        entry["region"] = [r_min, c_min,
                           min(r_min + 2, grid_rows - 1),
                           min(c_min + 2, grid_cols - 1)]
        entry["description"] = f"Toxic spill (severity={severity:.2f})"

    else:  # RESOURCE_RUSH
        entry["duration"] = int(rng.uniform(5, 10))
        entry["resource_boost"] = round(5.0 * severity, 3)
        r_min = rng.randint(0, max(0, grid_rows - 4))
        c_min = rng.randint(0, max(0, grid_cols - 4))
        entry["region"] = [r_min, c_min,
                           min(r_min + 3, grid_rows - 1),
                           min(c_min + 3, grid_cols - 1)]
        entry["description"] = f"Resource rush (severity={severity:.2f})"

    return entry


def make_scheduled_event_dict(
    cycle: int,
    event_type: str,
    severity: float,
    duration,               # int for finite events; None or omit for epidemics
    grid_rows: int = 20,
    grid_cols: int = 20,
    seed: int = 0,
) -> dict:
    """
    Helper to create a scheduled (deterministic) event dict for use with
    generate_scenario_plan(scheduled_events=[...]).

    Parameters
    ----------
    cycle       Cycle when the event starts.
    event_type  EventType value string, e.g. "drought", "epidemic".
    severity    Event severity multiplier.
    duration    How many cycles the event lasts. Pass None for epidemics (indefinite).
    grid_rows, grid_cols  Grid dimensions (needed for region/seed sampling).
    seed        RNG seed for location sampling.
    """
    rng = random.Random(seed)
    et = EventType(event_type)

    # Use the caller-provided duration. For epidemics, None means indefinite spread;
    # a finite int limits the spread window (agents remain infected until recovery).
    resolved_duration = duration

    entry: dict = {
        "cycle": cycle,
        "event_type": et.value,
        "severity": severity,
        "duration": resolved_duration,
        "drought_factor": 0.0,
        "storm_damage": 0.0,
        "disease_spread_rate": 0.0,
        "toxic_drain": 0.0,
        "resource_boost": 0.0,
        "region": None,
        "epidemic_seed_row": None,
        "epidemic_seed_col": None,
        "description": f"Scheduled {et.value} (severity={severity})",
        "source": "scheduled",
    }

    if et == EventType.DROUGHT:
        entry["drought_factor"] = round(0.35 * severity, 4)

    elif et == EventType.STORM:
        entry["storm_damage"] = round(0.04 * severity, 5)

    elif et == EventType.EPIDEMIC:
        entry["disease_spread_rate"] = round(0.15 * severity, 4)
        r0 = rng.randint(0, grid_rows - 1)
        c0 = rng.randint(0, grid_cols - 1)
        entry["epidemic_seed_row"] = r0
        entry["epidemic_seed_col"] = c0

    elif et == EventType.TOXIC_SPILL:
        entry["toxic_drain"] = round(0.08 * severity, 5)
        r_min = rng.randint(0, max(0, grid_rows - 3))
        c_min = rng.randint(0, max(0, grid_cols - 3))
        entry["region"] = [r_min, c_min,
                           min(r_min + 2, grid_rows - 1),
                           min(c_min + 2, grid_cols - 1)]

    elif et == EventType.RESOURCE_RUSH:
        entry["resource_boost"] = round(5.0 * severity, 3)
        r_min = rng.randint(0, max(0, grid_rows - 4))
        c_min = rng.randint(0, max(0, grid_cols - 4))
        entry["region"] = [r_min, c_min,
                           min(r_min + 3, grid_rows - 1),
                           min(c_min + 3, grid_cols - 1)]

    return entry


# ---------------------------------------------------------------------------
# Simulation factory from plan
# ---------------------------------------------------------------------------

def build_simulation_from_plan(
    plan: ScenarioPlan,
    government: "Government",
    agents: List["Agent"],
    verbose: bool = False,
    record_audit_trail: bool = True,
) -> "Simulation":
    """
    Build a Simulation whose grid, agent positions, and event schedule are
    taken entirely from *plan* — no randomness, enabling fair comparison.

    Parameters
    ----------
    plan        Pre-generated ScenarioPlan (from generate_scenario_plan or .load).
    government  Government instance to test (freshly constructed).
    agents      Agent list (same count as plan.config_params['num_agents']).
    verbose     Enable per-cycle console output.
    record_audit_trail
                Build the AuditTrail text log (default).  Pass False when the
                caller discards it — the benchmark harness does, and building
                it costs a full pass over every living agent every cycle.

    Returns
    -------
    Fully configured Simulation ready to call .run() on.
    """
    from .simulation import Simulation, SimulationConfig
    from .grid import Grid, Terrain, Cell

    cp = plan.config_params
    config = SimulationConfig(
        grid_rows=plan.rows,
        grid_cols=plan.cols,
        resource_density=cp.get("resource_density", 0.8),
        terrain_variety=cp.get("terrain_variety", 0.3),
        num_agents=len(agents),
        initial_health=cp.get("initial_health", 1.0),
        initial_food_stock=cp.get("initial_food_stock", 10.0),
        initial_water_stock=cp.get("initial_water_stock", 10.0),
        difficulty=cp.get("difficulty", 25),
        event_frequency=0.0,          # all events come from the plan
        event_severity=cp.get("event_severity", 1.0),
        event_warning_cycles=cp.get("event_warning_cycles", 5),
        regen_mult=cp.get("regen_mult", 1.0),
        metabolic_rate=cp.get("metabolic_rate", 0.08),
        # A plan saved without this key gets the schedule value at the plan's
        # difficulty (SimulationConfig.__post_init__ resolves None).
        ambient_hazard=cp.get("ambient_hazard"),
        visibility_radius=cp.get("visibility_radius", 5),
        max_steps_per_cycle=cp.get("max_steps_per_cycle", 1),
        seed=plan.seed,
        max_cycles=plan.n_cycles,
        record_every=cp.get("record_every", 1),
        verbose=verbose,
        record_audit_trail=record_audit_trail,
    )

    # Build simulation (creates its own grid and places agents randomly)
    sim = Simulation(config=config, government=government, agents=agents)

    # --- Restore grid from snapshot ---
    _restore_grid(sim.grid, plan.grid_cells)
    # Re-initialize global depletion params now that the grid reflects plan values
    # (Simulation.__init__ called initialize_depletion_params on the randomly-generated
    # grid before the restore, so the initial totals were stale). Also reset cumulative
    # extraction counters so each government starts from the same clean state.
    sim.grid.cumulative_food_taken = 0.0
    sim.grid.cumulative_water_taken = 0.0
    sim.grid.cumulative_medicine_taken = 0.0
    sim.grid._pending_food_depletion = 1.0
    sim.grid._pending_water_depletion = 1.0
    sim.grid._pending_medicine_depletion = 1.0
    sim.grid.initialize_depletion_params(len(agents), config.difficulty)

    # --- Re-place agents at plan positions ---
    for agent in agents:
        if agent.position:
            r, c = agent.position
            if sim.grid.in_bounds(r, c):
                cell = sim.grid.cell(r, c)
                if agent in cell.agents:
                    cell.agents.remove(agent)
        agent.position = None

    rng_fallback = random.Random(plan.seed + 9999)
    for i, agent in enumerate(agents):
        if i < len(plan.agent_placements):
            r, c = plan.agent_placements[i]
        else:
            r = rng_fallback.randint(0, plan.rows - 1)
            c = rng_fallback.randint(0, plan.cols - 1)
        sim.grid.place_agent(agent, r, c)
        agent.health = config.initial_health

    # --- Schedule plan events (no random events) ---
    sim.event_system.scheduled_events.clear()
    for ev_dict in plan.events_plan:
        ev = _dict_to_active_event(ev_dict)
        sim.event_system.schedule(ev_dict["cycle"], ev)

    return sim


def _restore_grid(grid: "Grid", grid_cells: List[List[dict]]) -> None:
    from .grid import Terrain
    for r, row_data in enumerate(grid_cells):
        for c, cell_data in enumerate(row_data):
            if grid.in_bounds(r, c):
                cell = grid.cell(r, c)
                # Assigning through Cell.terrain also re-derives the cell's
                # hazard, shelter and regen from the new terrain.
                cell.terrain = Terrain(cell_data["terrain"])
                cell.food = cell_data["food"]
                cell.water = cell_data["water"]
                cell.medicine = cell_data["medicine"]
                cell.hazard_extra = 0.0
                cell.contaminated = False


def _dict_to_active_event(ev: dict) -> ActiveEvent:
    region = ev.get("region")
    if region is not None:
        region = tuple(region)
    raw_dur = ev.get("duration")
    # None (JSON null) means indefinite — used for epidemics
    duration = None if raw_dur is None else int(raw_dur)
    return ActiveEvent(
        event_type=EventType(ev["event_type"]),
        start_cycle=ev["cycle"],
        duration=duration,
        severity=ev["severity"],
        region=region,
        description=ev.get("description", ""),
        drought_factor=ev.get("drought_factor", 0.0),
        storm_damage=ev.get("storm_damage", 0.0),
        disease_spread_rate=ev.get("disease_spread_rate", 0.0),
        toxic_drain=ev.get("toxic_drain", 0.0),
        resource_boost=ev.get("resource_boost", 0.0),
        epidemic_seed_row=ev.get("epidemic_seed_row"),
        epidemic_seed_col=ev.get("epidemic_seed_col"),
    )
