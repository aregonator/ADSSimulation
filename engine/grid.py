"""Grid environment: cells, terrain, resources, and spatial queries."""

from __future__ import annotations

import random
import math
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from .difficulty import effective_difficulty

# ---------------------------------------------------------------------------
# Global resource depletion constants
# ---------------------------------------------------------------------------
# Each time any agent collects a resource, ALL cells' same resource diminishes
# by a fraction that grows exponentially once cumulative extraction crosses
# DEPLETION_THRESHOLD (as a fraction of effective_total = initial * REGEN_FACTOR_ESTIMATE).
#
# REGEN_FACTOR_ESTIMATE: rough multiplier for how much the grid regenerates
# relative to initial stock over a full simulation (accounts for regen over time).
# Lower at harder difficulties (less regen), but we use a fixed estimate here.

_DEPLETION_BASE_RATE: float = 0.000004    # per-unit depletion rate (before scaling)
_DEPLETION_THRESHOLD: float = 0.28        # fraction of effective_total where exponential begins
_DEPLETION_EXP_K: float = 1.0             # growth rate of the exponential phase: past the
                                           # threshold, rate = base * exp(k * excess); at 1.0 the
                                           # rate at most ~doubles (excess <= 0.72 -> x2.05)
_REGEN_FACTOR_ESTIMATE: float = 4.0       # effective_total = initial * this factor

if TYPE_CHECKING:
    from .agent import Agent


class Terrain(Enum):
    PLAINS = "plains"
    FOREST = "forest"
    WATER = "water"
    MOUNTAIN = "mountain"
    WASTELAND = "wasteland"


TERRAIN_STATS = {
    Terrain.PLAINS:    {"food_regen": 0.5, "water_regen": 0.2, "hazard": 0.00, "shelter": False},
    Terrain.FOREST:    {"food_regen": 1.0, "water_regen": 0.4, "hazard": 0.01, "shelter": True},
    Terrain.WATER:     {"food_regen": 0.0, "water_regen": 5.0, "hazard": 0.02, "shelter": False},
    Terrain.MOUNTAIN:  {"food_regen": 0.1, "water_regen": 0.1, "hazard": 0.04, "shelter": True},
    Terrain.WASTELAND: {"food_regen": 0.0, "water_regen": 0.0, "hazard": 0.05, "shelter": False},
}


def _validated_ambient_hazard(value: float) -> float:
    """Return *value* as a float if it is a valid ambient hazard, else raise.

    The ambient hazard is a per-cycle probability-like drain added to every
    cell, so it must be a finite real number in [0, 1).  ``bool`` is rejected
    explicitly because it is an ``int`` subclass and ``True`` would otherwise
    pass as 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"ambient_hazard must be a real number, got {type(value).__name__} {value!r}"
        )
    value = float(value)
    if not 0.0 <= value < 1.0:        # also rejects NaN
        raise ValueError(f"ambient_hazard must be in [0, 1), got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Per-terrain resource caps
# ---------------------------------------------------------------------------
#
# Module-level because THREE sites must agree on them — the same "construction
# sites must agree" discipline the severity conversions at engine/events.py:18-27
# already apply, and for the same reason:
#
#   1. ``Grid.regenerate`` below — the authoritative consumer, and the only one
#      that caps a real cell's stock;
#   2. ``EvaluatorNode._snap_cells`` (``governments/ads.py``) — the lookahead's
#      snapshot of those same caps.  A drift here would make the projected
#      resource trajectory diverge from the engine's for a reason nobody would
#      think to look for, since both sites would individually look correct;
#   3. ``ParameterEstimator`` (``governments/parameter_inference.py``) — the
#      regen estimator excludes cells at or above ``REGEN_OBS_MAX_FILL`` of
#      their cap, so the exclusion rule is expressed as a fraction of these.
#
# Values match the initialisation ranges in ``Grid._generate``.
#
# Read-only by convention; nothing mutates these at runtime.
FOOD_CAP = {Terrain.PLAINS: 70, Terrain.FOREST: 100, Terrain.WATER: 0,
            Terrain.MOUNTAIN: 35, Terrain.WASTELAND: 0}
WATER_CAP = {Terrain.PLAINS: 50, Terrain.FOREST: 55, Terrain.WATER: 100,
             Terrain.MOUNTAIN: 30, Terrain.WASTELAND: 0}
MED_CAP = {Terrain.PLAINS: 15, Terrain.FOREST: 20, Terrain.WATER: 15,
           Terrain.MOUNTAIN: 15, Terrain.WASTELAND: 0}


class _TerrainField:
    """Descriptor for ``Cell.terrain``: assigning a terrain re-derives the cell's terrain constants.

    ``Cell`` keeps five per-terrain constants as plain instance attributes so
    that ``hazard``, ``shelter``, ``food_regen``, ``water_regen`` and
    ``to_dict()`` — read on the engine's and the lookahead's hot paths — avoid a
    ``TERRAIN_STATS`` dict-plus-enum lookup on every read:

        _terrain_hazard: float   _terrain_shelter: bool   _food_regen: float
        _water_regen: float      _terrain_value: str

    They are written here and nowhere else, except that ``_terrain_hazard`` is
    also re-derived by :class:`_AmbientHazardField` (below).  Every way of
    setting a terrain — the dataclass ``__init__``, ``dataclasses.replace``,
    and plain assignment such as ``engine/scenario_plan._restore_grid``
    relabelling a generated grid from a plan — goes through ``__set__``, so the
    constants always agree with ``TERRAIN_STATS[cell.terrain]``, with
    ``_terrain_hazard = TERRAIN_STATS[cell.terrain]["hazard"] +
    cell.ambient_hazard``.  ``test_terrain_consistency.py`` checks this on the
    production path.

    This is the dataclass "descriptor-typed field" protocol: ``@dataclass``
    reads the field default through ``__get__(None, Cell)``, and the generated
    ``__init__`` assigns through ``__set__``.  ``copy``, ``deepcopy`` and
    ``pickle`` restore the instance ``__dict__`` without calling ``__set__``;
    the dict they restore already holds a consistent label and constants.
    """

    def __init__(self, default: Terrain) -> None:
        self._default = default

    def __get__(self, cell: Optional["Cell"], owner: Optional[type] = None) -> Terrain:
        if cell is None:
            return self._default
        return cell._terrain

    def __set__(self, cell: "Cell", terrain: Terrain) -> None:
        # Validate before writing anything, so a rejected value leaves the cell
        # exactly as it was.
        if not isinstance(terrain, Terrain):
            raise TypeError(
                f"Cell.terrain must be a Terrain, got {type(terrain).__name__} {terrain!r}"
            )
        stats = TERRAIN_STATS[terrain]
        cell._terrain = terrain
        # The dataclass __init__ assigns ``terrain`` before the keyword-only
        # ``ambient_hazard`` field, so during construction the cell has no
        # ambient value yet; 0.0 is a placeholder that _AmbientHazardField.__set__
        # overwrites a few statements later in the same __init__.  On every
        # later relabel the cell's own ambient value is used.
        cell._terrain_hazard = stats["hazard"] + cell.__dict__.get("_ambient_hazard", 0.0)
        cell._terrain_shelter = stats["shelter"]
        cell._food_regen = stats["food_regen"]
        cell._water_regen = stats["water_regen"]
        cell._terrain_value = terrain.value


class _AmbientHazardField:
    """Descriptor for ``Cell.ambient_hazard``: the uniform hazard added to the terrain hazard.

    The value comes from the difficulty schedule
    (``engine.difficulty.ambient_hazard``) via ``SimulationConfig`` ->
    ``Grid`` -> ``Cell``, so each cell carries the ambient hazard of the run
    it belongs to.  Nothing reads a process-global: the sweep's ``spawn``
    workers get the value through the same config every other difficulty
    parameter travels in.

    Assigning re-derives ``_terrain_hazard`` as
    ``TERRAIN_STATS[terrain]["hazard"] + ambient_hazard``, the same expression
    (and the same floating-point association) :class:`_TerrainField` uses, so
    the result is independent of which of the two was assigned last.  Readers
    see it only through ``cell.hazard`` and ``cell.to_dict()["hazard"]``.
    """

    def __init__(self, default: float) -> None:
        self._default = default

    def __get__(self, cell: Optional["Cell"], owner: Optional[type] = None) -> float:
        if cell is None:
            return self._default
        return cell._ambient_hazard

    def __set__(self, cell: "Cell", value: float) -> None:
        value = _validated_ambient_hazard(value)     # raises before any write
        cell._ambient_hazard = value
        terrain = cell.__dict__.get("_terrain")
        if terrain is not None:
            cell._terrain_hazard = TERRAIN_STATS[terrain]["hazard"] + value


@dataclass
class Cell:
    row: int
    col: int
    terrain: Terrain = _TerrainField(default=Terrain.PLAINS)
    food: float = 50.0
    water: float = 50.0
    medicine: float = 20.0
    hazard_extra: float = 0.0       # dynamic hazard from events (stacks with terrain)
    contaminated: bool = False       # disease contamination
    agents: List["Agent"] = field(default_factory=list)
    _: KW_ONLY
    #: Uniform hazard from the difficulty schedule, added to the terrain hazard
    #: (see :class:`_AmbientHazardField`).  ``Grid`` always passes it; the 0.0
    #: default serves stand-alone cells built outside a grid.
    ambient_hazard: float = _AmbientHazardField(default=0.0)

    @property
    def hazard(self) -> float:
        return min(1.0, self._terrain_hazard + self.hazard_extra)

    @property
    def shelter(self) -> bool:
        return self._terrain_shelter

    @property
    def food_regen(self) -> float:
        return self._food_regen

    @property
    def water_regen(self) -> float:
        return self._water_regen

    def has_agent(self, agent: "Agent") -> bool:
        return agent in self.agents

    def to_dict(self) -> dict:
        # Avoid round() for performance — agents use these values for comparisons,
        # not display, so full float precision is fine.
        return {
            "row": self.row,
            "col": self.col,
            "terrain": self._terrain_value,
            "food": self.food,
            "water": self.water,
            "medicine": self.medicine,
            "hazard": min(1.0, self._terrain_hazard + self.hazard_extra),
            "contaminated": self.contaminated,
            "shelter": self._terrain_shelter,
            "num_agents": len(self.agents),
        }


class Grid:
    """The n×m world. Manages cells, resources, agent placement, and spatial queries."""

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        resource_density: float = 0.7,
        terrain_variety: float = 0.5,
        seed: Optional[int] = None,
        *,
        ambient_hazard: float,
    ):
        """
        *ambient_hazard* is required and keyword-only on purpose: it is a
        difficulty parameter (``engine.difficulty.ambient_hazard``) that every
        caller takes from its ``SimulationConfig``, and a default would let a
        new call site silently build a grid without it.  It is passed to every
        cell this grid creates.
        """
        self.rows = rows
        self.cols = cols
        self.ambient_hazard: float = _validated_ambient_hazard(ambient_hazard)
        self.rng = random.Random(seed)
        self._cells: List[List[Cell]] = []
        self._generate(resource_density, terrain_variety)

        # --- Global resource depletion tracking ---
        # Depletion params (set by initialize_depletion_params after agents are placed)
        self.n_initial_agents: int = 50
        self.difficulty: int = 1
        # Snapshots of initial grid totals (set in initialize_depletion_params)
        self.initial_grid_food: float = 0.0
        self.initial_grid_water: float = 0.0
        self.initial_grid_medicine: float = 0.0
        # Running totals of resources extracted by agents across all cycles
        self.cumulative_food_taken: float = 0.0
        self.cumulative_water_taken: float = 0.0
        self.cumulative_medicine_taken: float = 0.0
        # Per-cycle accumulated depletion multipliers (flushed in regenerate())
        self._pending_food_depletion: float = 1.0
        self._pending_water_depletion: float = 1.0
        self._pending_medicine_depletion: float = 1.0

        #: OBSERVATION CHANNEL — cells any agent collected from this cycle, as
        #: ``(row, col)``.  Populated by the three ``collect_*`` methods when
        #: ``taken > 0`` and cleared at the top of :meth:`regenerate`.
        #:
        #: **No engine consumer.**  Nothing in ``Grid`` or ``Simulation`` reads
        #: it; it exists so an external observer can tell "this cell's stock
        #: fell because somebody harvested it" apart from "this cell's stock
        #: fell because of the global depletion haircut", which is not otherwise
        #: recoverable from the cell alone.  ``ParameterEstimator`` uses it to
        #: exclude harvested cells from its regen observation.
        #:
        #: Available to every government; the epistemic asymmetry is that only
        #: ADS chooses to read it, which keeps the engine regime-agnostic.
        #:
        #: TIMING, which is what makes it usable: cleared at step 2 of cycle t
        #: (``regenerate``), populated at step 4 (``_execute_actions``), read at
        #: step 6 (``government.tick``).  It therefore describes exactly the
        #: collections that occurred inside the interval a government's
        #: cycle-over-cycle stock diff spans.  Bounded by the number of
        #: collecting agents.
        self.collected_cells_this_cycle: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _generate(self, resource_density: float, terrain_variety: float) -> None:
        rows, cols = self.rows, self.cols
        total = rows * cols

        # Target cell fractions for non-PLAINS terrains at terrain_variety=1.0.
        # terrain_variety=0 yields all PLAINS; terrain_variety=1 uses full fractions.
        full_fracs = {
            Terrain.FOREST:    0.20,
            Terrain.WATER:     0.15,
            Terrain.MOUNTAIN:  0.15,
            Terrain.WASTELAND: 0.10,
        }

        terrain_grid = [[Terrain.PLAINS] * cols for _ in range(rows)]
        claimed = [[False] * cols for _ in range(rows)]

        for terrain_type, base_frac in full_fracs.items():
            target_cells = int(base_frac * terrain_variety * total)
            if target_cells <= 0:
                continue

            # 1–4 patches; more patches at higher terrain_variety for realism.
            n_patches = max(1, min(4, round(1 + terrain_variety * 3)))
            target_per_patch = max(1, target_cells // n_patches)

            for _ in range(n_patches):
                # Find a random unclaimed seed cell.
                seed_r = seed_c = None
                for _ in range(200):
                    r0 = self.rng.randint(0, rows - 1)
                    c0 = self.rng.randint(0, cols - 1)
                    if not claimed[r0][c0]:
                        seed_r, seed_c = r0, c0
                        break
                if seed_r is None:
                    break

                # BFS patch growth: pick a random cell from the frontier each step
                # (not strict BFS order) to create irregular, natural patch shapes.
                frontier: list = [seed_r * cols + seed_c]
                in_frontier: set = {seed_r * cols + seed_c}
                patch_count = 0

                while frontier and patch_count < target_per_patch:
                    idx = self.rng.randrange(len(frontier))
                    key = frontier[idx]
                    frontier[idx] = frontier[-1]
                    frontier.pop()
                    in_frontier.discard(key)

                    r, c = key // cols, key % cols
                    if claimed[r][c]:
                        continue

                    terrain_grid[r][c] = terrain_type
                    claimed[r][c] = True
                    patch_count += 1

                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        nkey = nr * cols + nc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not claimed[nr][nc]
                                and nkey not in in_frontier):
                            frontier.append(nkey)
                            in_frontier.add(nkey)

        # Per-terrain resource ranges (base × resource_density gives final value).
        # format: (food_lo, food_hi, water_lo, water_hi, med_lo, med_hi)
        # Wastelands have no food, water, or medicine.
        # Water cells have abundant water, no food.
        # Forests have the richest food; mountains moderate food.
        _TERRAIN_RESOURCES = {
            Terrain.PLAINS:    (30,  70,  20,  50,  5,  15),
            Terrain.FOREST:    (60, 100,  25,  55,  8,  20),
            Terrain.WATER:     ( 0,   0,  80, 100,  5,  15),
            Terrain.MOUNTAIN:  (10,  35,  10,  30,  5,  15),
            Terrain.WASTELAND: ( 0,   0,   0,   0,  0,   0),
        }

        self._cells = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                terrain = terrain_grid[r][c]
                fl, fh, wl, wh, ml, mh = _TERRAIN_RESOURCES[terrain]
                food     = self.rng.uniform(fl, fh) * resource_density if fh > 0 else 0.0
                water    = self.rng.uniform(wl, wh) * resource_density if wh > 0 else 0.0
                medicine = self.rng.uniform(ml, mh) * resource_density if mh > 0 else 0.0
                row_cells.append(Cell(row=r, col=c, terrain=terrain,
                                      food=food, water=water, medicine=medicine,
                                      ambient_hazard=self.ambient_hazard))
            self._cells.append(row_cells)

    # ------------------------------------------------------------------
    # Depletion system
    # ------------------------------------------------------------------

    def initialize_depletion_params(self, n_initial_agents: int, difficulty: int) -> None:
        """
        Record initial grid totals and simulation params for the depletion mechanic.
        Call once after the grid is generated and agents are placed.
        """
        self.n_initial_agents = max(1, n_initial_agents)
        self.difficulty = max(1, min(100, difficulty))
        self.initial_grid_food     = sum(c.food     for row in self._cells for c in row)
        self.initial_grid_water    = sum(c.water    for row in self._cells for c in row)
        self.initial_grid_medicine = sum(c.medicine for row in self._cells for c in row)

    def _compute_depletion_rate_per_unit(self, resource_type: str) -> float:
        """
        Return the per-unit fractional depletion rate for all cells when one unit
        of `resource_type` is collected.  Grows exponentially after the threshold.

        Scaling:
          - diff_scale: higher difficulty → steeper ecological consequences
          - agent_scale: normalized to 100 agents so rate is comparable across runs
        """
        if resource_type == "food":
            cumulative = self.cumulative_food_taken
            initial    = self.initial_grid_food
        elif resource_type == "water":
            cumulative = self.cumulative_water_taken
            initial    = self.initial_grid_water
        else:
            cumulative = self.cumulative_medicine_taken
            initial    = self.initial_grid_medicine

        if initial <= 0:
            return 0.0

        effective_total = initial * _REGEN_FACTOR_ESTIMATE
        fraction = min(1.0, cumulative / effective_total)

        # Knee-folded difficulty — the one canonical difficulty schedule, shared
        # with SimulationConfig.from_difficulty and difficulty_multiplier.  Was
        # an inline `min(self.difficulty, 90)`.  self.difficulty is already
        # clamped to [1, 100] by initialize_depletion_params, so folding it
        # again is a no-op on the clamp and changes only the knee.
        d_eff = effective_difficulty(self.difficulty)
        diff_scale  = 0.20 + (d_eff / 100.0) * 1.60   # 0.20 at d=1, 1.64 at the knee
        agent_scale = math.sqrt(100.0 / self.n_initial_agents)   # normalised to 100 agents

        base = _DEPLETION_BASE_RATE * diff_scale * agent_scale

        if fraction < _DEPLETION_THRESHOLD:
            return base
        else:
            excess = fraction - _DEPLETION_THRESHOLD
            return base * math.exp(_DEPLETION_EXP_K * excess)

    @property
    def food_depletion_fraction(self) -> float:
        """Fraction of effective total food consumed (0–1 scale)."""
        eff = self.initial_grid_food * _REGEN_FACTOR_ESTIMATE
        return min(1.0, self.cumulative_food_taken / max(1.0, eff))

    @property
    def water_depletion_fraction(self) -> float:
        """Fraction of effective total water consumed (0–1 scale)."""
        eff = self.initial_grid_water * _REGEN_FACTOR_ESTIMATE
        return min(1.0, self.cumulative_water_taken / max(1.0, eff))

    @property
    def medicine_depletion_fraction(self) -> float:
        """Fraction of effective total medicine consumed (0–1 scale)."""
        eff = self.initial_grid_medicine * _REGEN_FACTOR_ESTIMATE
        return min(1.0, self.cumulative_medicine_taken / max(1.0, eff))

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def cell(self, r: int, c: int) -> Cell:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self._cells[r][c]
        raise IndexError(f"Cell ({r},{c}) out of bounds ({self.rows}×{self.cols})")

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def all_cells(self) -> List[Cell]:
        # Build once and cache; callers must not mutate the returned list.
        if not hasattr(self, "_all_cells_cache"):
            self._all_cells_cache = [cell for row in self._cells for cell in row]
        return self._all_cells_cache

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def neighbors(self, r: int, c: int, radius: int = 1) -> List[Cell]:
        """All cells within Chebyshev distance `radius`."""
        result = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    result.append(self._cells[nr][nc])
        return result

    def adjacent_cells(self, r: int, c: int) -> List[Cell]:
        """4-connected cardinal neighbors."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                result.append(self._cells[nr][nc])
        return result

    def cells_in_radius(self, r: int, c: int, radius: int) -> List[Cell]:
        """Euclidean radius — returns all cells within the circle.

        Bounded-rectangle optimisation: only examines the (2r+1)×(2r+1) box
        then applies a squared-distance check (no sqrt).  This is O(r²) instead
        of O(n_cells), giving a ~20× speedup for radius=5 on a 50×50 grid.
        """
        r2 = radius * radius
        r_lo = max(0, r - radius)
        r_hi = min(self.rows - 1, r + radius)
        c_lo = max(0, c - radius)
        c_hi = min(self.cols - 1, c + radius)
        result = []
        for row in range(r_lo, r_hi + 1):
            dr2 = (row - r) ** 2
            for col in range(c_lo, c_hi + 1):
                if dr2 + (col - c) ** 2 <= r2:
                    result.append(self._cells[row][col])
        return result

    def visible_cells(self, r: int, c: int, radius: int) -> List[dict]:
        """Return serialised view of visible cells (for agent observation)."""
        return [cell.to_dict() for cell in self.cells_in_radius(r, c, radius)]

    def agents_in_radius(self, r: int, c: int, radius: int) -> List["Agent"]:
        result = []
        for cell in self.cells_in_radius(r, c, radius):
            result.extend(cell.agents)
        return result

    def shelter_cells(self) -> List[Cell]:
        return [c for c in self.all_cells() if c.shelter]

    def nearest_shelter(self, r: int, c: int) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[float, int, int]] = None
        for cell in self.shelter_cells():
            d = abs(cell.row - r) + abs(cell.col - c)
            if best is None or d < best[0]:
                best = (d, cell.row, cell.col)
        return (best[1], best[2]) if best else None

    def richest_food_cell(self, r: int, c: int, radius: int) -> Optional[Tuple[int, int]]:
        cells = self.cells_in_radius(r, c, radius)
        if not cells:
            return None
        best = max(cells, key=lambda cell: cell.food)
        return (best.row, best.col) if best.food > 0 else None

    # ------------------------------------------------------------------
    # Agent placement
    # ------------------------------------------------------------------

    def place_agent(self, agent: "Agent", r: int, c: int) -> None:
        if agent.position:
            old_r, old_c = agent.position
            if self.in_bounds(old_r, old_c):
                cell = self._cells[old_r][old_c]
                if agent in cell.agents:
                    cell.agents.remove(agent)
        self._cells[r][c].agents.append(agent)
        agent.position = (r, c)

    def remove_agent(self, agent: "Agent") -> None:
        if agent.position:
            r, c = agent.position
            if self.in_bounds(r, c):
                cell = self._cells[r][c]
                if agent in cell.agents:
                    cell.agents.remove(agent)
        agent.position = None

    # ------------------------------------------------------------------
    # Resource operations
    # ------------------------------------------------------------------

    def collect_food(self, r: int, c: int, amount: float) -> float:
        cell = self._cells[r][c]
        taken = min(cell.food, max(0.0, amount))
        cell.food -= taken
        if taken > 0:
            # Observation channel only — see collected_cells_this_cycle.  Note
            # this is NOT inside the `initial_grid_food > 0` guard below: that
            # guard is about whether depletion accounting is initialised, while
            # the harvest itself happened either way and is what an observer
            # needs to know about.
            self.collected_cells_this_cycle.add((r, c))
        if taken > 0 and self.initial_grid_food > 0:
            self.cumulative_food_taken += taken
            rate = self._compute_depletion_rate_per_unit("food")
            self._pending_food_depletion *= max(0.0, 1.0 - rate * taken)
        return taken

    def collect_water(self, r: int, c: int, amount: float) -> float:
        cell = self._cells[r][c]
        taken = min(cell.water, max(0.0, amount))
        cell.water -= taken
        if taken > 0:
            self.collected_cells_this_cycle.add((r, c))
        if taken > 0 and self.initial_grid_water > 0:
            self.cumulative_water_taken += taken
            rate = self._compute_depletion_rate_per_unit("water")
            self._pending_water_depletion *= max(0.0, 1.0 - rate * taken)
        return taken

    def collect_medicine(self, r: int, c: int, amount: float) -> float:
        cell = self._cells[r][c]
        taken = min(cell.medicine, max(0.0, amount))
        cell.medicine -= taken
        if taken > 0:
            self.collected_cells_this_cycle.add((r, c))
        if taken > 0 and self.initial_grid_medicine > 0:
            self.cumulative_medicine_taken += taken
            rate = self._compute_depletion_rate_per_unit("medicine")
            self._pending_medicine_depletion *= max(0.0, 1.0 - rate * taken)
        return taken

    # ------------------------------------------------------------------
    # Resource regeneration (called once per cycle by simulation)
    # ------------------------------------------------------------------

    def regenerate(self, drought_factor: float = 0.0, base_regen_mult: float = 1.0) -> None:
        # Opens the next observation window.  Cleared HERE, at step 2 of the
        # cycle, rather than at the end of the previous one, so the set is
        # non-empty for exactly the span between this regen and the next — which
        # is the interval a cycle-over-cycle stock diff measures.  See
        # Grid.collected_cells_this_cycle.
        self.collected_cells_this_cycle.clear()

        # Apply accumulated global depletions from this cycle's collections BEFORE regen.
        # The pending multipliers are (1 - rate*amount) products across all collections
        # this cycle — applying them here is equivalent to applying after each collection
        # but avoids O(n_cells) work per collection event.
        if self._pending_food_depletion < 1.0:
            m = self._pending_food_depletion
            for row in self._cells:
                for c in row:
                    c.food = max(0.0, c.food * m)
            self._pending_food_depletion = 1.0

        if self._pending_water_depletion < 1.0:
            m = self._pending_water_depletion
            for row in self._cells:
                for c in row:
                    c.water = max(0.0, c.water * m)
            self._pending_water_depletion = 1.0

        if self._pending_medicine_depletion < 1.0:
            m = self._pending_medicine_depletion
            for row in self._cells:
                for c in row:
                    c.medicine = max(0.0, c.medicine * m)
            self._pending_medicine_depletion = 1.0

        for cell in self.all_cells():
            # Read once: Cell.terrain is a descriptor, and this loop runs over
            # every cell every cycle.
            terrain = cell.terrain
            if terrain == Terrain.WASTELAND:
                continue
            regen_mult = max(0.0, 1.0 - drought_factor) * max(0.0, base_regen_mult)
            cell.food = min(FOOD_CAP[terrain], cell.food + cell.food_regen * regen_mult)
            cell.water = min(WATER_CAP[terrain], cell.water + cell.water_regen * regen_mult)
            cell.medicine = min(MED_CAP[terrain], cell.medicine + 0.05 * regen_mult)

    # ------------------------------------------------------------------
    # Global stats (for OE observation)
    # ------------------------------------------------------------------

    def mean_food(self) -> float:
        cells = self.all_cells()
        return sum(c.food for c in cells) / len(cells) if cells else 0.0

    def mean_water(self) -> float:
        cells = self.all_cells()
        return sum(c.water for c in cells) / len(cells) if cells else 0.0

    def contamination_fraction(self) -> float:
        cells = self.all_cells()
        return sum(1 for c in cells if c.contaminated) / len(cells) if cells else 0.0

    def region_food(self, rows: range, cols: range) -> float:
        total, count = 0.0, 0
        for r in rows:
            for c in cols:
                if self.in_bounds(r, c):
                    total += self._cells[r][c].food
                    count += 1
        return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def cluster_positions(
    grid: Grid,
    n: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    Return n distinct (row, col) positions forming a connected cluster on
    non-WATER terrain, starting from a random seed cell.

    If the non-water region has fewer than n cells the shortfall is filled
    from remaining cells (water included) so the list always has length n.
    """
    non_water = [
        (r, c)
        for r in range(grid.rows)
        for c in range(grid.cols)
        if grid.cell(r, c).terrain != Terrain.WATER
    ]

    if non_water:
        start = rng.choice(non_water)
    else:
        start = (rng.randint(0, grid.rows - 1), rng.randint(0, grid.cols - 1))

    positions: List[Tuple[int, int]] = []
    visited: set = {start}
    frontier: List[Tuple[int, int]] = [start]

    while frontier and len(positions) < n:
        idx = rng.randrange(len(frontier))
        r, c = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()

        positions.append((r, c))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (grid.in_bounds(nr, nc)
                        and (nr, nc) not in visited
                        and grid.cell(nr, nc).terrain != Terrain.WATER):
                    visited.add((nr, nc))
                    frontier.append((nr, nc))

    # Fallback: fill remainder from any unoccupied cell (including water).
    if len(positions) < n:
        pos_set = set(positions)
        extras = [
            (r, c)
            for r in range(grid.rows)
            for c in range(grid.cols)
            if (r, c) not in pos_set
        ]
        rng.shuffle(extras)
        positions.extend(extras[: n - len(positions)])

    return positions
