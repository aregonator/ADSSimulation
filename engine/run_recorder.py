"""
Per-run structured detail output: turn one ``Simulation``'s per-cycle state into
newline-delimited JSON.

Single reason to exist: everything a run computes about *what actually happened*
— which events fired, which laws were enacted and why they lifted, which
groupings ADS evaluated and which candidate law won — is currently held in
memory, formatted into text by ``AuditTrail``, and discarded when the process
exits.  At 16,800 runs, "re-run it and watch" is not a debugging strategy.

Files written, per run directory:

===================== ===================== =================================
File                  When                  Contents
===================== ===================== =================================
``run_detail.jsonl``  always (default on)   header + cycle + decision + footer
``agent_states.jsonl`` only --detailed-agents  bounded per-agent snapshots
===================== ===================== =================================

Why JSONL rather than one JSON document
---------------------------------------

* **Crash resilience.**  A run that dies at cycle 118 leaves 118 valid,
  parseable records.  A single JSON document would be an unclosed fragment —
  worthless in the exact case the feature exists for.
* **Constant memory.**  Nothing accumulates before serialisation.
* **Streaming consumption.**  ``jq -c 'select(.rec=="cycle")'`` works on a
  half-written file; ``grep '"law_type":"QUARANTINE_EPIDEMIC"'`` needs no parser
  at all.

Contracts this module guarantees
--------------------------------

* **It never raises into the simulation.**  Observability must not be able to
  fail the science.  A failed record increments :attr:`RunRecorder.errors`,
  emits one ERROR line and is skipped; the count lands in
  ``footer.recorder_errors`` so a corrupted record is visible *as data* rather
  than discovered by absence.
* **``__exit__`` always writes a footer**, including on the exception path, and
  returns ``False`` so the exception still propagates.  A failed run therefore
  leaves *more* evidence than a successful one, not less.
* **Files are opened inside the worker process.**  Opening a handle before
  ``fork`` and writing to it from several children is the classic corruption
  bug; the invariant is that a recorder is constructed inside ``_run_one``.
* **No matplotlib, no numpy.**  Preserves the fork-safety invariant that the
  benchmark's worker path depends on, and keeps ``engine/`` dependency-light.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .metrics import _gini

if TYPE_CHECKING:                                    # pragma: no cover
    from .simulation import Simulation

_log = logging.getLogger("sim.bench")

#: Bumped whenever a record's shape changes incompatibly.  Present on the
#: header so a consumer can refuse a file it does not understand.
SCHEMA_VERSION = 1

#: Default rounding for floats with no more specific semantics.  Full repr
#: floats inflate the file ~20% and imply precision the model does not have.
_DEFAULT_ROUND = 4

#: Guard against pathological nesting in an unexpected object graph.
_MAX_DEPTH = 12

#: Columns of the ``agents`` record.  The record carries this list so it stays
#: self-describing: the schema travels with the data.
AGENT_FIELDS = (
    "agent_id", "alive", "r", "c", "health", "food", "water", "medicine",
    "hunger", "thirst", "age", "n_epidemics", "epidemic_ids", "n_laws",
)

#: Agent-level thresholds.  ``starving`` / ``dehydrated`` deliberately match the
#: thresholds ADS's FoodEvidence.pct_starving / WaterEvidence.pct_dehydrated use,
#: so the cycle series and the decision-round evidence are directly comparable.
_STARVING_STOCK = 1.0
_DEHYDRATED_STOCK = 1.0
#: ``hungry`` / ``thirsty`` match the point at which hunger and thirst begin to
#: drain health in ``Simulation._apply_health_dynamics``.
_HUNGRY = 0.4
_THIRSTY = 0.4


# ---------------------------------------------------------------------------
# Serialization contract
# ---------------------------------------------------------------------------

def _jsonable(value: Any, _depth: int = 0) -> Any:
    """
    Coerce *value* into something ``json.dumps(..., allow_nan=False)`` accepts.

    Never raises.  The non-finite-float rule is not cosmetic:
    ``FoodEvidence.cycles_until_empty`` is a division whose denominator can be
    zero, and Python's default ``json`` emits a bare ``Infinity`` — which is not
    valid JSON and is rejected by ``jq`` and by every strict parser.  Mapping
    non-finite floats to ``null`` here, plus ``allow_nan=False`` at the dump,
    turns a silently unreadable dataset into either clean data or a loud failure.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, _DEFAULT_ROUND)

    if _depth >= _MAX_DEPTH:
        return repr(value)[:120]

    if isinstance(value, Enum):
        return _jsonable(value.value, _depth + 1)

    if isinstance(value, (set, frozenset)):
        # Sorted so two identical runs produce byte-identical records.
        try:
            items = sorted(value)
        except TypeError:                            # pragma: no cover - mixed types
            items = sorted(value, key=repr)
        return [_jsonable(v, _depth + 1) for v in items]

    if isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in value]

    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _jsonable(dataclasses.asdict(value), _depth + 1)
        except Exception:                            # pragma: no cover - defensive
            return repr(value)[:120]

    # numpy scalars and anything else exposing .item()/.dtype, without importing
    # numpy (this module must stay fork-safe and dependency-light).
    if hasattr(value, "item") and hasattr(value, "dtype"):
        try:
            return _jsonable(value.item(), _depth + 1)
        except Exception:                            # pragma: no cover - defensive
            return repr(value)[:120]

    return repr(value)[:120]


def dumps_record(obj: Any) -> str:
    """Serialise one record to a single JSON line (no trailing newline)."""
    return json.dumps(
        _jsonable(obj),
        allow_nan=False,
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=False,
    )


def _r(value: Optional[float], places: int) -> Optional[float]:
    """Round, mapping non-finite values to ``None``."""
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):                  # pragma: no cover - defensive
        return None
    if not math.isfinite(fval):
        return None
    return round(fval, places)


def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of a pre-sorted list; ``None`` if empty."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _distribution(values: List[float], places: int) -> Dict[str, Optional[float]]:
    """mean/median/min/max of *values*; all-``None`` when there are none."""
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": _r(sum(ordered) / len(ordered), places),
        "median": _r(_percentile(ordered, 0.5), places),
        "min": _r(ordered[0], places),
        "max": _r(ordered[-1], places),
    }


# ---------------------------------------------------------------------------
# Law parameter summarising
# ---------------------------------------------------------------------------

def summarise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shrink ``Law.params`` to what a reader can act on.

    ``donor_amounts`` is a dict of hundreds of agent ids to floats and
    ``recipient_ids`` a list of hundreds.  Dumping them raw multiplies the file
    size by an order of magnitude and adds nothing actionable, so collections
    collapse to a count.  ``--detailed-agents`` (the "I am debugging one run"
    mode) emits the raw params instead.
    """
    if not isinstance(params, dict):                 # pragma: no cover - defensive
        return {}

    out: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          for v in value)
            if numeric and len(value) <= 8:
                out[key] = list(value)               # e.g. a region bbox
            else:
                out[f"{key}_count"] = len(value)
        elif isinstance(value, (dict, set, frozenset)):
            out[f"{key}_count"] = len(value)
        else:
            out[key] = repr(value)[:120]
    return out


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class RunRecorder:
    """
    Writes one run's JSONL detail files.  Never raises into the simulation.

    Typical use, inside a worker process::

        with RunRecorder(path, meta) as rec:
            rec.write_header(sim)
            for cycle in range(max_cycles):
                sim._step(cycle)
                rec.record_cycle(sim, cycle, sim.last_warnings)
            rec.set_summary(sim.metrics.summary())
    """

    def __init__(
        self,
        detail_path: str,
        meta: Dict[str, Any],
        *,
        decision_detail: bool = True,
        agent_states_path: Optional[str] = None,
        agent_states_every: Optional[int] = None,
        flush_every: int = 25,
    ) -> None:
        self.detail_path = detail_path
        self.meta = dict(meta or {})
        self.decision_detail = bool(decision_detail)
        self.agent_states_every = agent_states_every
        self.flush_every = max(1, int(flush_every))
        self.errors = 0

        # Timing is owned by the recorder so the footer is self-consistent even
        # if the caller forgets to hand one in.
        self._t0 = time.monotonic()
        self.started_at = self.meta.get("started_at") or utc_timestamp()
        self.meta.setdefault("started_at", self.started_at)

        self._detail = self._open(detail_path)
        self._agents = (
            self._open(agent_states_path)
            if (agent_states_path and agent_states_every) else None
        )
        #: Under --detailed-agents, law params are emitted raw: that mode is
        #: explicitly targeted debugging of one run.
        self._raw_law_params = self._agents is not None

        # -- per-run state --------------------------------------------------
        self._cycles_written = 0
        self._last_cycle: Optional[int] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._prev_alive: Optional[int] = None
        self._initial_agents = 0
        self._max_cycles: Optional[int] = None

        # Event identity.  ActiveEvent has no stable id (only epidemics carry
        # epidemic_id), so uids are assigned in first-appearance order.  A strong
        # reference to every event seen is retained so CPython cannot recycle an
        # id() onto a different object; assignment order is deterministic given
        # the seed, so the uids are reproducible across identical runs.
        self._event_uids: Dict[int, str] = {}
        self._event_refs: List[Any] = []
        self._prev_active_uids: Set[str] = set()

        # Laws already reported as enacted.  Derived recorder-side rather than
        # trusting Government._laws_enacted_this_cycle alone: that list is reset
        # at the start of tick(), which drops laws enacted earlier in the same
        # cycle from receive_event_warnings().
        self._seen_law_ids: Set[str] = set()

        # Cumulative totals for the footer.
        self._total_laws_enacted = 0
        self._total_laws_expired = 0
        self._total_decision_rounds = 0
        self._total_proposals = 0
        self._total_events_triggered = 0
        self._total_events_expired = 0

    # -- plumbing -------------------------------------------------------

    def _open(self, path: Optional[str]):
        if not path:
            return None
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return open(path, "w", encoding="utf-8")
        except OSError as exc:
            # An observability failure must not block the science: the run
            # continues without detail output, and the missing footer is itself
            # the record that something went wrong.
            self.errors += 1
            _log.error("run detail file could not be opened (%s): %s", path, exc)
            return None

    def _emit(self, handle, record: Dict[str, Any], what: str) -> None:
        if handle is None:
            return
        try:
            line = dumps_record(record)
        except Exception as exc:
            self.errors += 1
            _log.error("run detail record %s failed to serialise: %s", what, exc)
            return
        try:
            handle.write(line + "\n")
        except OSError as exc:
            self.errors += 1
            _log.error("run detail record %s failed to write: %s", what, exc)

    def _flush(self) -> None:
        for handle in (self._detail, self._agents):
            if handle is None:
                continue
            try:
                handle.flush()
            except OSError:                          # pragma: no cover - defensive
                self.errors += 1

    # -- context manager ------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """
        Always write a footer, then close.  Returns ``False`` so an in-flight
        exception continues to propagate to the caller's own handler — that is
        what makes "a failed run leaves more evidence" work with no new control
        flow at the call site.
        """
        try:
            self._write_footer(exc_type, exc, tb)
        except Exception as inner:                   # pragma: no cover - defensive
            self.errors += 1
            _log.error("run detail footer failed: %s", inner)
        for handle in (self._detail, self._agents):
            if handle is None:
                continue
            try:
                handle.flush()
                handle.close()
            except OSError:                          # pragma: no cover - defensive
                pass
        self._detail = None
        self._agents = None
        return False

    # -- header ---------------------------------------------------------

    def write_header(self, sim: "Simulation") -> None:
        """Write the single ``header`` record.  Must be the first line."""
        try:
            self._initial_agents = len(sim.agents)
            self._prev_alive = sum(1 for a in sim.agents if a.alive)
            self._max_cycles = int(getattr(sim.config, "max_cycles", 0)) or None
            record = {
                "rec": "header",
                "schema_version": SCHEMA_VERSION,
                "run": self.meta.get("run", {}),
                "config": self._config_block(sim),
                "scheduled_events": self._scheduled_events(sim),
                "government_params": self._government_params(sim),
                "started_at": self.meta.get("started_at"),
                "host": self.meta.get("host") or socket.gethostname(),
                "pid": self.meta.get("pid") or os.getpid(),
                "software": self.meta.get("software") or software_versions(),
                "options": {
                    "decision_detail": self.decision_detail,
                    "agent_states_every": self.agent_states_every,
                    "heartbeat_every": self.meta.get("heartbeat_every"),
                },
            }
            self._emit(self._detail, record, "header")
            self._flush()
        except Exception as exc:
            self.errors += 1
            _log.error("run detail header failed: %s", exc)

    @staticmethod
    def _config_block(sim: "Simulation") -> Dict[str, Any]:
        cfg = sim.config
        return {
            "grid_rows": cfg.grid_rows,
            "grid_cols": cfg.grid_cols,
            "num_agents": cfg.num_agents,
            "max_cycles": cfg.max_cycles,
            "max_steps_per_cycle": cfg.max_steps_per_cycle,
            "difficulty": cfg.difficulty,
            "drain_mult": _r(getattr(sim, "drain_mult", None), 4),
            "resource_density": cfg.resource_density,
            "terrain_variety": cfg.terrain_variety,
            "initial_health": cfg.initial_health,
            "initial_food_stock": cfg.initial_food_stock,
            "initial_water_stock": cfg.initial_water_stock,
            "event_frequency": cfg.event_frequency,
            "event_severity": cfg.event_severity,
            "event_warning_cycles": cfg.event_warning_cycles,
            "regen_mult": cfg.regen_mult,
            "metabolic_rate": cfg.metabolic_rate,
            "ambient_hazard": cfg.ambient_hazard,
            "visibility_radius": cfg.visibility_radius,
            "sim_config_seed": cfg.seed,
        }

    @staticmethod
    def _scheduled_events(sim: "Simulation") -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for trigger_cycle, event in getattr(sim.event_system, "scheduled_events", []):
            out.append({
                "cycle": trigger_cycle,
                "type": event.event_type.value,
                "severity": _r(event.severity, 4),
                "duration": event.duration,
                "region": list(event.region) if event.region else None,
            })
        out.sort(key=lambda e: (e["cycle"], e["type"]))
        return out

    @staticmethod
    def _government_params(sim: "Simulation") -> Dict[str, Any]:
        getter = getattr(sim.government, "get_params", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter() or {})
        except Exception:                            # pragma: no cover - defensive
            return {}

    # -- cycle ----------------------------------------------------------

    def record_cycle(self, sim: "Simulation", cycle: int, warnings: Any = ()) -> None:
        """
        Write one ``cycle`` record (plus a sibling ``decision`` record, and an
        ``agents`` record when the cadence calls for one).

        Must be called *after* ``sim._step(cycle)`` returns: that is the only
        point at which both ``_laws_enacted_this_cycle`` and
        ``_laws_expired_this_cycle`` hold this cycle's values, because
        ``Government._expire_laws()`` resets the enacted list at the start of
        ``tick()``.
        """
        if self._detail is None and self._agents is None:
            return
        try:
            self._last_cycle = cycle
            self._cycles_written += 1
            gov = sim.government
            decision = self._decision_payload(gov, cycle)

            record: Dict[str, Any] = {
                "rec": "cycle",
                "cycle": cycle,
                "pop": self._population_block(sim),
                "env": self._environment_block(sim),
                "events": self._events_block(sim, cycle, warnings),
                "laws": self._laws_block(gov, cycle),
                "government": {
                    "name": gov.name,
                    "decision_this_cycle": decision is not None,
                    "audit": self._audit_info(gov, cycle),
                },
            }

            # Optional, government-supplied block.  Added only when the
            # government implements `get_calibration_record` — today that is
            # ADS alone, and the key is therefore ABSENT (not null) from every
            # other government's records, which keeps their output
            # byte-identical to before self-calibration existed.  Appended last
            # rather than spliced next to `laws` for the same reason: inserting
            # mid-record would reorder nothing for ADS but invites a future
            # edit that reorders it for everyone.
            calibration = self._calibration_payload(gov, cycle)
            if calibration is not None:
                record["calibration"] = calibration

            self._emit(self._detail, record, f"cycle {cycle}")

            if decision is not None:
                self._total_decision_rounds += 1
                summary = decision.get("summary") or {}
                try:
                    self._total_proposals += int(summary.get("proposals_evaluated") or 0)
                except (TypeError, ValueError):      # pragma: no cover - defensive
                    pass
                if not self.decision_detail:
                    # groupings is ~95% of the bytes and the only part that is
                    # "show your work" rather than "what you decided".
                    decision.pop("groupings", None)
                self._emit(
                    self._detail,
                    {"rec": "decision", "cycle": cycle, **decision},
                    f"decision {cycle}",
                )

            if self._should_emit_agents(cycle):
                self._emit_agents(sim, cycle)

            if self._cycles_written % self.flush_every == 0:
                self._flush()
        except Exception as exc:
            self.errors += 1
            _log.error("run detail cycle %d failed: %s", cycle, exc)

    # -- cycle sub-blocks ------------------------------------------------

    def _population_block(self, sim: "Simulation") -> Dict[str, Any]:
        agents = sim.agents
        alive = [a for a in agents if a.alive]
        n_alive = len(alive)
        n_initial = self._initial_agents or len(agents) or 1

        deaths = 0 if self._prev_alive is None else max(0, self._prev_alive - n_alive)
        self._prev_alive = n_alive

        healths = sorted(a.health for a in alive)
        foods = [a.food_stock for a in alive]
        waters = [a.water_stock for a in alive]
        medicines = [a.medicine_stock for a in alive]

        if healths:
            health_block: Dict[str, Optional[float]] = {
                "mean": _r(sum(healths) / len(healths), 3),
                "median": _r(_percentile(healths, 0.5), 3),
                "min": _r(healths[0], 3),
                "max": _r(healths[-1], 3),
                "p25": _r(_percentile(healths, 0.25), 3),
                "p75": _r(_percentile(healths, 0.75), 3),
                "std": _r(statistics.pstdev(healths) if len(healths) > 1 else 0.0, 4),
                "gini": _r(_gini(list(healths)), 4),
            }
        else:
            health_block = {k: None for k in
                            ("mean", "median", "min", "max", "p25", "p75", "std", "gini")}

        return {
            "alive": n_alive,
            "initial": n_initial,
            "deaths_this_cycle": deaths,
            "normalized_health_score": _r(sim.normalized_health_score(), 4),
            "survival_rate": _r(n_alive / n_initial, 4),
            "health": health_block,
            "food": _distribution(foods, 2),
            "water": _distribution(waters, 2),
            "medicine": _distribution(medicines, 2),
            "infected": sum(1 for a in alive if a.epidemic_ids),
            "hungry": sum(1 for a in alive if a.hunger > _HUNGRY),
            "thirsty": sum(1 for a in alive if a.thirst > _THIRSTY),
            "starving": sum(1 for a in alive if a.food_stock < _STARVING_STOCK),
            "dehydrated": sum(1 for a in alive if a.water_stock < _DEHYDRATED_STOCK),
        }

    @staticmethod
    def _environment_block(sim: "Simulation") -> Dict[str, Any]:
        grid = sim.grid
        cells = grid.all_cells()
        n_cells = len(cells) or 1

        # One pass, not three: grid.mean_food()/mean_water() would each walk the
        # whole grid again, and this runs on every cycle of every run.
        total_food = 0.0
        total_water = 0.0
        hazard_cells = 0
        for cell in cells:
            total_food += cell.food
            total_water += cell.water
            # Every cell carries the difficulty's ambient hazard (from D2
            # upward), so ``cell.hazard > 0.0`` is true everywhere and this
            # counter would saturate at ``n_cells``. Compare against the
            # cell's own ambient floor instead, so the count/fraction track
            # cells with terrain or event hazard ON TOP OF the ambient floor.
            if cell.hazard > cell.ambient_hazard:
                hazard_cells += 1

        drought_factor = max(
            (ev.drought_factor for ev in sim.event_system.active_events
             if not ev.cancelled),
            default=0.0,
        )
        return {
            "grid_mean_food": _r(total_food / n_cells, 4),
            "grid_mean_water": _r(total_water / n_cells, 4),
            "grid_total_food": _r(total_food, 2),
            "grid_total_water": _r(total_water, 2),
            "hazard_cell_count": hazard_cells,
            "hazard_cell_fraction": _r(hazard_cells / n_cells, 4),
            "drain_mult": _r(getattr(sim, "drain_mult", None), 4),
            "drought_factor": _r(drought_factor, 4),
            "storm_damage": _r(sim.event_system.storm_damage_per_cycle(), 4),
        }

    def _event_uid(self, event: Any) -> str:
        key = id(event)
        uid = self._event_uids.get(key)
        if uid is None:
            uid = f"E-{len(self._event_uids) + 1:04d}"
            self._event_uids[key] = uid
            self._event_refs.append(event)     # keep alive so id() stays unique
        return uid

    def _events_block(self, sim: "Simulation", cycle: int, warnings: Any) -> Dict[str, Any]:
        active_records: List[Dict[str, Any]] = []
        current_uids: Set[str] = set()

        for event in sim.event_system.active_events:
            if event.cancelled or not event.is_active(cycle):
                continue
            uid = self._event_uid(event)
            current_uids.add(uid)
            effects = {
                name: _r(getattr(event, name, 0.0), 4)
                for name in ("drought_factor", "storm_damage", "disease_spread_rate",
                             "toxic_drain", "resource_boost")
                if getattr(event, name, 0.0)
            }
            active_records.append({
                "uid": uid,
                "type": event.event_type.value,
                "severity": _r(event.severity, 4),
                "cycles_remaining": event.cycles_remaining(cycle),
                "start_cycle": event.start_cycle,
                "duration": event.duration,
                "region": list(event.region) if event.region else None,
                "epidemic_id": event.epidemic_id,
                "effects": effects,
            })

        triggered = sorted(current_uids - self._prev_active_uids)
        expired = sorted(self._prev_active_uids - current_uids)
        self._prev_active_uids = current_uids
        self._total_events_triggered += len(triggered)
        self._total_events_expired += len(expired)

        warning_records = [
            {
                "type": w.event_type.value,
                "cycles_until": w.cycles_until,
                "severity": _r(w.severity, 4),
                "region": list(w.region) if w.region else None,
            }
            for w in (warnings or ())
            if getattr(w, "cycles_until", 0) > 0
        ]

        return {
            "active": active_records,
            "triggered": triggered,
            "expired": expired,
            "warnings": warning_records,
        }

    def _law_record(self, law: Any) -> Dict[str, Any]:
        applies_to = law.applies_to
        return {
            "law_id": law.law_id,
            "law_type": law.law_type,
            "source": getattr(law, "source", None),
            "enacted_cycle": law.enacted_cycle,
            "duration": law.duration,
            "expires_cycle": law.enacted_cycle + law.duration,
            "applies_to_all": applies_to is None,
            "applies_to_count": None if applies_to is None else len(applies_to),
            "event_type": law.event_type,
            "event_id": law.event_id,
            "description": law.description,
            "params": (_jsonable(law.params) if self._raw_law_params
                       else summarise_params(law.params)),
        }

    def _laws_block(self, gov: Any, cycle: int) -> Dict[str, Any]:
        reasons = getattr(gov, "_law_expiry_reasons", {}) or {}
        expired_laws = list(getattr(gov, "_laws_expired_this_cycle", []))

        # Which laws are new this cycle?  Not simply
        # Government._laws_enacted_this_cycle: that list is reset at the start
        # of tick(), so it loses any law enacted earlier in the same cycle from
        # receive_event_warnings() — including ADS's pre-emptive responses to a
        # warning, and every law an out-of-band decision round produced.  Taking
        # the union with active_laws recovers those, and adding the just-expired
        # laws catches the born-and-died-in-one-cycle case (a pre-emptive law
        # whose triggering event never materialised expires in the very tick
        # that follows).  Purely recorder-side: no behaviour changes, and the
        # enacted/expired counts in the footer stay consistent with each other.
        enacted_laws: List[Any] = []
        batch: Set[str] = set()
        for law in (list(getattr(gov, "_laws_enacted_this_cycle", []))
                    + list(gov.active_laws)
                    + expired_laws):
            if law.law_id in self._seen_law_ids or law.law_id in batch:
                continue
            batch.add(law.law_id)
            enacted_laws.append(law)
        self._seen_law_ids |= batch

        self._total_laws_enacted += len(enacted_laws)
        self._total_laws_expired += len(expired_laws)

        return {
            "active_count": sum(1 for l in gov.active_laws if l.is_active(cycle)),
            "enacted": [self._law_record(l) for l in enacted_laws],
            "expired": [
                {
                    "law_id": l.law_id,
                    "law_type": l.law_type,
                    "source": getattr(l, "source", None),
                    "enacted_cycle": l.enacted_cycle,
                    "duration": l.duration,
                    "reason": reasons.get(l.law_id, "unknown"),
                }
                for l in expired_laws
            ],
        }

    @staticmethod
    def _audit_info(gov: Any, cycle: int) -> Dict[str, Any]:
        try:
            return _jsonable(gov.get_audit_info(cycle) or {})
        except Exception:                            # pragma: no cover - defensive
            return {}

    def _calibration_payload(self, gov: Any, cycle: int) -> Optional[Dict[str, Any]]:
        """
        The government's per-cycle self-calibration block, or None if it has no
        calibration loop.

        Same duck-typed shape as :meth:`_decision_payload`: absence of the
        method means absence of the key, so adding this cost the other seven
        regimes exactly nothing.  A raising implementation is counted in
        ``errors`` and skipped rather than allowed to fail the run —
        observability must not be able to fail the science.
        """
        getter = getattr(gov, "get_calibration_record", None)
        if not callable(getter):
            return None
        try:
            payload = getter(cycle)
        except Exception as exc:                     # pragma: no cover - defensive
            self.errors += 1
            _log.error("calibration record for cycle %d failed: %s", cycle, exc)
            return None
        if payload is None:
            return None
        return dict(payload)

    def _decision_payload(self, gov: Any, cycle: int) -> Optional[Dict[str, Any]]:
        getter = getattr(gov, "get_decision_record", None)
        if not callable(getter):
            return None
        try:
            payload = getter(cycle)
        except Exception as exc:
            self.errors += 1
            _log.error("decision record for cycle %d failed: %s", cycle, exc)
            return None
        if not payload:
            return None
        return dict(payload)

    # -- agent states ----------------------------------------------------

    def _should_emit_agents(self, cycle: int) -> bool:
        if self._agents is None or not self.agent_states_every:
            return False
        if cycle == 0 or cycle % self.agent_states_every == 0:
            return True
        return self._max_cycles is not None and cycle == self._max_cycles - 1

    def _emit_agents(self, sim: "Simulation", cycle: int) -> None:
        gov = sim.government
        rows: List[List[Any]] = []
        for agent in sim.agents:
            position = agent.position if agent.alive else None
            try:
                n_laws = len(gov.laws_for_agent(agent.agent_id, cycle))
            except Exception:                        # pragma: no cover - defensive
                n_laws = None
            rows.append([
                agent.agent_id,
                bool(agent.alive),
                position[0] if position else None,
                position[1] if position else None,
                _r(agent.health, 3),
                _r(agent.food_stock, 2),
                _r(agent.water_stock, 2),
                _r(agent.medicine_stock, 2),
                _r(agent.hunger, 3),
                _r(agent.thirst, 3),
                agent.age,
                len(agent.epidemic_ids),
                sorted(agent.epidemic_ids),
                n_laws,
            ])

        self._emit(
            self._agents,
            {
                "rec": "agents",
                "cycle": cycle,
                "n": len(sim.agents),
                "n_alive": sum(1 for a in sim.agents if a.alive),
                # Columnar: repeating fourteen key strings 500 times per record
                # would cost ~55% of the file for zero information.
                "fields": list(AGENT_FIELDS),
                "rows": rows,
                "active_laws": [
                    {
                        "law_id": l.law_id,
                        "law_type": l.law_type,
                        "expires_cycle": l.enacted_cycle + l.duration,
                        "applies_to_count": (None if l.applies_to is None
                                             else len(l.applies_to)),
                    }
                    for l in gov.active_laws if l.is_active(cycle)
                ],
            },
            f"agents {cycle}",
        )

    # -- footer ----------------------------------------------------------

    def set_summary(self, summary: Dict[str, Any]) -> None:
        """Hand the ``MetricsCollector`` summary in before exit, for the footer."""
        self._summary = dict(summary or {})

    def _write_footer(self, exc_type, exc, tb) -> None:
        if self._detail is None:
            return
        record: Dict[str, Any] = {
            "rec": "footer",
            "status": "ok" if exc_type is None else "failed",
            "cycles_completed": self._cycles_written,
        }
        if exc_type is None:
            record["summary"] = self._summary or {}
            record["totals"] = {
                "deaths": max(0, self._initial_agents - (self._prev_alive or 0)),
                "laws_enacted": self._total_laws_enacted,
                "laws_expired": self._total_laws_expired,
                "decision_rounds": self._total_decision_rounds,
                "proposals_evaluated": self._total_proposals,
                "events_triggered": self._total_events_triggered,
                "events_expired": self._total_events_expired,
            }
        else:
            record["error"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(exc_type, exc, tb)
                )[-8000:],
            }
        record["timing"] = {
            "started_at": self.started_at,
            "ended_at": utc_timestamp(),
            "wall_seconds": _r(time.monotonic() - self._t0, 3),
        }
        record["recorder_errors"] = self.errors
        self._emit(self._detail, record, "footer")


# ---------------------------------------------------------------------------
# Small shared helpers (also used by the benchmark harness)
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    """Current UTC time as ``2026-09-01T14:31:07.412Z`` (millisecond precision)."""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{now.microsecond // 1000:03d}Z"


def software_versions() -> Dict[str, Optional[str]]:
    """Versions of the packages whose behaviour could change a result."""
    versions: Dict[str, Optional[str]] = {
        "python": ".".join(str(p) for p in sys.version_info[:3]),
    }
    for name in ("numpy", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:                            # pragma: no cover - defensive
            versions[name] = None
    return versions
