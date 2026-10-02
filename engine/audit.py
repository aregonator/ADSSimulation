"""Per-cycle structured log trail for simulation runs.

Produces a human-readable .log file with clear cycle headers, organized
sections for population, environment, events, government decisions, and laws.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation
    from governments.base import Government


class AuditTrail:
    """Collects per-cycle data and writes a structured log file."""

    def __init__(self, government_name: str):
        self.government_name = government_name
        self._entries: List[str] = []
        self._prev_alive: Optional[int] = None
        self._config_header: Optional[str] = None

    def record(
        self,
        cycle: int,
        sim: "Simulation",
        warnings: list,
    ) -> None:
        """Record one cycle's data as a formatted log block."""
        gov = sim.government
        agents = sim.agents
        living = [a for a in agents if a.alive]
        n_alive = len(living)
        n_initial = sim.metrics.initial_population

        # Store config header on first call
        if self._config_header is None:
            self._config_header = self._build_config_header(sim)

        # Deaths this cycle
        deaths = (self._prev_alive - n_alive) if self._prev_alive is not None else 0
        self._prev_alive = n_alive

        lines = []
        lines.append(self._cycle_header(cycle, sim))
        lines.append("")

        # --- POPULATION ---
        lines.append(self._population_section(living, n_initial, deaths))

        # --- HEALTH DRAINS ---
        lines.append(self._health_drains_section(living, sim))

        # --- ENVIRONMENT ---
        lines.append(self._environment_section(sim))

        # --- ACTIVE EVENTS ---
        active_events = sim.event_system.active_summary(cycle)
        if active_events:
            lines.append(self._events_section(active_events))

        # --- EVENT WARNINGS ---
        if warnings:
            lines.append(self._warnings_section(warnings))

        # --- SCHEDULED EVENTS ---
        scheduled_str = self._scheduled_section(sim, cycle)
        if scheduled_str:
            lines.append(scheduled_str)

        # --- GOVERNMENT DECISIONS ---
        log_info = gov.get_log_info(cycle)
        gov_section = self._government_section(log_info)
        if gov_section:
            lines.append(gov_section)

        # --- LAWS ---
        laws_section = self._laws_section(gov, cycle)
        if laws_section:
            lines.append(laws_section)

        # --- NOTES ---
        notes = self._notes_section(gov)
        if notes:
            lines.append(notes)

        lines.append("")
        self._entries.append("\n".join(lines))

    def save(self, path: str) -> None:
        """Write the full log to a file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if self._config_header:
                f.write(self._config_header)
                f.write("\n\n")
            for entry in self._entries:
                f.write(entry)
                f.write("\n")

    # ------------------------------------------------------------------
    # Header builders
    # ------------------------------------------------------------------

    def _build_config_header(self, sim: "Simulation") -> str:
        cfg = sim.config
        ruler = "=" * 80
        return (
            f"{ruler}\n"
            f"SIMULATION LOG — {self.government_name}\n"
            f"{ruler}\n"
            f"  Government:  {self.government_name}\n"
            f"  Grid:        {cfg.grid_rows} x {cfg.grid_cols}\n"
            f"  Agents:      {cfg.num_agents}\n"
            f"  Difficulty:  {cfg.difficulty}\n"
            f"  Max cycles:  {cfg.max_cycles}\n"
            f"  Seed:        {cfg.seed}\n"
            f"  Visibility:  {cfg.visibility_radius}\n"
            f"  Max steps:   {cfg.max_steps_per_cycle}/cycle\n"
            f"{ruler}"
        )

    def _cycle_header(self, cycle: int, sim: "Simulation") -> str:
        ruler = "=" * 80
        label = f"CYCLE {cycle}"
        tag = f"[{self.government_name}]"
        spacing = 80 - len(label) - len(tag) - 2
        return f"{ruler}\n{label}{' ' * max(1, spacing)}{tag}\n{ruler}"

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _population_section(self, living: list, n_initial: int, deaths: int) -> str:
        n = len(living)
        pct = 100.0 * n / max(1, n_initial)

        healths = [a.health for a in living] if living else [0.0]
        foods = [a.food_stock for a in living] if living else [0.0]
        waters = [a.water_stock for a in living] if living else [0.0]
        medicines = [a.medicine_stock for a in living] if living else [0.0]

        infected = sum(1 for a in living if a.infected)
        hungry = sum(1 for a in living if a.hunger > 0.5)
        thirsty = sum(1 for a in living if a.thirst > 0.5)

        lines = [
            "--- POPULATION ---",
            f"  Alive: {n}/{n_initial} ({pct:.1f}%)  |  Deaths this cycle: {deaths}",
            f"  Health:   median={_median_val(healths):.3f}  min={min(healths):.2f}  max={max(healths):.2f}  std={_std(healths):.3f}",
            f"  Food:     median={_median_val(foods):.2f}   min={min(foods):.1f}   max={max(foods):.1f}",
            f"  Water:    median={_median_val(waters):.2f}   min={min(waters):.1f}   max={max(waters):.1f}",
            f"  Medicine: median={_median_val(medicines):.2f}   min={min(medicines):.1f}   max={max(medicines):.1f}",
            f"  Infected: {infected} ({100*infected/max(1,n):.1f}%)  |  Hungry: {hungry} ({100*hungry/max(1,n):.1f}%)  |  Thirsty: {thirsty} ({100*thirsty/max(1,n):.1f}%)",
        ]
        return "\n".join(lines)

    def _health_drains_section(self, living: list, sim: "Simulation") -> str:
        if not living:
            return "--- HEALTH DRAINS ---\n  No living agents"
        dm = sim.drain_mult
        storm_damage = sim.event_system.storm_damage_per_cycle()
        drains = {}
        hunger_total = 0.0
        thirst_total = 0.0
        infection_total = 0.0
        hazard_total = 0.0
        storm_total = 0.0
        for a in living:
            if a.hunger > 0.4:
                hunger_total += sim.hunger_drain * dm * a.hunger
            if a.thirst > 0.4:
                thirst_total += sim.thirst_drain * dm * a.thirst
            if a.epidemic_ids:
                infection_total += sim.disease_drain * dm * len(a.epidemic_ids)
            if a.position:
                r, c = a.position
                cell = sim.grid.cell(r, c)
                hazard_total += cell.hazard * dm
                if storm_damage > 0 and not cell.shelter:
                    storm_total += storm_damage * dm
        n = len(living)
        drains["hunger"] = hunger_total / n
        drains["thirst"] = thirst_total / n
        drains["infection"] = infection_total / n
        drains["terrain_hazard"] = hazard_total / n
        drains["storm"] = storm_total / n
        ranked = sorted(drains.items(), key=lambda x: x[1], reverse=True)
        lines = ["--- HEALTH DRAINS (avg per agent, largest first) ---"]
        for name, val in ranked:
            if val > 0:
                lines.append(f"  {name:<16} {val:.5f}")
        if len(lines) == 1:
            lines.append("  (none)")
        return "\n".join(lines)

    def _environment_section(self, sim: "Simulation") -> str:
        grid = sim.grid
        all_cells = list(grid.all_cells())
        total_cells = len(all_cells)
        mean_food = sum(c.food for c in all_cells) / max(1, total_cells)
        mean_water = sum(c.water for c in all_cells) / max(1, total_cells)
        hazard_cells = sum(1 for c in all_cells if c.hazard + getattr(c, "hazard_extra", 0) > 0.1)

        lines = [
            "--- ENVIRONMENT ---",
            f"  Grid resources: food={mean_food:.1f}/cell  water={mean_water:.1f}/cell",
            f"  Hazardous cells: {hazard_cells} ({100*hazard_cells/max(1,total_cells):.1f}% of grid)",
        ]
        return "\n".join(lines)

    def _events_section(self, active_events: List[dict]) -> str:
        lines = ["--- ACTIVE EVENTS ---"]
        for i, e in enumerate(active_events, 1):
            remaining = e.get("cycles_remaining")
            rem_str = "indefinite" if remaining is None else f"{remaining} cycles"
            region = e.get("region")
            region_str = f"region={_format_region(region)}" if region else "full grid"
            epidemic_id = e.get("epidemic_id", "")
            extra = f"  id={epidemic_id}" if epidemic_id else ""
            lines.append(
                f"  [{i}] {e['type'].upper():<12} severity={e['severity']:.2f}  "
                f"remaining={rem_str}  {region_str}{extra}"
            )
        return "\n".join(lines)

    def _warnings_section(self, warnings: list) -> str:
        lines = ["--- EVENT WARNINGS ---"]
        for w in warnings:
            etype = w.event_type.value if hasattr(w.event_type, "value") else str(w.event_type)
            lines.append(
                f"  {etype.upper()} approaching in {w.cycles_until} cycles (severity={w.severity:.2f})"
            )
        return "\n".join(lines)

    def _scheduled_section(self, sim: "Simulation", cycle: int) -> Optional[str]:
        lookahead = 10
        upcoming = []
        for trigger_cycle, event in sorted(sim.event_system.scheduled_events, key=lambda x: x[0]):
            if cycle < trigger_cycle <= cycle + lookahead:
                etype = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
                in_cycles = trigger_cycle - cycle
                upcoming.append(f"  {etype.upper()} in {in_cycles} cycles (at cycle {trigger_cycle})")
        if not upcoming:
            return None
        lines = ["--- UPCOMING EVENTS (next 10 cycles) ---"] + upcoming
        return "\n".join(lines)

    def _government_section(self, log_info: dict) -> Optional[str]:
        summary = log_info.get("decision_summary", "")
        trigger = log_info.get("decision_trigger", "")
        vote_details = log_info.get("vote_details", {})
        narrative = log_info.get("state_narrative", "")

        has_content = summary or vote_details or narrative
        if not has_content:
            return None

        lines = ["--- GOVERNMENT DECISIONS ---"]

        if summary:
            for line in summary.split("\n"):
                lines.append(f"  {line}" if not line.startswith("  ") else line)

        if trigger:
            lines.append(f"  Trigger: {trigger}")

        if vote_details:
            lines.append("")
            lines.append("  Vote Distribution:")
            for cohort, detail in vote_details.items():
                lines.append(f"    {cohort}: {detail}")

        if narrative:
            lines.append("")
            lines.append("  Government State:")
            for line in narrative.split("\n"):
                lines.append(f"  {line}" if not line.startswith("  ") else line)

        return "\n".join(lines)

    def _laws_section(self, gov: "Government", cycle: int) -> Optional[str]:
        enacted = gov._laws_enacted_this_cycle
        expired = gov._laws_expired_this_cycle
        active = [l for l in gov.active_laws if l.is_active(cycle)]

        if not enacted and not expired and not active:
            return None

        lines = ["--- LAWS ---"]

        if enacted:
            lines.append(f"  Enacted this cycle ({len(enacted)}):")
            for law in enacted:
                applies = "all agents" if law.applies_to is None else f"{len(law.applies_to)} specific agents"
                lines.append(f"    {law.law_id}: {law.law_type}")
                lines.append(f"      \"{law.description}\"")
                lines.append(f"      Duration: {law.duration} cycles | Applies to: {applies}")

        if active:
            lines.append(f"  Active laws ({len(active)} total):")
            for law in active:
                rem = law.cycles_remaining(cycle)
                lines.append(f"    {law.law_id}: {law.law_type} ({rem} cycles remaining) — {law.description}")

        if expired:
            lines.append(f"  Expired this cycle ({len(expired)}):")
            for law in expired:
                lines.append(f"    {law.law_id}: {law.law_type} — completed after {law.duration} cycles")

        return "\n".join(lines)

    def _notes_section(self, gov: "Government") -> Optional[str]:
        parts = []
        if hasattr(gov, "_newly_promoted_elite") and gov._newly_promoted_elite:
            parts.append(f"  NEW_ELITE: {', '.join(gov._newly_promoted_elite)}")
        if hasattr(gov, "_newly_promoted_loyalists") and gov._newly_promoted_loyalists:
            parts.append(f"  NEW_LOYALIST: {', '.join(gov._newly_promoted_loyalists)}")
        if hasattr(gov, "_new_leader_this_cycle") and gov._new_leader_this_cycle:
            parts.append(f"  NEW_LEADER: {gov._new_leader_this_cycle}")
        if not parts:
            return None
        return "--- NOTES ---\n" + "\n".join(parts)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _mean(values: list) -> float:
    return sum(values) / max(1, len(values))


def _median_val(values: list) -> float:
    if not values:
        return 0.0
    import statistics
    return statistics.median(values)


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _format_region(region) -> str:
    if region is None:
        return "full grid"
    if isinstance(region, (list, tuple)) and len(region) == 4:
        return f"({region[0]},{region[1]})-({region[2]},{region[3]})"
    return str(region)
