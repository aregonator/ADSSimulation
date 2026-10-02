"""Metrics collection, storage, and reporting for simulation runs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .grid import Grid
    from governments.base import Government


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


@dataclass
class CycleSnapshot:
    cycle: int
    survival_rate: float
    median_health: float
    health_gini: float
    median_food: float
    median_water: float
    active_events: List[str]
    num_alive: int
    num_initial: int
    normalized_health_score: float = 0.0
    mean_calibration: Optional[float] = None
    proposals_submitted: Optional[int] = None
    proposals_enacted: Optional[int] = None
    laws_active: Optional[int] = None
    mean_prediction_error: Optional[float] = None


def _gini(values: List[float]) -> float:
    """Gini coefficient of a list of non-negative values. 0=equal, 1=maximally unequal."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    vals = sorted(max(0.0, v) for v in values)
    total = sum(vals)
    if total == 0:
        return 0.0
    # Rank-weighted sum with ASCENDING 1-based ranks, matching the standard
    # formula  G = 2*sum(i*x_i) / (n*sum(x)) - (n+1)/n  for x sorted ascending.
    #
    # REGRESSION GUARD — do not "simplify" this back into a running cumulative
    # sum.  Accumulating `cumsum += v` and then `weighted_sum += cumsum` sums
    # each value with weight (n - i + 1) — the ranks in DESCENDING order.
    # Because
    #     sum((n-i+1)*x_i) = (n+1)*S - sum(i*x_i),
    # that expression evaluates to exactly -G, and the `max(0.0, ...)` clamp
    # below then returns 0.0 for every non-degenerate distribution.  The
    # equal-values case still gives the correct 0.0, so a spot check on that
    # input alone would not catch the error.
    weighted_sum = 0.0
    for i, v in enumerate(vals):
        weighted_sum += (i + 1) * v
    # Clamp guards float error around the perfectly-equal case only; a correct
    # Gini is already in [0, 1) for non-negative inputs.
    return max(0.0, (2 * weighted_sum) / (n * total) - (n + 1) / n)


class MetricsCollector:
    """Collects per-cycle snapshots and computes summary statistics."""

    def __init__(self, initial_population: int):
        self.initial_population = initial_population
        self.snapshots: List[CycleSnapshot] = []
        self._ads_data: Dict[int, Dict[str, Any]] = {}

    def record(
        self,
        cycle: int,
        agents: List["Agent"],
        grid: "Grid",
        government: "Government",
        active_event_names: List[str],
    ) -> CycleSnapshot:
        alive = [a for a in agents if a.alive]
        healths = [a.health for a in alive]
        n_alive = len(alive)

        total_health = sum(healths)
        agent_foods = [a.food_stock for a in alive]
        agent_waters = [a.water_stock for a in alive]
        snap = CycleSnapshot(
            cycle=cycle,
            survival_rate=n_alive / self.initial_population,
            median_health=_median(healths),
            health_gini=_gini(healths),
            median_food=_median(agent_foods),
            median_water=_median(agent_waters),
            active_events=active_event_names,
            num_alive=n_alive,
            num_initial=self.initial_population,
            normalized_health_score=total_health / self.initial_population,
        )

        # ADS-specific metrics
        if hasattr(government, "get_ads_metrics"):
            ads = government.get_ads_metrics()
            snap.mean_calibration = ads.get("mean_calibration")
            snap.proposals_submitted = ads.get("proposals_submitted")
            snap.proposals_enacted = ads.get("proposals_enacted")
            snap.laws_active = ads.get("laws_active")
            snap.mean_prediction_error = ads.get("mean_prediction_error")

        self.snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {}

        last = self.snapshots[-1]
        healths_at_50pct = [
            s.cycle for s in self.snapshots if s.survival_rate <= 0.5
        ]
        time_to_50pct = healths_at_50pct[0] if healths_at_50pct else None

        return {
            "total_cycles": last.cycle,
            "normalized_health_score": round(last.normalized_health_score, 4),
            "final_survival_rate": round(last.survival_rate, 4),
            "final_median_health": round(last.median_health, 4),
            "final_health_gini": round(last.health_gini, 4),
            "min_survival_rate": round(min(s.survival_rate for s in self.snapshots), 4),
            "time_to_50pct_loss": time_to_50pct,
            "final_median_food": round(last.median_food, 2),
            "final_median_water": round(last.median_water, 2),
            "final_mean_calibration": last.mean_calibration,
            "final_proposals_enacted": last.proposals_enacted,
            "final_mean_prediction_error": last.mean_prediction_error,
        }

    def event_impact(self, event_name: str, window: int = 5) -> Dict[str, float]:
        """Measure health drop around events of a given type."""
        impacts = []
        for i, snap in enumerate(self.snapshots):
            if event_name in snap.active_events:
                pre = self.snapshots[max(0, i - 1)].median_health
                post_idx = min(len(self.snapshots) - 1, i + window)
                post = self.snapshots[post_idx].median_health
                impacts.append(pre - post)
        if not impacts:
            return {"mean_impact": 0.0, "max_impact": 0.0}
        return {
            "mean_impact": round(sum(impacts) / len(impacts), 4),
            "max_impact": round(max(impacts), 4),
        }

    def print_report(self, government_name: str = "Unknown") -> None:
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  Government: {government_name}")
        print(f"{'='*60}")
        print(f"  Total cycles:          {s['total_cycles']}")
        print(f"  Normalized health:     {s['normalized_health_score']:.4f}  ← primary metric")
        print(f"  Final survival rate:   {s['final_survival_rate']:.1%}")
        print(f"  Final median health:   {s['final_median_health']:.3f}")
        print(f"  Health inequality:     {s['final_health_gini']:.3f}  (Gini; 0=equal)")
        print(f"  Time to 50% loss:      {s.get('time_to_50pct_loss', 'N/A')} cycles")
        print(f"  Final median food:     {s['final_median_food']:.1f}")
        print(f"  Final median water:    {s['final_median_water']:.1f}")
        if s.get("final_mean_calibration") is not None:
            print(f"  Mean IA calibration:   {s['final_mean_calibration']:.3f}")
            print(f"  Proposals enacted:     {s['final_proposals_enacted']}")
            print(f"  Mean prediction err:   {s['final_mean_prediction_error']:.4f}")
        print(f"{'='*60}\n")

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "cycle": s.cycle,
                "normalized_health_score": round(s.normalized_health_score, 4),
                "survival_rate": s.survival_rate,
                "median_health": s.median_health,
                "health_gini": s.health_gini,
                "median_food": s.median_food,
                "median_water": s.median_water,
                "num_alive": s.num_alive,
                "events": "|".join(s.active_events),
                # `x if x is not None else ""` — NOT `x or ""`.  These three are
                # Optional: None means "this government reports no such metric"
                # and renders as an empty cell.  But 0.0 and 0 are legitimate
                # *values* and are falsy, so `or ""` blanks them — it cannot
                # distinguish "absent" from "exactly zero".  This matters most
                # for `mean_prediction_error` (populated by
                # ``AdsGovernment.get_ads_metrics``): a perfect forecast, delta
                # == 0.0, is exactly the case this column exists to show, and a
                # falsy check would blank the one value it should report.
                #
                # `is not None` also matches print_report's existing test above.
                # The other two fields are always None today, but the same
                # check protects them once a government starts populating them.
                "mean_calibration": (
                    s.mean_calibration if s.mean_calibration is not None else ""
                ),
                "proposals_enacted": (
                    s.proposals_enacted if s.proposals_enacted is not None else ""
                ),
                "mean_prediction_error": (
                    s.mean_prediction_error
                    if s.mean_prediction_error is not None else ""
                ),
            }
            for s in self.snapshots
        ]


class MultiRunComparison:
    """Holds results from multiple government types running same scenario."""

    def __init__(self):
        self._results: Dict[str, MetricsCollector] = {}

    def add(self, government_name: str, metrics: MetricsCollector) -> None:
        self._results[government_name] = metrics

    def print_comparison(self) -> None:
        print(f"\n{'='*78}")
        print(f"  COMPARATIVE RESULTS  (primary metric: normalized health score)")
        print(f"{'='*78}")
        header = (f"{'Government':<22} {'NormHealth':>10} {'Survival':>8} "
                  f"{'MedHealth':>10} {'Gini':>6} {'T50%':>6} {'Food':>7}")
        print(header)
        print("-" * 78)
        ranked = sorted(
            self._results.items(),
            key=lambda kv: kv[1].summary().get("normalized_health_score", 0),
            reverse=True,
        )
        for name, metrics in ranked:
            s = metrics.summary()
            t50 = str(s.get("time_to_50pct_loss") or "N/A")
            print(
                f"  {name:<20} "
                f"{s['normalized_health_score']:>10.4f} "
                f"{s['final_survival_rate']:>8.1%} "
                f"{s['final_median_health']:>10.3f} "
                f"{s['final_health_gini']:>6.3f} "
                f"{t50:>6} "
                f"{s['final_median_food']:>7.1f}"
            )
        print(f"{'='*78}\n")

    def best_by(self, metric: str) -> Optional[str]:
        if not self._results:
            return None
        return max(
            self._results.keys(),
            key=lambda name: self._results[name].summary().get(metric, 0) or 0,
        )
