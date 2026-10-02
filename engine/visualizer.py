"""
SimulationVisualizer: color-coded grid + health progression visualization.

Requires matplotlib (pip install matplotlib).  Every public method checks for
the dependency and raises a clear ImportError if it is missing.

Quick start
-----------
    from engine.visualizer import SimulationVisualizer

    viz = SimulationVisualizer()

    # During a run, call record_frame() each cycle:
    for cycle in range(max_cycles):
        sim._step(cycle)
        viz.record_frame(sim, cycle)

    # After the run:
    viz.render_final(sim)                          # grid + timeline dashboard
    viz.save_animation("run.gif")                  # animated gif
    viz.compare_governments(results_dict)          # bar chart comparison
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation
    from .grid import Grid
    from .agent import Agent
    from .metrics import MetricsCollector, CycleSnapshot


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

TERRAIN_COLORS: Dict[str, str] = {
    "plains":    "#8BC34A",   # light green
    "forest":    "#2E7D32",   # dark green
    "water":     "#1565C0",   # blue
    "mountain":  "#9E9E9E",   # grey
    "wasteland": "#795548",   # brown
}

EVENT_OVERLAY_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "drought":       (1.0,  0.55, 0.0,  0.35),   # amber, semi-transparent
    "storm":         (0.33, 0.43, 0.48, 0.35),   # blue-grey
    "epidemic":      (0.72, 0.11, 0.11, 0.40),   # dark red
    "toxic_spill":   (0.48, 0.11, 0.64, 0.40),   # purple
    "resource_rush": (0.98, 0.66, 0.15, 0.40),   # gold
}

# ---------------------------------------------------------------------------
# Print legibility
# ---------------------------------------------------------------------------
#
# The cycle dashboards are included in the manuscript at ~6.5 in (\textwidth).
# Every in-figure point size is therefore multiplied by
#
#     PRINT_SCALE = 6.5 / (final figure width in inches)
#
# on the printed page.  An 18.4 in-wide canvas would give PRINT_SCALE ≈ 0.35,
# shrinking a 7 pt legend to ~2.5 pt — illegible, and well under JPART's
# 12-14 pt accessibility guidance.
#
# The figure is therefore authored close to the size it is printed at, so
# point sizes chosen here mean roughly what they say.  At 9.75 in wide the
# scale is ≈ 0.68, so the sizes below land at 10-12 pt on the page.  Resolution
# is recovered with DPI rather than with canvas inches.
#
# This only holds if the saved width really is DASHBOARD_FIGSIZE[0].  It is not
# under ``bbox_inches="tight"``: tight-bbox grows the canvas to enclose artists
# placed outside their axes (an outside-axes legend grows 9.75 in to 11.78 in,
# cutting every effective size by a further 17%).  So the legend lives in its
# own GridSpec row (inside the canvas) and the dashboard is saved WITHOUT tight
# bbox.  Keep both properties together or the arithmetic below stops being true.
#
# If you change DASHBOARD_FIGSIZE, re-derive these: effective_pt = pt * 6.5 / W.
DASHBOARD_FIGSIZE: Tuple[float, float] = (9.75, 7.2)
DASHBOARD_DPI: int = 300
DASHBOARD_PRINT_WIDTH_IN: float = 6.5   # \textwidth the PNG is included at

FS_SUPTITLE: int = 18     # → 12.0 pt at 6.5 in
FS_PANEL_TITLE: int = 17  # → 11.3 pt
FS_AXIS_LABEL: int = 16   # → 10.7 pt
FS_TICK: int = 15         # → 10.0 pt
FS_LEGEND: int = 15       # → 10.0 pt
FS_REGION_LABEL: int = 12 # →  8.0 pt (in-map overlay label, not body text)

# Health below this is drawn as "critical".  Matches the lowest band of
# _health_color(), so shape and hue always change together.
CRITICAL_HEALTH: float = 0.3


def _agent_marker(health: float, infected: bool) -> str:
    """
    Marker shape for an agent, so health is never encoded by colour alone.

    The red→green health ramp collapses under deuteranopia and protanopia, which
    would leave "healthy" and "critical" agents indistinguishable by colour
    alone.  Shape carries the same distinction redundantly, mirroring the
    colour+linestyle+marker convention the aggregate plots already use (see
    GOV_COLORS / GOV_MARKERS in benchmark_core).

        o  healthy        (health >= CRITICAL_HEALTH)
        v  critical       (health <  CRITICAL_HEALTH) — triangle points down
        X  infected       (takes precedence; filled, so it accepts an edgecolor)
    """
    if infected:
        return "X"
    return "o" if health >= CRITICAL_HEALTH else "v"


# Agent health → colour (green → yellow → red)
def _health_color(health: float) -> str:
    """Return hex colour for a health value in [0, 1]."""
    h = max(0.0, min(1.0, health))
    if h >= 0.6:
        # green (1.0) → yellow-green (0.6)
        ratio = (h - 0.6) / 0.4
        r = int(255 * (1 - ratio))
        g = 200
        b = 0
    elif h >= 0.3:
        # yellow (0.6) → orange (0.3)
        ratio = (h - 0.3) / 0.3
        r = 220
        g = int(160 * ratio)
        b = 0
    else:
        # orange (0.3) → red (0.0)
        ratio = h / 0.3
        r = int(180 + 40 * ratio)
        g = int(40 * ratio)
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _resource_alpha(food: float, water: float) -> float:
    """Brightness overlay based on available resources (0=depleted, 1=full)."""
    return max(0.05, min(0.85, (food + water) / 200.0))


def _blend_terrain(hex_col: str, alpha: float) -> Tuple[float, float, float]:
    """
    The single definition of how a cell's on-screen colour is produced.

    Every render path applied this same expression inline, and the legend used
    the raw TERRAIN_COLORS entry instead — so the swatches never matched the
    map.  Routing both through one function is what keeps them honest.
    """
    return tuple(v * 0.5 + 0.5 * alpha for v in _hex_to_rgb(hex_col))


def _rendered_terrain_swatches(
    cells: List[List[dict]],
) -> Dict[str, Tuple[float, float, float]]:
    """
    Mean *as-drawn* colour of each terrain type present in a frame.

    Cell colour is terrain hue blended with a resource-brightness term, so a
    single terrain spans a range of on-screen colours and no swatch can be
    exactly right.  The mean of what was actually drawn is the honest choice:
    the legend entry is the average of the pixels it claims to describe.
    Terrains absent from the frame are omitted rather than shown in a colour
    that appears nowhere on the map.
    """
    sums: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    for row in cells:
        for cd in row:
            terrain = cd["terrain"]
            rgb = _blend_terrain(
                TERRAIN_COLORS.get(terrain, "#AAAAAA"),
                _resource_alpha(cd["food"], cd["water"]),
            )
            acc = sums.setdefault(terrain, [0.0, 0.0, 0.0])
            for i in range(3):
                acc[i] += rgb[i]
            counts[terrain] = counts.get(terrain, 0) + 1

    # Preserve TERRAIN_COLORS ordering so the legend is stable across frames.
    return {
        t: (sums[t][0] / counts[t], sums[t][1] / counts[t], sums[t][2] / counts[t])
        for t in TERRAIN_COLORS
        if counts.get(t)
    }


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class SimulationVisualizer:
    """
    Records lightweight frame snapshots during a simulation run and renders
    colour-coded grids, health timelines, and government comparison charts.
    """

    def __init__(self) -> None:
        self._frames: List[dict] = []   # lightweight per-cycle snapshots

    # ------------------------------------------------------------------
    # Frame recording
    # ------------------------------------------------------------------

    def record_frame(self, sim: "Simulation", cycle: int) -> None:
        """
        Capture a lightweight snapshot at *cycle* for later rendering.
        Call this once per cycle during sim.run() or inside a custom loop.
        """
        import statistics as _stats
        alive = [a for a in sim.agents if a.alive]
        n_init = len(sim.agents)
        n_alive = len(alive)
        nhs = sim.normalized_health_score()
        median_h = _stats.median([a.health for a in alive]) if alive else 0.0

        agent_data = [
            {
                "row": a.position[0] if a.position else -1,
                "col": a.position[1] if a.position else -1,
                "health": a.health,
                "food": a.food_stock,
                "water": a.water_stock,
                "medicine": a.medicine_stock,
                "infected": a.infected,
            }
            for a in alive
        ]

        cells_data = []
        for row in sim.grid._cells:
            row_snap = []
            for cell in row:
                row_snap.append({
                    "terrain": cell.terrain.value,
                    "food": cell.food,
                    "water": cell.water,
                    "medicine": cell.medicine,
                    "hazard": cell.hazard,
                    "contaminated": cell.contaminated,
                })
            cells_data.append(row_snap)

        # Capture federation region boundaries if available
        region_boundaries: List[Tuple[int, int, int, int, str]] = []
        gov = sim.government
        if hasattr(gov, "regions") and gov.regions:
            for reg in gov.regions:
                r_min, c_min, r_max, c_max = reg.bounds
                region_boundaries.append((r_min, c_min, r_max, c_max, reg.region_id))

        self._frames.append({
            "cycle": cycle,
            "agents": agent_data,
            "cells": cells_data,
            "active_events": sim.event_system.active_summary(cycle),
            "normalized_health_score": nhs,
            "survival_rate": n_alive / max(1, n_init),
            "median_health": median_h,
            "num_alive": n_alive,
            "num_initial": n_init,
            "region_boundaries": region_boundaries,
        })

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _require_matplotlib(self=None):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with:  pip install matplotlib"
            )

    def render_grid(
        self,
        sim: "Simulation",
        cycle: Optional[int] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """
        Render the current grid state as a colour-coded 2-D image.

        Colour key
        ----------
        - Cell background  → terrain type (see TERRAIN_COLORS)
        - Cell brightness  → resource level (food + water)
        - Coloured patches → active event regions
        - Dots             → agents, coloured by health (green→red)
        - Dot size         → proportional to agent's total resources

        Returns matplotlib Figure.
        """
        self._require_matplotlib()
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        grid = sim.grid
        rows, cols = grid.rows, grid.cols
        active_events = sim.event_system.active_summary(cycle or sim.cycle)

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(8, cols * 0.55), max(7, rows * 0.55)))
        else:
            fig = ax.get_figure()

        # --- Terrain base layer ---
        terrain_img = np.zeros((rows, cols, 3))
        resource_alpha = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                cell = grid.cell(r, c)
                hex_col = TERRAIN_COLORS.get(cell.terrain.value, "#AAAAAA")
                rgb = _hex_to_rgb(hex_col)
                alpha = _resource_alpha(cell.food, cell.water)
                # Blend terrain colour toward white by resource level
                blended = tuple(v * 0.5 + 0.5 * alpha for v in rgb)
                terrain_img[r, c] = blended
                resource_alpha[r, c] = alpha

        ax.imshow(terrain_img, origin="upper", aspect="equal", interpolation="nearest")

        # --- Event overlays ---
        for event in active_events:
            etype = event.get("type", "")
            rgba = EVENT_OVERLAY_COLORS.get(etype)
            if rgba is None:
                continue
            region = event.get("region")
            if region:
                r_min, c_min, r_max, c_max = region
                rect = mpatches.Rectangle(
                    (c_min - 0.5, r_min - 0.5),
                    (c_max - c_min + 1), (r_max - r_min + 1),
                    linewidth=1.5,
                    edgecolor=rgba[:3],
                    facecolor=rgba,
                    zorder=2,
                )
                ax.add_patch(rect)
            else:
                # Global event — full grid overlay
                rect = mpatches.Rectangle(
                    (-0.5, -0.5), cols, rows,
                    linewidth=0, facecolor=rgba, zorder=2,
                )
                ax.add_patch(rect)

        # --- Contaminated cells (epidemic) ---
        for r in range(rows):
            for c in range(cols):
                if grid.cell(r, c).contaminated:
                    rect = mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0,
                        facecolor=(0.72, 0.11, 0.11, 0.25),
                        zorder=3,
                    )
                    ax.add_patch(rect)

        # --- Agent markers ---
        for agent in sim.agents:
            if not agent.alive or not agent.position:
                continue
            ar, ac = agent.position
            color = _health_color(agent.health)
            resources = agent.food_stock + agent.water_stock + agent.medicine_stock
            size = max(20, min(120, resources * 3 + 20))
            ax.scatter(
                ac, ar,
                c=color, s=size,
                marker=_agent_marker(agent.health, bool(agent.infected)),
                zorder=5, edgecolors="black", linewidths=0.4,
            )

        # --- Federation region boundaries ---
        gov = sim.government
        if hasattr(gov, "regions") and gov.regions:
            region_boundaries = [
                (reg.bounds[0], reg.bounds[1], reg.bounds[2], reg.bounds[3], reg.region_id)
                for reg in gov.regions
            ]
            self._draw_region_boundaries(ax, region_boundaries, cols)

        # --- Grid lines ---
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(range(0, cols, max(1, cols // 10)))
        ax.set_yticks(range(0, rows, max(1, rows // 10)))

        # --- Legend ---
        legend_elements = [
            mpatches.Patch(facecolor=TERRAIN_COLORS[t], label=t.capitalize())
            for t in TERRAIN_COLORS
        ]
        legend_elements += [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca25f",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=8, label="Agent (healthy)"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#dd1c1a",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=8, label=f"Agent (critical, <{CRITICAL_HEALTH:g})"),
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#dd1c1a",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=8, label="Agent (infected)"),
        ]
        for etype, rgba in EVENT_OVERLAY_COLORS.items():
            legend_elements.append(
                mpatches.Patch(
                    facecolor=rgba, edgecolor=rgba[:3], linewidth=1,
                    label=etype.replace("_", " ").title(),
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            framealpha=0.85,
        )

        cycle_lbl = cycle if cycle is not None else sim.cycle
        nhs = sim.normalized_health_score()
        alive_count = sum(1 for a in sim.agents if a.alive)
        ax.set_title(
            title or
            f"Cycle {cycle_lbl}  |  Alive: {alive_count} of {len(sim.agents)}  "
            f"|  Normalized health: {nhs:.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Region boundary overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_region_boundaries(
        ax,
        region_boundaries: List[Tuple[int, int, int, int, str]],
        cols: int,
    ) -> None:
        """
        Draw horizontal federation region boundary lines and region ID labels.

        Each entry in *region_boundaries* is (r_min, c_min, r_max, c_max, region_id).
        Boundaries are drawn as white dashed lines between regions, with region
        labels on the left margin.
        """
        if not region_boundaries:
            return

        # Sort by r_min to identify boundary rows between consecutive strips
        sorted_regions = sorted(region_boundaries, key=lambda x: x[0])

        for i, (r_min, c_min, r_max, c_max, region_id) in enumerate(sorted_regions):
            # Draw a dividing line above each region except the first
            if i > 0:
                # The boundary sits between the previous region's r_max and this r_min.
                # Draw it at r_min - 0.5 (between rows in image coordinates).
                y = r_min - 0.5
                ax.axhline(
                    y=y,
                    xmin=0, xmax=1,
                    color="white",
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.85,
                    zorder=6,
                )

            # Region label on the left side, centred vertically within the strip
            mid_row = (r_min + r_max) / 2.0
            ax.text(
                -0.5, mid_row,
                region_id,
                color="white",
                fontsize=FS_REGION_LABEL,
                fontweight="bold",
                va="center",
                ha="right",
                zorder=7,
                bbox=dict(
                    facecolor="black",
                    alpha=0.55,
                    pad=1.5,
                    boxstyle="round,pad=0.2",
                ),
            )

    # ------------------------------------------------------------------
    # Health timeline
    # ------------------------------------------------------------------

    def render_health_timeline(
        self,
        metrics: "MetricsCollector",
        government_name: str = "",
        ax=None,
    ):
        """
        Plot normalized health score and survival rate over cycles.
        Returns matplotlib Figure.
        """
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        snaps = metrics.snapshots
        if not snaps:
            raise ValueError("No snapshots in MetricsCollector — run the simulation first.")

        cycles = [s.cycle for s in snaps]
        nhs = [s.normalized_health_score for s in snaps]
        surv = [s.survival_rate for s in snaps]
        median_h = [s.median_health for s in snaps]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.fill_between(cycles, nhs, alpha=0.25, color="#1565C0")
        ax.plot(cycles, nhs, label="Normalized health score", color="#1565C0", linewidth=2)
        ax.plot(cycles, surv, label="Survival rate", color="#2E7D32", linewidth=1.5,
                linestyle="--")
        ax.plot(cycles, median_h, label="Median health (alive)", color="#F57F17",
                linewidth=1.5, linestyle=":")

        # Event markers
        event_cycles = sorted(
            {s.cycle for s in snaps if s.active_events},
        )
        for ec in event_cycles:
            ax.axvline(ec, color="red", alpha=0.08, linewidth=0.8)

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Score / Rate")
        ax.set_title(f"Health Progression — {government_name}" if government_name
                     else "Health Progression")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Full dashboard
    # ------------------------------------------------------------------

    def render_final(
        self,
        sim: "Simulation",
        government_name: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Create a two-panel dashboard: grid view (left) + health timeline (right).
        Optionally saves to *save_path* (PNG/PDF).
        Returns matplotlib Figure.
        """
        self._require_matplotlib()
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 8))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])

        ax_grid = fig.add_subplot(gs[0])
        ax_time = fig.add_subplot(gs[1])

        self.render_grid(sim, ax=ax_grid, title=f"Final state — {government_name}")
        self.render_health_timeline(sim.metrics, government_name=government_name, ax=ax_time)

        nhs = sim.normalized_health_score()
        fig.suptitle(
            f"{government_name}  |  Normalized health score: {nhs:.4f}",
            fontsize=13, fontweight="bold",
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Animation / Video
    # ------------------------------------------------------------------

    def save_animation(self, output_path: str, fps: int = 5) -> None:
        """
        Save an animated GIF of recorded frames (grid panel only).
        For a full two-panel MP4/GIF use save_video() instead.
        Requires imageio.
        """
        if not self._frames:
            raise RuntimeError("No frames recorded. Call record_frame() during the run.")

        self._require_matplotlib()
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for animation. "
                "Install with:  pip install imageio"
            )

        import matplotlib.pyplot as plt
        import io

        images = []
        for frame in self._frames:
            fig = self._render_frame_dict(frame)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            images.append(imageio.imread(buf))
            plt.close(fig)

        imageio.mimsave(output_path, images, fps=fps)
        print(f"Animation saved to {output_path}  ({len(images)} frames)")

    def save_video(
        self,
        output_path: str,
        fps: int = 5,
        dpi: int = 100,
        government_name: str = "",
    ) -> None:
        """
        Save a two-panel video (grid + cumulative health chart) matching the
        render_final() dashboard layout.

        Supports MP4 (preferred) and GIF based on file extension.

        MP4 requires imageio-ffmpeg:
            pip install imageio[ffmpeg]
        GIF requires only imageio:
            pip install imageio
        """
        if not self._frames:
            raise RuntimeError("No frames recorded. Call record_frame() during the run.")

        self._require_matplotlib()
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with:  pip install imageio[ffmpeg]"
            )

        import matplotlib.pyplot as plt
        import io
        import os

        ext = os.path.splitext(output_path)[1].lower()
        is_mp4 = ext in (".mp4", ".avi", ".mov", ".mkv")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        total = len(self._frames)
        print(f"  Rendering {total} frames for video ...", flush=True)

        if is_mp4:
            writer_kwargs = {"fps": fps, "codec": "libx264",
                             "quality": 8, "pixelformat": "yuv420p"}
            writer = imageio.get_writer(output_path, **writer_kwargs)
        else:
            writer = imageio.get_writer(output_path, fps=fps, loop=0)

        for i, frame in enumerate(self._frames):
            fig = self._render_dashboard_frame(frame, i, government_name)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            img = imageio.imread(buf)
            # MP4 requires even pixel dimensions
            if is_mp4:
                h, w = img.shape[:2]
                if h % 2 != 0 or w % 2 != 0:
                    img = img[: h - h % 2, : w - w % 2]
            writer.append_data(img)
            plt.close(fig)
            if (i + 1) % 20 == 0 or i == total - 1:
                print(f"    {i+1}/{total} frames written", flush=True)

        writer.close()
        print(f"  Video saved to {output_path}  ({total} frames @ {fps} fps)")

    def _render_dashboard_frame(
        self,
        frame: dict,
        frame_idx: int,
        government_name: str = "",
    ):
        """
        Render one two-panel video frame: grid (left) + health timeline to
        the current cycle (right). Matches the render_final() layout.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as mpatches
        import numpy as np

        cells = frame["cells"]
        agents = frame["agents"]
        active_events = frame["active_events"]
        cycle = frame["cycle"]
        rows = len(cells)
        cols = len(cells[0]) if rows else 1
        nhs = frame["normalized_health_score"]
        n_alive = frame["num_alive"]
        n_init = frame["num_initial"]

        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])
        ax_grid = fig.add_subplot(gs[0])
        ax_time = fig.add_subplot(gs[1])

        # --- Grid panel ---
        terrain_img = np.zeros((rows, cols, 3))
        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                hex_col = TERRAIN_COLORS.get(cd["terrain"], "#AAAAAA")
                alpha = _resource_alpha(cd["food"], cd["water"])
                rgb = _hex_to_rgb(hex_col)
                terrain_img[r, c] = tuple(v * 0.5 + 0.5 * alpha for v in rgb)

        ax_grid.imshow(terrain_img, origin="upper", aspect="equal",
                       interpolation="nearest")

        for event in active_events:
            etype = event.get("type", "")
            rgba = EVENT_OVERLAY_COLORS.get(etype)
            if rgba is None:
                continue
            region = event.get("region")
            if region:
                r_min, c_min, r_max, c_max = region
                ax_grid.add_patch(mpatches.Rectangle(
                    (c_min - 0.5, r_min - 0.5),
                    c_max - c_min + 1, r_max - r_min + 1,
                    linewidth=1.5, edgecolor=rgba[:3], facecolor=rgba, zorder=2,
                ))
            else:
                ax_grid.add_patch(mpatches.Rectangle(
                    (-0.5, -0.5), cols, rows,
                    linewidth=0, facecolor=rgba, zorder=2,
                ))

        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                if cd.get("contaminated"):
                    ax_grid.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor=(0.72, 0.11, 0.11, 0.25), zorder=3,
                    ))

        for ad in agents:
            if ad["row"] < 0:
                continue
            color = _health_color(ad["health"])
            resources = ad["food"] + ad["water"] + ad["medicine"]
            size = max(20, min(120, resources * 3 + 20))
            ax_grid.scatter(ad["col"], ad["row"], c=color, s=size,
                            marker=_agent_marker(ad["health"],
                                                 bool(ad.get("infected"))),
                            zorder=5, edgecolors="black", linewidths=0.4)

        # --- Federation region boundaries ---
        region_boundaries = frame.get("region_boundaries", [])
        if region_boundaries:
            self._draw_region_boundaries(ax_grid, region_boundaries, cols)

        # --- Legend (matches render_grid) ---
        legend_elements = [
            mpatches.Patch(facecolor=TERRAIN_COLORS[t], label=t.capitalize())
            for t in TERRAIN_COLORS
        ]
        import matplotlib.pyplot as _plt
        legend_elements += [
            _plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca25f",
                        markeredgecolor="black", markeredgewidth=0.4,
                        markersize=8, label="Agent (healthy)"),
            _plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#dd1c1a",
                        markeredgecolor="black", markeredgewidth=0.4,
                        markersize=8, label=f"Agent (critical, <{CRITICAL_HEALTH:g})"),
            _plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#dd1c1a",
                        markeredgecolor="black", markeredgewidth=0.4,
                        markersize=8, label="Agent (infected)"),
        ]
        for etype, rgba in EVENT_OVERLAY_COLORS.items():
            legend_elements.append(
                mpatches.Patch(
                    facecolor=rgba, edgecolor=rgba[:3], linewidth=1,
                    label=etype.replace("_", " ").title(),
                )
            )
        ax_grid.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            framealpha=0.85,
        )

        ax_grid.set_title(
            f"Cycle {cycle}  |  Alive: {n_alive} of {n_init}  |  "
            f"Normalized health: {nhs:.2f}",
            fontsize=10,
        )
        ax_grid.set_xlabel("Column", fontsize=8)
        ax_grid.set_ylabel("Row", fontsize=8)
        ax_grid.tick_params(labelsize=7)

        # --- Health timeline panel (data up to current frame) ---
        history = self._frames[: frame_idx + 1]
        h_cycles = [f["cycle"] for f in history]
        h_nhs    = [f["normalized_health_score"] for f in history]
        h_surv   = [f["survival_rate"] for f in history]
        h_median = [f["median_health"] for f in history]

        ax_time.fill_between(h_cycles, h_nhs, alpha=0.20, color="#1565C0")
        ax_time.plot(h_cycles, h_nhs,  label="Norm. health score",
                     color="#1565C0", linewidth=2)
        ax_time.plot(h_cycles, h_surv, label="Survival rate",
                     color="#2E7D32", linewidth=1.5, linestyle="--")
        ax_time.plot(h_cycles, h_median, label="Median health (alive)",
                     color="#F57F17", linewidth=1.5, linestyle=":")

        # Mark active-event cycles
        for f in history:
            if f["active_events"]:
                ax_time.axvline(f["cycle"], color="red", alpha=0.07, linewidth=0.7)

        ax_time.set_xlim(0, max(self._frames[-1]["cycle"], 1))
        ax_time.set_ylim(-0.02, 1.05)
        ax_time.set_xlabel("Cycle", fontsize=8)
        ax_time.set_ylabel("Score / Rate", fontsize=8)
        ax_time.set_title(
            f"Health Progression — {government_name}" if government_name
            else "Health Progression",
            fontsize=10,
        )
        ax_time.legend(fontsize=8)
        ax_time.grid(alpha=0.3)
        ax_time.tick_params(labelsize=7)

        fig.suptitle(
            f"{government_name}  |  Normalized health score: {nhs:.4f}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def _render_frame_dict(self, frame: dict):
        """Render a stored lightweight frame dict to a Figure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        cells = frame["cells"]
        agents = frame["agents"]
        active_events = frame["active_events"]
        cycle = frame["cycle"]
        rows = len(cells)
        cols = len(cells[0]) if rows else 0

        fig, ax = plt.subplots(figsize=(max(6, cols * 0.5), max(5, rows * 0.5)))

        terrain_img = np.zeros((rows, cols, 3))
        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                hex_col = TERRAIN_COLORS.get(cd["terrain"], "#AAAAAA")
                alpha = _resource_alpha(cd["food"], cd["water"])
                rgb = _hex_to_rgb(hex_col)
                terrain_img[r, c] = tuple(v * 0.5 + 0.5 * alpha for v in rgb)

        ax.imshow(terrain_img, origin="upper", aspect="equal", interpolation="nearest")

        # Event overlays
        for event in active_events:
            etype = event.get("type", "")
            rgba = EVENT_OVERLAY_COLORS.get(etype)
            if rgba is None:
                continue
            region = event.get("region")
            if region:
                r_min, c_min, r_max, c_max = region
                rect = mpatches.Rectangle(
                    (c_min - 0.5, r_min - 0.5),
                    (c_max - c_min + 1), (r_max - r_min + 1),
                    linewidth=1.5, edgecolor=rgba[:3], facecolor=rgba, zorder=2,
                )
                ax.add_patch(rect)
            else:
                ax.add_patch(mpatches.Rectangle(
                    (-0.5, -0.5), cols, rows,
                    linewidth=0, facecolor=rgba, zorder=2,
                ))

        # Contaminated cells
        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                if cd.get("contaminated"):
                    ax.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor=(0.72, 0.11, 0.11, 0.25), zorder=3,
                    ))

        # Agent markers
        for ad in agents:
            if ad["row"] < 0:
                continue
            color = _health_color(ad["health"])
            resources = ad["food"] + ad["water"] + ad["medicine"]
            size = max(20, min(120, resources * 3 + 20))
            ax.scatter(ad["col"], ad["row"], c=color, s=size,
                       marker=_agent_marker(ad["health"],
                                            bool(ad.get("infected"))),
                       zorder=5, edgecolors="black", linewidths=0.4)

        # Federation region boundaries
        region_boundaries = frame.get("region_boundaries", [])
        if region_boundaries:
            self._draw_region_boundaries(ax, region_boundaries, cols)

        nhs = frame["normalized_health_score"]
        n_alive = frame["num_alive"]
        n_init = frame["num_initial"]
        ax.set_title(
            f"Cycle {cycle}  |  Alive: {n_alive} of {n_init}  |  "
            f"Normalized health: {nhs:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Column", fontsize=7)
        ax.set_ylabel("Row", fontsize=7)
        ax.tick_params(labelsize=6)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Rich snapshot dashboard (grid + health timeline from stored history)
    # ------------------------------------------------------------------

    def render_snapshot_dashboard(
        self,
        frame: dict,
        health_history: List[dict],
        government_name: str = "",
    ):
        """
        Two-panel dashboard for a single captured snapshot.

        Left  — colour-coded grid with full terrain/agent/event legend.
        Right — health timeline (Normalized Health Score, Survival Rate,
                Mean Health alive) using all history up to frame["cycle"].

        Parameters
        ----------
        frame           Snapshot dict captured by record_frame().
        health_history  List of lightweight dicts with keys:
                        cycle, nhs, survival_rate, median_health, has_events.
                        All entries with cycle <= frame["cycle"] are plotted.
        government_name Display name used in titles.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as mpatches
        import numpy as np

        cells = frame["cells"]
        agents = frame["agents"]
        active_events = frame["active_events"]
        cycle = frame["cycle"]
        rows = len(cells)
        cols = len(cells[0]) if rows else 1
        nhs = frame["normalized_health_score"]
        n_alive = frame["num_alive"]
        n_init = frame["num_initial"]

        # constrained_layout, not tight_layout: it is the only one of the two
        # that reserves space for a legend hosted in its own axes instead of
        # letting it overlap its neighbours.
        fig = plt.figure(figsize=DASHBOARD_FIGSIZE, layout="constrained")
        # Row 1 is a full-width, invisible host for the legend.  Keeping the
        # legend inside the canvas is what makes the saved width equal
        # DASHBOARD_FIGSIZE[0], which the point-size arithmetic depends on.
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1],
                      height_ratios=[1, 0.34])
        ax_grid = fig.add_subplot(gs[0, 0])
        ax_time = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, :])
        ax_legend.axis("off")

        # ── Grid panel ───────────────────────────────────────────────────────
        terrain_img = np.zeros((rows, cols, 3))
        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                terrain_img[r, c] = _blend_terrain(
                    TERRAIN_COLORS.get(cd["terrain"], "#AAAAAA"),
                    _resource_alpha(cd["food"], cd["water"]),
                )

        ax_grid.imshow(terrain_img, origin="upper", aspect="equal",
                       interpolation="nearest")

        for event in active_events:
            etype = event.get("type", "")
            rgba = EVENT_OVERLAY_COLORS.get(etype)
            if rgba is None:
                continue
            region = event.get("region")
            if region:
                r_min, c_min, r_max, c_max = region
                ax_grid.add_patch(mpatches.Rectangle(
                    (c_min - 0.5, r_min - 0.5),
                    c_max - c_min + 1, r_max - r_min + 1,
                    linewidth=1.5, edgecolor=rgba[:3], facecolor=rgba, zorder=2,
                ))
            else:
                ax_grid.add_patch(mpatches.Rectangle(
                    (-0.5, -0.5), cols, rows,
                    linewidth=0, facecolor=rgba, zorder=2,
                ))

        for r, row_data in enumerate(cells):
            for c, cd in enumerate(row_data):
                if cd.get("contaminated"):
                    ax_grid.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor=(0.72, 0.11, 0.11, 0.25), zorder=3,
                    ))

        for ad in agents:
            if ad["row"] < 0:
                continue
            color = _health_color(ad["health"])
            resources = ad["food"] + ad["water"] + ad["medicine"]
            size = max(20, min(120, resources * 3 + 20))
            ax_grid.scatter(
                ad["col"], ad["row"], c=color, s=size,
                marker=_agent_marker(ad["health"], bool(ad.get("infected"))),
                zorder=5, edgecolors="black", linewidths=0.4,
            )

        region_boundaries = frame.get("region_boundaries", [])
        if region_boundaries:
            self._draw_region_boundaries(ax_grid, region_boundaries, cols)

        # Terrain swatches are the mean colour actually drawn for that terrain in
        # THIS frame, not the raw TERRAIN_COLORS hue — the map blends hue with a
        # resource-brightness term, so the raw hue appears nowhere on it.
        legend_elements = [
            mpatches.Patch(facecolor=rgb, edgecolor="#444444", linewidth=0.5,
                           label=t.capitalize())
            for t, rgb in _rendered_terrain_swatches(cells).items()
        ]
        # Shape, not just hue, separates healthy from critical: the red→green
        # ramp is indistinguishable under deuteranopia/protanopia.
        legend_elements += [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca25f",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=9, label="Agent (healthy)"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#dd1c1a",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=9, label=f"Agent (critical, <{CRITICAL_HEALTH:g})"),
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#dd1c1a",
                       markeredgecolor="black", markeredgewidth=0.4,
                       markersize=9, label="Agent (infected)"),
        ]
        for etype, rgba in EVENT_OVERLAY_COLORS.items():
            legend_elements.append(
                mpatches.Patch(
                    facecolor=rgba, edgecolor=rgba[:3], linewidth=1,
                    label=etype.replace("_", " ").title(),
                )
            )
        # Full-width strip along the bottom, not a column stacked to the right
        # of the map.  At the enlarged point sizes a single right-hand column is
        # taller than the figure and steals width from the timeline panel.
        ax_legend.legend(
            handles=legend_elements,
            loc="center",
            ncol=4,
            fontsize=FS_LEGEND,
            framealpha=0.85,
            handlelength=1.4,
            columnspacing=1.1,
            borderpad=0.5,
            title="Cell hue = terrain,  brightness = food+water\n"
                  "Marker shape = agent status,  size = agent resources",
            title_fontsize=FS_LEGEND,
        )
        # Two lines: at FS_PANEL_TITLE this does not fit the panel on one.
        #
        # The cycle number is deliberately NOT repeated here.  frame["cycle"] is
        # 0-based while the caller's suptitle uses snapshot_points()' 1-based
        # label, so printing both put "Cycle 149" directly beneath
        # "Cycle 150 (final)" in the same figure.  The suptitle is the one place
        # that states it; nhs here is rounded to 2 dp for reader-facing display.
        ax_grid.set_title(
            f"Alive: {n_alive} of {n_init}\nNormalized health: {nhs:.2f}",
            fontsize=FS_PANEL_TITLE,
        )
        ax_grid.set_xlabel("Column", fontsize=FS_AXIS_LABEL)
        ax_grid.set_ylabel("Row", fontsize=FS_AXIS_LABEL)
        ax_grid.tick_params(labelsize=FS_TICK)

        # ── Health timeline panel ─────────────────────────────────────────────
        hist = [h for h in health_history if h["cycle"] <= cycle]
        if hist:
            h_cycles = [h["cycle"] for h in hist]
            h_nhs    = [h["nhs"] for h in hist]
            h_surv   = [h["survival_rate"] for h in hist]
            h_median = [h["median_health"] for h in hist]

            ax_time.fill_between(h_cycles, h_nhs, alpha=0.20, color="#1565C0")
            ax_time.plot(h_cycles, h_nhs,    label="Normalized Health Score",
                         color="#1565C0", linewidth=2)
            ax_time.plot(h_cycles, h_surv,   label="Survival Rate",
                         color="#2E7D32", linewidth=1.5, linestyle="--")
            # h_median is `s.median_health` (engine/metrics.py:
            # `median_health=_median(healths)`), a real median, not a mean.
            ax_time.plot(h_cycles, h_median, label="Median Health (alive)",
                         color="#F57F17", linewidth=1.5, linestyle=":")

            for h in hist:
                if h["has_events"]:
                    ax_time.axvline(h["cycle"], color="red", alpha=0.07, linewidth=0.7)

        max_cycle = health_history[-1]["cycle"] if health_history else cycle
        ax_time.set_xlim(0, max(max_cycle, 1))
        ax_time.set_ylim(-0.02, 1.05)
        ax_time.set_xlabel("Cycle", fontsize=FS_AXIS_LABEL)
        ax_time.set_ylabel("Score / Rate", fontsize=FS_AXIS_LABEL)
        ax_time.set_title(
            f"Health Progression — {government_name}" if government_name
            else "Health Progression",
            fontsize=FS_PANEL_TITLE,
        )
        # loc="best", not a fixed corner: the curves fall to very different
        # depths across governments and difficulties (ADS stays near 1.0,
        # anarchy at D=100 collapses), so any hard-coded position covers the
        # data in some cells.  "best" is chosen per figure to minimise overlap.
        ax_time.legend(fontsize=FS_LEGEND, framealpha=0.85, loc="best")
        ax_time.grid(alpha=0.3)
        ax_time.tick_params(labelsize=FS_TICK)

        # Keep suptitles short.  Without tight bbox the canvas does not grow
        # to fit overflowing text — it clips it.  At FS_SUPTITLE bold, roughly
        # 60 characters is the limit for DASHBOARD_FIGSIZE[0].
        fig.suptitle(
            f"{government_name}  |  NHS {nhs:.4f}" if government_name
            else f"Normalized Health Score {nhs:.4f}",
            fontsize=FS_SUPTITLE, fontweight="bold",
        )
        # No tight_layout(): the figure was created with layout="constrained".
        return fig

    # ------------------------------------------------------------------
    # Government comparison chart
    # ------------------------------------------------------------------

    @staticmethod
    def compare_governments(
        results: Dict[str, "MetricsCollector"],
        save_path: Optional[str] = None,
    ):
        """
        Plot bar charts comparing normalized health score and survival rate
        across multiple government types.

        Parameters
        ----------
        results    Dict mapping government name → MetricsCollector.
        save_path  Optional file path to save the figure (PNG/PDF).

        Returns matplotlib Figure.
        """
        SimulationVisualizer._require_matplotlib()
        import matplotlib.pyplot as plt
        import numpy as np

        names = list(results.keys())
        summaries = [results[n].summary() for n in names]

        nhs_vals = [s.get("normalized_health_score", 0.0) for s in summaries]
        surv_vals = [s.get("final_survival_rate", 0.0) for s in summaries]
        health_vals = [s.get("final_median_health", 0.0) for s in summaries]
        gini_vals = [s.get("final_health_gini", 0.0) for s in summaries]

        # Sort by normalized health score descending
        order = sorted(range(len(names)), key=lambda i: nhs_vals[i], reverse=True)
        names = [names[i] for i in order]
        nhs_vals = [nhs_vals[i] for i in order]
        surv_vals = [surv_vals[i] for i in order]
        health_vals = [health_vals[i] for i in order]
        gini_vals = [gini_vals[i] for i in order]

        x = np.arange(len(names))
        w = 0.2

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: primary metric
        bars = axes[0].bar(x, nhs_vals, color="#1565C0", alpha=0.85, label="Norm. health")
        axes[0].bar(x + w, surv_vals, color="#2E7D32", alpha=0.75, label="Survival rate")
        axes[0].bar(x + 2 * w, health_vals, color="#F57F17", alpha=0.75,
                    # health_vals is `final_median_health` (engine/metrics.py:137,
                    # `round(last.median_health, 4)`) -- a median, not a mean.
                    label="Median health (alive)")
        axes[0].set_xticks(x + w)
        axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_ylabel("Score / Rate")
        axes[0].set_title("Primary Comparison Metrics")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)

        # Annotate bars with values
        for bar, val in zip(bars, nhs_vals):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

        # Right: health inequality (Gini)
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
        axes[1].bar(x, gini_vals, color=colors, alpha=0.9)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        axes[1].set_ylim(0, max(gini_vals) * 1.3 if gini_vals else 1)
        axes[1].set_ylabel("Gini coefficient of health")
        axes[1].set_title("Health Inequality (Gini)")
        axes[1].grid(axis="y", alpha=0.3)
        for i, val in enumerate(gini_vals):
            axes[1].text(i, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(
            "Government Comparison",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Progression multi-panel
    # ------------------------------------------------------------------

    def render_progression_panels(
        self,
        n_panels: int = 6,
        save_path: Optional[str] = None,
    ):
        """
        Render *n_panels* equally-spaced snapshots from recorded frames
        as a grid of sub-plots showing simulation progression.

        Requires frames recorded with record_frame().
        Returns matplotlib Figure.
        """
        if not self._frames:
            raise RuntimeError("No frames recorded. Call record_frame() during the run.")

        self._require_matplotlib()
        import matplotlib.pyplot as plt

        step = max(1, len(self._frames) // n_panels)
        selected = self._frames[::step][:n_panels]

        ncols = min(3, n_panels)
        nrows = (len(selected) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 6, nrows * 5.5))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]

        for idx, (frame, ax) in enumerate(
            zip(selected, [ax for row in axes for ax in row])
        ):
            sub_fig = self._render_frame_dict(frame)
            # Copy content into the subplot (render to image then imshow)
            import io
            buf = io.BytesIO()
            sub_fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            plt.close(sub_fig)

            import numpy as np
            try:
                from PIL import Image
                img = np.array(Image.open(buf))
            except ImportError:
                img = plt.imread(buf)

            ax.imshow(img)
            ax.axis("off")

        # Hide unused axes
        all_axes = [ax for row in axes for ax in row]
        for ax in all_axes[len(selected):]:
            ax.set_visible(False)

        fig.suptitle("Simulation Progression", fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")

        return fig


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert a '#RRGGBB' hex string to a (R, G, B) tuple in [0, 1]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
