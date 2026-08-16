"""Figure 4 — urban-form counterfactual outcomes.

Plots the urban-form scenario results (panels 4a-4e) from staged data in
``data/figure_source/figure4/``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

from figure_contracts import (
    CATEGORY_COLORS,
    PANEL_DPI,
    PANEL_SIZES_MM,
    FIG4_PANELS,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    FIG4_SOURCE_DATA,
)
from map_contract import (
    MAP_CONTRACT,
    add_boundary_overlay,
    add_boundary_underlay,
    apply_map_frame,
)

INK = "#3A352E"
REFERENCE = "#6B6357"
GRID = "#D8CFBF"
CREAM = "#F4EEE2"
CATEGORY_LABELS = {
    "metropolitan_core": "Metropolitan core",
    "regional_city": "Regional city",
    "rural": "Rural",
}
CATEGORY_MARKERS = {
    "metropolitan_core": "o",
    "regional_city": "s",
    "rural": "^",
}
# Representative meshes from PV-increment-weighted spatial clusters of
# gen_Sx - gen_Sxm. Levels encode relative cluster totals qualitatively.
PV_CYLINDER_SPECS = {
    "S1": (
        (139.778125, 35.697917, 5),
        (139.584375, 35.535417, 1),
        (139.659375, 35.831250, 1),
        (140.115625, 35.631250, 1),
    ),
    "S2": (
        (139.665625, 35.652083, 4),
        (139.871875, 35.777083, 4),
        (139.453125, 35.406250, 3),
        (139.553125, 35.939583, 3),
        (139.409375, 35.697917, 3),
        (140.178125, 35.635417, 3),
    ),
    "S3": (
        (139.159375, 35.260417, 2),
        (139.340625, 35.660417, 2),
        (139.478125, 35.910417, 2),
        (139.490625, 35.343750, 2),
        (139.621875, 35.439583, 2),
        (139.621875, 35.902083, 2),
        (139.971875, 35.860417, 2),
        (140.115625, 35.610417, 2),
        (140.265625, 35.522917, 2),
        (140.496875, 35.635417, 2),
        (139.228125, 35.414583, 2),
        (139.278125, 35.797917, 2),
        (139.546875, 35.652083, 2),
    ),
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.3,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "legend.fontsize": 5.5,
    "axes.edgecolor": REFERENCE,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.linewidth": .7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def figure4_layout_contract() -> dict:
    return {
        "fig4a": {
            "limits": (MAP_CONTRACT.xlim, MAP_CONTRACT.ylim),
            "ticks": (MAP_CONTRACT.xticks, MAP_CONTRACT.yticks),
            "boundary_linewidth": MAP_CONTRACT.boundary_linewidth,
            "map_count": 3,
            "pv_cylinders_per_map": {
                family: len(specs) for family, specs in PV_CYLINDER_SPECS.items()
            },
            "pv_cylinder_coordinates": "data",
            "colorbar_orientation": "vertical",
            "north_arrow_count": 3,
            "scenario_title_size": 7.7,
            "colorbar_tick_size": 5.0,
            "colorbar_gap": .035,
            "s1_annotation": "BIPV",
            "colorbar_label_position": "below_right",
            "colorbar_label_alignment": "center",
            "colorbar_label_size": 8.3,
            "colorbar_label_lines": 3,
            "colorbar_label_linespacing": 1.12,
            "colorbar_x": .918,
            "colorbar_width": .018,
            "colorbar_bottom": .37,
            "scenario_title_tone": "darkened",
            "map_left": .040,
            "map_height": .74,
        },
        "fig4b": {"scenario_rows": 6, "estimand_columns": 4},
        "fig4c": {
            "metric_columns": 2,
            "category_count": 3,
            "color_semantics": "urban_form_scenario",
            "category_encoding": "marker",
        },
        "fig4d": {
            "xlim": (-2, 2),
            "ylim": (-1, 2),
            "sample_points": 2700,
            "color_semantics": "urban_form_scenario",
        },
        "fig4e": {"aggregation_columns": 2, "padding_fraction": .08},
    }


def _new(panel: str):
    width, height = PANEL_SIZES_MM[panel]
    return plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")


def _save(fig, panel: str):
    FIG4_PANELS.mkdir(parents=True, exist_ok=True)
    width, height = PANEL_SIZES_MM[panel]
    fig.set_size_inches(width / 25.4, height / 25.4, forward=True)
    fig.savefig(FIG4_PANELS / f"{panel}.svg", bbox_inches=None, pad_inches=0)
    fig.savefig(FIG4_PANELS / f"{panel}.png", dpi=PANEL_DPI[panel], bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _clean(ax, axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=.45, alpha=.65, linestyle=(0, (2.2, 2.2)))
    ax.set_axisbelow(True)


def _format_signed_pp(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.2f}"


def _add_map_pv_cylinders(ax, family):
    """Add qualitative PV cylinders at scenario-aligned map anchors."""
    patches = []
    width = .036
    cap_height = .014
    for lon, lat, level in PV_CYLINDER_SPECS[family]:
        height = .040 * level
        body = Rectangle(
            (lon - width / 2, lat), width, height,
            transform=ax.transData, facecolor="#D5A83F", edgecolor="none",
            linewidth=0, zorder=101,
        )
        base = Ellipse(
            (lon, lat), width, cap_height,
            transform=ax.transData, facecolor="#C7942C", edgecolor="none",
            linewidth=0, zorder=101,
        )
        cap = Ellipse(
            (lon, lat + height), width, cap_height,
            transform=ax.transData, facecolor="#E7BD58", edgecolor="none",
            linewidth=0, zorder=102,
        )
        for patch in (body, base, cap):
            ax.add_patch(patch)
            patches.append(patch)
    return patches


def _figure4a_floor_norm():
    return mcolors.SymLogNorm(
        linthresh=.30, linscale=.5, vmin=-5, vmax=5, base=10,
    )


def _figure4a_display_values(values, family):
    base = np.asarray(values, dtype=float).copy()
    focus = np.full_like(base, np.nan)
    base_alpha = 1.0
    focus_alpha = 0.0
    if family == "S2":
        focus = base.copy()
        focus[np.abs(focus) < .30] = np.nan
        base_alpha = 1.0
        focus_alpha = .48
    return base, focus, base_alpha, focus_alpha


def _add_figure4a_north_arrow(ax):
    c = MAP_CONTRACT
    top = (c.north_x, c.north_y + c.north_height)
    bottom = (c.north_x, c.north_y)
    left = (c.north_x - c.north_half_width, c.north_y)
    right = (c.north_x + c.north_half_width, c.north_y)
    outer = Polygon(
        [top, left, right], transform=ax.transAxes,
        facecolor="white", edgecolor=c.ink, linewidth=c.north_linewidth,
        zorder=105,
    )
    filled_half = Polygon(
        [top, left, bottom], transform=ax.transAxes,
        facecolor=c.ink, edgecolor="none", linewidth=0, zorder=106,
    )
    ax.add_patch(outer)
    ax.add_patch(filled_half)
    ax.text(
        c.north_x, c.north_y + c.north_height + c.north_label_gap, "N",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=c.north_font_size, color=c.ink, zorder=107,
    )
    return [outer, filled_half]


def plot_4a():
    data = pd.read_parquet(FIG4_SOURCE_DATA / "a_hexagons.parquet")
    summary = pd.read_csv(FIG4_SOURCE_DATA / "a_scenario_summary.csv").set_index("family")
    fig = _new("fig4a")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "floor_change", [SCENARIO_COLORS["S3"], CREAM, SCENARIO_COLORS["S1"]]
    )
    norm = _figure4a_floor_norm()
    title_colors = {
        "S1": "#A9635E",
        "S2": "#AD832A",
        "S3": "#617FA1",
    }
    image = None
    map_left = .040
    map_width = .276
    map_step = .282
    for index, family in enumerate(("S1", "S2", "S3")):
        ax = fig.add_axes([map_left + index * map_step, .16, map_width, .74])
        part = data.loc[data.family.eq(family)]
        add_boundary_underlay(ax)
        vertices = np.stack([
            part[[f"v{i}_lon", f"v{i}_lat"]].to_numpy(float) for i in range(6)
        ], axis=1)
        base_values, focus_values, base_alpha, focus_alpha = _figure4a_display_values(
            part.value.to_numpy(float), family,
        )
        image = PolyCollection(
            vertices, array=base_values, cmap=cmap, norm=norm, alpha=base_alpha,
            edgecolors="none", linewidths=0, rasterized=True, zorder=3,
        )
        ax.add_collection(image)
        if family == "S2":
            focus = PolyCollection(
                vertices, array=focus_values, cmap=cmap, norm=norm, alpha=focus_alpha,
                edgecolors="none", linewidths=0, rasterized=True, zorder=4,
            )
            ax.add_collection(focus)
        add_boundary_overlay(ax)
        apply_map_frame(ax, show_xlabels=True, show_ylabels=index == 0)
        ax.tick_params(labelsize=5.0, pad=.8)
        if index:
            ax.tick_params(axis="y", length=0)
        ax.text(.98, .985, SCENARIO_LABELS[family], transform=ax.transAxes,
                ha="right", va="top", fontsize=7.7, color=title_colors[family],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.84, pad=.7))
        row = summary.loc[family]
        ax.text(.02, .025, f"{int(row.changed_meshes):,} meshes · "
                f"{row.redistributed_demand_twh:.1f} TWh moved",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=4.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=.8),
                visible=False)
        _add_map_pv_cylinders(ax, family)
        _add_figure4a_north_arrow(ax)
        if family == "S1":
            ax.text(139.88, 35.86, "BIPV", fontsize=5.6, color=INK,
                    ha="left", va="bottom", zorder=110,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=.8))
    cax = fig.add_axes([.918, .37, .018, .40])
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical",
                            ticks=[-3, -1, 0, 1, 3], extend="both")
    colorbar.ax.tick_params(labelsize=5.0, pad=1.0)
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    fig.text(.927, .335, "Change in\naverage\nfloors", fontsize=8.3,
             ha="center", va="top", linespacing=1.12)
    _save(fig, "fig4a")


def plot_4b():
    data = pd.read_csv(FIG4_SOURCE_DATA / "b_four_estimands.csv")
    metrics = (
        ("cssr_mesh_mean", "Collective self-sufficiency", "Mesh mean"),
        ("cssr_system", "Collective self-sufficiency", "System ratio"),
        ("cpi_mesh_mean", "Sharing contribution", "Mesh mean"),
        ("cpi_system", "Sharing contribution", "System ratio"),
    )
    order = ("S1m", "S1", "S2m", "S2", "S3m", "S3")
    table = data.pivot(index="urban_form_scenario", columns="metric", values="delta_pp").reindex(order)
    fig = _new("fig4b")
    ax = fig.add_axes([.12, .10, .86, .82])
    vmax = float(np.abs(table.to_numpy()).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "signed_delta", [SCENARIO_COLORS["S1"], CREAM, SCENARIO_COLORS["S3"]]
    )
    x_positions = (0.0, .78, 1.72, 2.50)
    for yi, scenario in enumerate(order):
        for xi, (metric, _, _) in enumerate(metrics):
            value = float(table.loc[scenario, metric])
            x = x_positions[xi]
            size = 130 + 210 * np.sqrt(abs(value) / vmax)
            ax.scatter(x, yi, s=size, color=cmap(norm(value)), edgecolor=INK,
                       linewidth=.55 if scenario.endswith("m") else .2, zorder=3)
            ax.text(x, yi, _format_signed_pp(value), ha="center", va="center", fontsize=5.0,
                    color="white" if abs(value) > .45 * vmax else INK, zorder=4)
    ax.set_xlim(-.43, 2.93)
    ax.set_ylim(6.55, -1.35)
    ax.set_xticks([])
    ax.set_yticks(range(6), order)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.text(np.mean(x_positions[:2]), -1.05, "Collective self-sufficiency", ha="center", va="bottom",
            fontsize=6.1, clip_on=False)
    ax.text(np.mean(x_positions[2:]), -1.05, "Sharing contribution", ha="center", va="bottom",
            fontsize=6.1, clip_on=False)
    for xi, (_, _, line2) in enumerate(metrics):
        ax.text(x_positions[xi], -.72, line2, ha="center", va="bottom", fontsize=4.8,
                color=REFERENCE, clip_on=False)
    for y in (1.5, 3.5):
        ax.axhline(y, color=GRID, lw=.55)
    ax.spines[:].set_visible(False)
    ax.text(.50, .03, "Change from S0 (pp) · m = morphology only",
            transform=ax.transAxes, fontsize=5.35, ha="center", va="bottom",
            color=REFERENCE)
    _save(fig, "fig4b")


def plot_4c():
    data = pd.read_csv(FIG4_SOURCE_DATA / "c_category_response.csv")
    fig = _new("fig4c")
    metric_specs = (
        ("collective_ssr", "Collective self-sufficiency"),
        ("sharing_contribution", "Sharing contribution"),
    )
    for column, (metric, title) in enumerate(metric_specs):
        ax = fig.add_axes([.13 + column * .49, .15, .35, .78])
        part = data.loc[data.metric.eq(metric)]
        for category_index, category in enumerate(CATEGORY_LABELS):
            for family_index, family in enumerate(("S1", "S2", "S3")):
                values = part.loc[
                    part.family.eq(family) & part.category_3.eq(category)
                ].set_index("configuration").delta_pp
                x = family_index + (category_index - 1) * .15
                color = SCENARIO_COLORS[family]
                marker = CATEGORY_MARKERS[category]
                ax.plot([x, x], [values.morphology_only, values.morphology_plus_pv],
                        color=color, lw=1.0, alpha=.72)
                ax.scatter(x, values.morphology_only, s=17, marker=marker,
                           facecolor="white", edgecolor=color, lw=.8, zorder=3)
                ax.scatter(x, values.morphology_plus_pv, s=18, marker=marker,
                           facecolor=color, edgecolor="white",
                           lw=.25, zorder=4)
        ax.axhline(0, color=INK, lw=.5)
        ax.set_xticks(range(3), ("S1", "S2", "S3"))
        ax.set_title(title, fontsize=5.9, loc="left", pad=2)
        _clean(ax, "y")
        if column:
            ax.set_ylabel("")
    fig.text(.045, .56, "Mesh-mean change from S0 (pp)", rotation=90,
             va="center", ha="center", fontsize=5.9)
    handles = [
        Line2D([], [], marker=CATEGORY_MARKERS[key], color=REFERENCE, linestyle="none",
               markerfacecolor="white", markeredgecolor=REFERENCE,
               label=label, markersize=4) for key, label in CATEGORY_LABELS.items()
    ]
    axes = fig.axes
    axes[0].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.10, 1.0), frameon=False,
                   fontsize=4.4, borderaxespad=.15, handletextpad=.2,
                   labelspacing=.38)
    axes[1].plot([.76, .88], [.94, .94], transform=axes[1].transAxes,
                 color=REFERENCE, lw=.8, clip_on=False)
    axes[1].scatter([.76], [.94], transform=axes[1].transAxes, s=16,
                    facecolor="white", edgecolor=REFERENCE, lw=.7, zorder=6)
    axes[1].scatter([.88], [.94], transform=axes[1].transAxes, s=17,
                    facecolor=REFERENCE, edgecolor="white", lw=.2, zorder=6)
    axes[1].text(.72, .94, "m", transform=axes[1].transAxes,
                 ha="right", va="center", fontsize=4.4, color=REFERENCE)
    axes[1].text(.92, .94, "+PV", transform=axes[1].transAxes,
                 ha="left", va="center", fontsize=4.4, color=REFERENCE)
    _save(fig, "fig4c")


def plot_4d():
    data = pd.read_parquet(FIG4_SOURCE_DATA / "d_mesh_sample.parquet")
    medians = pd.read_csv(FIG4_SOURCE_DATA / "d_scenario_medians.csv").set_index("family")
    fig = _new("fig4d")
    ax = fig.add_axes([.18, .23, .77, .70])
    for family in ("S1", "S2", "S3"):
        part = data.loc[data.family.eq(family)]
        ax.scatter(part.morph_delta_cpi_pp, part.full_delta_cpi_pp, s=5.2,
                   color=SCENARIO_COLORS[family], alpha=.22, linewidths=0,
                   rasterized=True)
        median = medians.loc[family]
        x = float(median.median_morph_delta_cpi_pp)
        y = float(median.median_full_delta_cpi_pp)
        ax.scatter(x, y, s=28, facecolor=SCENARIO_COLORS[family], edgecolor="white",
                   linewidth=.65, zorder=5)
        offset = {"S1": (-62, -14), "S2": (32, 8), "S3": (24, -17)}[family]
        label_colors = {"S1": "#A9635E", "S2": "#AD832A", "S3": "#617FA1"}
        ax.annotate(f"{family} ({_format_signed_pp(x)}, {_format_signed_pp(y)})", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=5.8,
                    color=label_colors[family],
                    arrowprops=dict(arrowstyle="-", color=label_colors[family], lw=.5))
    ax.plot([-1, 2], [-1, 2], color=REFERENCE, lw=.7, linestyle=(0, (3, 2)))
    ax.axhline(0, color=INK, lw=.5)
    ax.axvline(0, color=INK, lw=.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("Morphology-only change in sharing contribution (pp)",
                  fontsize=5.9, labelpad=2)
    ax.set_ylabel("Morphology + PV change\nin sharing contribution (pp)",
                  fontsize=5.9, labelpad=2)
    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_yticks(np.arange(-1, 3, 1))
    ax.text(-1.88, 1.82, "Paired PV amplification", fontsize=5.8, color=REFERENCE)
    ax.text(-1.88, -.88, "Paired PV offset", fontsize=5.8, color=REFERENCE)
    ax.text(.38, .08, "Limited change", fontsize=5.8, color=REFERENCE)
    ax.scatter(.72, .075, s=28, transform=ax.transAxes, facecolor="white",
               edgecolor=INK, linewidth=.7, zorder=6)
    ax.text(.77, .075, "Median marker", transform=ax.transAxes,
            ha="left", va="center", fontsize=5.8, color=REFERENCE)
    _clean(ax)
    _save(fig, "fig4d")


def _padded_limits(values: np.ndarray, fraction: float = .08) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    if span == 0:
        span = max(abs(low), 1.0)
    return low - fraction * span, high + fraction * span


def plot_4e():
    data = pd.read_csv(FIG4_SOURCE_DATA / "e_paired_endpoints.csv")
    fig = _new("fig4e")
    for column, (aggregation, title) in enumerate((
        ("system", "Demand-weighted system"),
        ("mesh_mean", "Unweighted mesh mean"),
    )):
        ax = fig.add_axes([.10 + column * .49, .21, .39, .70])
        part = data.loc[data.aggregation.eq(aggregation)]
        x_values = part.delta_cssr_pp.to_numpy(float)
        y_values = part.delta_cpi_pp.to_numpy(float)
        ax.set_xlim(*_padded_limits(x_values))
        ax.set_ylim(*_padded_limits(y_values))
        for family in ("S1", "S2", "S3"):
            morph = part.loc[
                part.family.eq(family) & part.configuration.eq("morphology_only")
            ].iloc[0]
            full = part.loc[
                part.family.eq(family) & part.configuration.eq("morphology_plus_pv")
            ].iloc[0]
            color = SCENARIO_COLORS[family]
            arrow = FancyArrowPatch(
                (morph.delta_cssr_pp, morph.delta_cpi_pp),
                (full.delta_cssr_pp, full.delta_cpi_pp),
                arrowstyle="-|>", mutation_scale=7, color=color, lw=1.25,
                shrinkA=4, shrinkB=4, zorder=2,
            )
            ax.add_patch(arrow)
            ax.scatter(morph.delta_cssr_pp, morph.delta_cpi_pp, s=22,
                       facecolor="white", edgecolor=color, lw=.85, zorder=3)
            ax.scatter(full.delta_cssr_pp, full.delta_cpi_pp, s=23,
                       facecolor=color, edgecolor="white", lw=.3, zorder=4)
        ax.set_title(title, loc="left", fontsize=6.2, pad=2)
        if column == 0:
            ax.text(.02, .98, "Paired PV increment", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.2, color=REFERENCE)
        _clean(ax)
        if column:
            ax.set_ylabel("")
    fig.text(.50, .065, "Collective self-sufficiency change (pp)", ha="center", fontsize=6.7)
    fig.text(.018, .55, "Sharing contribution change (pp)", rotation=90,
             va="center", ha="center", fontsize=6.7)
    _save(fig, "fig4e")


def render_all():
    plot_4a()
    plot_4b()
    plot_4c()
    plot_4d()
    plot_4e()


if __name__ == "__main__":
    render_all()
