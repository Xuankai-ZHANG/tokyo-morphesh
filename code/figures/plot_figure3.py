from __future__ import annotations

import struct
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle

from figure_contracts import (
    AUX_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    FIG3_CATEGORY_COLORS,
    FIG3_PANELS,
    FIG3_SOURCE_DATA,
    MAP_FRAME,
    MAP_LIMITS,
    MAP_TICKS,
    NORTH_ARROW,
    PANEL_DPI,
    PANEL_SIZES_MM,
)
from map_contract import (
    add_boundary_overlay,
    add_boundary_underlay,
    add_north_arrow,
    apply_map_frame,
)

INK = AUX_COLORS["brown_dark"]
GRID = AUX_COLORS["warm_grid"]
_PREFECTURE_PARTS = None

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.2,
    "axes.edgecolor": AUX_COLORS["brown"],
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.linewidth": .7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

def _new(panel: str):
    width, height = PANEL_SIZES_MM[panel]
    return plt.figure(figsize=(width / 25.4, height / 25.4))

def _save(fig, panel: str):
    FIG3_PANELS.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(
        PANEL_SIZES_MM[panel][0] / 25.4,
        PANEL_SIZES_MM[panel][1] / 25.4,
        forward=True,
    )
    fig.savefig(FIG3_PANELS / f"{panel}.svg", bbox_inches=None, pad_inches=0)
    fig.savefig(FIG3_PANELS / f"{panel}.png", dpi=PANEL_DPI[panel], bbox_inches=None, pad_inches=0)
    plt.close(fig)

def _clean(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(
            axis=grid_axis, color=GRID, lw=.45, alpha=.55,
            linestyle=figure3_layout_contract()["grid_linestyle"],
        )
        ax.set_axisbelow(True)

def figure3_layout_contract() -> dict:
    return {
        "grid_linestyle": (0, (2.2, 2.2)),
        "fig3a": {
            "left_axes": (.105, .28, .285, .62),
            "legend_axes": (.405, .12, .185, .78),
            "right_axes": (.655, .28, .255, .62),
            "xlabel_pad": 1,
            "legend_rows": 6,
            "legend_fontsize": 7.0,
            "legend_marker_fraction": .5,
            "left_tick_pad": 2.0,
            "right_tick_pad": 2.0,
            "symlog_ticks": (-10_000, -1_000, -100, 0, 100, 1_000, 10_000),
            "symlog_ticklabels": ("−10⁴", "−10³", "−10²", "0", "10²", "10³", "10⁴"),
        },
        "fig3b": {
            "limits": MAP_LIMITS,
            "ticks": MAP_TICKS,
            "frame": MAP_FRAME,
            "north_arrow": NORTH_ARROW,
            "point_size": .36,
            "boundary_linewidth": .35,
            "near_balance_color": AUX_COLORS["warm_grid"],
            "signal_alpha": .90,
            "arrow_extension_km": 4.0,
        },
        "fig3d": {"legend_fontsize": 7.0},
        "fig3e": {"funnel_width": .46, "funnel_row_gap": .62, "funnel_fontsize": 7.0},
        "fig3f": {
            "hex_gridsize": (66, 27),
            "colorbar_axes": (.855, .37, .032, .42),
            "colorbar_title_fontsize": 7.0,
        },
        "fig3h": {
            "line_after_marker": False,
            "pooling_text_background": AUX_COLORS["cream"],
            "pooling_value_side": "left",
            "aggregation_annotation_mode": "title_above_values_inside",
            "mechanism_label_x": 2.0,
            "pooling_value_offset": -7.0,
            "aggregation_inside_labels": ("81.8% absorbed", "0.305 TWh"),
            "aggregation_title_x": 2.0,
            "xlim": (0, 105),
        },
        "fig3g": {
            "connect_series": True,
            "marker_size": 5.2,
            "ylim": (-7, 138),
            "legend_inside": True,
            "legend_fontsize": 7.0,
            "legend_ncol": 2,
        },
    }

def plot_3b():
    data = pd.read_csv(FIG3_SOURCE_DATA / "b_radius_sensitivity.csv")
    fig = _new("fig3c")
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"left": .20, "right": .86, "bottom": .12, "top": .86, "hspace": .20, "height_ratios": [1.18, 1]})
    ax = axes[0]
    for column, label, color, marker, ls in (
        ("lr_tp030_pct", "Reachability, 30%", FIG3_CATEGORY_COLORS["metropolitan_core"], "o", "-"),
        ("da_tp030_pct", "Availability, 30%", FIG3_CATEGORY_COLORS["regional_city"], "s", "-"),
        ("lr_tp100_pct", "Reachability, 100%", FIG3_CATEGORY_COLORS["metropolitan_core"], "o", "--"),
        ("da_tp100_pct", "Availability, 100%", FIG3_CATEGORY_COLORS["regional_city"], "s", "--"),
    ):
        subset = data.dropna(subset=[column])
        ax.plot(subset.radius_km, subset[column], marker=marker, color=color, ls=ls, lw=1.05, ms=2.6, label=label)
    ax.set_ylabel("Eligible meshes (%)", labelpad=2)
    ax.set_ylim(0, 105)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(.53, .985), handlelength=1.5,
        columnspacing=.55, labelspacing=.3, fontsize=7.0,
    )
    _clean(ax, "y")

    ax = axes[1]
    cpi = data.dropna(subset=["cpi_tp030_pp"])
    sur = data.dropna(subset=["sur_tp030_pct"])
    ax.plot(cpi.radius_km, cpi.cpi_tp030_pp, "^-", color=AUX_COLORS["green"], lw=1.2, ms=3, label="Mean sharing gain")
    ax.set_ylabel("Mean mesh-level sharing gain (pp)", color=AUX_COLORS["green"], labelpad=2)
    ax.tick_params(axis="y", colors=AUX_COLORS["green"])
    twin = ax.twinx()
    twin.plot(sur.radius_km, sur.sur_tp030_pct, "D--", color=AUX_COLORS["brown"], lw=1.05, ms=2.7, label="Surplus utilisation")
    twin.set_ylabel("Surplus utilisation (%)", color=AUX_COLORS["brown"], labelpad=2)
    twin.tick_params(axis="y", colors=AUX_COLORS["brown"])
    twin.spines["top"].set_visible(False)
    twin.set_ylim(65, 80)
    _clean(ax, "y")
    for ax in axes:
        ax.axvline(5, color=AUX_COLORS["brown"], ls=":", lw=.8)
    axes[0].text(5.25, 5, "Primary", fontsize=7.0, color=AUX_COLORS["brown"])
    axes[-1].set_xscale("log")
    axes[-1].set_xticks([1, 2, 3, 5, 10, 15], ["1", "2", "3", "5", "10", "15"])
    axes[-1].set_xlabel("Candidate radius (km)")
    _save(fig, "fig3c")

def _decorate_map_like_figure2(ax):
    add_boundary_underlay(ax)
    apply_map_frame(ax)

def plot_3c():
    points = pd.read_parquet(FIG3_SOURCE_DATA / "c_direction_points.parquet")
    arrows = pd.read_parquet(FIG3_SOURCE_DATA / "c_direction_arrows.parquet")
    fig = _new("fig3b")
    frame = figure3_layout_contract()["fig3b"]["frame"]
    ax = fig.add_axes([frame["left"], frame["bottom"], .84, .86])
    _decorate_map_like_figure2(ax)
    layout = figure3_layout_contract()["fig3b"]
    palette = {"near_balance": layout["near_balance_color"], "net_output": AUX_COLORS["green"], "net_receipt": "#CF7A5C"}
    names = {"near_balance": "Near balance", "net_output": "Net output", "net_receipt": "Net receipt"}
    shares = 100 * points.point_class.value_counts(normalize=True)
    handles = []
    for key in ("near_balance", "net_output", "net_receipt"):
        subset = points[points.point_class == key]
        ax.scatter(
            subset.lon, subset.lat,
            s=figure3_layout_contract()["fig3b"]["point_size"], c=palette[key],
            alpha=layout["signal_alpha"] if key != "near_balance" else .70, lw=0,
            rasterized=True, zorder=3,
        )
        handles.append(Line2D([0], [0], marker="o", lw=0, color=palette[key], ms=3.5, label=f"{names[key]}  {shares.get(key, 0):.0f}%"))
    for row in arrows.sort_values("internal_net_flow_share_pct").itertuples(index=False):
        mean_lat = np.radians((row.donor_lat + row.receiver_lat) / 2)
        x_km = (row.receiver_lon - row.donor_lon) * 111.0 * np.cos(mean_lat)
        y_km = (row.receiver_lat - row.donor_lat) * 111.0
        distance_km = float(np.hypot(x_km, y_km))
        if distance_km:
            end = (
                row.receiver_lon + (x_km / distance_km * layout["arrow_extension_km"]) / (111.0 * np.cos(mean_lat)),
                row.receiver_lat + (y_km / distance_km * layout["arrow_extension_km"]) / 111.0,
            )
        else:
            end = (row.receiver_lon, row.receiver_lat)
        curve = .15 if int(row.donor_cell) < int(row.receiver_cell) else -.15
        arrow = FancyArrowPatch(
            (row.donor_lon, row.donor_lat), end,
            arrowstyle="-|>", mutation_scale=4.8, connectionstyle=f"arc3,rad={curve}",
            linewidth=float(row.line_width) * .62, color=INK, alpha=.82,
            path_effects=[patheffects.Stroke(linewidth=float(row.line_width) * .62 + .55, foreground="white", alpha=.9), patheffects.Normal()],
            zorder=10,
        )
        ax.add_patch(arrow)
    add_boundary_overlay(ax)
    add_north_arrow(ax)
    handles.append(Line2D([0], [1], color=INK, lw=.6, marker=">", markevery=[1], ms=3, label="Annual net flow (Top 25%)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.25, handletextpad=.35, labelspacing=.28, borderaxespad=.05, fontsize=9.0)
    ax.tick_params(
        labelsize=frame["tick_size"], pad=frame["tick_pad"],
    )
    for spine in ax.spines.values():
        spine.set_color(AUX_COLORS["brown"])
    _save(fig, "fig3b")

def plot_3d():
    data = pd.read_csv(FIG3_SOURCE_DATA / "d_category_distribution.csv")
    fig = _new("fig3d")
    ax = fig.add_axes([.31, .10, .64, .86])
    offsets = {"metropolitan_core": -.20, "regional_city": 0, "rural": .20}
    deployments = sorted(data.deployment.unique())
    ybase = np.arange(len(deployments))
    for category in CATEGORY_ORDER:
        d = data[data.category_3 == category].set_index("deployment").loc[deployments]
        y = ybase + offsets[category]
        ax.hlines(y, d.p10_pp, d.p90_pp, color=FIG3_CATEGORY_COLORS[category], lw=.8, alpha=.7)
        ax.hlines(y, d.p25_pp, d.p75_pp, color=FIG3_CATEGORY_COLORS[category], lw=3.2)
        ax.plot(d.mean_pp, y, marker={"metropolitan_core": "o", "regional_city": "s", "rural": "^"}[category], ls="none", color=FIG3_CATEGORY_COLORS[category], mec=INK, mew=.35, ms=3.5, label=CATEGORY_LABELS[category])
    ax.set_yticks(ybase, [f"{x}%" for x in deployments])
    ax.invert_yaxis()
    ax.set_xlabel("Mesh-level sharing gain (pp)")
    ax.set_ylabel("Rooftop PV technical potential (%)")
    _clean(ax, "x")
    ax.legend(frameon=False, loc="upper right", handlelength=.8, labelspacing=.2,
              fontsize=figure3_layout_contract()["fig3d"]["legend_fontsize"])
    _save(fig, "fig3d")

def plot_3f():
    data = pd.read_parquet(FIG3_SOURCE_DATA / "f_convergence.parquet")
    sigma = pd.read_csv(FIG3_SOURCE_DATA / "f_sigma_summary.csv").set_index("deployment")
    fig = _new("fig3f")
    axes = fig.subplots(3, 1, sharex=True, sharey=True,
        gridspec_kw={"left": .18, "right": .93, "bottom": .11, "top": .97, "hspace": .15})
    ramps = (
        mcolors.LinearSegmentedColormap.from_list("density_brown_final", [AUX_COLORS["cream"], AUX_COLORS["brown"]]),
        mcolors.LinearSegmentedColormap.from_list("density_yellow_final", [AUX_COLORS["cream"], AUX_COLORS["yellow"]]),
        mcolors.LinearSegmentedColormap.from_list("density_green_final", [AUX_COLORS["cream"], AUX_COLORS["green"]]),
    )
    hexbins = []
    for ax, deployment, cmap in zip(axes, (10, 30, 100), ramps):
        d = data[data.deployment == deployment]
        hb = ax.hexbin(100 * d.ssr, 100 * d["cssr_5.0"],
            gridsize=figure3_layout_contract()["fig3f"]["hex_gridsize"], cmap=cmap,
            mincnt=1, norm=mcolors.LogNorm(vmin=1), linewidths=0, rasterized=True)
        hexbins.append(hb)
        ax.plot([0, 100], [0, 100], ls="--", lw=.7, color=AUX_COLORS["brown"])
        ax.text(.03, .88, f"Rooftop PV technical potential={deployment}%", transform=ax.transAxes, fontsize=8.1)
        ax.text(.98, .09, f"σ ratio = {sigma.loc[deployment, 'sigma_convergence']:.3f}",
                transform=ax.transAxes, ha="right", fontsize=8.1)
        _clean(ax)
    maximum = max(float(hb.get_array().max()) for hb in hexbins)
    shared_norm = mcolors.LogNorm(vmin=1, vmax=maximum)
    for hb in hexbins:
        hb.set_norm(shared_norm)
    cax = axes[0].inset_axes(figure3_layout_contract()["fig3f"]["colorbar_axes"])
    colorbar = fig.colorbar(hexbins[0], cax=cax)
    colorbar.ax.tick_params(labelsize=5.6, pad=1)
    colorbar.ax.set_ylabel("Mesh count", fontsize=figure3_layout_contract()["fig3f"]["colorbar_title_fontsize"], labelpad=2)
    colorbar.outline.set_linewidth(.9)
    axes[-1].set_xlabel("Pre-sharing self-sufficiency (%)", fontsize=9.0)
    fig.supylabel("Post-sharing self-sufficiency (%)", x=.015, fontsize=9.0)
    axes[-1].set_xlim(0, 70); axes[-1].set_ylim(0, 75)
    _save(fig, "fig3f")

def _draw_quantile_envelopes(ax, data):
    y = np.arange(len(data))
    receiver = FIG3_CATEGORY_COLORS["metropolitan_core"]
    donor = FIG3_CATEGORY_COLORS["rural"]

    def signed_interval(y0, low, high, linewidth):
        if low < 0:
            ax.plot([low, min(high, 0)], [y0, y0], color=receiver, lw=linewidth,
                    solid_capstyle="round", zorder=2)
        if high > 0:
            ax.plot([max(low, 0), high], [y0, y0], color=donor, lw=linewidth,
                    solid_capstyle="round", zorder=2)

    for y0, row in zip(y, data.itertuples(index=False)):
        signed_interval(y0, row.q001_mwh, row.q999_mwh, .8)
        signed_interval(y0, row.q01_mwh, row.q99_mwh, 3.0)
        ax.plot(row.median_mwh, y0, marker="o", ms=2.7, color=AUX_COLORS["brown"],
                mec=INK, mew=.3, zorder=3)
    ax.axvline(0, color=INK, lw=.65, ls=figure3_layout_contract()["grid_linestyle"], zorder=1)
    ax.set_xscale("symlog", linthresh=100, linscale=.75)
    ax.set_yticks(y, [f"{value}%" for value in data.deployment])
    ax.invert_yaxis()

def plot_3a():
    quantiles = pd.read_csv(FIG3_SOURCE_DATA / "a_distribution_quantiles.csv")
    trajectory = pd.read_csv(FIG3_SOURCE_DATA / "a_trajectory.csv").sort_values("deployment")
    layout = figure3_layout_contract()["fig3a"]
    fig = _new("fig3a")

    left = fig.add_axes(layout["left_axes"])
    _draw_quantile_envelopes(left, quantiles)
    left.set_xticks(layout["symlog_ticks"], layout["symlog_ticklabels"])
    left.set_xlabel("Annual net position (MWh)", labelpad=layout["xlabel_pad"])
    left.set_ylabel("PV technical\npotential (%)", labelpad=2)
    left.tick_params(axis="x", labelsize=6.2, pad=layout["left_tick_pad"])
    _clean(left, "x")

    right = fig.add_axes(layout["right_axes"])
    right.plot(trajectory.deployment, trajectory.donor_pct, "o-",
               color=FIG3_CATEGORY_COLORS["metropolitan_core"], lw=1.2, ms=3)
    right.plot(trajectory.deployment, trajectory.sur_pct, "s--",
               color=AUX_COLORS["green"], lw=1.1, ms=2.8)
    right.set_xlabel("PV technical potential (%)", labelpad=layout["xlabel_pad"])
    right.set_ylabel("Share (%)", labelpad=1)
    right.set_ylim(-3, 104)
    right.tick_params(axis="x", labelsize=6.2, pad=layout["right_tick_pad"])
    _clean(right, "y")
    twin = right.twinx()
    twin.plot(trajectory.deployment, trajectory.transfer_twh, "^:",
              color=AUX_COLORS["yellow"], lw=1.2, ms=3)
    twin.set_ylabel("Transfer (TWh)", labelpad=2)
    twin.spines["top"].set_visible(False)

    legend = fig.add_axes(layout["legend_axes"])
    legend.axis("off")
    rows = (
        (AUX_COLORS["brown"], "-", None, .8, "0.1–99.9%"),
        (AUX_COLORS["brown"], "-", None, 3.0, "1–99%"),
        (AUX_COLORS["brown"], "none", "o", 0, "Median"),
        (FIG3_CATEGORY_COLORS["metropolitan_core"], "-", "o", 1.2, "Annual donor"),
        (AUX_COLORS["green"], "--", "s", 1.1, "Surplus utilisation"),
        (AUX_COLORS["yellow"], ":", "^", 1.2, "Transfer"),
    )
    y_rows = np.linspace(.91, .09, len(rows))
    for y0, (color, linestyle, marker, linewidth, label) in zip(y_rows, rows):
        if linestyle != "none":
            sample_x = [.04, .19, .34] if marker else [.04, .34]
            legend.plot(sample_x, [y0] * len(sample_x), color=color, ls=linestyle, lw=linewidth,
                        marker=marker, markevery=[1] if marker else None, ms=2.8, clip_on=False)
        else:
            legend.plot([.19], [y0], marker=marker, color=color, ms=3.2, lw=0)
        legend.text(.69, y0, label, ha="center", va="center", fontsize=layout["legend_fontsize"])
    legend.set_xlim(0, 1); legend.set_ylim(0, 1)
    _save(fig, "fig3a")

def plot_3e():
    cat = pd.read_csv(FIG3_SOURCE_DATA / "e_category_flows.csv").set_index("category_3").loc[list(CATEGORY_ORDER)]
    shift = pd.read_csv(FIG3_SOURCE_DATA / "e_category_shift.csv").set_index("category_3").loc[list(CATEGORY_ORDER)]
    funnel = pd.read_csv(FIG3_SOURCE_DATA / "g_funnel.csv")
    fig = _new("fig3e")
    ax = fig.add_axes([.30, .72, .66, .20])
    rows = [("Available\nsurplus", cat.surplus_share_pct), ("Realised\nreceipts", cat.receipt_share_pct)]
    for y0, (_, values) in enumerate(rows[::-1]):
        left = 0.0
        for category in CATEGORY_ORDER:
            value = float(values.loc[category])
            ax.barh(y0, value, left=left, height=.58, color=FIG3_CATEGORY_COLORS[category], ec=INK, lw=.35)
            if value >= 6:
                ax.text(left + value / 2, y0, f"{value:.0f}", ha="center", va="center", fontsize=6.2)
            left += value
    ax.set_yticks([0, 1], [rows[1][0], rows[0][0]])
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("System composition (%)", labelpad=1)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=7.0)
    _clean(ax, "x")

    ax_shift = fig.add_axes([.30, .45, .66, .16])
    y = np.arange(len(CATEGORY_ORDER)); values = shift.receipt_minus_surplus_pp.to_numpy(float)
    bars = ax_shift.barh(y, values, color=[FIG3_CATEGORY_COLORS[c] for c in CATEGORY_ORDER], ec=INK, lw=.4, height=.58)
    for bar, value in zip(bars, values):
        ax_shift.text(value + (.12 if value >= 0 else -.12), bar.get_y() + bar.get_height() / 2,
                      f"{value:+.1f}", ha="left" if value >= 0 else "right", va="center", fontsize=6.2)
    limit = max(4.5, np.ceil(np.abs(values).max()))
    ax_shift.axvline(0, color=INK, lw=.65)
    ax_shift.set_xlim(-limit, limit); ax_shift.set_yticks(y, [CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
    ax_shift.invert_yaxis(); ax_shift.set_xlabel("Receipt − surplus (pp)", labelpad=1)
    ax_shift.tick_params(axis="y", length=0, pad=2, labelsize=7.0); _clean(ax_shift, "x")

    ax_fun = fig.add_axes([.03, .04, .94, .29])
    maximum = funnel.energy_twh.max(); contract = figure3_layout_contract()["fig3e"]
    widths = funnel.energy_twh / maximum * contract["funnel_width"]
    colors = [AUX_COLORS["green_light"], AUX_COLORS["yellow"], AUX_COLORS["brown"]]
    ys = [1.80, 1.18, .56]; center = .28
    for index, (y0, width, color, row) in enumerate(zip(ys, widths, colors, funnel.itertuples(index=False))):
        x0, x1 = center - width / 2, center + width / 2; inset = min(.022, width * .12)
        ax_fun.fill([x0, x1, x1 - inset, x0 + inset], [y0 + .27, y0 + .27, y0 - .27, y0 - .27],
                    color=color, ec=INK, lw=.35)
        ax_fun.text(center, y0, f"{row.stage}\n{row.energy_twh:.3f} TWh", ha="center", va="center",
                    fontsize=contract["funnel_fontsize"], linespacing=1.16, color="white" if index == 2 else INK)
    ax_fun.text(.61, 1.49, f"Reachability gap\n{funnel.energy_twh.iloc[0] - funnel.energy_twh.iloc[1]:.3f} TWh",
                fontsize=contract["funnel_fontsize"], va="center")
    ax_fun.text(.61, .87, f"Allocation gap\n{funnel.energy_twh.iloc[1] - funnel.energy_twh.iloc[2]:.3f} TWh",
                fontsize=contract["funnel_fontsize"], va="center")
    ax_fun.set_xlim(0, 1); ax_fun.set_ylim(.20, 2.12); ax_fun.axis("off")
    _save(fig, "fig3e")

def _fig3h_marker_sizes(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.full_like(values, 32.0)
    return 20.0 + 40.0 * (values - values.min()) / span

def plot_3gh():
    g = pd.read_csv(FIG3_SOURCE_DATA / "g_support_comparison.csv")
    h = pd.read_csv(FIG3_SOURCE_DATA / "h_municipality_mechanism.csv").set_index("stage")
    contract = figure3_layout_contract()["fig3g"]
    fig = _new("fig3g")

    # Left: transfer-retained chart (3g)
    ax = fig.add_axes([.08, .28, .34, .60])
    x = np.arange(len(g))
    ax.plot(x - .10, g.edge_retention_pct, marker="s", ms=contract["marker_size"],
            color=AUX_COLORS["yellow"], mec=INK, mew=.35, lw=1.05,
            label="Edge-deletion mesh", zorder=3)
    ax.plot(x + .10, g.municipality_retention_pct, marker="^", ms=contract["marker_size"],
            color=AUX_COLORS["brown"], mec=INK, mew=.35, lw=1.05,
            label="Municipality node", zorder=3)
    ax.set_xticks(x, [str(value) for value in g.technical_potential_pct])
    ax.set_xlabel("PV technical potential (%)", labelpad=1, fontsize=9.0)
    ax.set_ylabel("Transfer retained\n(% of baseline)", labelpad=2, fontsize=9.0)
    ax.set_ylim(*contract["ylim"]); ax.set_yticks([0, 50, 100]); _clean(ax, "y")
    ax.legend(frameon=False, loc="upper center", ncol=contract["legend_ncol"], borderaxespad=.25,
              handlelength=.85, handletextpad=.25, columnspacing=.45, labelspacing=.15,
              fontsize=contract["legend_fontsize"])

    # Right: municipality mechanism chart (3h)
    ax2 = fig.add_axes([.46, .28, .51, .60])
    gross = float(h.loc["gross_mesh_surplus", "value_twh"])
    residual = float(h.loc["residual_municipal_surplus", "value_twh"]); absorbed = gross - residual
    aggregation_y = 4.05
    absorbed_pct = 100 * absorbed / gross
    residual_pct = 100 * residual / gross
    ax2.barh(aggregation_y, absorbed_pct, height=.52, color=AUX_COLORS["green_light"], ec=INK, lw=.35)
    ax2.barh(aggregation_y, residual_pct, left=absorbed_pct, height=.52,
             color=AUX_COLORS["cream"], ec=INK, lw=.35)
    h_contract = figure3_layout_contract()["fig3h"]
    ax2.text(h_contract["aggregation_title_x"], aggregation_y + .38, "Municipal aggregation",
             ha="left", va="bottom", fontsize=7.0)
    ax2.text(absorbed_pct / 2, aggregation_y, f"{absorbed_pct:.1f}% absorbed",
             ha="center", va="center", fontsize=7.0)
    ax2.text(absorbed_pct + residual_pct / 2, aggregation_y, f"{residual:.3f} TWh",
             ha="center", va="center", fontsize=7.0)
    stages = ["primary", "maximum_r_lambda", "multi_round", "pooling"]
    labels = ["Primary", "Maximum R/λ", "Multi-round", "Pooling"]
    colors = [AUX_COLORS["brown"], AUX_COLORS["yellow"], AUX_COLORS["green_light"], AUX_COLORS["green"]]
    values = h.loc[stages, "value_twh"].to_numpy(float)
    shares = h.loc[stages, "share_pct"].to_numpy(float); sizes = _fig3h_marker_sizes(values)
    for yi, label, share, value, size, color, stage in zip(np.arange(3, -1, -1), labels, shares, values, sizes, colors, stages):
        ax2.plot([0, share], [yi, yi], color=color, lw=2.0, solid_capstyle="butt")
        ax2.scatter([share], [yi], s=size, color=color, edgecolor=INK, linewidth=.35, zorder=3)
        ax2.text(h_contract["mechanism_label_x"], yi + .20, label, ha="left", va="bottom", fontsize=7.0)
        bbox = {"facecolor": AUX_COLORS["cream"], "edgecolor": "none", "pad": 1.0, "alpha": .92} if stage == "pooling" else None
        text_x = share + h_contract["pooling_value_offset"] if stage == "pooling" else share + 2
        text_ha = "right" if stage == "pooling" else "left"
        ax2.text(text_x, yi, f"{value:.3f} TWh  ({share:.1f}%)", ha=text_ha, va="center", fontsize=7.0, bbox=bbox)
    ax2.set_xlim(*h_contract["xlim"]); ax2.set_ylim(-.45, 4.70); ax2.set_xticks([0, 50, 100])
    ax2.set_xlabel("Share of municipal residual surplus realised (%)", labelpad=1, fontsize=9.0); ax2.set_yticks([])
    _clean(ax2, "x")
    _save(fig, "fig3g")

def plot_all():
    plot_3a()
    plot_3b(); plot_3c()
    plot_3d(); plot_3e(); plot_3f()
    plot_3gh()

if __name__ == "__main__":
    plot_all()
