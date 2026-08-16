from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from map_contract import (
    add_boundary_overlay,
    add_boundary_underlay,
    add_north_arrow,
    apply_map_frame,
)
from scipy.stats import gaussian_kde

from figure_contracts import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    CREAM_PANEL,
    FIG2_LISA_COLORS,
    FONT_SM,
    INK,
    MAP_FRAME,
    MAP_LIMITS,
    MAP_TICKS,
    NORTH_ARROW,
    PANEL_DPI,
    PANEL_SIZES_MM,
    FIG2_PANELS,
    REFERENCE,
    FIG2_SOURCE_DATA,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 6.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.edgecolor": REFERENCE,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

def map_geometry_contract() -> dict:
    return {
        "limits": MAP_LIMITS,
        "ticks": MAP_TICKS,
        "frame": MAP_FRAME,
        "north_arrow": NORTH_ARROW,
    }

def compound_layout_contract() -> dict:
    return {
        "fig2a": {
            "colorbar_label": "Net balance\n(MW)",
            "legend_location": "upper left",
            "colorbar_label_coords": (1.35, -0.08),
        },
        "fig2c": {
            "category_positions": (0.0, 0.70, 1.40),
            "jitter_width": 0.12,
            "violin_width": 0.48,
            "violin_alpha": 0.88,
            "scatter_alpha": 0.28,
            "n_label_y": 1.02,
            "y_max": 80.0,
            "legend_location": "upper left",
            "legend_bbox": (0.0, 1.06),
        },
        "fig2d": {
            "y_label": "Annual rooftop PV\ngeneration / annual demand",
            "point_deployment": 100,
            "show_deployment_trajectory": False,
            "main_axes": (0.075, 0.22, 0.615, 0.72),
            "companion_frame": (0.760, 0.22, 0.205, 0.72),
            "companion_axes": (0.760, 0.220, 0.205, 0.720),
            "inset_axes": (0.035, 0.071, 0.473, 0.374),
            "inset_facecolor": "none",
            "inset_kind": "category_mismatch_ridges",
            "sample_by_category": {
                "metropolitan_core": 1000,
                "regional_city": 2000,
                "rural": 500,
            },
            "background_point_size": 4.8,
            "background_alpha": 0.25,
            "y_min": 0.02,
            "companion_kind": "continuous_mismatch_onset",
            "companion_xlabel": "Rooftop PV technical potential (%)",
            "companion_ylabel": "Meshes with >1% mismatch (%)",
            "xlabel_y": 0.10,
            "legend_left": ("Category median", "Full self-sufficiency"),
            "legend_right": ("Metro Core", "Regional City", "Rural"),
            "ridge_median_markers": True,
            "ridge_alpha": 0.68,
            "transition_quantiles": (0.25, 0.50, 0.75),
        },
        "fig2f": {
            "scatter": (0.17, 0.63, 0.78, 0.30),
            "trajectory": (0.17, 0.13, 0.78, 0.30),
            "trajectory_kind": "arrangement_penalty",
            "legend_location": "upper left",
            "y_limits": (-2.0, 30.0),
            "gap": 0.20,
            "label_font_size": 11.2,
            "annotation_font_size": 11.2,
            "legend_font_size": 10.2,
            "tick_font_size": 7.5,
            "trajectory_xlabel_font_size": 11.2,
            "show_manifest_ranges": False,
            "profile_span_fill": False,
            "emphasised_deployments": (30, 100),
            "normal_marker_size": 3.0,
            "endpoint_marker_size": 5.5,
        },
        "fig2e": {
            "map_point_size": 0.36,
            "boundary_source": "N03 official",
            "boundary_linewidth": 0.35,
        },
    }

def visual_style_contract() -> dict:
    return {"grid_linestyle": "--"}

def _new(panel: str):
    width, height = PANEL_SIZES_MM[panel]
    return plt.figure(figsize=(width / 25.4, height / 25.4))

def _save(fig, panel: str):
    FIG2_PANELS.mkdir(parents=True, exist_ok=True)
    width, height = PANEL_SIZES_MM[panel]
    fig.set_size_inches(width / 25.4, height / 25.4, forward=True)
    fig.savefig(FIG2_PANELS / f"{panel}.svg", bbox_inches=None, pad_inches=0)
    fig.savefig(
        FIG2_PANELS / f"{panel}.png",
        dpi=PANEL_DPI[panel],
        bbox_inches=None,
        pad_inches=0,
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)

def _clean(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#D8CFBF",
            lw=0.45,
            alpha=0.55,
            linestyle=visual_style_contract()["grid_linestyle"],
        )
        ax.set_axisbelow(True)

def _draw_basemap(ax):
    add_boundary_underlay(ax)
    apply_map_frame(ax)
    return ax

def _decorate_map(fig, ax):
    add_boundary_overlay(ax)
    add_north_arrow(ax)
    fig.subplots_adjust(
        left=MAP_FRAME["left"],
        right=MAP_FRAME["right"],
        bottom=MAP_FRAME["bottom"],
        top=MAP_FRAME["top"],
    )

def plot_2a():
    data = pd.read_parquet(FIG2_SOURCE_DATA / "a_timeseries.parquet")
    profile = pd.read_csv(FIG2_SOURCE_DATA / "a_intraday.csv")
    matrix = (
        data.pivot(index="slot", columns="day", values="net_balance_kwh").to_numpy()
        / 500.0
    )
    limit = float(np.quantile(np.abs(matrix), 0.98))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "balance", ["#c47c76", "#ddb4af", CREAM_PANEL, "#b2c7dc", "#7d9fc4"]
    )
    fig = _new("fig2a")
    ax = fig.add_axes([0.075, 0.25, 0.655, 0.68])
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=(1, 366, 0, 23.5),
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        rasterized=True,
    )
    ax.set_xlabel("Day of year", labelpad=1)
    ax.set_ylabel("Time of day", labelpad=2)
    ax.set_yticks([0, 6, 12, 18, 23.5], ["00:00", "06:00", "12:00", "18:00", "23:30"])
    cax = fig.add_axes([0.744, 0.25, 0.018, 0.68])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.set_xlabel(
        compound_layout_contract()["fig2a"]["colorbar_label"],
        fontsize=7.0,
        labelpad=2,
        linespacing=0.95,
    )
    colorbar.ax.xaxis.set_label_coords(
        *compound_layout_contract()["fig2a"]["colorbar_label_coords"]
    )
    colorbar.ax.tick_params(labelsize=FONT_SM, pad=1)
    ax2 = fig.add_axes([0.82, 0.25, 0.15, 0.68], sharey=ax)
    hours = profile.slot / 2
    ax2.plot(profile.gen_kwh / 500, hours, color="#E0A33C", lw=1.3, label="Generation")
    ax2.plot(profile.demand_kwh / 500, hours, color=REFERENCE, lw=1.3, label="Demand")
    ax2.set_xlabel("MW", labelpad=1)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    _clean(ax2, "x")
    ax2.legend(
        frameon=False,
        loc=compound_layout_contract()["fig2a"]["legend_location"],
        handlelength=1.4,
        labelspacing=0.2,
    )
    _save(fig, "fig2a")

def plot_2b():
    hexagons = pd.read_csv(FIG2_SOURCE_DATA / "b_availability_hexagons.csv")
    scale = pd.read_csv(FIG2_SOURCE_DATA / "b_map_scale.csv").iloc[0]
    fig = _new("fig2b")
    ax = fig.add_axes([MAP_FRAME["left"], MAP_FRAME["bottom"], 0.84, 0.86])
    _draw_basemap(ax)
    vertices = [
        hexagons[[f"v{i}_lon", f"v{i}_lat"]].to_numpy()
        for i in range(6)
    ]
    polygons = np.stack(vertices, axis=1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "availability", ["#c47c76", "#e0b7b3", CREAM_PANEL, "#b4cbe2", "#7d9fc4"]
    )
    collection = PolyCollection(
        polygons,
        array=hexagons.value.to_numpy(),
        cmap=cmap,
        norm=mcolors.Normalize(vmin=scale.vmin, vmax=scale.vmax),
        edgecolors="white",
        linewidths=0.06,
        zorder=3,
    )
    ax.add_collection(collection)
    cax = ax.inset_axes([0.53, 0.88, 0.42, 0.032])
    colorbar = fig.colorbar(collection, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=FONT_SM, pad=1)
    colorbar.ax.set_title("Receiver-side availability", fontsize=7.0, pad=1)
    _decorate_map(fig, ax)
    _save(fig, "fig2b")

def plot_2c():
    distribution = pd.read_parquet(FIG2_SOURCE_DATA / "c_category_distribution.parquet")
    summary = pd.read_csv(FIG2_SOURCE_DATA / "c_category_summary.csv").set_index("category_3")
    trajectory = pd.read_csv(FIG2_SOURCE_DATA / "c_trajectory.csv")
    fig = _new("fig2c")
    ax = fig.add_axes([0.18, 0.47, 0.77, 0.43])
    rng = np.random.default_rng(20260806)
    layout = compound_layout_contract()["fig2c"]
    positions = layout["category_positions"]
    for position, category in zip(positions, CATEGORY_ORDER):
        values = 100 * distribution.loc[
            distribution.category_3 == category, "local_donor_availability_solar"
        ].to_numpy()
        violin = ax.violinplot(
            values,
            positions=[position],
            widths=layout["violin_width"],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.18,
        )["bodies"][0]
        for path in violin.get_paths():
            path.vertices[:, 0] = np.minimum(path.vertices[:, 0], position)
        violin.set_facecolor(CATEGORY_COLORS[category])
        violin.set_edgecolor(INK)
        violin.set_linewidth(0.55)
        violin.set_alpha(layout["violin_alpha"])
        violin.set_gid("raincloud-violin")

        sample_size = min(1400, len(values))
        sampled = rng.choice(values, size=sample_size, replace=False)
        x = position + 0.045 + rng.uniform(0, layout["jitter_width"], sample_size)
        ax.scatter(
            x,
            sampled,
            s=1.1,
            alpha=layout["scatter_alpha"],
            color=CATEGORY_COLORS[category],
            linewidths=0,
            rasterized=True,
        )
        q10, q25, median, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
        ax.vlines(position, q10, q90, color=INK, lw=0.6, zorder=4)
        box = mpatches.Rectangle(
            (position - 0.035, q25),
            0.07,
            q75 - q25,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.55,
            zorder=5,
        )
        box.set_gid("raincloud-box")
        ax.add_patch(box)
        ax.hlines(median, position - 0.035, position + 0.035, color=INK, lw=0.75, zorder=6)
        ax.scatter(
            position + 0.205,
            100 * summary.loc[category, "da_shortfall_weighted"],
            marker="D",
            s=16,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.55,
            zorder=7,
        )
        ax.text(
            position,
            layout["n_label_y"],
            f"n={len(values):,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.0,
            clip_on=False,
        )
    ax.set_xticks(positions, [CATEGORY_LABELS[x] for x in CATEGORY_ORDER])
    ax.tick_params(axis="both", labelsize=7.0)
    ax.set_xlim(-0.34, 1.70)
    ax.set_ylabel("Receiver-side availability (%)", labelpad=2, fontsize=7.0)
    ax.set_ylim(-1, layout["y_max"])
    _clean(ax, "y")
    ax2 = fig.add_axes([0.18, 0.12, 0.77, 0.25])
    ax2.plot(
        trajectory.deployment_pct,
        100 * trajectory.recurrent_share,
        "o-",
        color="#c47c76",
        lw=1.15,
        ms=3.2,
        label="Recurrent contributor",
    )
    ax2.plot(
        trajectory.deployment_pct,
        100 * trajectory.receiver_availability,
        "s--",
        color="#7d9fc4",
        lw=1.15,
        ms=3.0,
        label="Receiver availability",
    )
    ax2.set_xlim(5, 105)
    ax2.set_ylim(0, 100)
    ax2.set_xticks([10, 20, 30, 40, 50, 100])
    ax2.set_xlabel("Rooftop PV technical potential (%)", fontsize=7.0, labelpad=1)
    ax2.set_ylabel("System share (%)", labelpad=2)
    ax2.legend(
        frameon=False,
        ncol=1,
        loc=layout["legend_location"],
        bbox_to_anchor=layout["legend_bbox"],
        fontsize=7.0,
        handlelength=1.5,
        labelspacing=0.18,
    )
    _clean(ax2, "y")
    _save(fig, "fig2c")

def _draw_mismatch_ridges(ax, main):
    ridge_alpha = compound_layout_contract()["fig2d"]["ridge_alpha"]
    x_grid = np.linspace(0, 100, 241)
    baselines = np.array([2.0, 1.0, 0.0])
    for baseline, category in zip(baselines, CATEGORY_ORDER):
        values = 100 * (
            1
            - main.loc[
                main.category_3 == category, "coincident_generation_share"
            ].dropna().to_numpy()
        )
        density = gaussian_kde(values, bw_method=0.18)(x_grid)
        density = 0.72 * density / density.max()
        color = CATEGORY_COLORS[category]
        ax.fill_between(
            x_grid,
            baseline,
            baseline + density,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.45,
            alpha=ridge_alpha,
            zorder=3,
        )
        median = float(np.median(values))
        median_height = float(np.interp(median, x_grid, density))
        ax.vlines(
            median,
            baseline,
            baseline + max(0.18, median_height),
            color=INK,
            linewidth=0.65,
            zorder=4,
        )
        ax.text(
            1.0,
            baseline + 0.08,
            CATEGORY_LABELS[category],
            fontsize=6.5,
            ha="left",
            va="bottom",
            color=INK,
            zorder=5,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.08, 2.88)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([])
    ax.text(
        80,
        0.08,
        "Non-coincident share (%)",
        fontsize=7.0,
        ha="center",
        va="bottom",
        color=INK,
        zorder=5,
    )
    ax.tick_params(axis="x", labelsize=6.5, pad=0.5, length=0)
    ax.patch.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

def plot_2d():
    points = pd.read_parquet(FIG2_SOURCE_DATA / "d_phase_points.parquet")
    onset = pd.read_csv(FIG2_SOURCE_DATA / "d_mismatch_onset_curve.csv")
    onset_summary = pd.read_csv(FIG2_SOURCE_DATA / "d_mismatch_onset_summary.csv").iloc[0]
    category_summary = pd.read_csv(FIG2_SOURCE_DATA / "d_phase_summary.csv").set_index("category_3")
    layout = compound_layout_contract()["fig2d"]
    main = points.loc[points.deployment_pct == layout["point_deployment"]]
    fig = _new("fig2d")
    ax = fig.add_axes(layout["main_axes"])

    for category in CATEGORY_ORDER:
        group = main.loc[main.category_3 == category]
        sample_count = layout["sample_by_category"][category]
        if len(group) > sample_count:
            group = group.sample(sample_count, random_state=20260806)
        ax.scatter(
            group.coincident_generation_share,
            group.deployed_generation_demand_ratio,
            s=layout["background_point_size"],
            alpha=layout["background_alpha"],
            color=CATEGORY_COLORS[category],
            linewidths=0,
            rasterized=True,
        )
    x = np.linspace(0.05, 1, 300)
    ax.plot(x, 1 / x, color=REFERENCE, ls="--", lw=1.0)
    for category in CATEGORY_ORDER:
        median = category_summary.loc[category]
        ax.scatter(
            median.median_coincident_share,
            median.median_generation_demand_ratio,
            s=38,
            color=CATEGORY_COLORS[category],
            edgecolor=INK,
            linewidth=0.55,
            zorder=6,
        )
    handles = [
        Line2D(
            [0], [0], marker="o", lw=0, color=CATEGORY_COLORS[c],
            markeredgecolor=INK, label=CATEGORY_LABELS[c], markersize=4.5,
        )
        for c in CATEGORY_ORDER
    ] + [
        Line2D(
            [0], [0], marker="o", lw=0, markerfacecolor="white",
            markeredgecolor=INK, label="Category median", markersize=4.5,
        ),
        Line2D(
            [0], [0], color=REFERENCE, ls="--", lw=1,
            label="Full self-sufficiency",
        ),
    ]
    handles_by_label = {handle.get_label(): handle for handle in handles}
    left_legend = ax.legend(
        handles=[handles_by_label[label] for label in layout["legend_left"]],
        frameon=False, loc="upper right", bbox_to_anchor=(0.72, 1.0),
        handlelength=1.4, labelspacing=0.5, fontsize=7.0,
    )
    ax.add_artist(left_legend)
    ax.legend(
        handles=[handles_by_label[label] for label in layout["legend_right"]],
        frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0),
        handlelength=1.4, labelspacing=0.5, fontsize=7.0,
    )
    ax.set_yscale("log")
    ax.set_xlim(0.03, 1.02)
    ax.set_ylim(layout["y_min"], 20)
    ax.set_ylabel(layout["y_label"], labelpad=2, fontsize=9.0)
    _clean(ax)

    inset = ax.inset_axes(layout["inset_axes"])
    _draw_mismatch_ridges(inset, main)

    companion_ax = fig.add_axes(layout["companion_axes"])
    p25 = float(onset_summary.p25_potential_pct)
    p50 = float(onset_summary.p50_potential_pct)
    p75 = float(onset_summary.p75_potential_pct)
    companion_ax.axvspan(
        p25, p75, color="#E8DCC4", alpha=0.58, linewidth=0, zorder=0
    )
    companion_ax.fill_between(
        onset.potential_pct, 0, onset.prevalence_pct,
        color="#D9BE8E", alpha=0.20, linewidth=0, zorder=1,
    )
    companion_ax.plot(
        onset.potential_pct, onset.prevalence_pct,
        color="#8D6039", lw=1.45, zorder=3,
    )
    companion_ax.axvline(p50, color=REFERENCE, lw=0.65, ls="--", zorder=2)
    companion_ax.axhline(
        50, xmax=p50 / 100, color=REFERENCE, lw=0.65, ls="--", zorder=2
    )
    companion_ax.scatter(
        [p50], [50], s=18, facecolor="#8D6039", edgecolor=INK,
        linewidth=0.45, zorder=4,
    )
    a30_y = float(
        onset.loc[np.isclose(onset.potential_pct, 30.0), "prevalence_pct"].iloc[0]
    )
    companion_ax.scatter(
        [30], [a30_y], s=14, facecolor="white", edgecolor=INK,
        linewidth=0.55, zorder=4,
    )
    companion_ax.annotate(
        "30%", (30, a30_y), xytext=(-2, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=7.0,
    )
    companion_ax.text(
        p50 - 1, 91, "System-wide\ntransition",
        ha="right", va="top", fontsize=7.0, color=REFERENCE,
    )
    companion_ax.annotate(
        f"Median onset\n{p50:.1f}%", (p50, 50), xytext=(8, -8),
        textcoords="offset points", fontsize=7.0, ha="left", va="top",
    )
    companion_ax.set_xlim(0, 100)
    companion_ax.set_ylim(0, 100)
    companion_ax.set_xticks([0, 25, 50, 75, 100])
    companion_ax.set_yticks([0, 25, 50, 75, 100])
    companion_ax.set_ylabel(layout["companion_ylabel"], labelpad=2, fontsize=9.0)
    _clean(companion_ax, "y")

    companion = layout["companion_frame"]
    fig.text(
        companion[0] + companion[2] / 2, layout["xlabel_y"],
        layout["companion_xlabel"], ha="center", va="center", fontsize=9.0,
    )
    fig.text(
        layout["main_axes"][0] + layout["main_axes"][2] / 2,
        layout["xlabel_y"], "Coincident generation share",
        ha="center", va="center", fontsize=9.0,
    )
    _save(fig, "fig2d")

def plot_2e():
    data = pd.read_parquet(FIG2_SOURCE_DATA / "e_lisa.parquet")
    fig = _new("fig2e")
    ax = fig.add_axes([MAP_FRAME["left"], MAP_FRAME["bottom"], 0.84, 0.86])
    _draw_basemap(ax)
    point_size = compound_layout_contract()["fig2e"]["map_point_size"]
    for quadrant in ("NS", "LL", "HH", "HL", "LH"):
        group = data.loc[data.quadrant == quadrant]
        ax.scatter(
            group.lon,
            group.lat,
            s=point_size,
            color=FIG2_LISA_COLORS[quadrant],
            linewidths=0,
            zorder=3,
        )
    labels = {
        "HH": "High–High",
        "LL": "Low–Low",
        "HL": "High–Low",
        "LH": "Low–High",
        "NS": "Not significant",
    }
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=FIG2_LISA_COLORS[q], label=labels[q], markersize=4.5)
        for q in ("HH", "LL", "HL", "LH", "NS")
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    _decorate_map(fig, ax)
    _save(fig, "fig2e")

def plot_2f():
    data = pd.read_parquet(FIG2_SOURCE_DATA / "e_lisa.parquet")
    stats = pd.read_csv(FIG2_SOURCE_DATA / "e_global.csv").iloc[0]
    trajectory = pd.read_csv(FIG2_SOURCE_DATA / "f_penalty_trajectory.csv")
    fig = _new("fig2f")
    layout = compound_layout_contract()["fig2f"]
    ax = fig.add_axes(layout["scatter"])
    sampled = data.sample(min(6000, len(data)), random_state=7)
    ax.scatter(
        sampled.z_floors,
        sampled.spatial_lag,
        c=[FIG2_LISA_COLORS[x] for x in sampled.quadrant],
        s=5.5,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(0, color=REFERENCE, lw=0.55)
    ax.axvline(0, color=REFERENCE, lw=0.55)
    xline = np.linspace(stats.x_lo, stats.x_hi, 100)
    ax.plot(xline, stats.registry_morans_i * xline, color=INK, lw=1.0)
    ax.text(
        0.96,
        0.08,
        f"Moran's I = {stats.registry_morans_i:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=layout["annotation_font_size"],
    )
    ax.set_xlim(stats.x_lo, stats.x_hi)
    ax.set_ylim(stats.y_lo, stats.y_hi)
    ax.set_xlabel("Standardized average storeys", fontsize=layout["label_font_size"], labelpad=1)
    ax.set_ylabel("Spatial lag", fontsize=layout["label_font_size"], labelpad=2)
    ax.tick_params(axis="both", labelsize=layout["tick_font_size"])
    _clean(ax)

    ax2 = fig.add_axes(layout["trajectory"])
    deployments = [10, 15, 20, 30, 40, 50, 100]
    positions = np.arange(len(deployments), dtype=float)
    styles = {
        "Mesh-specific profiles": {"color": CATEGORY_COLORS["metropolitan_core"], "ls": "-", "marker": "o"},
        "Common irradiance": {"color": CATEGORY_COLORS["rural"], "ls": "--", "marker": "o"},
    }
    for profile, style in styles.items():
        values = trajectory.loc[trajectory.profile == profile].set_index("deployment_pct")
        y = values.loc[deployments, "penalty_pct"].to_numpy()
        marker_face = style["color"] if profile == "Mesh-specific profiles" else "white"
        ax2.plot(
            positions,
            y,
            color=style["color"],
            ls=style["ls"],
            lw=1.15,
            marker=style["marker"],
            ms=layout["normal_marker_size"],
            markerfacecolor=marker_face,
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
            label=profile,
            zorder=3,
        )
        for deployment in layout["emphasised_deployments"]:
            index = deployments.index(deployment)
            ax2.plot(
                positions[index],
                y[index],
                marker=style["marker"],
                ms=layout["endpoint_marker_size"],
                linestyle="none",
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                zorder=4,
            )
    ax2.axhline(0, color=REFERENCE, lw=0.55)
    ax2.set_xticks(positions, deployments)
    ax2.set_xlim(-0.25, len(deployments) - 0.75)
    ax2.set_ylim(*layout["y_limits"])
    ax2.set_yticks([0, 10, 20, 30])
    ax2.set_xlabel(
        "Rooftop PV technical potential (%)",
        fontsize=layout["trajectory_xlabel_font_size"],
        labelpad=1,
    )
    ax2.set_ylabel("Arrangement penalty (%)", fontsize=layout["label_font_size"], labelpad=2)
    ax2.tick_params(axis="both", labelsize=layout["tick_font_size"])
    ax2.legend(
        frameon=False,
        ncol=1,
        loc=layout["legend_location"],
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.5,
        labelspacing=0.18,
        borderaxespad=0.25,
        fontsize=layout["legend_font_size"],
    )
    _clean(ax2, "y")
    _save(fig, "fig2f")

PLOTTERS = {
    "fig2a": plot_2a,
    "fig2b": plot_2b,
    "fig2c": plot_2c,
    "fig2d": plot_2d,
    "fig2e": plot_2e,
    "fig2f": plot_2f,
}

def render_all():
    for plotter in PLOTTERS.values():
        plotter()

if __name__ == "__main__":
    render_all()
