from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]


N03_PREFECTURES = (
    ROOT / "figures" / "reference_boundaries" / "gsi_n03_kanto" / "N03-20260101_53_prefecture.shp"
)


KANTO_PREFECTURES = {"Tokyo", "Kanagawa", "Saitama", "Chiba"}


N03_KANTO_NAMES = {
    "東京都": "Tokyo",
    "神奈川県": "Kanagawa",
    "埼玉県": "Saitama",
    "千葉県": "Chiba",
}


@dataclass(frozen=True)
class BoundaryRecord:
    name: str
    path: MplPath


def _dbf_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    n_records = struct.unpack_from("<I", raw, 4)[0]
    header_len, record_len = struct.unpack_from("<2H", raw, 8)
    fields: list[tuple[str, int, int]] = []
    position, offset = 32, 1
    while raw[position] != 0x0D:
        name = raw[position : position + 11].split(b"\0", 1)[0].decode("ascii")
        length = raw[position + 16]
        fields.append((name, offset, length))
        offset += length
        position += 32
    rows = []
    for index in range(n_records):
        start = header_len + index * record_len
        row = {}
        for name, field_offset, field_length in fields:
            value = raw[
                start + field_offset : start + field_offset + field_length
            ].strip(b" \0")
            row[name] = value.decode("utf-8", errors="replace")
        rows.append(row)
    return rows


def _polygon_rings(content: bytes) -> list[np.ndarray]:
    shape_type = struct.unpack_from("<i", content, 0)[0]
    if shape_type not in (5, 15, 25):
        raise ValueError(f"unsupported polygon shape type: {shape_type}")
    n_parts, n_points = struct.unpack_from("<2i", content, 36)
    starts = list(struct.unpack_from(f"<{n_parts}i", content, 44)) + [n_points]
    point_offset = 44 + 4 * n_parts
    points = np.array(
        struct.unpack_from(f"<{2 * n_points}d", content, point_offset),
        dtype=float,
    ).reshape(-1, 2)
    rings = []
    for start, end in zip(starts[:-1], starts[1:]):
        ring = points[start:end]
        if len(ring) < 3:
            continue
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        rings.append(ring)
    if not rings:
        raise ValueError("polygon record contains no usable rings")
    return rings


def _rings_to_path(rings: list[np.ndarray]) -> MplPath:
    vertices = []
    codes = []
    for ring in rings:
        ring_codes = np.full(len(ring), MplPath.LINETO, dtype=np.uint8)
        ring_codes[0] = MplPath.MOVETO
        ring_codes[-1] = MplPath.CLOSEPOLY
        vertices.append(ring)
        codes.append(ring_codes)
    return MplPath(np.vstack(vertices), np.concatenate(codes))


def _edge_loops(edges: set, scale: int) -> list[np.ndarray]:
    """Build vertex paths from a set of undirected quantized edges."""
    remaining = set(edges)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for a, b in remaining:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    loops = []
    while remaining:
        first_edge = next(iter(remaining))
        start, current = first_edge
        previous = start
        vertices = [start, current]
        remaining.remove(first_edge)
        while current != start:
            candidates = []
            for neighbour in adjacency.get(current, ()):
                key = (current, neighbour) if current < neighbour else (neighbour, current)
                if key in remaining:
                    candidates.append(neighbour)
            if not candidates:
                break
            if len(candidates) == 1:
                next_vertex = candidates[0]
            else:
                candidates.sort(key=lambda point: point == previous)
                next_vertex = candidates[0]
            key = (
                (current, next_vertex)
                if current < next_vertex
                else (next_vertex, current)
            )
            remaining.remove(key)
            previous, current = current, next_vertex
            vertices.append(current)
        if len(vertices) >= 2:
            loops.append(np.asarray(vertices, dtype=float) / scale)
    return loops


def _split_rings(rings: list[np.ndarray], precision: int = 9):
    """Return (outer_loops, shared_loops) for a set of polygon rings.

    Outer loops are edges appearing an odd number of times (exterior
    boundaries); shared loops are edges appearing an even number of times
    (boundaries shared between two input polygons).
    """
    scale = 10**precision
    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for ring in rings:
        quantized = np.rint(ring * scale).astype(np.int64)
        for first, second in zip(quantized[:-1], quantized[1:]):
            a = (int(first[0]), int(first[1]))
            b = (int(second[0]), int(second[1]))
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    outer = {edge for edge, count in edge_counts.items() if count % 2 == 1}
    shared = {edge for edge, count in edge_counts.items() if count % 2 == 0}
    return _edge_loops(outer, scale), _edge_loops(shared, scale)


def _outer_rings(rings: list[np.ndarray], precision: int = 9) -> list[np.ndarray]:
    outer, _ = _split_rings(rings, precision)
    closed = [
        loop for loop in outer
        if len(loop) >= 4 and np.array_equal(loop[0], loop[-1])
    ]
    if not closed:
        raise ValueError("shared-edge dissolve produced no closed boundary rings")
    return closed


def _load_n03_grouped() -> dict[str, list[np.ndarray]]:
    rows = _dbf_rows(N03_PREFECTURES.with_suffix(".dbf"))
    selected = {
        index: N03_KANTO_NAMES[row["N03_001"]]
        for index, row in enumerate(rows)
        if row.get("N03_001") in N03_KANTO_NAMES
    }
    grouped: dict[str, list[np.ndarray]] = {
        name: [] for name in N03_KANTO_NAMES.values()
    }
    with N03_PREFECTURES.open("rb") as stream:
        stream.seek(100)
        record_index = 0
        while True:
            header = stream.read(8)
            if len(header) < 8:
                break
            _, words = struct.unpack(">2i", header)
            content = stream.read(words * 2)
            if record_index in selected:
                grouped[selected[record_index]].extend(_polygon_rings(content))
            record_index += 1
    return grouped


def load_n03_kanto() -> list[BoundaryRecord]:
    grouped = _load_n03_grouped()
    records = [
        BoundaryRecord(name, _rings_to_path(_outer_rings(grouped[name])))
        for name in ("Tokyo", "Kanagawa", "Saitama", "Chiba")
    ]
    if {record.name for record in records} != KANTO_PREFECTURES:
        raise ValueError("N03 input does not contain all four Kanto prefectures")
    return records


SCIENTIFIC_LAYER_ZORDER = 3


BOUNDARY_UNDERLAY_ZORDER = 1


BOUNDARY_OVERLAY_ZORDER = 20


DECORATION_ZORDER = 100


@dataclass(frozen=True)
class MapContract:
    xlim: tuple[float, float] = (138.703125, 140.945625)
    ylim: tuple[float, float] = (34.88625, 36.32375)
    xticks: tuple[float, ...] = (139.0, 139.5, 140.0, 140.5)
    yticks: tuple[float, ...] = (35.0, 35.25, 35.5, 35.75, 36.0, 36.25)
    aspect_latitude: float = 35.6
    boundary_color: str = "#9E9E9E"
    boundary_linewidth: float = 0.35
    tick_size: float = 6.5
    tick_pad: float = 1.5
    axes_left: float = 0.12
    axes_bottom: float = 0.10
    axes_width: float = 0.84
    axes_height: float = 0.86
    north_x: float = 0.91
    north_y: float = 0.105
    north_half_width: float = 0.026
    north_height: float = 0.072
    north_label_gap: float = 0.012
    north_linewidth: float = 0.7
    north_font_size: float = 8.0
    ink: str = "#3A352E"


MAP_CONTRACT = MapContract()


@lru_cache(maxsize=1)
def boundary_records() -> tuple[BoundaryRecord, ...]:
    return tuple(load_n03_kanto())


def _add_boundaries(ax, *, facecolor: str, edgecolor: str, linewidth: float, zorder: int):
    for record in boundary_records():
        ax.add_patch(
            mpatches.PathPatch(
                record.path,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                joinstyle="round",
                capstyle="round",
                zorder=zorder,
            )
        )


def add_boundary_underlay(ax) -> None:
    _add_boundaries(
        ax,
        facecolor="white",
        edgecolor="none",
        linewidth=0.0,
        zorder=BOUNDARY_UNDERLAY_ZORDER,
    )


def add_boundary_overlay(ax) -> None:
    _add_boundaries(
        ax,
        facecolor="none",
        edgecolor=MAP_CONTRACT.boundary_color,
        linewidth=MAP_CONTRACT.boundary_linewidth,
        zorder=BOUNDARY_OVERLAY_ZORDER,
    )


def apply_map_frame(ax, *, show_xlabels: bool = True, show_ylabels: bool = True) -> None:
    ax.set_xlim(*MAP_CONTRACT.xlim)
    ax.set_ylim(*MAP_CONTRACT.ylim)
    ax.set_xticks(MAP_CONTRACT.xticks)
    ax.set_yticks(MAP_CONTRACT.yticks)
    ax.set_aspect(1.0 / np.cos(np.radians(MAP_CONTRACT.aspect_latitude)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}°E"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}°N"))
    ax.tick_params(
        axis="both",
        labelsize=MAP_CONTRACT.tick_size,
        pad=MAP_CONTRACT.tick_pad,
        labelbottom=show_xlabels,
        labelleft=show_ylabels,
    )


def add_north_arrow(ax) -> None:
    c = MAP_CONTRACT
    top = (c.north_x, c.north_y + c.north_height)
    bottom = (c.north_x, c.north_y)
    left = (c.north_x - c.north_half_width, c.north_y)
    right = (c.north_x + c.north_half_width, c.north_y)
    ax.add_patch(
        mpatches.Polygon(
            [top, left, right],
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=c.ink,
            linewidth=c.north_linewidth,
            zorder=DECORATION_ZORDER,
        )
    )
    ax.add_patch(
        mpatches.Polygon(
            [top, left, bottom],
            transform=ax.transAxes,
            facecolor=c.ink,
            edgecolor="none",
            linewidth=0,
            zorder=DECORATION_ZORDER + 1,
        )
    )
    ax.text(
        c.north_x,
        c.north_y + c.north_height + c.north_label_gap,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=c.north_font_size,
        color=c.ink,
        zorder=DECORATION_ZORDER + 2,
    )


def standalone_map_axes(fig):
    c = MAP_CONTRACT
    return fig.add_axes([c.axes_left, c.axes_bottom, c.axes_width, c.axes_height])
