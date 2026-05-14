"""
Streamlit application for DXF Inspection Tool.

Replaces the legacy HTTP server (main.py) + HTML/JS frontend (index.html)
with an interactive web UI built on Streamlit + Plotly.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import base64
import io
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ezdxf import recover

from dxf_statistical import analyze_dxf_bytes
from endpoint_boards import analyze_endpoint_boards_bytes
from final_print import analyze_final_payload
from interpolate import analyze_interpolate_payload
from lack_print import analyze_lack_payload
from projection_3_axis import analyze_projection_3_axis_payload

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DXF Inspection Tool",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGHLIGHT_COLORS: dict[str, str] = {
    "endpoint_board": "#0f766e",
    "circle_board": "#db2777",
    "distance_x": "#dc2626",
    "distance_y": "#f59e0b",
    "distance_edge": "#f59e0b",
    "distance_angle": "#7c3aed",
    "distance_radius": "#16a34a",
    "distance_other": "#6b7280",
    "full_distance_x": "#ea580c",
    "full_distance_y": "#0891b2",
    "side_way": "#14b8a6",
    "top_way": "#f97316",
    "font_way": "#8b5cf6",
    "lack_distance_x": "#ef4444",
    "lack_distance_y": "#3b82f6",
    "lack_distance_other": "#a855f7",
    "final_distance_x": "#b91c1c",
    "final_distance_y": "#115e59",
    "final_distance_oz": "#f97316",
}

# ---------------------------------------------------------------------------
# DXF → Plotly renderer
# ---------------------------------------------------------------------------


def read_dxf_entities(dxf_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a DXF file and return a list of geometric entity dicts.

    Each dict has keys: ``type``, plus type-specific coordinate data.
    """
    try:
        doc, _auditor = recover.read(io.BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(io.BytesIO(dxf_bytes))
    msp = doc.modelspace()
    entities: list[dict[str, Any]] = []

    for entity in msp:
        etype = entity.dxftype().upper()
        try:
            if etype == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                entities.append({
                    "type": "LINE",
                    "x1": round(float(start.x), 4),
                    "y1": round(float(start.y), 4),
                    "x2": round(float(end.x), 4),
                    "y2": round(float(end.y), 4),
                })
            elif etype == "LWPOLYLINE":
                pts = [(round(float(p[0]), 4), round(float(p[1]), 4))
                       for p in entity.get_points("xy")]
                closed = bool(entity.closed)
                entities.append({"type": "LWPOLYLINE", "points": pts, "closed": closed})
            elif etype == "POLYLINE":
                pts = []
                for v in entity.vertices():
                    loc = v.dxf.location
                    pts.append((round(float(loc.x), 4), round(float(loc.y), 4)))
                closed = bool(entity.closed)
                entities.append({"type": "POLYLINE", "points": pts, "closed": closed})
            elif etype == "CIRCLE":
                c = entity.dxf.center
                entities.append({
                    "type": "CIRCLE",
                    "cx": round(float(c.x), 4),
                    "cy": round(float(c.y), 4),
                    "r": round(float(entity.dxf.radius), 4),
                })
            elif etype == "ARC":
                c = entity.dxf.center
                entities.append({
                    "type": "ARC",
                    "cx": round(float(c.x), 4),
                    "cy": round(float(c.y), 4),
                    "r": round(float(entity.dxf.radius), 4),
                    "start_angle": float(entity.dxf.start_angle),
                    "end_angle": float(entity.dxf.end_angle),
                })
            elif etype == "DIMENSION":
                dim_type = int(getattr(entity.dxf, "dimtype", 0)) & 7
                entry: dict[str, Any] = {
                    "type": "DIMENSION",
                    "dim_type": dim_type,
                    "kind": "distance_other",
                }

                for name in ("defpoint", "defpoint2", "defpoint3", "defpoint4", "defpoint5", "text_midpoint"):
                    point = getattr(entity.dxf, name, None)
                    if point is None:
                        continue
                    entry[name] = [round(float(point.x), 4), round(float(point.y), 4)]

                if dim_type == 0:
                    angle = float(getattr(entity.dxf, "angle", 0.0))
                    entry["angle"] = angle
                    if "defpoint2" in entry:
                        entry["start"] = entry["defpoint2"]
                    if "defpoint3" in entry:
                        entry["end"] = entry["defpoint3"]
                    if "defpoint" in entry:
                        entry["dimension_point"] = entry["defpoint"]
                    entry["kind"] = "distance_x" if abs((angle % 180.0)) < 1e-6 else "distance_y"
                elif dim_type == 1:
                    if "defpoint2" in entry:
                        entry["start"] = entry["defpoint2"]
                    if "defpoint3" in entry:
                        entry["end"] = entry["defpoint3"]
                    if "defpoint" in entry:
                        entry["dimension_point"] = entry["defpoint"]
                    entry["kind"] = "distance_edge"
                elif dim_type == 2:
                    entry["kind"] = "distance_angle"
                elif dim_type == 3:
                    if "defpoint" in entry:
                        entry["center"] = entry["defpoint"]
                    if "defpoint4" in entry:
                        entry["point_on_circle"] = entry["defpoint4"]
                    entry["kind"] = "distance_radius"
                else:
                    entry["kind"] = "distance_other"

                entities.append(entry)
        except Exception:
            continue  # skip malformed entities

    return entities


def build_dxf_figure(entities: list[dict[str, Any]]) -> go.Figure:
    """Build a Plotly figure from DXF entities."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain", title="X"),
        yaxis=dict(title="Y"),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        showlegend=False,
        plot_bgcolor="rgba(248,250,252,1)",
    )

    for ent in entities:
        try:
            if ent["type"] == "LINE":
                fig.add_trace(go.Scatter(
                    x=[ent["x1"], ent["x2"], None],
                    y=[ent["y1"], ent["y2"], None],
                    mode="lines",
                    line=dict(color="#334155", width=1.5),
                    hoverinfo="skip",
                ))
            elif ent["type"] in ("LWPOLYLINE", "POLYLINE"):
                pts = ent["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if ent["closed"]:
                    xs.append(xs[0])
                    ys.append(ys[0])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(color="#334155", width=1.5),
                    hoverinfo="skip",
                ))
            elif ent["type"] == "CIRCLE":
                theta = np.linspace(0, 2 * math.pi, 100)
                fig.add_trace(go.Scatter(
                    x=ent["cx"] + ent["r"] * np.cos(theta),
                    y=ent["cy"] + ent["r"] * np.sin(theta),
                    mode="lines",
                    line=dict(color="#334155", width=1.5),
                    hoverinfo="skip",
                ))
            elif ent["type"] == "ARC":
                sa = math.radians(ent["start_angle"])
                ea = math.radians(ent["end_angle"])
                if ea <= sa:
                    ea += 2 * math.pi
                theta = np.linspace(sa, ea, 80)
                fig.add_trace(go.Scatter(
                    x=ent["cx"] + ent["r"] * np.cos(theta),
                    y=ent["cy"] + ent["r"] * np.sin(theta),
                    mode="lines",
                    line=dict(color="#334155", width=1.5),
                    hoverinfo="skip",
                ))
        except Exception:
            continue

    # Ensure equal aspect
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def overlay_analysis_highlights(
    fig: go.Figure,
    payload: dict[str, Any],
) -> None:
    """Overlay highlighted groups on the DXF figure from analysis payload."""
    highlight_groups = payload.get("highlight_groups") or {}
    for group_name, items in highlight_groups.items():
        color = HIGHLIGHT_COLORS.get(group_name, "#111827")
        if not isinstance(items, list):
            continue
        for item in items:
            # Points-based highlighting
            points = item.get("points") if isinstance(item, dict) else None
            if points and len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=6, color=color),
                    name=group_name,
                    legendgroup=group_name,
                    hoverinfo="text",
                    hovertext=f"{group_name}: {item.get('kind', '')}",
                ))
            # Single point highlight
            point = item.get("point") if isinstance(item, dict) else None
            if point and len(point) == 2:
                fig.add_trace(go.Scatter(
                    x=[point[0]], y=[point[1]],
                    mode="markers+text",
                    marker=dict(size=8, color=color, symbol="circle"),
                    text=item.get("name", ""),
                    textposition="top center",
                    name=group_name,
                    legendgroup=group_name,
                ))
            # Center-based highlight (circles)
            center = item.get("center") if isinstance(item, dict) else None
            if center and len(center) == 2:
                r = item.get("radius", 0)
                theta = np.linspace(0, 2 * math.pi, 50)
                fig.add_trace(go.Scatter(
                    x=center[0] + r * np.cos(theta),
                    y=center[1] + r * np.sin(theta),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=group_name,
                    legendgroup=group_name,
                ))


def overlay_initial_constraints(
    fig: go.Figure,
    entities: list[dict[str, Any]],
) -> None:
    """Overlay raw DIMENSION entities with a fixed 10-unit offset."""
    for ent in entities:
        if not isinstance(ent, dict) or ent.get("type") != "DIMENSION":
            continue

        kind = str(ent.get("kind") or "")
        if kind in ("distance_x", "distance_y"):
            start = ent.get("defpoint2") or ent.get("start")
            end = ent.get("defpoint3") or ent.get("end")
            angle = float(ent.get("angle", 0.0))
            if not start or not end:
                continue
            _overlay_axis_dimension(fig, start, end, angle, kind, "#0f20dc", 10.0)
            continue

        if kind == "distance_edge":
            start = ent.get("defpoint2") or ent.get("start")
            end = ent.get("defpoint3") or ent.get("end")
            dim_point = ent.get("defpoint") or ent.get("dimension_point")
            if not start or not end or not dim_point:
                continue
            _overlay_dimension_with_fixed_offset(fig, start, end, dim_point, "#0f20dc", 10.0)
            continue

        if kind == "distance_radius" or (
            kind == "distance_other"
            and (ent.get("center") or ent.get("defpoint"))
            and (ent.get("point_on_circle") or ent.get("defpoint4"))
        ):
            center = ent.get("center") or ent.get("defpoint")
            point_on_circle = ent.get("point_on_circle") or ent.get("defpoint4")
            if not center or not point_on_circle:
                continue
            c = _as_point(center)
            p = _as_point(point_on_circle)
            _add_line(fig, c, p, "#0f20dc", 0.8)
            _arrow_head(fig, p[0], p[1], math.atan2(c[1] - p[1], c[0] - p[0]), "#0f20dc", 2.5)
            radius = _distance(c, p)
            _add_text(fig, ((c[0] + p[0]) / 2, (c[1] + p[1]) / 2), f"{radius:.2f}".rstrip("0").rstrip("."), "#0f20dc")


def _arrow_head(
    fig: go.Figure,
    tip_x: float, tip_y: float,
    angle: float,
    color: str,
    size: float = 5.0,
) -> None:
    """Draw a filled triangle arrowhead at (tip_x, tip_y) pointing in *angle* (radians)."""
    a1 = angle + math.pi * 0.85
    a2 = angle - math.pi * 0.85
    xs = [tip_x, tip_x + math.cos(a1) * size, tip_x + math.cos(a2) * size, tip_x]
    ys = [tip_y, tip_y + math.sin(a1) * size, tip_y + math.sin(a2) * size, tip_y]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        fill="toself",
        fillcolor=color,
        line=dict(color=color, width=0.5),
        showlegend=False,
        hoverinfo="skip",
    ))


def _overlay_dimension_with_fixed_offset(
    fig: go.Figure,
    start: list[float] | tuple[float, float],
    end: list[float] | tuple[float, float],
    dimension_point: list[float] | tuple[float, float],
    color: str,
    offset_distance: float,
) -> None:
    p1 = _as_point(start)
    p2 = _as_point(end)
    _ = dimension_point  # keep the signature aligned with the analysis overlay helpers

    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx >= dy:
        offset_vec = (0.0, offset_distance)
    else:
        offset_vec = (offset_distance, 0.0)

    d1 = (p1[0] + offset_vec[0], p1[1] + offset_vec[1])
    d2 = (p2[0] + offset_vec[0], p2[1] + offset_vec[1])

    _add_line(fig, p1, d1, "#6b7280", 0.6)
    _add_line(fig, p2, d2, "#6b7280", 0.6)
    _add_line(fig, d1, d2, color, 0.8, dash="dash")

    measured = _distance(p1, p2)
    arrow_size = max(measured * 0.02, 2.0)
    _arrow_head(fig, d1[0], d1[1], math.atan2(d2[1] - d1[1], d2[0] - d1[0]), color, arrow_size)
    _arrow_head(fig, d2[0], d2[1], math.atan2(d1[1] - d2[1], d1[0] - d2[0]), color, arrow_size)
    _add_text(fig, ((d1[0] + d2[0]) / 2, (d1[1] + d2[1]) / 2), f"{measured:.2f}", color)


def _overlay_axis_dimension(
    fig: go.Figure,
    start: list[float] | tuple[float, float],
    end: list[float] | tuple[float, float],
    angle: float,
    kind: str,
    color: str,
    offset_distance: float,
) -> None:
    p1 = _as_point(start)
    p2 = _as_point(end)

    if kind == "distance_x":
        axis_end = (p2[0], p1[1])
        offset_vec = (0.0, offset_distance)
        measured = abs(axis_end[0] - p1[0])
    else:
        axis_end = (p1[0], p2[1])
        offset_vec = (offset_distance, 0.0)
        measured = abs(axis_end[1] - p1[1])

    d1 = (p1[0] + offset_vec[0], p1[1] + offset_vec[1])
    d2 = (axis_end[0] + offset_vec[0], axis_end[1] + offset_vec[1])
    _add_line(fig, p1, d1, "#6b7280", 0.6)
    _add_line(fig, p2, d2, "#6b7280", 0.6)
    _add_line(fig, d1, d2, color, 0.8, dash="dash")

    arrow_size = max(measured * 0.02, 2.0)
    _arrow_head(fig, d1[0], d1[1], math.atan2(d2[1] - d1[1], d2[0] - d1[0]), color, arrow_size)
    _arrow_head(fig, d2[0], d2[1], math.atan2(d1[1] - d2[1], d1[0] - d2[0]), color, arrow_size)

    label = f"{measured:.2f}".rstrip("0").rstrip(".")
    if not label:
        label = "0"
    _add_text(fig, ((d1[0] + d2[0]) / 2, (d1[1] + d2[1]) / 2), label, color)


def overlay_constraints(
    fig: go.Figure,
    payload: dict[str, Any],
) -> None:
    """Draw all original constraints (distance_x/y/edge/angle/radius/other)
    as red dimension arrows with arrowheads at both ends and labels above."""
    ARROW_COLOR = "#0f20dc"  # red
    HEAD_SIZE = 5.0

    constraint_keys = [
        "distance_x", "distance_y", "distance_edge",
        "distance_angle", "distance_radius", "distance_other",
    ]
    for key in constraint_keys:
        items = payload.get(key)
        if not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            dim_point = item.get("dimension_point")
            if key in ("distance_x", "distance_y") and start and end and dim_point:
                _overlay_linear_dimension(fig, start, end, dim_point, ARROW_COLOR, HEAD_SIZE)
                continue
            if key == "distance_radius" and item.get("center") and item.get("point_on_circle"):
                _overlay_radius_dimension(fig, item, ARROW_COLOR)
                continue
            if not start or not end or len(start) < 2 or len(end) < 2:
                continue

            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-8:
                continue
            angle = math.atan2(dy, dx)

            # Main red line
            fig.add_trace(go.Scatter(
                x=[x1, x2, None],
                y=[y1, y2, None],
                mode="lines",
                line=dict(color=ARROW_COLOR, width=1.8),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{key}: ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})",
            ))

            # Arrowhead at start (pointing toward end)
            _arrow_head(fig, x1, y1, angle, ARROW_COLOR, HEAD_SIZE)
            # Arrowhead at end (pointing opposite direction = angle+π)
            _arrow_head(fig, x2, y2, angle + math.pi, ARROW_COLOR, HEAD_SIZE)

            # Label above the line (perpendicular offset)
            perp_angle = angle + math.pi / 2
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = max(length * 0.04, 3.0)
            lx = mx + math.cos(perp_angle) * offset
            ly = my + math.sin(perp_angle) * offset

            dist = round(length, 2)
            label = f"{dist}"

            fig.add_trace(go.Scatter(
                x=[lx], y=[ly],
                mode="text",
                text=[label],
                textposition="middle center",
                textfont=dict(size=10, color=ARROW_COLOR, family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
            ))


def _overlay_linear_dimension(
    fig: go.Figure,
    start: list[float] | tuple[float, float],
    end: list[float] | tuple[float, float],
    dimension_point: list[float] | tuple[float, float],
    color: str,
    head_size: float,
) -> None:
    p1 = (float(start[0]), float(start[1]))
    p2 = (float(end[0]), float(end[1]))
    dp = (float(dimension_point[0]), float(dimension_point[1]))

    direction = _snap_direction(_unit_vector(p2[0] - p1[0], p2[1] - p1[1]), 5.0)
    normal = (-direction[1], direction[0])
    offset = _dot(dp[0] - p1[0], dp[1] - p1[1], normal[0], normal[1])
    d1 = (p1[0] + normal[0] * offset, p1[1] + normal[1] * offset)
    d2 = (p2[0] + normal[0] * offset, p2[1] + normal[1] * offset)
    measured = abs(_dot(p2[0] - p1[0], p2[1] - p1[1], direction[0], direction[1]))
    arrow_size = max(measured * 0.02, 2.0)
    mid = ((d1[0] + d2[0]) / 2, (d1[1] + d2[1]) / 2)
    text_point = (mid[0] + normal[0] * (arrow_size * 2.2), mid[1] + normal[1] * (arrow_size * 2.2))

    _add_line(fig, p1, d1, "#6b7280", 0.6)
    _add_line(fig, p2, d2, "#6b7280", 0.6)
    _add_line(fig, d1, d2, color, 0.8, dash="dash")
    _arrow_head(fig, d1[0], d1[1], math.atan2(d2[1] - d1[1], d2[0] - d1[0]), color, arrow_size)
    _arrow_head(fig, d2[0], d2[1], math.atan2(d1[1] - d2[1], d1[0] - d2[0]), color, arrow_size)
    fig.add_trace(go.Scatter(
        x=[text_point[0]],
        y=[text_point[1]],
        mode="text",
        text=[f"{round(measured, 2)}"],
        textposition="middle center",
        textfont=dict(size=10, color=color, family="Arial Black"),
        showlegend=False,
        hoverinfo="skip",
    ))


def _overlay_radius_dimension(
    fig: go.Figure,
    item: dict[str, Any],
    color: str,
) -> None:
    center = item.get("center")
    point = item.get("point_on_circle")
    if not center or not point or len(center) < 2 or len(point) < 2:
        return

    c = (float(center[0]), float(center[1]))
    p = (float(point[0]), float(point[1]))
    _add_line(fig, c, p, color, 0.8)
    _arrow_head(fig, p[0], p[1], math.atan2(c[1] - p[1], c[0] - p[0]), color, max(_distance(c, p) * 0.04, 2.0))


def _add_line(
    fig: go.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: float,
    dash: str | None = None,
) -> None:
    fig.add_trace(go.Scatter(
        x=[start[0], end[0], None],
        y=[start[1], end[1], None],
        mode="lines",
        line=dict(color=color, width=width, dash=dash or "solid"),
        showlegend=False,
        hoverinfo="skip",
    ))


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def _snap_direction(direction: tuple[float, float], degrees_threshold: float) -> tuple[float, float]:
    angle = math.atan2(direction[1], direction[0])
    step = math.pi / 2
    snapped = round(angle / step) * step
    threshold = math.radians(degrees_threshold)
    if abs(angle - snapped) <= threshold:
        return math.cos(snapped), math.sin(snapped)
    return direction


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _as_point(point: list[float] | tuple[float, float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _add_text(
    fig: go.Figure,
    point: tuple[float, float],
    text: str,
    color: str,
) -> None:
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode="text",
        text=[text],
        textposition="middle center",
        textfont=dict(size=10, color=color, family="Arial Black"),
        showlegend=False,
        hoverinfo="skip",
    ))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(dxf_bytes: bytes) -> dict[str, Any] | None:
    """Run the entire 6-stage analysis pipeline on DXF bytes.

    Returns the merged payload dict, or ``None`` on failure.
    """
    # Stage 1: Endpoint boards
    result = analyze_endpoint_boards_bytes(dxf_bytes)
    if isinstance(result, dict) and result.get("error"):
        st.error(f"Stage 1 failed: {result['error']}")
        return None

    # Stage 2: DXF statistical analysis
    extra = analyze_dxf_bytes(dxf_bytes)
    if isinstance(extra, dict) and extra.get("error"):
        st.error(f"Stage 2 failed: {extra['error']}")
        return None
    result = _merge_payload(result, extra)

    # Stage 3: Interpolation
    extra = analyze_interpolate_payload(result)
    if isinstance(extra, dict) and extra.get("error"):
        st.error(f"Stage 3 failed: {extra['error']}")
        return result  # partial result
    result = _merge_payload(result, extra)

    # Attach endpoint names
    from dxf_statistical import attach_endpoint_names
    result = attach_endpoint_names(result)

    # Stage 4: Projection 3-axis
    extra = analyze_projection_3_axis_payload(result)
    if isinstance(extra, dict) and extra.get("error"):
        st.error(f"Stage 4 failed: {extra['error']}")
        return result
    result = _merge_payload(result, extra)

    # Stage 5: Lack analysis
    extra = analyze_lack_payload(result)
    if isinstance(extra, dict) and extra.get("error"):
        st.error(f"Stage 5 failed: {extra['error']}")
        return result
    result = _merge_payload(result, extra)

    # Stage 6: Final dimensions
    extra = analyze_final_payload(result)
    if isinstance(extra, dict) and extra.get("error"):
        st.error(f"Stage 6 failed: {extra['error']}")
        return result
    result = _merge_payload(result, extra)

    return result


def _merge_payload(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge two analysis payloads (same logic as main.py)."""
    if not isinstance(extra, dict) or not extra:
        return base
    merged = dict(base)
    for key, value in extra.items():
        if key == "highlight_groups" and isinstance(value, dict):
            existing = merged.get("highlight_groups")
            combined: dict[str, Any] = {}
            if isinstance(existing, dict):
                combined.update(existing)
            combined.update(value)
            merged["highlight_groups"] = combined
        elif key not in merged:
            merged[key] = value
    return merged


def _normalize_frontend_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return payload

    normalized = deepcopy(payload)
    for key in ("final_distance_x", "final_distance_y", "final_distance_oz"):
        normalized[key] = _normalize_final_records(normalized.get(key), key)

    highlight_groups = normalized.get("highlight_groups")
    if isinstance(highlight_groups, dict):
        for key in ("final_distance_x", "final_distance_y", "final_distance_oz"):
            if key in highlight_groups:
                highlight_groups[key] = _normalize_final_records(highlight_groups.get(key), key)

    return normalized


def _normalize_final_records(
    records: list[dict[str, Any]] | None,
    group_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return records or []

    normalized: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        copy_item = dict(item)
        points = copy_item.get("points")
        if isinstance(points, list) and len(points) >= 2:
            p1 = list(points[0]) if isinstance(points[0], (list, tuple)) else None
            p2 = list(points[1]) if isinstance(points[1], (list, tuple)) else None
            if p1 and p2 and len(p1) >= 2 and len(p2) >= 2:
                if group_name in ("final_distance_x", "final_distance_oz") or copy_item.get("kind") in ("distance_x", "distance_oz"):
                    p2[1] = p1[1]
                elif group_name == "final_distance_y" or copy_item.get("kind") == "distance_y":
                    p2[0] = p1[0]
                copy_item["points"] = [p1[:2], p2[:2]]
        normalized.append(copy_item)
    return normalized


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _to_df(data: list[Any] | None, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert a list of dicts to a DataFrame, or return empty DataFrame."""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if columns:
        # Only keep requested columns that exist
        existing = [c for c in columns if c in df.columns]
        df = df[existing]
    return df


def _points_to_df(points: list[dict[str, Any]] | None, label: str = "P") -> pd.DataFrame:
    """Convert endpoint board to a DataFrame with X, Y columns."""
    if not points:
        return pd.DataFrame()
    rows = []
    for item in points:
        pt = item.get("point")
        if pt and len(pt) == 2:
            rows.append({
                "Name": item.get("name", ""),
                "X": pt[0],
                "Y": pt[1],
            })
    return pd.DataFrame(rows)


def _constraints_to_df(constraints: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert constraint list to a readable DataFrame."""
    if not constraints:
        return pd.DataFrame()
    rows = []
    for item in constraints:
        start = item.get("start") or [None, None]
        end = item.get("end") or [None, None]
        rows.append({
            "Kind": item.get("kind", ""),
            "Start X": start[0] if len(start) > 0 else None,
            "Start Y": start[1] if len(start) > 1 else None,
            "End X": end[0] if len(end) > 0 else None,
            "End Y": end[1] if len(end) > 1 else None,
            "Angle": item.get("angle"),
            "Endpoints": ", ".join(item.get("endpoint_names", [])),
        })
    return pd.DataFrame(rows)


def _circles_to_df(circles: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert circle board to a DataFrame."""
    if not circles:
        return pd.DataFrame()
    rows = []
    for item in circles:
        center = item.get("center") or [None, None]
        rows.append({
            "Center X": center[0] if len(center) > 0 else None,
            "Center Y": center[1] if len(center) > 1 else None,
            "Radius": item.get("radius"),
        })
    return pd.DataFrame(rows)


def _projection_to_df(proj: list[list[float]] | None) -> pd.DataFrame:
    """Convert projection array to DataFrame."""
    if not proj:
        return pd.DataFrame()
    rows = []
    for item in proj:
        if len(item) >= 2:
            rows.append({"Coordinate": item[0], "Offset": item[1]})
    return pd.DataFrame(rows)


def _constraint_triplets_to_df(triplets: list[list[float]] | None) -> pd.DataFrame:
    """Convert constraint triplets [c1, c2, w] to DataFrame."""
    if not triplets:
        return pd.DataFrame()
    rows = []
    for item in triplets:
        if len(item) >= 3:
            rows.append({"From": item[0], "To": item[1], "Weight": item[2]})
        elif len(item) >= 2:
            rows.append({"From": item[0], "To": item[1], "Weight": ""})
    return pd.DataFrame(rows)


def _offset_to_df(offset: list[list[float]] | None) -> pd.DataFrame:
    """Convert offset array to DataFrame."""
    if not offset:
        return pd.DataFrame()
    rows = []
    for item in offset:
        if len(item) >= 2:
            rows.append({"Coordinate": item[0], "Offset": item[1]})
    return pd.DataFrame(rows)


def _final_to_df(final: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert final dimension list to DataFrame."""
    if not final:
        return pd.DataFrame()
    rows = []
    for item in final:
        start = item.get("start") or [None, None]
        end = item.get("end") or [None, None]
        rows.append({
            "Kind": item.get("kind", ""),
            "Start X": start[0],
            "Start Y": start[1],
            "End X": end[0],
            "End Y": end[1],
            "Endpoints": ", ".join(item.get("endpoint_names", [])),
        })
    return pd.DataFrame(rows)


def _highlight_groups_to_df(groups: dict[str, Any] | None) -> pd.DataFrame:
    """Convert highlight groups to summary DataFrame."""
    if not groups:
        return pd.DataFrame()
    rows = []
    for name, items in groups.items():
        if isinstance(items, list):
            rows.append({"Group": name, "Count": len(items)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Projection visualization
# ---------------------------------------------------------------------------


def build_projection_figure(
    title: str,
    projection: list[list[float]] | None,
    constraints: list[list[float]] | None,
    color: str = "#2563eb",
) -> go.Figure:
    """Build a Plotly figure for a single projection axis."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Coordinate",
        yaxis_title="Value",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        showlegend=True,
        height=350,
    )

    # Plot projection points
    proj_df = _projection_to_df(projection)
    if not proj_df.empty:
        fig.add_trace(go.Scatter(
            x=proj_df["Coordinate"],
            y=proj_df["Offset"],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            name="Projection points",
        ))

    # Plot constraints as lines
    con_df = _constraint_triplets_to_df(constraints)
    if not con_df.empty:
        for _, row in con_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["From"], row["To"], None],
                y=[0, 0, None],
                mode="lines",
                line=dict(color="#f59e0b", width=2),
                name="Constraints" if _ == 0 else None,
                legendgroup="constraints",
                hovertext=f"w={row['Weight']}",
                hoverinfo="text",
            ))

    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📐 DXF Checker Tool - Solution 1")
st.sidebar.markdown("Tải lên file DXF để phân tích")

uploaded_file = st.sidebar.file_uploader(
    "Chọn file .dxf",
    type=["dxf"],
    label_visibility="collapsed",
)

current_signature = None
if uploaded_file is not None:
    dxf_bytes = uploaded_file.getvalue()
    current_signature = hashlib.sha1(dxf_bytes).hexdigest()
    if st.session_state.get("source_signature") != current_signature:
        st.session_state.pop("payload", None)
        st.session_state["source_signature"] = current_signature
    st.sidebar.success(f"✅ **{uploaded_file.name}** ({len(dxf_bytes):,} bytes)")
    run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
else:
    dxf_bytes = None
    run_button = False
    st.sidebar.info("📄 Chưa có file nào được chọn")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📐 DXF Checker Tool - Solution 1")
st.markdown("Công cụ kiểm tra bản vẽ DXF — phát hiện kích thước thiếu")

if dxf_bytes is None:
    st.info("👈 Tải lên file DXF từ sidebar để bắt đầu")
    st.stop()

# --- Read DXF entities for rendering (always available) ---
dxf_entities = read_dxf_entities(dxf_bytes)
entity_count = len(dxf_entities)

st.sidebar.markdown("---")
st.sidebar.metric("Entities detected", entity_count)

# --- Run pipeline ---
if run_button:
    with st.spinner("🔄 Đang phân tích DXF..."):
        payload = run_pipeline(dxf_bytes)
        if payload is None:
            st.error("Phân tích thất bại. Vui lòng kiểm tra file DXF.")
            st.stop()
        payload = _normalize_frontend_payload(payload)
        st.session_state["payload"] = payload
        st.session_state["dxf_entities"] = dxf_entities
        if current_signature is not None:
            st.session_state["payload_signature"] = current_signature
    st.rerun()

payload = st.session_state.get("payload")
analysis_ready = payload is not None
if analysis_ready:
    payload = _normalize_frontend_payload(payload) or {}
    st.session_state["payload"] = payload
else:
    payload = {}
if not analysis_ready:
    st.info("👆 Bấm **Run Analysis** để bắt đầu phân tích")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Bản vẽ",
    "📊 Dữ liệu",
    "🔄 Nội suy",
    "📈 Hình chiếu",
    "🔍 Thiếu hụt",
    "✅ Kết quả",
])

# ===== Tab 1: Drawing =====
with tab1:
    st.subheader("Bản vẽ DXF")

    cc1, cc2 = st.columns(2)
    with cc1:
        show_highlights = st.checkbox(
            "🔦 Highlight analysis",
            value=True,
            key="tab1_highlights",
            disabled=not analysis_ready,
        )
    with cc2:
        show_constraints = st.checkbox(
            "📏 Constraint gốc (mũi tên đỏ)",
            value=True,
            key="tab1_constraints",
        )
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = build_dxf_figure(dxf_entities)
        if analysis_ready and show_highlights:
            overlay_analysis_highlights(fig, payload)
        if show_constraints:
            overlay_initial_constraints(fig, dxf_entities)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Thông tin")
        st.metric("Entities", entity_count)
        st.metric("Endpoints", len(payload.get("endpoint_board") or []))
        st.metric("Circles", len(payload.get("circle_board") or []))
        st.metric("Constraints", sum(
            len(payload.get(k) or [])
            for k in ("distance_x", "distance_y", "distance_edge",
                      "distance_angle", "distance_radius", "distance_other")
        ))

        # Highlight legend
        st.markdown("### Chú thích")
        groups = payload.get("highlight_groups") or {}
        for name, items in groups.items():
            color = HIGHLIGHT_COLORS.get(name, "#111827")
            if isinstance(items, list):
                st.markdown(
                    f'<span style="color:{color};font-weight:bold">●</span> '
                    f'{name} ({len(items)})',
                    unsafe_allow_html=True,
                )

# ===== Tab 2: Data =====
with tab2:
    st.subheader("Endpoints & Constraints")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Endpoint Board")
        df_ep = _points_to_df(payload.get("endpoint_board"))
        if not df_ep.empty:
            st.dataframe(df_ep, use_container_width=True, hide_index=True)
        else:
            st.info("Không có endpoint")

    with col_b:
        st.markdown("### Circle Board")
        df_cir = _circles_to_df(payload.get("circle_board"))
        if not df_cir.empty:
            st.dataframe(df_cir, use_container_width=True, hide_index=True)
        else:
            st.info("Không có circle")

    st.markdown("---")
    st.markdown("### Constraints")
    all_constraints = []
    for kind in ("distance_x", "distance_y", "distance_edge", "distance_angle", "distance_radius", "distance_other"):
        items = payload.get(kind) or []
        all_constraints.extend(items)

    df_con = _constraints_to_df(all_constraints)
    if not df_con.empty:
        # Count by kind
        kind_counts = df_con["Kind"].value_counts().reset_index()
        kind_counts.columns = ["Kind", "Count"]
        st.dataframe(kind_counts, use_container_width=True, hide_index=True)
        with st.expander("📋 Xem chi tiết tất cả constraints"):
            st.dataframe(df_con, use_container_width=True, hide_index=True)
    else:
        st.info("Không có constraint nào")

# ===== Tab 3: Interpolation =====
with tab3:
    st.subheader("Kết quả nội suy")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### full_distance_x")
        df_fdx = _points_to_df(
            [{"point": k, "name": v} for k, v in (payload.get("full_distance_x") or {}).items()],
        ) if isinstance(payload.get("full_distance_x"), dict) else pd.DataFrame()
        # Handle dict format: full_distance_x is dict of {point_str: info}
        raw_fdx = payload.get("full_distance_x") or {}
        if isinstance(raw_fdx, dict) and raw_fdx:
            rows = []
            for point_key, info in raw_fdx.items():
                if isinstance(point_key, (list, tuple)) and len(point_key) == 2:
                    rows.append({"X": point_key[0], "Y": point_key[1]})
            df_fdx = pd.DataFrame(rows)
        if not df_fdx.empty:
            st.dataframe(df_fdx, use_container_width=True, hide_index=True)
            st.metric("Số điểm", len(df_fdx))
        else:
            st.info("Không có dữ liệu")

    with col_b:
        st.markdown("### full_distance_y")
        raw_fdy = payload.get("full_distance_y") or {}
        if isinstance(raw_fdy, dict) and raw_fdy:
            rows = []
            for point_key, info in raw_fdy.items():
                if isinstance(point_key, (list, tuple)) and len(point_key) == 2:
                    rows.append({"X": point_key[0], "Y": point_key[1]})
            df_fdy = pd.DataFrame(rows)
        else:
            df_fdy = pd.DataFrame()
        if not df_fdy.empty:
            st.dataframe(df_fdy, use_container_width=True, hide_index=True)
            st.metric("Số điểm", len(df_fdy))
        else:
            st.info("Không có dữ liệu")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### Side")
        st.metric("Points", len(payload.get("side_points") or []))
        st.metric("Ways", len(payload.get("side_way") or []))
        st.metric("Constraints", len(payload.get("side_constraints") or []))
        with st.expander("Side points"):
            df_sp = _points_to_df(payload.get("side_points"))
            if not df_sp.empty:
                st.dataframe(df_sp, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### Top")
        st.metric("Points", len(payload.get("top_points") or []))
        st.metric("Ways", len(payload.get("top_way") or []))
        st.metric("Constraints", len(payload.get("top_constraints") or []))
        with st.expander("Top points"):
            df_tp = _points_to_df(payload.get("top_points"))
            if not df_tp.empty:
                st.dataframe(df_tp, use_container_width=True, hide_index=True)

    with col_c:
        st.markdown("### Front")
        st.metric("Points", len(payload.get("font_points") or payload.get("front_points") or []))
        st.metric("Ways", len(payload.get("font_way") or []))
        st.metric("Constraints", len(payload.get("font_constraints") or []))
        with st.expander("Front points"):
            df_fp = _points_to_df(payload.get("font_points") or payload.get("front_points"))
            if not df_fp.empty:
                st.dataframe(df_fp, use_container_width=True, hide_index=True)

# ===== Tab 4: Projection =====
with tab4:
    st.subheader("3-Axis Projection")

    delta_oz = payload.get("delta_oz", 0)
    st.metric("Δ₀₂ (delta_oz)", f"{delta_oz:.4f}")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_ox = build_projection_figure(
            "OX Projection (Top → X axis)",
            payload.get("ox_projection"),
            payload.get("ox_projection_constraints"),
            color="#2563eb",
        )
        st.plotly_chart(fig_ox, use_container_width=True)

    with col_b:
        fig_oy = build_projection_figure(
            "OY Projection (Front → Y axis)",
            payload.get("oy_projection"),
            payload.get("oy_projection_constraints"),
            color="#16a34a",
        )
        st.plotly_chart(fig_oy, use_container_width=True)

    fig_oz = build_projection_figure(
        "OZ Projection (Side → X axis)",
        payload.get("oz_projection"),
        payload.get("oz_projection_constraints"),
        color="#dc2626",
    )
    st.plotly_chart(fig_oz, use_container_width=True)

    with st.expander("📋 OZ Projection Merge"):
        df_ozm = _constraint_triplets_to_df(payload.get("oz_projection_merge"))
        if not df_ozm.empty:
            st.dataframe(df_ozm, use_container_width=True, hide_index=True)

# ===== Tab 5: Lack Analysis =====
with tab5:
    st.subheader("Missing Constraints (Floyd-Warshall)")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### OX Lack")
        df_oxl = _constraint_triplets_to_df(payload.get("ox_lack_constraints"))
        if not df_oxl.empty:
            st.dataframe(df_oxl, use_container_width=True, hide_index=True)
        else:
            st.info("✅ Không thiếu")
        st.markdown("**OX Offset**")
        df_oxo = _offset_to_df(payload.get("ox_projection_offset"))
        if not df_oxo.empty:
            st.dataframe(df_oxo, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### OY Lack")
        df_oyl = _constraint_triplets_to_df(payload.get("oy_lack_constraints"))
        if not df_oyl.empty:
            st.dataframe(df_oyl, use_container_width=True, hide_index=True)
        else:
            st.info("✅ Không thiếu")
        st.markdown("**OY Offset**")
        df_oyo = _offset_to_df(payload.get("oy_projection_offset"))
        if not df_oyo.empty:
            st.dataframe(df_oyo, use_container_width=True, hide_index=True)

    with col_c:
        st.markdown("### OZ Lack")
        df_ozl = _constraint_triplets_to_df(payload.get("oz_lack_constraints"))
        if not df_ozl.empty:
            st.dataframe(df_ozl, use_container_width=True, hide_index=True)
        else:
            st.info("✅ Không thiếu")
        st.markdown("**OZ Offset**")
        df_ozo = _offset_to_df(payload.get("oz_projection_offset"))
        if not df_ozo.empty:
            st.dataframe(df_ozo, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Missing Radius / Other")
    df_lack_other = _constraints_to_df(payload.get("lack_distance_other"))
    if not df_lack_other.empty:
        st.dataframe(df_lack_other, use_container_width=True, hide_index=True)
    else:
        st.info("✅ Không thiếu radius/other")

    # Highlight groups summary
    st.markdown("### Highlight Groups Summary")
    df_hg = _highlight_groups_to_df(payload.get("highlight_groups"))
    if not df_hg.empty:
        st.dataframe(df_hg, use_container_width=True, hide_index=True)

# ===== Tab 6: Final Results =====
with tab6:
    st.subheader("Kết quả cuối cùng")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### final_distance_x")
        df_fx = _final_to_df(payload.get("final_distance_x"))
        if not df_fx.empty:
            st.dataframe(df_fx, use_container_width=True, hide_index=True)
            st.metric("Số kích thước", len(df_fx))
        else:
            st.info("Không có kích thước X nào")

    with col_b:
        st.markdown("### final_distance_y")
        df_fy = _final_to_df(payload.get("final_distance_y"))
        if not df_fy.empty:
            st.dataframe(df_fy, use_container_width=True, hide_index=True)
            st.metric("Số kích thước", len(df_fy))
        else:
            st.info("Không có kích thước Y nào")

    with col_c:
        st.markdown("### final_distance_oz")
        df_foz = _final_to_df(payload.get("final_distance_oz"))
        if not df_foz.empty:
            st.dataframe(df_foz, use_container_width=True, hide_index=True)
            st.metric("Số kích thước", len(df_foz))
        else:
            st.info("Không có kích thước OZ nào")

    # Summary
    st.markdown("---")
    st.markdown("### Tổng quan")
    total_final = (
        len(payload.get("final_distance_x") or [])
        + len(payload.get("final_distance_y") or [])
        + len(payload.get("final_distance_oz") or [])
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Tổng kích thước cuối", total_final)
    col_s2.metric("Endpoints", len(payload.get("endpoint_board") or []))
    col_s3.metric("Delta OZ", f"{delta_oz:.4f}")

    # Raw JSON for debugging
    with st.expander("📄 Raw JSON payload"):
        st.json(payload)
