"""Page 3 — Results: review ranked centerlines, 3D views, disqualify, export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

import cache

dash.register_page(__name__, path="/results", title="NeuralPipe — Results")

layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H4("Route Candidates", className="mt-3 mb-1")),
        dbc.Col(
            dbc.Button("Export Cloud (.ply)", id="export-cloud-btn",
                       color="secondary", outline=True, size="sm",
                       className="mt-3 float-end"),
            width="auto",
        ),
    ]),
    dbc.Row(dbc.Col(html.P(
        "Ranked centerlines from the last routing run. "
        "Disqualify routes with a reason to build the feedback record.",
        className="text-muted small mb-1",
    ))),
    html.Div(id="export-cloud-status", className="small text-info mb-3"),
    html.Div(id="results-body"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route_fig(c: dict, start, end, store: dict) -> go.Figure:
    fig = go.Figure()

    pts = cache.get_cloud()
    if pts is not None:
        n    = min(30_000, len(pts))
        prev = pts[np.random.default_rng().choice(len(pts), n, replace=False, shuffle=False)]
        fig.add_trace(go.Scatter3d(
            x=prev[:, 0], y=prev[:, 1], z=prev[:, 2],
            mode="markers",
            marker=dict(size=2, color=prev[:, 2], colorscale="Viridis", opacity=0.5),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            name="Cloud",
        ))

    wps = c["waypoints"]
    fig.add_trace(go.Scatter3d(
        x=[w[0] for w in wps], y=[w[1] for w in wps], z=[w[2] for w in wps],
        mode="lines",
        line=dict(color="#3498db", width=5),
        name="Route",
        hovertemplate="X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
    ))
    if start:
        fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode="markers+text", text=["START"], textposition="top center",
            marker=dict(size=10, color="#ff00ff"), name="START",
        ))
    if end:
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode="markers+text", text=["END"], textposition="top center",
            marker=dict(size=10, color="#e74c3c", symbol="diamond"), name="END",
        ))

    bmin = store.get("cloud_bmin")
    bmax = store.get("cloud_bmax")
    if bmin and bmax:
        bmn     = np.asarray(bmin, dtype=np.float64)
        bmx     = np.asarray(bmax, dtype=np.float64)
        extents = bmx - bmn
        max_ext = float(extents.max()) or 1.0
        ratios  = (extents / max_ext).clip(0.05, 1.0)
        scene = dict(
            xaxis=dict(title="X (m)", range=[float(bmn[0]), float(bmx[0])]),
            yaxis=dict(title="Y (m)", range=[float(bmn[1]), float(bmx[1])]),
            zaxis=dict(title="Z (m)", range=[float(bmn[2]), float(bmx[2])]),
            aspectmode="manual",
            aspectratio=dict(x=float(ratios[0]), y=float(ratios[1]), z=float(ratios[2])),
            bgcolor="rgba(10,10,20,1)",
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.6),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)),
        )
    else:
        scene = dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            bgcolor="rgba(10,10,20,1)",
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.6),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)),
        )

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,20,30,0.8)", font=dict(color="#ccc")),
    )
    return fig


def _write_ply(pts: np.ndarray, out_path: Path) -> None:
    """Write binary little-endian PLY point cloud."""
    pts32 = pts.astype(np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts32)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(pts32.tobytes())


def _write_dxf(waypoints: list, route_id: str, out_path: Path) -> None:
    """Write a 3D polyline DXF (R12 format, no external libraries needed)."""
    lines = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        "0", "POLYLINE",
        "8", route_id,
        "66", "1",
        "70", "8",
        "10", "0.0", "20", "0.0", "30", "0.0",
    ]
    for wp in waypoints:
        lines += [
            "0", "VERTEX",
            "8", route_id,
            "10", f"{wp[0]:.6f}",
            "20", f"{wp[1]:.6f}",
            "30", f"{wp[2]:.6f}",
            "70", "32",
        ]
    lines += ["0", "SEQEND", "8", route_id, "0", "ENDSEC", "0", "EOF"]
    out_path.write_text("\n".join(lines), encoding="ascii")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("results-body", "children"),
    Input("store", "data"),
)
def build_results(store):
    store = store or {}
    candidates = store.get("candidates")
    if not candidates:
        return dbc.Alert("No candidates yet. Go to Route and generate routes.", color="secondary")

    start = store.get("start")
    end   = store.get("end")
    status_color = {"VALID": "success", "FLAGGED": "warning", "FAILED": "danger"}
    cards = []

    for i, c in enumerate(candidates):
        color = status_color.get(c["status"], "secondary")
        fig   = _route_fig(c, start, end, store)

        cards.append(dbc.Card(className="mb-4", children=dbc.CardBody([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H6(f"#{i+1}  {c['route_id']}", className="mb-1"),
                    dbc.Badge(c["status"], color=color, className="me-2"),
                    html.Span(f"Score: {c['score']:.2f}", className="small text-muted"),
                ], md=6),
                dbc.Col(
                    html.Span(
                        f"Length: {c['length_m']:.1f} m  |  "
                        f"Elbows: {c['num_elbows']}  |  "
                        f"Supports: {c['num_supports']}",
                        className="small",
                    ), md=6,
                ),
            ], className="mb-2"),

            # Flags
            *([dbc.Alert(f, color="warning", className="small py-1 mb-1")
               for f in c.get("flags", [])]),

            # Inline 3D route view
            dcc.Graph(
                figure=fig,
                style={"height": "320px"},
                config={"scrollZoom": True, "displayModeBar": True},
            ),

            # Action buttons
            dbc.Row([
                dbc.Col(
                    dbc.Button("Export JSON",
                               id={"type": "export-btn", "index": c["route_id"]},
                               color="secondary", outline=True, size="sm",
                               className="w-100"),
                    md=3,
                ),
                dbc.Col(
                    dbc.Button("Export .dxf",
                               id={"type": "export-dxf-btn", "index": c["route_id"]},
                               color="info", outline=True, size="sm",
                               className="w-100"),
                    md=3,
                ),
                dcc.Download(id={"type": "route-dl",     "index": c["route_id"]}),
                dcc.Download(id={"type": "route-dxf-dl", "index": c["route_id"]}),
            ], className="g-2 mt-2 mb-3"),

            # DQ row
            dbc.Row([
                dbc.Col(dbc.Textarea(
                    id={"type": "dq-text", "index": c["route_id"]},
                    placeholder="Reason for disqualification (plain English)…",
                    rows=2,
                ), md=9),
                dbc.Col(dbc.Button(
                    "Disqualify",
                    id={"type": "dq-btn", "index": c["route_id"]},
                    color="danger", outline=True, size="sm", className="w-100 mt-1",
                ), md=3),
            ], className="g-2 mt-1"),
            html.Div(id={"type": "dq-result", "index": c["route_id"]}, className="small mt-1"),
        ])))

    return cards


@callback(
    Output("export-cloud-status", "children"),
    Input("export-cloud-btn",     "n_clicks"),
    State("store", "data"),
    prevent_initial_call=True,
)
def export_cloud(n_clicks, store):
    pts = cache.get_cloud()
    if pts is None:
        return "No cloud loaded."
    store = store or {}
    cloud_path = store.get("cloud_path", "")
    stem = Path(cloud_path).stem if cloud_path else "cloud"
    out = Path(cloud_path).parent / f"{stem}.ply" if cloud_path else Path("cloud.ply")
    try:
        _write_ply(pts, out)
        size_mb = out.stat().st_size / 1e6
        return f"Saved {len(pts):,} pts → {out}  ({size_mb:.0f} MB)  — open in CloudCompare"
    except Exception as e:
        return f"Error: {e}"


@callback(
    Output({"type": "route-dl",   "index": dash.MATCH}, "data"),
    Input({"type":  "export-btn", "index": dash.MATCH}, "n_clicks"),
    State({"type":  "export-btn", "index": dash.MATCH}, "id"),
    State("store", "data"),
    prevent_initial_call=True,
)
def export_route_json(n_clicks, btn_id, store):
    if not n_clicks:
        return dash.no_update
    store = store or {}
    route_id  = btn_id["index"]
    candidate = next(
        (c for c in store.get("candidates", []) if c["route_id"] == route_id), None
    )
    if not candidate:
        return dash.no_update
    return dict(content=json.dumps(candidate, indent=2), filename=f"{route_id}.json")


@callback(
    Output({"type": "route-dxf-dl",  "index": dash.MATCH}, "data"),
    Input({"type": "export-dxf-btn", "index": dash.MATCH}, "n_clicks"),
    State({"type": "export-dxf-btn", "index": dash.MATCH}, "id"),
    State("store", "data"),
    prevent_initial_call=True,
)
def export_route_dxf(n_clicks, btn_id, store):
    if not n_clicks:
        return dash.no_update
    store     = store or {}
    route_id  = btn_id["index"]
    candidate = next(
        (c for c in store.get("candidates", []) if c["route_id"] == route_id), None
    )
    if not candidate:
        return dash.no_update
    import tempfile, os
    tmp = Path(tempfile.mktemp(suffix=".dxf"))
    _write_dxf(candidate["waypoints"], route_id, tmp)
    content = tmp.read_text(encoding="ascii")
    tmp.unlink(missing_ok=True)
    return dict(content=content, filename=f"{route_id}.dxf")


@callback(
    Output({"type": "dq-result", "index": dash.MATCH}, "children"),
    Input({"type":  "dq-btn",    "index": dash.MATCH}, "n_clicks"),
    State({"type":  "dq-text",   "index": dash.MATCH}, "value"),
    State({"type":  "dq-btn",    "index": dash.MATCH}, "id"),
    prevent_initial_call=True,
)
def disqualify(n_clicks, reason, btn_id):
    if not n_clicks or not reason or not reason.strip():
        return dbc.Alert("Enter a reason first.", color="warning", className="py-1")
    route_id = btn_id["index"]
    try:
        record = cache.get_agent().disqualify(route_id=route_id, reason=reason.strip())
        return dbc.Alert(f"DQ recorded: {record.dq_id[:8]}… ({record.dq_category})",
                         color="success", className="py-1")
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="py-1")
