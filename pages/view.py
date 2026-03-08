"""Route + Cloud viewer — opened in a new tab from the Results page."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

import cache

dash.register_page(__name__, path="/view", title="NeuralPipe — View")

layout = dbc.Container(fluid=True, style={"padding": "0"}, children=[
    dcc.Location(id="view-url"),
    dbc.Row(dbc.Col(html.H5(id="view-title", className="mt-3 mb-2 px-3"))),
    dcc.Graph(
        id="view-graph",
        style={"height": "calc(100vh - 90px)", "background": "#0a0a14"},
        config={"scrollZoom": True, "displayModeBar": True},
    ),
])


def _empty_3d() -> go.Figure:
    """Blank figure that is explicitly 3D so Plotly doesn't fall back to 2D."""
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            bgcolor="rgba(10,10,20,1)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@callback(
    Output("view-graph", "figure"),
    Output("view-title", "children"),
    Input("view-url",    "search"),
    Input("store",       "data"),   # re-fires once store hydrates in the new tab
)
def build_view(search, store):
    store      = store or {}
    candidates = store.get("candidates", [])
    start      = store.get("start")
    end        = store.get("end")

    # Parse ?i=N from URL query string
    idx = 0
    if search:
        for part in search.lstrip("?").split("&"):
            if part.startswith("i="):
                try:
                    idx = int(part.split("=", 1)[1])
                except ValueError:
                    pass

    if not candidates or idx >= len(candidates):
        return _empty_3d(), "Waiting for route data…"

    c   = candidates[idx]
    fig = go.Figure()

    # Point cloud
    pts = cache.get_cloud()
    if pts is not None:
        n    = min(50_000, len(pts))
        prev = pts[np.random.default_rng().choice(len(pts), n, replace=False, shuffle=False)]
        fig.add_trace(go.Scatter3d(
            x=prev[:, 0], y=prev[:, 1], z=prev[:, 2],
            mode="markers",
            marker=dict(size=2, color=prev[:, 2], colorscale="Viridis", opacity=0.5),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            name="Cloud",
        ))

    # Route line
    wps = c["waypoints"]
    fig.add_trace(go.Scatter3d(
        x=[w[0] for w in wps], y=[w[1] for w in wps], z=[w[2] for w in wps],
        mode="lines",
        line=dict(color="#3498db", width=5),
        name=f"#{idx+1} {c['route_id']}",
        hovertemplate="X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
    ))

    # Start / end markers
    if start:
        fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode="markers+text", text=["START"], textposition="top center",
            marker=dict(size=10, color="#2ecc71"), name="START",
        ))
    if end:
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode="markers+text", text=["END"], textposition="top center",
            marker=dict(size=10, color="#e74c3c", symbol="diamond"), name="END",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            bgcolor="rgba(10,10,20,1)",
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.6),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,20,30,0.8)", font=dict(color="#ccc")),
    )

    title = (f"#{idx+1}  {c['route_id']}  —  "
             f"Score: {c['score']:.2f}  |  {c['length_m']:.1f} m  |  {c['num_elbows']} elbows")
    return fig, title
