"""Route — load point cloud, pick start/end, generate pipe route candidates."""
from __future__ import annotations

import threading
import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, clientside_callback, dcc, html

import cache
from neuralpipe import NumpyVoxelGrid, downsample_to_npy
from neuralpipe.geometry.voxel_grid import load_e57_points, load_pts_points

dash.register_page(__name__, path="/", title="NeuralPipe — Route")

# ---------------------------------------------------------------------------
# Shared background-thread state
# ---------------------------------------------------------------------------

_browse_result: dict = {"path": None, "ready": False}
_browse_gates_result: dict = {"path": None, "ready": False}
_load_state: dict = {"status": "", "progress": 0, "done": False,
                     "fig": None, "store": None, "error": None}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H4("Route", className="mt-3 mb-1"))),

    # --- Cloud load section ---
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="cloud-path", placeholder="C:/scans/cloud.npy",
                          debounce=False, persistence=True, persistence_type="session"),
                dbc.Button("Browse…",    id="cloud-browse-btn",  color="secondary", n_clicks=0),
                dbc.Button("Cache .npy", id="cloud-cache-btn",   color="warning",   n_clicks=0),
                dbc.Button("Load",       id="cloud-load-btn",    color="primary",   n_clicks=0),
                dbc.InputGroupText(html.Span("", id="load-timer",
                                             style={"fontFamily": "monospace",
                                                    "minWidth": "52px"})),
            ]),
            dbc.Checklist(
                id="cloud-opts",
                options=[
                    {"label": "Pre-downsampled .npy",         "value": "npy"},
                    {"label": "Skip downsampling (raw scan)",  "value": "skip-ds"},
                    {"label": "Auto-align to axes (PCA)",      "value": "align"},
                ],
                value=["npy"],
                inline=True,
                className="mt-2 small",
                persistence=True,
                persistence_type="session",
            ),
            dbc.Row([
                dbc.Col(html.Label("Voxel cell (mm)", className="small mt-2"), width="auto"),
                dbc.Col(dcc.Slider(id="cell-mm", min=0.1, max=5, step=None, value=3,
                                   marks={v: str(v) for v in [0.1, 0.5, 1, 3, 5]},
                                   className="mt-2"), width=6),
            ], align="center"),
        ], md=8),
    ], className="mb-3"),

    dcc.Interval(id="browse-poll",       interval=200, disabled=True, max_intervals=0),
    dcc.Interval(id="load-poll",         interval=400, disabled=True, max_intervals=0),
    dcc.Interval(id="gates-browse-poll", interval=200, disabled=True, max_intervals=0),
    dcc.Store(id="gates-store"),

    html.Div(id="cloud-status", className="small text-info mb-2"),
    dbc.Progress(id="cloud-progress", value=0, animated=True, striped=True,
                 style={"height": "6px"}, className="mb-3"),

    # --- Gates section ---
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="gates-path", placeholder="C:/results/gates.json",
                          debounce=False, persistence=True, persistence_type="session"),
                dbc.Button("Browse Gates…", id="gates-browse-btn", color="info",
                           outline=True, n_clicks=0),
            ]),
        ], md=6),
        dbc.Col(html.Div(id="gates-status",
                         className="small text-info d-flex align-items-center h-100"), md=6),
    ], className="mb-3"),

    # --- Pick mode + 3D graph ---
    dcc.Store(id="pick-mode", data=None),

    dbc.Row([
        dbc.Col(dbc.Button("Set Start", id="btn-set-start", n_clicks=0,
                           color="success", outline=True, size="sm", className="w-100"),
                xs=6, sm=3, md=2),
        dbc.Col(dbc.Button("Set End",   id="btn-set-end",   n_clicks=0,
                           color="danger",  outline=True, size="sm", className="w-100"),
                xs=6, sm=3, md=2),
        dbc.Col(html.Div(id="pick-hint",
                         className="text-muted small d-flex align-items-center h-100"),
                className="ps-2"),
    ], className="mb-2 g-2"),

    dcc.Graph(
        id="route-graph",
        style={"height": "70vh", "background": "#0a0a14"},
        config={"scrollZoom": True, "displayModeBar": True},
        figure={
            "data": [],
            "layout": {
                "scene": {
                    "xaxis": {"title": "X (m)"},
                    "yaxis": {"title": "Y (m)"},
                    "zaxis": {"title": "Z (m)"},
                    "bgcolor": "rgba(10,10,20,1)",
                },
                "paper_bgcolor": "rgba(0,0,0,0)",
                "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            },
        },
    ),

    # --- Coordinate inputs ---
    dbc.Row([
        dbc.Col([
            html.Label("Start (m)", className="small fw-bold", style={"color": "#ff00ff"}),
            dbc.Row([
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("X"), dbc.Input(id="start-x", type="number", step=0.1)]), md=4),
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("Y"), dbc.Input(id="start-y", type="number", step=0.1)]), md=4),
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("Z"), dbc.Input(id="start-z", type="number", step=0.1)]), md=4),
            ], className="g-2"),
        ], md=6),
        dbc.Col([
            html.Label("End (m)", className="small fw-bold text-danger"),
            dbc.Row([
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("X"), dbc.Input(id="end-x", type="number", step=0.1)]), md=4),
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("Y"), dbc.Input(id="end-y", type="number", step=0.1)]), md=4),
                dbc.Col(dbc.InputGroup([dbc.InputGroupText("Z"), dbc.Input(id="end-z", type="number", step=0.1)]), md=4),
            ], className="g-2"),
        ], md=6),
    ], className="mt-3 mb-2"),

    dbc.Row(dbc.Col(
        dbc.Button("Apply Coordinates", id="apply-coords-btn", color="secondary",
                   outline=True, size="sm", n_clicks=0),
    ), className="mb-4"),

    html.Hr(),

    # --- Pipe parameters ---
    dbc.Row(dbc.Col(html.H5("Pipe Parameters", className="mb-2"))),
    dbc.Row([
        dbc.Col([
            html.Label("NPS (inches)", className="small"),
            dcc.Dropdown(id="nps", options=[1, 2, 4, 6, 8, 12, 16, 18, 24],
                         value=6, clearable=False, className="dbc"),
        ], md=2),
        dbc.Col([
            html.Label("Pipe Spec", className="small"),
            dbc.Input(id="pipe-spec", value="A1A"),
        ], md=2),
        dbc.Col([
            html.Label("Fluid Service", className="small"),
            dbc.Input(id="fluid-service", value="HGO"),
        ], md=2),
        dbc.Col([
            html.Label("Line Number", className="small"),
            dbc.Input(id="line-number", value="6-HGO-1001-A1A"),
        ], md=3),
        dbc.Col([
            html.Label("Candidates", className="small"),
            dcc.Slider(id="num-cands", min=1, max=10, step=1, value=5,
                       marks={i: str(i) for i in [1, 3, 5, 10]}),
        ], md=3),
    ], className="g-3 mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Insulation", className="small"),
            dcc.Dropdown(id="insulation-type",
                         options=["None", "PERS", "HOTC", "COLD", "TRACE"],
                         value="None", clearable=False, className="dbc"),
        ], md=2),
        dbc.Col([
            html.Label("Insulation thickness (mm)", className="small"),
            dbc.Input(id="insulation-mm", type="number", value=0, min=0),
        ], md=2),
        dbc.Col([
            dbc.Checkbox(id="grade-allowed", label="Grade routing allowed",
                         value=False, className="mt-4"),
        ], md=2),
        dbc.Col([
            dbc.Button("Generate Routes", id="generate-btn",
                       color="primary", size="lg", className="mt-3 w-100"),
            html.Span("", id="route-timer",
                      style={"fontFamily": "monospace", "fontSize": "0.8rem",
                             "marginLeft": "8px", "color": "#2ecc71"}),
        ], md=3),
    ], className="g-3 mb-4"),

    html.Div(id="route-status", className="small text-info"),

    # Init interval — restores graph on page revisit
    dcc.Interval(id="route-init", interval=500, max_intervals=1),
])

# ---------------------------------------------------------------------------
# Clientside — load timer
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(poll_disabled, current_text) {
        if (window._loadTimerInterval) {
            clearInterval(window._loadTimerInterval);
            window._loadTimerInterval = null;
        }
        if (poll_disabled) { return current_text; }
        window._loadTimerStart = Date.now();
        window._loadTimerInterval = setInterval(function() {
            var el = document.getElementById('load-timer');
            if (!el) return;
            var secs = Math.floor((Date.now() - window._loadTimerStart) / 1000);
            var m = Math.floor(secs / 60);
            var s = secs % 60;
            el.innerText = (m > 0 ? m + 'm ' : '') + s + 's';
        }, 1000);
        return '0s';
    }
    """,
    Output("load-timer", "children"),
    Input("load-poll", "disabled"),
    State("load-timer", "children"),
    prevent_initial_call=True,
)

# ---------------------------------------------------------------------------
# Clientside — generate routes timer with colour thresholds
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(n_clicks, current_text) {
        if (window._routeTimerInterval) {
            clearInterval(window._routeTimerInterval);
            window._routeTimerInterval = null;
        }
        if (!n_clicks) return current_text;
        window._routeTimerStart = Date.now();
        window._routeTimerInterval = setInterval(function() {
            var el = document.getElementById('route-timer');
            if (!el) return;
            var secs = Math.floor((Date.now() - window._routeTimerStart) / 1000);
            var m = Math.floor(secs / 60);
            var s = secs % 60;
            el.innerText = (m > 0 ? m + 'm ' : '') + s + 's';
            if      (secs < 30)  { el.style.color = '#2ecc71'; }
            else if (secs < 120) { el.style.color = '#f39c12'; }
            else                 { el.style.color = '#e74c3c'; }
        }, 1000);
        return '0s';
    }
    """,
    Output("route-timer", "children"),
    Input("generate-btn", "n_clicks"),
    State("route-timer", "children"),
    prevent_initial_call=True,
)

# ---------------------------------------------------------------------------
# Browse callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("browse-poll", "disabled"),
    Output("browse-poll", "max_intervals"),
    Input("cloud-browse-btn", "n_clicks"),
    State("cloud-opts",       "value"),
    prevent_initial_call=True,
)
def browse_file(_, opts):
    opts = opts or []
    _browse_result["ready"] = False
    _browse_result["path"]  = None

    def _pick():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        if "npy" in opts:
            ftypes = [("NumPy array", "*.npy"), ("Scan files", "*.e57 *.pts *.las *.laz"), ("All files", "*.*")]
        else:
            ftypes = [("Scan files", "*.e57 *.pts *.las *.laz"), ("NumPy array", "*.npy"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select file", filetypes=ftypes)
        root.destroy()
        _browse_result["path"]  = path or None
        _browse_result["ready"] = True

    threading.Thread(target=_pick, daemon=True).start()
    return False, -1


@callback(
    Output("cloud-path",  "value",         allow_duplicate=True),
    Output("browse-poll", "disabled",      allow_duplicate=True),
    Output("browse-poll", "max_intervals", allow_duplicate=True),
    Input("browse-poll",  "n_intervals"),
    prevent_initial_call=True,
)
def poll_browse(n):
    if not _browse_result["ready"]:
        return dash.no_update, dash.no_update, dash.no_update
    path = _browse_result["path"]
    _browse_result["ready"] = False
    if not path:
        return dash.no_update, True, 0
    return path, True, 0

# ---------------------------------------------------------------------------
# Gates browse / load callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("gates-browse-poll", "disabled"),
    Output("gates-browse-poll", "max_intervals"),
    Input("gates-browse-btn", "n_clicks"),
    prevent_initial_call=True,
)
def browse_gates(_):
    _browse_gates_result["ready"] = False
    _browse_gates_result["path"]  = None

    def _pick():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        ftypes = [("JSON files", "*.json"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select gates.json", filetypes=ftypes)
        root.destroy()
        _browse_gates_result["path"]  = path or None
        _browse_gates_result["ready"] = True

    threading.Thread(target=_pick, daemon=True).start()
    return False, -1


@callback(
    Output("gates-path",        "value",         allow_duplicate=True),
    Output("gates-browse-poll", "disabled",      allow_duplicate=True),
    Output("gates-browse-poll", "max_intervals", allow_duplicate=True),
    Input("gates-browse-poll",  "n_intervals"),
    prevent_initial_call=True,
)
def poll_gates_browse(n):
    if not _browse_gates_result["ready"]:
        return dash.no_update, dash.no_update, dash.no_update
    path = _browse_gates_result["path"]
    _browse_gates_result["ready"] = False
    if not path:
        return dash.no_update, True, 0
    return path, True, 0


@callback(
    Output("gates-store",  "data",     allow_duplicate=True),
    Output("gates-status", "children", allow_duplicate=True),
    Output("route-graph",  "figure",   allow_duplicate=True),
    Input("gates-path",    "value"),
    State("store",         "data"),
    State("start-x", "value"), State("start-y", "value"), State("start-z", "value"),
    State("end-x",   "value"), State("end-y",   "value"), State("end-z",   "value"),
    prevent_initial_call=True,
)
def load_gates_file(path, store, sx, sy, sz, ex, ey, ez):
    from pathlib import Path
    no = dash.no_update
    if not path or not path.strip():
        return no, no, no
    p = Path(path.strip())
    if not p.exists():
        return None, f"Not found: {p.name}", no
    try:
        from neuralpipe.models.gate import load_gates
        gates = load_gates(p)
        gate_dicts = [g.to_dict() for g in gates]
        store = store or {}
        sx = float(sx) if sx is not None else float((store.get("start") or [0,0,0])[0])
        sy = float(sy) if sy is not None else float((store.get("start") or [0,0,0])[1])
        sz = float(sz) if sz is not None else float((store.get("start") or [0,0,0])[2])
        ex = float(ex) if ex is not None else float((store.get("end") or [0,0,0])[0])
        ey = float(ey) if ey is not None else float((store.get("end") or [0,0,0])[1])
        ez = float(ez) if ez is not None else float((store.get("end") or [0,0,0])[2])
        fig = _picker_fig(store, sx, sy, sz, ex, ey, ez,
                          candidates=store.get("candidates"), gates_data=gate_dicts)
        return gate_dicts, f"{len(gates)} gates loaded from {p.name}", fig
    except Exception as e:
        return None, f"Error loading gates: {e}", no


# ---------------------------------------------------------------------------
# Load callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("cloud-status",   "children",      allow_duplicate=True),
    Output("cloud-progress", "value",         allow_duplicate=True),
    Output("load-poll",      "disabled",      allow_duplicate=True),
    Output("load-poll",      "max_intervals", allow_duplicate=True),
    Input("cloud-load-btn",  "n_clicks"),
    State("cloud-path",  "value"),
    State("cloud-opts",  "value"),
    State("cell-mm",     "value"),
    State("store",       "data"),
    prevent_initial_call=True,
)
def load_cloud(_, path, opts, cell_mm, store):
    from pathlib import Path
    opts  = opts or []
    store = store or {}

    if not path or not path.strip():
        return "No path provided.", 0, True, 0
    p = Path(path.strip())
    if not p.exists():
        return f"File not found: {p}", 0, True, 0

    _load_state.update({"status": "Starting...", "progress": 0, "done": False,
                         "fig": None, "store": None, "error": None})

    (Path(__file__).parent.parent / "cloud_load.log").write_text("", encoding="utf-8")

    def _run():
        import time
        from pathlib import Path as _Path
        t0 = time.time()
        _logfile = _Path(__file__).parent.parent / "cloud_load.log"

        def _log(msg):
            line = f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:6.1f}s] {msg}"
            print(line, flush=True)
            with open(_logfile, "a", encoding="utf-8") as _f:
                _f.write(line + "\n")

        try:
            def _cb(msg):
                _log(msg)
                if "scan" in msg.lower():
                    try:
                        part, total = msg.split("/")[0].split()[-1], msg.split("/")[1].split()[0]
                        pct = int(int(part) / int(total) * 60)
                    except Exception:
                        pct = _load_state["progress"]
                elif "merging" in msg.lower():
                    pct = 62
                elif "voxelis" in msg.lower():
                    pct = 70
                elif "saving" in msg.lower():
                    pct = 90
                else:
                    pct = _load_state["progress"]
                _load_state["status"] = msg
                _load_state["progress"] = pct

            _log(f"Starting load: {p}  opts={opts}  cell_mm={cell_mm}")
            _load_state["status"] = "Reading file..."
            _load_state["progress"] = 2

            if "npy" in opts:
                _log("Mode: .npy")
                _load_state["status"] = "Loading .npy..."
                _load_state["progress"] = 10
                pts = np.load(str(p), allow_pickle=True)
                _log(f".npy loaded: {len(pts):,} pts  shape={pts.shape}  dtype={pts.dtype}")
                _load_state["progress"] = 70
            elif "skip-ds" in opts:
                _log("Mode: skip downsample (raw load)")
                pts = load_e57_points(p) if p.suffix.lower() == ".e57" else load_pts_points(p)
                _log(f"Raw load done: {len(pts):,} pts")
                _load_state["progress"] = 70
            else:
                _log("Mode: downsample E57/PTS")
                pts = downsample_to_npy(p, p.parent / f"{p.stem}_{cell_mm}mm.npy",
                                        cell_size_m=cell_mm / 1000.0,
                                        progress_callback=_cb)
                _log(f"Downsample done: {len(pts):,} pts")
                _load_state["progress"] = 70

            if "align" in opts:
                _log("Aligning to axes (PCA)...")
                _load_state["status"] = "Aligning to axes..."
                pts = _align_to_axes(pts)
                _log("Alignment done.")

            _log("Computing bounds...")
            _load_state["status"] = "Computing bounds..."
            _load_state["progress"] = 80
            bmin = pts.min(axis=0)
            bmax = pts.max(axis=0)
            _log(f"Bounds: X {bmin[0]:.1f}..{bmax[0]:.1f}  Y {bmin[1]:.1f}..{bmax[1]:.1f}  Z {bmin[2]:.1f}..{bmax[2]:.1f}")

            cache.set_cloud(pts)

            # 2M pts is enough for 152mm occupancy radius — cKDTree queries stay fast
            _GRID_CAP = 2_000_000
            if len(pts) > _GRID_CAP:
                _log(f"Subsampling {len(pts):,} -> {_GRID_CAP:,} for KDTree...")
                _load_state["status"] = f"Subsampling {len(pts):,} pts to {_GRID_CAP:,} for routing grid..."
                rng = np.random.default_rng(0)
                grid_pts = pts[rng.choice(len(pts), _GRID_CAP, replace=False, shuffle=False)]
            else:
                grid_pts = pts

            _ROUTING_STEP_M = 0.305
            _log(f"Building KDTree on {len(grid_pts):,} pts (occupancy radius = {_ROUTING_STEP_M/2*1000:.0f}mm)...")
            _load_state["status"] = "Building routing grid (KDTree)..."
            _load_state["progress"] = 85
            grid = NumpyVoxelGrid(grid_pts, occupancy_radius_m=_ROUTING_STEP_M / 2)
            cache.get_agent().voxel_grid = grid
            _log("KDTree built.")

            start = [round(float(bmin[i]) + 3.0, 3) for i in range(3)]
            end   = [round(float(bmax[i]) - 3.0, 3) for i in range(3)]
            s = dict(store)
            s.update({
                "cloud_path":    str(p),
                "cloud_bmin":    bmin.tolist(),
                "cloud_bmax":    bmax.tolist(),
                "cloud_preview": True,
                "start": start,
                "end":   end,
            })

            _log("Building figure...")
            _load_state["status"] = "Building figure..."
            _load_state["progress"] = 90
            fig = _picker_fig(s, start[0], start[1], start[2],
                                 end[0],   end[1],   end[2])
            status = (f"Loaded {len(pts):,} pts  |  "
                      f"X {bmin[0]:.1f}–{bmax[0]:.1f}  "
                      f"Y {bmin[1]:.1f}–{bmax[1]:.1f}  "
                      f"Z {bmin[2]:.1f}–{bmax[2]:.1f} m")
            _log(f"All done. Total: {time.time()-t0:.1f}s")
            _load_state.update({"status": status, "progress": 100,
                                 "fig": fig, "store": s, "done": True})
        except Exception as e:
            import traceback
            print(f"[CLOUD ERROR] {e}\n{traceback.format_exc()}", flush=True)
            _load_state.update({"status": f"Error: {e}", "progress": 0,
                                 "done": True, "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return "Loading...", 2, False, -1


@callback(
    Output("route-graph",    "figure",         allow_duplicate=True),
    Output("cloud-status",   "children",       allow_duplicate=True),
    Output("cloud-progress", "value",          allow_duplicate=True),
    Output("store",          "data",           allow_duplicate=True),
    Output("start-x", "value", allow_duplicate=True),
    Output("start-y", "value", allow_duplicate=True),
    Output("start-z", "value", allow_duplicate=True),
    Output("end-x",   "value", allow_duplicate=True),
    Output("end-y",   "value", allow_duplicate=True),
    Output("end-z",   "value", allow_duplicate=True),
    Output("load-poll",      "disabled",       allow_duplicate=True),
    Output("load-poll",      "max_intervals",  allow_duplicate=True),
    Input("load-poll",       "n_intervals"),
    State("store",           "data"),
    prevent_initial_call=True,
)
def poll_load(_, store):
    import time
    from pathlib import Path
    no = dash.no_update
    status   = _load_state["status"]
    progress = _load_state["progress"]

    if not _load_state["done"]:
        return no, status, progress, no, no, no, no, no, no, no, no, no

    fig = _load_state["fig"]
    s   = _load_state["store"] or store
    _load_state["done"] = False

    logfile = Path(__file__).parent.parent / "cloud_load.log"
    if _load_state["error"]:
        with open(logfile, "a") as _f:
            _f.write(f"[{time.strftime('%H:%M:%S')}] poll_load: returning ERROR\n")
        return no, status, 0, no, no, no, no, no, no, no, True, 0

    n_traces = len(fig.data) if fig is not None else 0
    with open(logfile, "a") as _f:
        _f.write(f"[{time.strftime('%H:%M:%S')}] poll_load: dispatching figure  traces={n_traces}\n")

    start = s.get("start", [0.0, 0.0, 0.0])
    end   = s.get("end",   [0.0, 0.0, 0.0])
    return (fig, status, 100, s,
            start[0], start[1], start[2],
            end[0],   end[1],   end[2],
            True, 0)


@callback(
    Output("cloud-status",   "children",  allow_duplicate=True),
    Output("cloud-progress", "value",     allow_duplicate=True),
    Input("cloud-cache-btn", "n_clicks"),
    State("cloud-path",      "value"),
    prevent_initial_call=True,
)
def cache_to_npy(_, path):
    from pathlib import Path
    if not path or not path.strip():
        return "No path provided.", 0
    p = Path(path.strip())
    if not p.exists():
        return f"File not found: {p}", 0
    if p.suffix.lower() not in (".e57", ".pts", ".xyz", ".txt"):
        return "Cache to .npy only supports .e57 / .pts / .xyz files.", 0
    out = p.parent / f"{p.stem}_raw.npy"
    try:
        pts = load_e57_points(p) if p.suffix.lower() == ".e57" else load_pts_points(p)
        np.save(str(out), pts)
        size_mb = out.stat().st_size / 1e6
        return f"Cached {len(pts):,} pts -> {out.name} ({size_mb:.0f} MB)", 100
    except Exception as e:
        return f"Error: {e}", 0

# ---------------------------------------------------------------------------
# Pick mode toggle
# ---------------------------------------------------------------------------

@callback(
    Output("pick-mode",     "data"),
    Output("btn-set-start", "outline"),
    Output("btn-set-end",   "outline"),
    Output("pick-hint",     "children"),
    Input("btn-set-start",  "n_clicks"),
    Input("btn-set-end",    "n_clicks"),
    State("pick-mode",      "data"),
    prevent_initial_call=True,
)
def toggle_pick_mode(n_start, n_end, current_mode):
    triggered = dash.callback_context.triggered_id
    if triggered == "btn-set-start":
        new_mode = None if current_mode == "start" else "start"
    else:
        new_mode = None if current_mode == "end" else "end"
    hints = {
        "start": "Active — click a point in the cloud to place the Start marker.",
        "end":   "Active — click a point in the cloud to place the End marker.",
        None:    "Select Set Start or Set End, then click a point in the cloud.",
    }
    return new_mode, new_mode != "start", new_mode != "end", hints[new_mode]

# ---------------------------------------------------------------------------
# Graph init / restore on page revisit
# ---------------------------------------------------------------------------

@callback(
    Output("route-graph", "figure",   allow_duplicate=True),
    Output("start-x", "value",        allow_duplicate=True),
    Output("start-y", "value",        allow_duplicate=True),
    Output("start-z", "value",        allow_duplicate=True),
    Output("end-x",   "value",        allow_duplicate=True),
    Output("end-y",   "value",        allow_duplicate=True),
    Output("end-z",   "value",        allow_duplicate=True),
    Input("route-init", "n_intervals"),
    State("store",      "data"),
    prevent_initial_call=True,
)
def restore_route(_, store):
    store = store or {}
    if not store.get("cloud_preview") or cache.get_cloud() is None:
        return (dash.no_update,) * 7
    sx, sy, sz = store.get("start") or [0.0, 0.0, 0.0]
    ex, ey, ez = store.get("end")   or [0.0, 0.0, 0.0]
    fig = _picker_fig(store, sx, sy, sz, ex, ey, ez,
                      candidates=store.get("candidates"))
    return fig, sx, sy, sz, ex, ey, ez

# ---------------------------------------------------------------------------
# Apply coordinates button
# ---------------------------------------------------------------------------

@callback(
    Output("route-graph", "figure", allow_duplicate=True),
    Output("store",        "data",  allow_duplicate=True),
    Input("apply-coords-btn", "n_clicks"),
    State("start-x", "value"), State("start-y", "value"), State("start-z", "value"),
    State("end-x",   "value"), State("end-y",   "value"), State("end-z",   "value"),
    State("store",   "data"),
    prevent_initial_call=True,
)
def apply_coords(_, sx, sy, sz, ex, ey, ez, store):
    store = store or {}
    sx = float(sx) if sx is not None else 0.0
    sy = float(sy) if sy is not None else 0.0
    sz = float(sz) if sz is not None else 0.0
    ex = float(ex) if ex is not None else 0.0
    ey = float(ey) if ey is not None else 0.0
    ez = float(ez) if ez is not None else 0.0
    store["start"] = [sx, sy, sz]
    store["end"]   = [ex, ey, ez]
    return _picker_fig(store, sx, sy, sz, ex, ey, ez), store

# ---------------------------------------------------------------------------
# Click-to-pick
# ---------------------------------------------------------------------------

@callback(
    Output("route-graph", "figure",  allow_duplicate=True),
    Output("start-x", "value",       allow_duplicate=True),
    Output("start-y", "value",       allow_duplicate=True),
    Output("start-z", "value",       allow_duplicate=True),
    Output("end-x",   "value",       allow_duplicate=True),
    Output("end-y",   "value",       allow_duplicate=True),
    Output("end-z",   "value",       allow_duplicate=True),
    Output("store",   "data",        allow_duplicate=True),
    Input("route-graph", "clickData"),
    State("pick-mode", "data"),
    State("start-x", "value"), State("start-y", "value"), State("start-z", "value"),
    State("end-x",   "value"), State("end-y",   "value"), State("end-z",   "value"),
    State("store",   "data"),
    prevent_initial_call=True,
)
def handle_pick(click_data, mode, sx, sy, sz, ex, ey, ez, store):
    if not click_data or not mode:
        return (dash.no_update,) * 8
    pt = click_data["points"][0]
    x = round(float(pt["x"]), 3)
    y = round(float(pt["y"]), 3)
    z = round(float(pt["z"]), 3)
    store = store or {}
    if mode == "start":
        sx, sy, sz = x, y, z
        store["start"] = [sx, sy, sz]
    else:
        ex, ey, ez = x, y, z
        store["end"] = [ex, ey, ez]
    fig = _picker_fig(store, sx, sy, sz, ex, ey, ez)
    return fig, sx, sy, sz, ex, ey, ez, store

# ---------------------------------------------------------------------------
# Generate Routes
# ---------------------------------------------------------------------------

@callback(
    Output("route-graph",  "figure",   allow_duplicate=True),
    Output("store",        "data",     allow_duplicate=True),
    Output("route-status", "children"),
    Input("generate-btn",  "n_clicks"),
    State("start-x", "value"), State("start-y", "value"), State("start-z", "value"),
    State("end-x",   "value"), State("end-y",   "value"), State("end-z",   "value"),
    State("nps",             "value"),
    State("pipe-spec",       "value"),
    State("fluid-service",   "value"),
    State("line-number",     "value"),
    State("insulation-type", "value"),
    State("insulation-mm",   "value"),
    State("grade-allowed",   "value"),
    State("num-cands",       "value"),
    State("store",           "data"),
    State("gates-store",     "data"),
    prevent_initial_call=True,
)
def generate_routes(_, sx, sy, sz, ex, ey, ez,
                    nps, pipe_spec, fluid_service, line_number,
                    ins_type, ins_mm, grade_allowed, num_cands, store,
                    gates_data):
    import time
    from pathlib import Path
    t0 = time.time()

    def _log(msg):
        line = f"[ROUTE {time.strftime('%H:%M:%S')} +{time.time()-t0:5.1f}s] {msg}"
        print(line, flush=True)
        with open(Path(__file__).parent.parent / "route.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    (Path(__file__).parent.parent / "route.log").write_text("", encoding="utf-8")

    store = store or {}

    # Safe float conversion — fall back to store values if inputs are None
    _start = store.get("start", [0.0, 0.0, 0.0])
    _end   = store.get("end",   [0.0, 0.0, 0.0])
    sx = float(sx) if sx is not None else float(_start[0])
    sy = float(sy) if sy is not None else float(_start[1])
    sz = float(sz) if sz is not None else float(_start[2])
    ex = float(ex) if ex is not None else float(_end[0])
    ey = float(ey) if ey is not None else float(_end[1])
    ez = float(ez) if ez is not None else float(_end[2])
    nps = float(nps) if nps is not None else 6.0

    _log(f"start=({sx:.2f}, {sy:.2f}, {sz:.2f})  end=({ex:.2f}, {ey:.2f}, {ez:.2f})  nps={nps}")

    agent = cache.get_agent()
    agent.num_candidates = num_cands or 5
    ins       = None if ins_type == "None" else ins_type
    ins_thick = float(ins_mm) if ins_mm else None

    gates = None
    if gates_data:
        from neuralpipe.models.gate import GateOpening
        gates = [GateOpening.from_dict(g) for g in gates_data]
        _log(f"Using {len(gates)} gates for gate-sequenced routing")

    try:
        candidates = agent.route(
            start=(sx, sy, sz),
            end=(ex, ey, ez),
            nominal_diameter=nps,
            pipe_spec=pipe_spec,
            fluid_service=fluid_service,
            grade_routing_allowed=bool(grade_allowed),
            line_number=line_number,
            insulation_type=ins,
            insulation_thickness_mm=ins_thick,
            gates=gates,
        )
        _log(f"agent.route returned {len(candidates)} candidate(s) in {time.time()-t0:.1f}s")
    except Exception as e:
        import traceback
        _log(f"ERROR: {e}\n{traceback.format_exc()}")
        return dash.no_update, store, f"Routing error: {e}"

    if not candidates:
        _log("No candidates — A* found no paths.")
        return dash.no_update, store, "No valid routes found."

    store["candidates"] = [
        {
            "route_id":    r.route_id,
            "status":      r.status,
            "score":       r.score,
            "breakdown":   r.score_breakdown,
            "flags":       r.flags,
            "length_m":    r.total_length_m,
            "num_elbows":  r.num_elbows,
            "num_supports":r.num_supports,
            "waypoints":   [[w.x, w.y, w.z] for w in r.waypoints],
        }
        for r in candidates
    ]
    store["line_number"] = line_number

    fig = _picker_fig(store, sx, sy, sz, ex, ey, ez,
                      candidates=store["candidates"], gates_data=gates_data)
    return fig, store, f"{len(candidates)} candidate(s) generated. Go to Results to review."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_to_axes(pts: np.ndarray) -> np.ndarray:
    n = min(200_000, len(pts))
    sample_xy = pts[np.random.choice(len(pts), n, replace=False), :2]
    c = sample_xy.mean(axis=0)
    cov = np.cov((sample_xy - c).T)
    _, vecs = np.linalg.eigh(cov)
    ang = -np.arctan2(vecs[1, -1], vecs[0, -1])
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    R = np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])
    cx = np.array([c[0], c[1], 0.])
    return (pts - cx) @ R.T + cx


def _gate_edges(gates_data: list) -> go.Scatter3d:
    """Draw gate bboxes as orange wireframe boxes (12 edges each)."""
    xs, ys, zs = [], [], []
    for g in gates_data:
        b = g.get("bbox_3d")
        if not b or len(b) < 6:
            continue
        x0, y0, z0, x1, y1, z1 = b
        for edge in [
            (x0,y0,z0,x1,y0,z0), (x1,y0,z0,x1,y1,z0),
            (x1,y1,z0,x0,y1,z0), (x0,y1,z0,x0,y0,z0),
            (x0,y0,z1,x1,y0,z1), (x1,y0,z1,x1,y1,z1),
            (x1,y1,z1,x0,y1,z1), (x0,y1,z1,x0,y0,z1),
            (x0,y0,z0,x0,y0,z1), (x1,y0,z0,x1,y0,z1),
            (x1,y1,z0,x1,y1,z1), (x0,y1,z0,x0,y1,z1),
        ]:
            ax, ay, az, bx, by, bz = edge
            xs += [ax, bx, None]
            ys += [ay, by, None]
            zs += [az, bz, None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="#f39c12", width=2),
        name="Gates",
        hoverinfo="skip",
    )


def _picker_fig(store, sx, sy, sz, ex, ey, ez, candidates=None,
                gates_data=None) -> go.Figure:
    fig   = go.Figure()
    store = store or {}

    pts = cache.get_cloud()
    if pts is not None:
        n    = min(75_000, len(pts))
        prev = pts[np.random.default_rng().choice(len(pts), n, replace=False, shuffle=False)]
        fig.add_trace(go.Scatter3d(
            x=prev[:, 0], y=prev[:, 1], z=prev[:, 2],
            mode="markers",
            marker=dict(size=2, color=prev[:, 2], colorscale="Viridis", opacity=0.7),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            name="Cloud",
        ))

    fig.add_trace(go.Scatter3d(
        x=[sx], y=[sy], z=[sz],
        mode="markers+text", text=["START"], textposition="top center",
        marker=dict(size=10, color="#ff00ff"), name="START",
        hovertemplate="START<br>X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[ex], y=[ey], z=[ez],
        mode="markers+text", text=["END"], textposition="top center",
        marker=dict(size=10, color="#e74c3c", symbol="diamond"), name="END",
        hovertemplate="END<br>X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
    ))

    if gates_data:
        fig.add_trace(_gate_edges(gates_data))

    if candidates:
        colors = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#f1c40f"]
        for i, c in enumerate(candidates):
            wps = c["waypoints"]
            fig.add_trace(go.Scatter3d(
                x=[w[0] for w in wps], y=[w[1] for w in wps], z=[w[2] for w in wps],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=4),
                name=f"#{i+1} {c['route_id']} ({c['score']:.1f})",
            ))

    bmin = store.get("cloud_bmin")
    bmax = store.get("cloud_bmax")
    if bmin and bmax:
        bmn = np.asarray(bmin, dtype=np.float64)
        bmx = np.asarray(bmax, dtype=np.float64)
        extents  = bmx - bmn
        max_ext  = float(extents.max()) or 1.0
        ratios   = (extents / max_ext).clip(0.05, 1.0)
        xr = [float(bmn[0]) - 1, float(bmx[0]) + 1]
        yr = [float(bmn[1]) - 1, float(bmx[1]) + 1]
        zr = [float(bmn[2]) - 1, float(bmx[2]) + 1]
        aspect = dict(x=float(ratios[0]), y=float(ratios[1]), z=float(ratios[2]))
        amode  = "manual"
    else:
        xr, yr, zr = [-5, 25], [-5, 25], [0, 10]
        aspect = dict(x=1, y=1, z=1)
        amode  = "cube"

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)", range=xr),
            yaxis=dict(title="Y (m)", range=yr),
            zaxis=dict(title="Z (m)", range=zr),
            aspectmode=amode,
            aspectratio=aspect,
            bgcolor="rgba(10,10,20,1)",
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.6),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,20,30,0.8)", font=dict(color="#ccc")),
    )
    return fig
