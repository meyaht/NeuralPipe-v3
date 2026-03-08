# NeuralPipe v0.3 — Claude Context

## What This Is
A Dash + Plotly web app for AI-assisted pipe routing through real building scan data.
Engineer loads a point cloud, picks start/end points in 3D, and the system generates
ranked pipe route candidates that avoid obstacles (walls, floors, equipment).

## Stack
- **App entry:** `app.py` (Dash multi-page, port 8050)
- **Pages:** `pages/cloud.py` (load scan) → `pages/route.py` (pick points + generate) → `pages/results.py` (review candidates) → `pages/view.py` (full-screen view)
- **Core routing:** `neuralpipe/routing/astar.py` — 3D orthogonal A* on voxel grid
- **Obstacle grid:** `neuralpipe/geometry/voxel_grid.py` — `NumpyVoxelGrid` (KDTree-backed)
- **Agent:** `neuralpipe/agent/router.py` — `NeuralPipeAgent`, orchestrates routing + constraints + scoring
- **Server cache:** `cache.py` — holds numpy array + agent instance in memory

## Scan Data
- Raw E57: `pointData/E57.e57` — 9.82 GB, 20 scans, 349M points (source of truth)
- **Test file: `pointData/test.npy`** — 547 MB, 45.6M pts — USE THIS FOR ALL ROUTING
  - Source: voxel-downsampled from E57.e57 at 8mm cell size via fast_downsample.py
  - Uniform surface coverage (voxel grid preserves wall density unlike random subsample)
- Raw cached: `pointData/E57_raw.npy` — 4.2 GB, 349M pts (display only, do not use for routing)
- Coordinate frame: absolute world coords, Z ~148–165m (building elevation), X ~22–66m, Y ~124–147m
- Building is a basement with foundation walls running through it — irregular shape, ~4–5 elbows expected per route
- Routing must stay in open space between foundations; start/end points should be at same floor elevation

## Key Constraints for Development
- **Scatter3d display cap: 75k pts** — anything higher causes ~20MB JSON and freezes the browser
- **KDTree input cap: 20M pts** — subsample from full cloud before building KDTree
- **Occupancy check radius: 0.152m (half of 305mm A* step)** — must be >= step/2 or router walks through walls
- **A* routing pass: pass 5 = 305mm step size** — coarse enough to be fast, fine enough for buildings
- `is_occupied` uses `tree.query(k=1)` nearest-neighbour — NOT `query_ball_point` (too slow at large radius)
- Background thread pattern for cloud loading — never block the Dash main thread
- `cloud_load.log` in project root — written every load, check with Read tool for diagnostics

## Routing Logic
- `NeuralPipeAgent.route()` in `router.py` calls `generate_candidates()` → `AStarRouter.search()`
- A* avoids nodes where `voxel_grid.is_occupied()` returns True (nearest scan point within 152mm)
- Open air/corridors = no nearby scan points = free = routable
- Walls/floors = dense scan points = occupied = blocked
- Elbows: unlimited, no support requirement enforced at routing time
- Candidates varied by `elbow_penalty` values: [15, 5, 30, 10, 50]
- `max_nodes=100_000` — abort threshold, raise if routes fail in large spaces

## v0.3 — Gate-Aware Routing
- New: `neuralpipe/models/gate.py` — GateOpening dataclass, load_gates(path)
- New: `neuralpipe/routing/gate_router.py` — filter_gates_on_path, route_through_gates, generate_gate_candidates
- `NeuralPipeAgent.route()` accepts `gates` param — uses gate-sequenced A* when provided
- UI: "Browse Gates…" button loads gates.json (from GateDetector / AutoGateDetector)
- Gates shown as orange wireframe boxes on 3D view
- Routing sequences A* through each gate center along the travel axis

## Run the App
```
cd C:/Users/zkrep/NeuralPipe-v3
python app.py
# or double-click NeuralPipe.bat
```

## Diagnostics
- `cloud_load.log` — per-step timestamps from cloud load background thread
- Terminal stdout — same content printed live
- Check log: Read("C:/Users/zkrep/NeuralPipe-v3/cloud_load.log")
- Screenshots: C:/Users/zkrep/Pictures/Screenshots/ — check latest file there

## Git
- Remote: https://github.com/meyaht/NeuralPipe-v3
- Identity: Zachary Kreps <zkreps@gmail.com>
- **Do NOT push until user confirms fix is working**
