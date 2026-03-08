"""VoxelGridProvider ABC + InMemoryVoxelGrid stub + NumpyVoxelGrid for real scan data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


Bounds = tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax
XYZ = tuple[float, float, float]


class VoxelGridProvider(ABC):
    """Abstract interface for spatial occupancy queries.

    Future implementations can swap in point-cloud (COPC) or IFC-derived grids
    without touching the routing or constraint code.
    """

    @abstractmethod
    def is_occupied(self, xyz: XYZ) -> bool:
        """Return True if the voxel at xyz is occupied by a solid obstacle."""

    @abstractmethod
    def get_occupied_in_bounds(self, bounds: Bounds) -> list[XYZ]:
        """Return all occupied voxel centres within the given AABB."""


@dataclass
class BoxObstacle:
    """Axis-aligned box obstacle for InMemoryVoxelGrid."""
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    label: str = ""

    def contains(self, xyz: XYZ) -> bool:
        x, y, z = xyz
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )

    def intersects(self, bounds: Bounds) -> bool:
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        return (
            self.xmin < xmax and self.xmax > xmin
            and self.ymin < ymax and self.ymax > ymin
            and self.zmin < zmax and self.zmax > zmin
        )

    def to_bounds(self) -> Bounds:
        return (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax)


class InMemoryVoxelGrid(VoxelGridProvider):
    """Simple box-obstacle grid for v1 testing.

    Add obstacles via add_obstacle(). No actual voxelisation — queries are
    point-in-box tests against the registered list. Adequate for unit tests
    and the Streamlit demo with a handful of obstacles.
    """

    def __init__(self) -> None:
        self._obstacles: list[BoxObstacle] = []

    def add_obstacle(self, obstacle: BoxObstacle) -> None:
        self._obstacles.append(obstacle)

    def is_occupied(self, xyz: XYZ) -> bool:
        return any(obs.contains(xyz) for obs in self._obstacles)

    def get_occupied_in_bounds(self, bounds: Bounds) -> list[XYZ]:
        """Return representative centres of obstacles intersecting bounds."""
        result: list[XYZ] = []
        for obs in self._obstacles:
            if obs.intersects(bounds):
                cx = (obs.xmin + obs.xmax) / 2
                cy = (obs.ymin + obs.ymax) / 2
                cz = (obs.zmin + obs.zmax) / 2
                result.append((cx, cy, cz))
        return result

    @property
    def obstacles(self) -> list[BoxObstacle]:
        return list(self._obstacles)


class NumpyVoxelGrid(VoxelGridProvider):
    """Real scan-data VoxelGridProvider backed by a numpy XYZ point array.

    Points are queried via a scipy KDTree for fast radius and box searches.
    Designed to load from a pre-downsampled .npy file produced by
    downsample_to_npy() below.

    Args:
        points: float32 array of shape (N, 3) in metres
        occupancy_radius_m: a query point is "occupied" if any scan point
            is within this radius (default = half the voxel cell size used
            when subsampling, e.g. 0.0015 for 3 mm cells)
    """

    def __init__(self, points: np.ndarray, occupancy_radius_m: float = 0.0015) -> None:
        from scipy.spatial import cKDTree
        self._points = points.astype(np.float32)
        self._radius = occupancy_radius_m
        self._tree = cKDTree(self._points)

    @classmethod
    def from_npy(cls, npy_path: str | Path, occupancy_radius_m: float = 0.0015) -> "NumpyVoxelGrid":
        """Load from a .npy file saved by downsample_to_npy()."""
        points = np.load(str(npy_path))
        return cls(points, occupancy_radius_m)

    def is_occupied(self, xyz: XYZ) -> bool:
        dist, _ = self._tree.query(xyz, k=1)
        return float(dist) <= self._radius

    def get_occupied_in_bounds(self, bounds: Bounds) -> list[XYZ]:
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        # Use a bounding sphere centred at the AABB centre for the KDTree query,
        # then filter to exact AABB.
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2
        r = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) / 2
        candidate_idx = self._tree.query_ball_point([cx, cy, cz], r)
        if not candidate_idx:
            return []
        pts = self._points[candidate_idx]
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        return [tuple(p) for p in pts[mask]]

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def bounds(self) -> Bounds:
        mn = self._points.min(axis=0)
        mx = self._points.max(axis=0)
        return (float(mn[0]), float(mn[1]), float(mn[2]),
                float(mx[0]), float(mx[1]), float(mx[2]))


# ---------------------------------------------------------------------------
# Downsampling utilities — streaming, scan-by-scan, low memory footprint
# ---------------------------------------------------------------------------

_STRIDE = np.int64(4_000_001)  # supports +-2000 m at 1 mm resolution


def _voxel_keys(pts: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Compute int64 voxel key for each point. pts is (N, 3) float."""
    ix = np.floor(pts[:, 0] / cell_size_m).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_size_m).astype(np.int64)
    iz = np.floor(pts[:, 2] / cell_size_m).astype(np.int64)
    return ix * (_STRIDE * _STRIDE) + iy * _STRIDE + iz


def _numpy_voxel_downsample(pts: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Pure-numpy voxel centroid downsample — no Python loops, all C via numpy.

    Args:
        pts: float32 (N, 3) array
        cell_size_m: voxel cell size in metres

    Returns:
        float32 (M, 3) centroid array, M <= N
    """
    pts64 = pts.astype(np.float64)
    keys = _voxel_keys(pts64, cell_size_m)
    order = np.argsort(keys, kind="stable")
    sorted_pts = pts64[order]
    sorted_keys = keys[order]
    del keys, order

    _, first_idx, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    del sorted_keys

    sums = np.add.reduceat(sorted_pts, first_idx, axis=0)
    del sorted_pts, first_idx

    return (sums / counts[:, np.newaxis]).astype(np.float32)


def _iter_e57_scans(path: Path):
    """Yield (scan_index, n_scans, pts_float32) for each scan in an E57 file."""
    import pye57
    e57 = pye57.E57(str(path))
    n_scans = e57.scan_count
    for idx in range(n_scans):
        data = e57.read_scan(idx, ignore_missing_fields=True)
        x = np.asarray(data["cartesianX"], dtype=np.float32)
        y = np.asarray(data["cartesianY"], dtype=np.float32)
        z = np.asarray(data["cartesianZ"], dtype=np.float32)
        yield idx, n_scans, np.column_stack([x, y, z])
        del data, x, y, z


def _iter_pts_chunks(path: Path, chunk_lines: int = 5_000_000):
    """Yield float32 (N, 3) chunks from a .pts / .xyz text file."""
    buf = []
    chunk_idx = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                buf.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
            if len(buf) >= chunk_lines:
                yield chunk_idx, np.array(buf, dtype=np.float32)
                buf.clear()
                chunk_idx += 1
    if buf:
        yield chunk_idx, np.array(buf, dtype=np.float32)


def downsample_to_npy(
    input_path: str | Path,
    output_path: str | Path,
    cell_size_m: float = 0.003,
    progress_callback=None,
) -> np.ndarray:
    """Load a scan file, voxel-downsample it, and save as .npy.

    Loads all points into memory then performs a single numpy sort+reduce —
    significantly faster than per-chunk dict accumulation for large files.
    Peak RAM is roughly 3x the raw XYZ float32 size of the input.

    Args:
        input_path: path to .e57, .pts, or .xyz file
        output_path: where to write the .npy result
        cell_size_m: voxel cell size in metres (default 3 mm)
        progress_callback: optional callable(status_str) for UI updates

    Returns:
        Downsampled float32 (M, 3) numpy array.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    chunks = []

    def _cb(msg):
        if progress_callback:
            progress_callback(msg)

    if suffix == ".e57":
        for scan_idx, n_scans, pts in _iter_e57_scans(input_path):
            _cb(f"Reading scan {scan_idx + 1}/{n_scans} — {len(pts):,} pts")
            chunks.append(pts)

    elif suffix in (".pts", ".xyz", ".txt"):
        for chunk_idx, pts in _iter_pts_chunks(input_path):
            _cb(f"Reading chunk {chunk_idx + 1} — {len(pts):,} pts")
            chunks.append(pts)

    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .e57, .pts, or .xyz")

    _cb("Merging scans...")
    all_pts = np.concatenate(chunks, axis=0)
    del chunks
    _cb(f"Voxelising {len(all_pts):,} pts at {cell_size_m * 1000:.1f} mm cell size...")

    downsampled = _numpy_voxel_downsample(all_pts, cell_size_m)
    del all_pts

    _cb(f"Saving {len(downsampled):,} pts -> {Path(output_path).name}...")
    np.save(str(output_path), downsampled)
    _cb(f"Done. {len(downsampled):,} points saved.")
    return downsampled


# Keep these as convenience wrappers for external callers / tests
def voxel_downsample(points: np.ndarray, cell_size_m: float) -> np.ndarray:
    """One-shot voxel downsampling for an already-loaded array."""
    return _numpy_voxel_downsample(points, cell_size_m)


def load_e57_points(path: str | Path) -> np.ndarray:
    """Load all XYZ from E57 into a single array (small files only)."""
    chunks = [pts for _, _, pts in _iter_e57_scans(Path(path))]
    return np.concatenate(chunks, axis=0)


def load_pts_points(path: str | Path) -> np.ndarray:
    """Load all XYZ from a PTS/XYZ text file into a single array (small files only)."""
    chunks = [pts for _, pts in _iter_pts_chunks(Path(path))]
    return np.concatenate(chunks, axis=0)
