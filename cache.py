"""Server-side cache for objects that can't live in dcc.Store (numpy arrays, agent)."""
from __future__ import annotations

import numpy as np
from neuralpipe import NeuralPipeAgent

# Single shared instances for a local single-user deployment.
# For multi-user, key these by session ID.
_cloud: np.ndarray | None = None
_agent: NeuralPipeAgent = NeuralPipeAgent()


def set_cloud(pts: np.ndarray) -> None:
    global _cloud
    _cloud = pts


def get_cloud() -> np.ndarray | None:
    return _cloud


def get_agent() -> NeuralPipeAgent:
    return _agent
