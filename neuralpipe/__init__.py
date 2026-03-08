"""NeuralPipe — AI-assisted pipe routing agent for industrial plant design.

Primary entry point:
    from neuralpipe import NeuralPipeAgent
"""
from neuralpipe.agent.router import NeuralPipeAgent
from neuralpipe.models.route import Route, Segment, Waypoint, ElbowFitting, SupportPoint
from neuralpipe.models.pipe import PipeSpec, PipeSpecRegistry
from neuralpipe.models.feedback import DQRecord
from neuralpipe.geometry.voxel_grid import (
    VoxelGridProvider, InMemoryVoxelGrid, BoxObstacle,
    NumpyVoxelGrid, voxel_downsample, downsample_to_npy,
)
from neuralpipe.constraints.base import AbstractConstraint, ConstraintResult
from neuralpipe.constraints.registry import ConstraintRegistry
from neuralpipe.scoring.base import AbstractScoringFactor
from neuralpipe.scoring.registry import ScoringRegistry
from neuralpipe.feedback.store import FeedbackStore, SQLiteFeedbackStore

__version__ = "0.1.0"

__all__ = [
    # Agent (primary API)
    "NeuralPipeAgent",
    # Models
    "Route",
    "Segment",
    "Waypoint",
    "ElbowFitting",
    "SupportPoint",
    "PipeSpec",
    "PipeSpecRegistry",
    "DQRecord",
    # Geometry
    "VoxelGridProvider",
    "InMemoryVoxelGrid",
    "BoxObstacle",
    # Extension points
    "AbstractConstraint",
    "ConstraintResult",
    "ConstraintRegistry",
    "AbstractScoringFactor",
    "ScoringRegistry",
    "FeedbackStore",
    "SQLiteFeedbackStore",
]
