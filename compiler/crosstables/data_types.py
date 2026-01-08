"""
Data types for cross table building.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GraphPoint:
    """Game graph vertex extracted from level.spawn"""
    index: int
    name: str
    position: Tuple[float, float, float]
    level_vertex_id: int
    location_types: bytes
    connection_point_name: str = ""
    connection_level_name: str = ""


@dataclass
class CrossTableCell:
    """Single entry in cross table (6 bytes)"""
    game_vertex_id: int  # u16
    distance: float  # f32
