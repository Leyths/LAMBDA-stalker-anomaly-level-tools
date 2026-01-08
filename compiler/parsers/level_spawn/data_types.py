"""
Data types for level spawn parsing.

Contains dataclasses used by all level_spawn submodules.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class Shape:
    """Parsed shape data from CSE_Shape."""
    shape_type: int  # 0 = sphere, 1 = box
    # For sphere: center (relative to entity) and radius
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    # For box: 3x3 rotation matrix (axis vectors scaled by half-extents) and translation
    box_axes: Optional[List[Tuple[float, float, float]]] = None  # 3 axis vectors
    box_translation: Optional[Tuple[float, float, float]] = None

    @property
    def is_sphere(self) -> bool:
        return self.shape_type == 0

    @property
    def is_box(self) -> bool:
        return self.shape_type == 1


@dataclass
class ALifeObject:
    """Parsed CSE_ALifeObject data."""
    game_graph_id: int = 0xFFFF
    distance: float = 0.0
    direct_control: bool = False
    node_id: int = 0xFFFFFFFF
    object_flags: int = 0
    ini_string: str = ""
    story_id: int = 0xFFFFFFFF
    spawn_story_id: int = 0xFFFFFFFF


@dataclass
class LevelChangerData:
    """Parsed CSE_ALifeLevelChanger destination data."""
    dest_game_vertex_id: int = 0xFFFF
    dest_level_vertex_id: int = 0xFFFFFFFF
    dest_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dest_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dest_level_name: str = ""
    dest_graph_point: str = ""
    silent_mode: bool = False
    # Offset of dest_game_vertex_id within the spawn packet (for in-place updates)
    dest_gvid_offset: int = -1


@dataclass
class SpawnEntity:
    """Parsed spawn entity data."""
    section_name: str
    entity_name: str
    position: Tuple[float, float, float]
    angle: Tuple[float, float, float]
    game_vertex_id: int
    level_vertex_id: int
    spawn_packet: bytes  # Full spawn packet with size prefix
    update_packet: Optional[bytes]  # M_UPDATE packet if present

    # Optional parsed fields
    spawn_id: int = 0
    version: int = 0

    # Offsets within spawn_packet for in-place updates (-1 if not available)
    game_vertex_id_offset: int = -1
    level_vertex_id_offset: int = -1


@dataclass
class GraphPoint:
    """Game graph vertex extracted from level.spawn."""
    index: int
    name: str
    position: Tuple[float, float, float]
    level_vertex_id: int
    location_types: bytes  # 4 bytes
    connection_point_name: str = ""
    connection_level_name: str = ""
