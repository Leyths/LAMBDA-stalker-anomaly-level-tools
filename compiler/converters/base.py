"""
Base module for converters.

Contains shared constants and dataclasses used by restrictor converters.
"""

from dataclasses import dataclass


# Sentinel values for unset vertex IDs (will be remapped during build)
LVID_UNSET = 0xFFFFFFFF  # u32 max
GVID_UNSET = 0xFFFF      # u16 max


@dataclass
class LevelRestrictorInfo:
    """Represents a single restrictor for an entire level"""
    level_name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    level_vertex_id: int = LVID_UNSET  # Set during remapping
    game_vertex_id: int = GVID_UNSET   # Set during remapping
