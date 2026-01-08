"""
Remapping Package

Handles GVID (Game Vertex ID) remapping for spawn entities and patrol paths.
All remapping uses the central GameGraph object for GVID resolution.
"""

from .spawn_remapper import remap_entity_gvids
from .patrol_remapper import remap_patrol_gvids, validate_and_remap_patrols
from .level_changer_config import LevelChangerConfig, LevelChangerDestination
from .start_location_remapper import remap_start_locations
