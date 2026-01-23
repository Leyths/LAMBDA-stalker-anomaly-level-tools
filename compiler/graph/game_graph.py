#!/usr/bin/env python3
"""
Game Graph Data Structure

Central data structure holding all game graph information and mappings.
This is the single source of truth for game vertex data after merging.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from levels import LevelConfig, LevelsConfig
    from game_graph_merger import GameVertex, GameEdge, DeathPoint
    from parsers import CrossTableParser
    from crosstables import LevelGraphNavigator
    from config import ModConfig


@dataclass
class GameGraph:
    """
    Central data structure holding all game graph info and mappings.

    This class encapsulates all the data produced by the game graph merger
    and provides methods for GVID resolution.
    """

    # Core graph data
    vertices: List['GameVertex']
    edges: List['GameEdge']
    death_points: List['DeathPoint']

    # Level information
    levels: List['LevelConfig']

    # Base mod that is being targeted
    base_mod: str = None

    # Mod configuration (from {basemod}.ini)
    mod_config: Optional['ModConfig'] = None

    # Level offsets: level_name -> cumulative GVID offset
    # The offset for a level is the sum of all previous levels' vertex counts
    level_offsets: Dict[str, int] = field(default_factory=dict)

    # Mappings for GVID resolution
    # Qualified key "level_name:graph_point_name" -> global GVID
    name_to_gvid: Dict[str, int] = field(default_factory=dict)

    # Internal mapping: (level_id, local_game_vertex_id) -> global_game_vertex_id
    vertex_mapping: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Path configuration for caching (set after construction)
    cross_table_dir: Optional[Path] = None
    base_path: Optional[Path] = None

    # Caches (lazily populated)
    _cross_table_cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    _level_ai_cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Optional reference to LevelsConfig for centralized lookups
    levels_config: Optional['LevelsConfig'] = None

    # Serialization data (populated during pipeline)
    cross_tables: List[Any] = field(default_factory=list)
    level_guids: Dict[int, bytes] = field(default_factory=dict)
    graph_guid: Optional[bytes] = None

    # =========================================================================
    # Cache Management
    # =========================================================================

    def set_paths(self, base_path: Path, cross_table_dir: Path):
        """
        Set paths for cache loading.

        Args:
            base_path: Base path for resolving level paths
            cross_table_dir: Directory containing .gct files
        """
        self.base_path = Path(base_path) if not isinstance(base_path, Path) else base_path
        self.cross_table_dir = Path(cross_table_dir) if not isinstance(cross_table_dir, Path) else cross_table_dir

    def _get_level_config(self, level_name: str) -> Optional['LevelConfig']:
        """Get LevelConfig by name."""
        if self.levels_config:
            return self.levels_config.get_level_by_name(level_name)
        return None

    def get_level_id(self, level_name: str) -> Optional[int]:
        """Get level ID by name."""
        if self.levels_config:
            return self.levels_config.get_level_id_by_name(level_name)
        return None

    def _get_cross_table(self, level_name: str) -> Optional['CrossTableParser']:
        """
        Get cross table for a level, loading from cache or file.

        Args:
            level_name: Name of the level

        Returns:
            CrossTableParser instance, or None if not available
        """
        if level_name in self._cross_table_cache:
            return self._cross_table_cache[level_name]

        if self.cross_table_dir is None:
            return None

        cross_table_path = self.cross_table_dir / f"{level_name}.gct"
        if not cross_table_path.exists():
            return None

        try:
            from parsers import CrossTableParser
            parser = CrossTableParser(cross_table_path)
            self._cross_table_cache[level_name] = parser
            return parser
        except Exception:
            return None

    def _get_level_ai(self, level_name: str) -> Optional['LevelGraphNavigator']:
        """
        Get level AI navigator for a level, loading from cache or file.

        Args:
            level_name: Name of the level

        Returns:
            LevelGraphNavigator instance, or None if not available
        """
        if level_name in self._level_ai_cache:
            return self._level_ai_cache[level_name]

        if self.base_path is None:
            return None

        level_config = self._get_level_config(level_name)
        if level_config is None:
            return None

        level_ai_path = self.base_path / level_config.path / "level.ai"
        if not level_ai_path.exists():
            return None

        try:
            from crosstables import LevelGraphNavigator
            navigator = LevelGraphNavigator(str(level_ai_path))
            self._level_ai_cache[level_name] = navigator
            return navigator
        except Exception:
            return None

    def clear_caches(self):
        """Clear all cached data to free memory."""
        self._cross_table_cache.clear()
        self._level_ai_cache.clear()
        if hasattr(self, '_level_positions_cache'):
            self._level_positions_cache.clear()
        if hasattr(self, '_cross_table_data_cache'):
            self._cross_table_data_cache.clear()

    def get_level_ai_positions(self, level_name: str):
        """
        Get cached level.ai vertex positions.

        Uses LevelAIParser for correct coordinate decoding.

        Args:
            level_name: Name of the level

        Returns:
            Nx3 numpy array of positions, or None if not available
        """
        if not hasattr(self, '_level_positions_cache'):
            self._level_positions_cache = {}

        if level_name in self._level_positions_cache:
            return self._level_positions_cache[level_name]

        level_config = self._get_level_config(level_name)
        if level_config is None or self.base_path is None:
            return None

        level_ai_path = self.base_path / level_config.path / "level.ai"
        if not level_ai_path.exists():
            return None

        try:
            from parsers import load_level_ai_positions
            positions = load_level_ai_positions(level_ai_path)
            self._level_positions_cache[level_name] = positions
            return positions
        except Exception:
            return None

    def get_cross_table_cache(self, level_name: str):
        """
        Get cached cross table data for fast lookups.

        Args:
            level_name: Name of the level

        Returns:
            CrossTableCache instance, or None if not available
        """
        if not hasattr(self, '_cross_table_data_cache'):
            self._cross_table_data_cache = {}

        if level_name in self._cross_table_data_cache:
            return self._cross_table_data_cache[level_name]

        if self.cross_table_dir is None:
            return None

        cross_table_path = self.cross_table_dir / f"{level_name}.gct"
        if not cross_table_path.exists():
            return None

        try:
            from converters.waypoint import CrossTableCache
            cache = CrossTableCache(cross_table_path)
            self._cross_table_data_cache[level_name] = cache
            return cache
        except Exception:
            return None

    # =========================================================================
    # GVID Resolution Methods
    # =========================================================================

    def get_gvid_by_name(self, level_name: str, graph_point_name: str) -> Optional[int]:
        """
        Look up GVID by qualified graph point name.

        Used for level_changer destination resolution.

        Args:
            level_name: Destination level name
            graph_point_name: Name of graph point on that level

        Returns:
            Global game vertex ID, or None if not found
        """
        qualified_key = f"{level_name}:{graph_point_name}"
        return self.name_to_gvid.get(qualified_key)

    def get_level_offset(self, level_name: str) -> int:
        """
        Get the GVID offset for a level.

        The offset is the sum of all previous levels' vertex counts.
        Adding this offset to a local GVID gives the global GVID.

        Args:
            level_name: Name of the level

        Returns:
            GVID offset (0 if level not found)
        """
        return self.level_offsets.get(level_name, 0)

    def get_vertex_count(self) -> int:
        """Get total number of game vertices."""
        return len(self.vertices)

    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return len(self.edges)

    def get_death_point_count(self) -> int:
        """Get total number of death points."""
        return len(self.death_points)

    # =========================================================================
    # Position-Based GVID Resolution
    # =========================================================================

    def get_gvid_for_position(self, level_name: str, position: Tuple[float, float, float]) -> Optional[int]:
        """
        Find global GVID for an arbitrary XYZ position on a level.

        This is the primary lookup method - given any world position,
        find which game vertex "owns" that position.

        Steps:
        1. Find nearest level.ai vertex to position (spatial search)
        2. Look up that level vertex in the cross table -> local GVID
        3. Add level offset -> global GVID

        Args:
            level_name: Name of the level (e.g., "zaton")
            position: (x, y, z) world coordinates

        Returns:
            Global game vertex ID (0 to N-1 across all levels), or None if lookup fails
        """
        # Get level vertex first
        level_vertex_id = self.get_level_vertex_for_position(level_name, position)
        if level_vertex_id is None:
            return None

        # Look up in cross table
        cross_table = self._get_cross_table(level_name)
        if cross_table is None:
            return None

        try:
            # Get local game vertex ID from cross table
            if level_vertex_id >= cross_table.level_vertex_count:
                return None
            local_gvid = cross_table.get_game_vertex_id(level_vertex_id)

            # Add level offset to get global GVID
            offset = self.get_level_offset(level_name)
            return local_gvid + offset

        except (IndexError, ValueError):
            return None

    def get_level_vertex_for_position(self, level_name: str, position: Tuple[float, float, float]) -> Optional[int]:
        """
        Find level.ai vertex ID for a position (not global GVID).

        Useful when you need the local vertex ID for other purposes.

        Args:
            level_name: Name of the level
            position: (x, y, z) world coordinates

        Returns:
            Level vertex ID, or None if lookup fails
        """
        level_ai = self._get_level_ai(level_name)
        if level_ai is None:
            return None

        try:
            return level_ai.find_nearest_vertex(position)
        except Exception:
            return None

    def get_cross_table_for_level(self, level_name: str) -> Optional['CrossTableParser']:
        """
        Get cross table parser for a level.

        Public accessor for the cached cross table.

        Args:
            level_name: Name of the level

        Returns:
            CrossTableParser instance, or None if not available
        """
        return self._get_cross_table(level_name)

    def get_level_ai_for_level(self, level_name: str) -> Optional['LevelGraphNavigator']:
        """
        Get level AI navigator for a level.

        Public accessor for the cached level AI.

        Args:
            level_name: Name of the level

        Returns:
            LevelGraphNavigator instance, or None if not available
        """
        return self._get_level_ai(level_name)
