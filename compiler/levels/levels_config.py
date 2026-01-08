#!/usr/bin/env python3
"""
Levels Configuration

Parser for levels.ini configuration file.
Defines which levels to include in the game graph build.
"""

import configparser
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from utils import log, logWarning, logError


@dataclass
class LevelConfig:
    """Configuration for a single level"""
    section: str  # Section name in INI
    name: str  # Internal level name (e.g., l01_escape)
    id: int  # Unique level ID (0-255)
    path: str  # Path to level folder
    offset: Tuple[float, float, float]  # World space offset (x, y, z)
    caption: str  # Display name
    original_spawn: Optional[str] = None  # Path to original spawn file for merging
    original_patrols: Optional[str] = None  # Path to original patrols file (.patrols binary)
    original_edges: Optional[str] = None  # Path to original edges file (.edges.json)
    connect_orphans_automatically: bool = False  # Auto-connect orphan nodes (default: preserve existing behavior)

    def __post_init__(self):
        """Validate level configuration"""
        if not 0 <= self.id <= 255:
            raise ValueError(f"Level ID {self.id} out of range (0-255)")

        if not self.name:
            raise ValueError(f"Level {self.section} has no name")

        if not self.path:
            raise ValueError(f"Level {self.section} has no path")


class LevelsConfig:
    """
    Levels configuration manager

    Loads and validates levels.ini configuration file.
    Provides ordered list of levels for compilation.
    """

    def __init__(self, config_path: str = "levels.ini"):
        """
        Load levels configuration

        Args:
            config_path: Path to levels.ini file
        """
        self.config_path = config_path
        self.levels: List[LevelConfig] = []
        self._load_config()

        # Base paths for deriving file locations (set via set_paths)
        self.base_path: Optional[Path] = None
        self.cross_table_dir: Optional[Path] = None

        # Cached lookup dicts (built lazily)
        self._name_to_id_cache: Optional[Dict[str, int]] = None
        self._id_to_name_cache: Optional[Dict[int, str]] = None

    def _load_config(self):
        """Load and parse levels.ini"""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)

        # Process each level section
        for section in config.sections():
            try:
                level = self._parse_level_section(section, config[section])
                self.levels.append(level)
            except Exception as e:
                logWarning(f"Skipping level [{section}]: {e}")

        # Validate
        self._validate_levels()

    def _parse_level_section(self, section: str, data: dict) -> LevelConfig:
        """Parse a single level section"""
        # Required fields
        name = data.get('name')
        if not name:
            raise ValueError("Missing 'name' field")

        id_str = data.get('id')
        if not id_str:
            raise ValueError("Missing 'id' field")

        path = data.get('path')
        if not path:
            raise ValueError("Missing 'path' field")

        # Parse offset (default to 0,0,0)
        offset_str = data.get('offset', '0.0, 0.0, 0.0')
        offset = self._parse_offset(offset_str)

        # Optional caption (default to name)
        caption = data.get('caption', name)

        # Optional original_spawn path
        original_spawn = data.get('original_spawn', None)
        if original_spawn:
            original_spawn = original_spawn.strip()

        # Optional original_patrols path
        original_patrols = data.get('original_patrols', None)
        if original_patrols:
            original_patrols = original_patrols.strip()

        # Optional original_edges path
        original_edges = data.get('original_edges', None)
        if original_edges:
            original_edges = original_edges.strip()

        # Optional connect_orphans_automatically flag (default False - preserve existing behavior)
        connect_orphans_str = data.get('connect_orphans_automatically', 'false')
        connect_orphans_automatically = connect_orphans_str.lower() in ('true', '1', 'yes', 'on')

        return LevelConfig(
            section=section,
            name=name,
            id=int(id_str),
            path=path,
            offset=offset,
            caption=caption,
            original_spawn=original_spawn,
            original_patrols=original_patrols,
            original_edges=original_edges,
            connect_orphans_automatically=connect_orphans_automatically
        )

    def _parse_offset(self, offset_str: str) -> Tuple[float, float, float]:
        """Parse offset string 'x, y, z' to tuple"""
        try:
            parts = [float(x.strip()) for x in offset_str.split(',')]
            if len(parts) != 3:
                raise ValueError("Offset must have 3 components")
            return tuple(parts)
        except Exception as e:
            raise ValueError(f"Invalid offset format: {offset_str}") from e

    def _validate_levels(self):
        """Validate level configuration"""
        if not self.levels:
            raise ValueError("No levels defined in configuration")

        # Check for duplicate IDs
        ids = [level.id for level in self.levels]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate level IDs found")

        # Check for duplicate names
        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate level names found")

    def get_level_by_name(self, name: str) -> Optional[LevelConfig]:
        """Get level by internal name"""
        for level in self.levels:
            if level.name == name:
                return level
        return None

    def get_level_by_id(self, level_id: int) -> Optional[LevelConfig]:
        """Get level by ID"""
        for level in self.levels:
            if level.id == level_id:
                return level
        return None

    # =========================================================================
    # Path Configuration
    # =========================================================================

    def set_paths(self, base_path: Path, cross_table_dir: Path):
        """
        Set base paths for deriving file locations.

        Args:
            base_path: Base path for resolving level paths
            cross_table_dir: Directory containing .gct files
        """
        self.base_path = Path(base_path) if not isinstance(base_path, Path) else base_path
        self.cross_table_dir = Path(cross_table_dir) if not isinstance(cross_table_dir, Path) else cross_table_dir

    def get_cross_table_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to .gct file for a level."""
        if not self.cross_table_dir:
            return None
        return self.cross_table_dir / f"{level.name}.gct"

    def get_level_ai_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to level.ai file."""
        if not self.base_path:
            return None
        return self.base_path / level.path / "level.ai"

    def get_edges_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to edges.json file (if configured)."""
        if not self.base_path or not level.original_edges:
            return None
        return self.base_path / level.original_edges

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    def get_level_id_by_name(self, name: str) -> Optional[int]:
        """Get level ID by name."""
        level = self.get_level_by_name(name)
        return level.id if level else None

    def get_level_name_by_id(self, level_id: int) -> Optional[str]:
        """Get level name by ID."""
        level = self.get_level_by_id(level_id)
        return level.name if level else None

    @property
    def name_to_id(self) -> Dict[str, int]:
        """Cached mapping from level name to level ID."""
        if self._name_to_id_cache is None:
            self._name_to_id_cache = {level.name: level.id for level in self.levels}
        return self._name_to_id_cache

    @property
    def id_to_name(self) -> Dict[int, str]:
        """Cached mapping from level ID to level name."""
        if self._id_to_name_cache is None:
            self._id_to_name_cache = {level.id: level.name for level in self.levels}
        return self._id_to_name_cache

    def print_summary(self):
        """Print configuration summary"""
        log(f"Loaded {len(self.levels)} levels from {self.config_path}")
        log()
        log("Levels:")
        for i, level in enumerate(self.levels, 1):
            log(f"  {i:2d}. [{level.id:3d}] {level.name:20s} - {level.caption}")
            log(f"      Path: {level.path}")
            log(f"      Offset: ({level.offset[0]:.1f}, {level.offset[1]:.1f}, {level.offset[2]:.1f})")
            if level.original_spawn:
                log(f"      Original spawn: {level.original_spawn}")
            if level.original_patrols:
                log(f"      Original patrols: {level.original_patrols}")
            if level.original_edges:
                log(f"      Original edges: {level.original_edges}")


def main():
    """Test the configuration parser"""
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "levels.ini"

    try:
        config = LevelsConfig(config_path)
        config.print_summary()

        log("\n" + "=" * 60)
        log("Configuration valid!")
        log("=" * 60)

    except Exception as e:
        logError(f"{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()