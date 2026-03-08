#!/usr/bin/env python3
"""
Levels Configuration

Parser for levels.ini configuration file.
Defines which levels to include in the game graph build.
"""

import configparser
import shutil
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
    path: Path  # Absolute path to level folder
    offset: Tuple[float, float, float]  # World space offset (x, y, z)
    caption: str  # Display name
    original_spawn: Optional[Path] = None  # Absolute path to original spawn file for merging
    original_patrols: Optional[Path] = None  # Absolute path to original patrols file (.patrols binary)
    original_edges: Optional[Path] = None  # Absolute path to original edges file (.edges.json)
    connect_orphans_automatically: bool = False  # Auto-connect orphan nodes (default: preserve existing behavior)
    base_anomaly_spawns_only: bool = False  # Only use anomaly/ original spawns, skip level.spawn entities/graph points
    base_anomaly_waypoints_only: bool = False  # Only use anomaly/ original patrols, skip level.game waypoints
    vanilla_hash_ai: Optional[str] = None  # SHA-256 prefix of vanilla level.ai
    vanilla_hash_spawn: Optional[str] = None  # SHA-256 prefix of vanilla level.spawn
    vanilla_hash_game: Optional[str] = None  # SHA-256 prefix of vanilla level.game

    def __post_init__(self):
        """Validate level configuration"""
        if not 0 <= self.id <= 255:
            raise ValueError(f"Level ID {self.id} out of range (0-255)")

        if not self.name:
            raise ValueError(f"Level {self.section} has no name")


class LevelsConfig:
    """
    Levels configuration manager

    Loads and validates levels.ini configuration file.
    Provides ordered list of levels for compilation.
    """

    def __init__(self, config_path: str = "levels.ini",
                 levels_dir=None, cross_table_dir: Path = None,
                 resolve_root: Path = None,
                 levels_override_dir: Path = None,
                 build_dir: Path = None):
        """
        Load levels configuration

        Args:
            config_path: Path to levels.ini file
            levels_dir: Directory containing level folders. Can be a string or Path.
                        If relative, resolved against resolve_root.
            cross_table_dir: Directory containing .gct cross table files
            resolve_root: Root for resolving relative paths in the INI
                          (original_spawn, original_patrols, etc.).
                          These INI paths were historically relative to compiler/.
                          Defaults to the parent of config_path.
            levels_override_dir: Directory with partial level file overrides.
                                 Per-file: if present in override dir, use it;
                                 otherwise fall back to levels_dir.
            build_dir: Build cache directory (for staging overrides into .tmp/)
        """
        self.config_path = config_path
        self.cross_table_dir = Path(cross_table_dir) if cross_table_dir else None

        # Determine the root for resolving relative INI paths.
        # INI paths like ../anomaly/foo.spawn were historically relative to compiler/.
        # The caller should pass compiler_dir (or paths.compiler_dir) as resolve_root.
        if resolve_root is not None:
            self._resolve_root = Path(resolve_root).resolve()
        else:
            # Fallback: assume INI is at project root, use parent as resolve root
            self._resolve_root = Path(config_path).resolve().parent

        # Resolve levels_dir
        if levels_dir is not None:
            ld = Path(levels_dir)
            if ld.is_absolute():
                self._levels_dir = ld
            else:
                self._levels_dir = (self._resolve_root / ld).resolve()
        else:
            self._levels_dir = self._resolve_root.parent / "levels"

        self.levels: List[LevelConfig] = []
        self.level_file_sources: Dict[str, Dict[str, str]] = {}
        self._load_config()

        # Apply per-file overrides if an override directory is provided
        if levels_override_dir and build_dir:
            self._apply_level_overrides(Path(levels_override_dir), Path(build_dir))

        # Cached lookup dicts (built lazily)
        self._name_to_id_cache: Optional[Dict[str, int]] = None
        self._id_to_name_cache: Optional[Dict[int, str]] = None

    def _apply_level_overrides(self, override_dir: Path, build_dir: Path):
        """
        Create a staging directory that merges level files from the override
        directory and the base levels directory. Per-file: if present in
        override dir, use it; otherwise fall back to base.
        """
        from utils.logging import Colors, _write_to_file

        staged_root = build_dir / "staged_levels"
        level_files = ["level.ai", "level.game", "level.spawn"]

        log(f"  Level override directory: {override_dir}")

        for level in self.levels:
            base_dir = level.path
            override_level_dir = override_dir / level.name
            staged_dir = staged_root / level.name
            staged_dir.mkdir(parents=True, exist_ok=True)

            sources = {}
            for filename in level_files:
                override_file = override_level_dir / filename
                base_file = base_dir / filename

                if override_file.exists():
                    shutil.copy2(override_file, staged_dir / filename)
                    sources[filename] = "override"
                elif base_file.exists():
                    shutil.copy2(base_file, staged_dir / filename)
                    sources[filename] = "base"
                else:
                    sources[filename] = "missing"

            self.level_file_sources[level.name] = sources
            level.path = staged_dir

    def print_override_summary(self):
        """Print a colored table showing where each level file was sourced from."""
        if not self.level_file_sources:
            return

        from utils.logging import Colors, _write_to_file

        level_files = ["level.ai", "level.spawn", "level.game"]
        col_width = 13

        log("\nLEVEL FILE SOURCES (override active)")
        log("=" * 70)

        # Header
        header = f"{'Level':<22}" + "".join(f"{f:<{col_width}}" for f in level_files)
        log(header)
        log("-" * 70)

        # Rows — colored console output, plain log file
        for level_name, sources in self.level_file_sources.items():
            # Build colored console line
            line_parts = [f"{level_name:<22}"]
            plain_parts = [f"{level_name:<22}"]

            for filename in level_files:
                source = sources.get(filename, "missing")
                label = source.upper() if source == "override" else source
                plain_parts.append(f"{label:<{col_width}}")

                if source == "override":
                    line_parts.append(f"{Colors.BLUE}{label:<{col_width}}{Colors.RESET}")
                elif source == "base":
                    line_parts.append(f"{Colors.GREEN}{label:<{col_width}}{Colors.RESET}")
                else:
                    line_parts.append(f"{Colors.RED}{label:<{col_width}}{Colors.RESET}")

            print("".join(line_parts))
            _write_to_file("".join(plain_parts))

        log("=" * 70)

    def _resolve_ini_path(self, raw_path: str) -> Path:
        """Resolve a relative path from the INI file to an absolute path."""
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return (self._resolve_root / p).resolve()

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
                # Derive path from levels_dir if not specified in INI
                if level.path is None:
                    level.path = self._levels_dir / level.name
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

        # Path (optional — derived from levels_dir/name if not in INI)
        raw_path = data.get('path', '').strip()
        path = self._resolve_ini_path(raw_path) if raw_path else None

        # Parse offset (default to 0,0,0)
        offset_str = data.get('offset', '0.0, 0.0, 0.0')
        offset = self._parse_offset(offset_str)

        # Optional caption (default to name)
        caption = data.get('caption', name)

        # Optional original_spawn path
        original_spawn = None
        raw = data.get('original_spawn', None)
        if raw and raw.strip():
            original_spawn = self._resolve_ini_path(raw.strip())

        # Optional original_patrols path
        original_patrols = None
        raw = data.get('original_patrols', None)
        if raw and raw.strip():
            original_patrols = self._resolve_ini_path(raw.strip())

        # Optional original_edges path
        original_edges = None
        raw = data.get('original_edges', None)
        if raw and raw.strip():
            original_edges = self._resolve_ini_path(raw.strip())

        # Optional connect_orphans_automatically flag (default False - preserve existing behavior)
        connect_orphans_str = data.get('connect_orphans_automatically', 'false')
        connect_orphans_automatically = connect_orphans_str.lower() in ('true', '1', 'yes', 'on')

        # Optional base_anomaly_spawns_only flag (default False)
        base_anomaly_spawns_only_str = data.get('base_anomaly_spawns_only', 'false')
        base_anomaly_spawns_only = base_anomaly_spawns_only_str.lower() in ('true', '1', 'yes', 'on')

        # Optional base_anomaly_waypoints_only flag (default False)
        base_anomaly_waypoints_only_str = data.get('base_anomaly_waypoints_only', 'false')
        base_anomaly_waypoints_only = base_anomaly_waypoints_only_str.lower() in ('true', '1', 'yes', 'on')

        # Optional vanilla file hashes (SHA-256 prefix, 16 hex chars)
        vanilla_hash_ai = data.get('vanilla_hash_ai', None)
        if vanilla_hash_ai:
            vanilla_hash_ai = vanilla_hash_ai.strip()
        vanilla_hash_spawn = data.get('vanilla_hash_spawn', None)
        if vanilla_hash_spawn:
            vanilla_hash_spawn = vanilla_hash_spawn.strip()
        vanilla_hash_game = data.get('vanilla_hash_game', None)
        if vanilla_hash_game:
            vanilla_hash_game = vanilla_hash_game.strip()

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
            connect_orphans_automatically=connect_orphans_automatically,
            base_anomaly_spawns_only=base_anomaly_spawns_only,
            base_anomaly_waypoints_only=base_anomaly_waypoints_only,
            vanilla_hash_ai=vanilla_hash_ai,
            vanilla_hash_spawn=vanilla_hash_spawn,
            vanilla_hash_game=vanilla_hash_game
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
    # Path Accessors
    # =========================================================================

    def get_cross_table_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to .gct file for a level."""
        if not self.cross_table_dir:
            return None
        return self.cross_table_dir / f"{level.name}.gct"

    def get_level_ai_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to level.ai file."""
        return level.path / "level.ai"

    def get_edges_path(self, level: LevelConfig) -> Optional[Path]:
        """Get path to edges.json file (if configured)."""
        return level.original_edges

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
            if level.base_anomaly_spawns_only:
                log(f"      Base anomaly spawns only: YES")
            if level.base_anomaly_waypoints_only:
                log(f"      Base anomaly waypoints only: YES")


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
