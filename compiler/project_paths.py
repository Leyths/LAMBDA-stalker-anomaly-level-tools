#!/usr/bin/env python3
"""
Project Paths

Central path resolution for the stalkertool build system.
All paths are resolved once at startup and passed through the pipeline.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import ModConfig


@dataclass(frozen=True)
class ProjectPaths:
    """
    Immutable container for all resolved project paths.

    Constructed once at startup (from GUI or CLI) and threaded through
    the entire build pipeline. Eliminates CWD-relative path resolution.
    """
    project_root: Path
    compiler_dir: Path
    levels_dir: Path
    levels_override_dir: Optional[Path]  # partial overrides for level files
    output_dir: Path        # the "gamedata" directory
    output_spawn: Path      # output_dir / "spawns" / "all.spawn"
    mods_dir: Path
    build_dir: Path         # .tmp cache directory

    @classmethod
    def from_root(cls, project_root: Path,
                  levels_dir: Path = None,
                  output_dir: Path = None,
                  levels_override_dir: Path = None) -> 'ProjectPaths':
        """
        Construct ProjectPaths from a project root directory.

        Args:
            project_root: Root directory of the project
            levels_dir: Override for levels directory (default: project_root/levels)
            output_dir: Override for output gamedata directory (default: project_root/gamedata)
            levels_override_dir: Directory with partial level file overrides.
                                 Files here take priority over levels_dir on a per-file basis.
        """
        project_root = Path(project_root).resolve()
        levels_dir = Path(levels_dir).resolve() if levels_dir else project_root / "levels"
        output_dir = Path(output_dir).resolve() if output_dir else project_root / "gamedata"
        levels_override = Path(levels_override_dir).resolve() if levels_override_dir else None

        build_dir = project_root / ".tmp"
        build_dir.mkdir(exist_ok=True)
        # Mark as hidden on Windows
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.kernel32.SetFileAttributesW(str(build_dir), 0x02)
            except Exception:
                pass

        return cls(
            project_root=project_root,
            compiler_dir=project_root / "compiler",
            levels_dir=levels_dir,
            levels_override_dir=levels_override,
            output_dir=output_dir,
            output_spawn=output_dir / "spawns" / "all.spawn",
            mods_dir=project_root / "mods",
            build_dir=build_dir,
        )

    def get_base_mod_ini(self, base_mod: str) -> Path:
        """Get path to the base mod INI file (e.g., anomaly.ini)."""
        return self.project_root / f"{base_mod}.ini"

    def get_blacklist_ini(self) -> Optional[Path]:
        """Get path to spawn_blacklist.ini if it exists."""
        p = self.project_root / "spawn_blacklist.ini"
        return p if p.exists() else None

    def get_level_changers_ini(self) -> Optional[Path]:
        """Get path to level_changers.ini if it exists."""
        p = self.project_root / "level_changers.ini"
        return p if p.exists() else None

    def get_levels_ini(self) -> Path:
        """Get path to levels.ini."""
        return self.project_root / "levels.ini"

    def find_mod_config(self, mod_config: 'ModConfig', config_rel: str) -> Optional[Path]:
        """
        Search enabled mods for a config file by relative path.

        Args:
            mod_config: ModConfig instance with enabled mods list
            config_rel: Relative path within a mod directory
                        (e.g., "configs/items/settings/dynamic_item_spawn_locations.ltx")

        Returns:
            Path to the first matching file, or None
        """
        if not mod_config:
            return None
        config_rel_path = Path(config_rel)
        for mod_name in mod_config.get_enabled_mods():
            candidate = self.mods_dir / mod_name / config_rel_path
            if candidate.exists():
                return candidate
        return None
