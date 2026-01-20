#!/usr/bin/env python3
"""
Mod Copier

Copies mod files from mods/<mod_name>/ to gamedata/, preserving directory structure.
Supports selective copying based on enabled mods from ModConfig.
Supports tag-based file rewriting for files specified in rewrite_files.
"""

import shutil
from pathlib import Path
from typing import List, Set, Optional, TYPE_CHECKING

from utils import log

from .mod_config import ModConfig
from .tag_rewriter import TagRewriter

if TYPE_CHECKING:
    from graph import GameGraph


class ModCopier:
    """
    Copies mod files to gamedata directory.
    Supports tag-based rewriting for files specified in mod config.
    """

    def __init__(self, mods_dir: Path, gamedata_dir: Path, game_graph: Optional['GameGraph'] = None):
        """
        Initialize mod copier.

        Args:
            mods_dir: Path to mods/ directory
            gamedata_dir: Path to gamedata/ directory
            game_graph: Optional GameGraph for tag rewriting (required for rewrite_files)
        """
        self.mods_dir = Path(mods_dir)
        self.gamedata_dir = Path(gamedata_dir)
        self.game_graph = game_graph
        self._tag_rewriter: Optional[TagRewriter] = None

    @property
    def tag_rewriter(self) -> Optional[TagRewriter]:
        """Lazy-initialize tag rewriter when needed."""
        if self._tag_rewriter is None and self.game_graph is not None:
            self._tag_rewriter = TagRewriter(self.game_graph)
        return self._tag_rewriter

    def copy_mod(self, mod_name: str, config: Optional[ModConfig] = None,
                 skip_files: Optional[Set[str]] = None) -> int:
        """
        Copy a single mod's files to gamedata.

        Args:
            mod_name: Name of the mod folder
            config: Optional ModConfig to check rewrite_files
            skip_files: Set of filenames to skip (e.g., {'new_game_start_locations.ltx'})

        Returns:
            Number of files copied/processed
        """
        skip_files = skip_files or set()
        mod_path = self.mods_dir / mod_name

        if not mod_path.exists():
            log(f"    Warning: Mod directory not found: {mod_path}")
            return 0

        if not mod_path.is_dir():
            log(f"    Warning: Mod path is not a directory: {mod_path}")
            return 0

        # Get rewrite files for this mod
        rewrite_files: Set[str] = set()
        if config:
            rewrite_files = set(config.get_rewrite_files(mod_name))

        copied_count = 0

        for src_path in mod_path.rglob('*'):
            if src_path.is_file():
                # Skip files in the skip set
                if src_path.name in skip_files:
                    continue

                # Skip .DS_Store and other hidden files
                if src_path.name.startswith('.'):
                    continue

                # Calculate destination path (strip mod folder prefix)
                rel_path = src_path.relative_to(mod_path)
                dst_path = self.gamedata_dir / rel_path

                # Create parent directories
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if this file needs tag rewriting
                rel_path_str = str(rel_path).replace('\\', '/')
                if rel_path_str in rewrite_files:
                    if self.tag_rewriter:
                        log(f"    Rewriting: {rel_path}")
                        self.tag_rewriter.rewrite_file(src_path, dst_path)
                    else:
                        log(f"    Warning: Cannot rewrite {rel_path} (no game_graph), copying instead")
                        shutil.copy2(src_path, dst_path)
                else:
                    # Normal copy
                    shutil.copy2(src_path, dst_path)
                    log(f"    Copied: {rel_path}")

                copied_count += 1

        return copied_count

    def copy_all_enabled_mods(self, config: ModConfig, skip_files: Optional[Set[str]] = None) -> int:
        """
        Copy all enabled mods to gamedata.

        Args:
            config: ModConfig instance with enabled mods
            skip_files: Set of filenames to skip

        Returns:
            Total number of files copied
        """
        skip_files = skip_files or set()
        enabled_mods = config.get_enabled_mods()

        if not enabled_mods:
            log("  No mods enabled in configuration")
            return 0

        log(f"  Copying {len(enabled_mods)} enabled mods...")
        total_copied = 0

        for mod_name in enabled_mods:
            log(f"  Processing mod: {mod_name}")
            copied = self.copy_mod(mod_name, config, skip_files)
            total_copied += copied
            if copied > 0:
                log(f"    Files processed: {copied}")

        log(f"  Total files processed: {total_copied}")
        return total_copied

    def get_mod_file(self, mod_name: str, relative_path: str) -> Optional[Path]:
        """
        Get the path to a specific file within a mod.

        Args:
            mod_name: Name of the mod
            relative_path: Relative path within the mod (e.g., 'configs/plugins/file.ltx')

        Returns:
            Full path to the file, or None if it doesn't exist
        """
        file_path = self.mods_dir / mod_name / relative_path
        if file_path.exists():
            return file_path
        return None

    def find_file_in_enabled_mods(self, config: ModConfig, relative_path: str) -> Optional[Path]:
        """
        Find a file in enabled mods (first match wins, based on mod order).

        Args:
            config: ModConfig instance
            relative_path: Relative path to search for

        Returns:
            Path to the first matching file, or None
        """
        for mod_name in config.get_enabled_mods():
            file_path = self.get_mod_file(mod_name, relative_path)
            if file_path:
                return file_path
        return None
