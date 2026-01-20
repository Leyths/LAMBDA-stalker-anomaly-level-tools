#!/usr/bin/env python3
"""
Mod Configuration Parser

Parses INI files (anomaly.ini, gamma.ini) that define which mods are enabled
for a given basemod configuration.

INI Format:
    [Mod Name]
    include = true
    rewrite_files = configs/plugins/file1.ltx
                    configs/plugins/file2.ltx

Each mod is a section where the section name is the mod name.
- include = true/false controls whether mod is enabled
- rewrite_files lists files (one per line) that need tag processing
"""

from pathlib import Path
from typing import List, Optional, Set

from utils import log, logWarning


class ModConfig:
    """
    Parses and provides access to mod configuration from INI files.
    """

    def __init__(self, config_path: Path):
        """
        Load mod configuration from an INI file.

        Args:
            config_path: Path to the INI file (e.g., anomaly.ini, gamma.ini)
        """
        self.config_path = Path(config_path)
        self._enabled_mods: List[str] = []
        self._rewrite_files: dict[str, List[str]] = {}  # mod_name -> list of files

        self._load()

    def _load(self) -> None:
        """Load and parse the INI file."""
        if not self.config_path.exists():
            logWarning(f"Mod config not found: {self.config_path}")
            return

        self._parse_mod_sections()

    def _parse_mod_sections(self) -> None:
        """Parse mod sections, each section name is a mod name."""
        if not self.config_path.exists():
            return

        content = self.config_path.read_text(encoding='utf-8')

        current_section = None
        current_include = False
        current_rewrite_files: List[str] = []
        in_rewrite_files = False

        for line in content.split('\n'):
            stripped = line.strip()

            # Skip comments and empty lines
            if stripped.startswith(';') or stripped.startswith('#') or not stripped:
                in_rewrite_files = False  # Multiline rewrite_files ends on blank/comment
                continue

            # Section header
            if stripped.startswith('[') and stripped.endswith(']'):
                # Save previous section if it was a mod
                if current_section and current_section.lower() != 'config':
                    if current_include:
                        self._enabled_mods.append(current_section)
                    if current_rewrite_files:
                        self._rewrite_files[current_section] = current_rewrite_files

                # Start new section
                current_section = stripped[1:-1].strip()
                current_include = False
                current_rewrite_files = []
                in_rewrite_files = False
                continue

            # Skip [config] section
            if current_section and current_section.lower() == 'config':
                continue

            # Parse key = value pairs
            if '=' in stripped:
                parts = stripped.split('=', 1)
                key = parts[0].strip().lower()
                value = parts[1].strip()

                if key == 'include':
                    current_include = value.lower() in ('true', '1', 'yes', 'on')
                    in_rewrite_files = False
                elif key == 'rewrite_files':
                    in_rewrite_files = True
                    # Value may be on same line or on following lines
                    if value:
                        # Handle comma-separated or single value
                        files = [f.strip() for f in value.split(',') if f.strip()]
                        current_rewrite_files.extend(files)
                else:
                    in_rewrite_files = False
            elif in_rewrite_files:
                # Continuation line for rewrite_files
                # Handle comma-separated or single value
                files = [f.strip() for f in stripped.split(',') if f.strip()]
                current_rewrite_files.extend(files)

        # Save final section
        if current_section and current_section.lower() != 'config':
            if current_include:
                self._enabled_mods.append(current_section)
            if current_rewrite_files:
                self._rewrite_files[current_section] = current_rewrite_files

    def get_enabled_mods(self) -> List[str]:
        """
        Get list of enabled mod names in order.

        Returns:
            List of mod names where include = true
        """
        return self._enabled_mods.copy()

    def is_mod_enabled(self, mod_name: str) -> bool:
        """
        Check if a specific mod is enabled.

        Args:
            mod_name: Name of the mod to check

        Returns:
            True if mod is enabled, False otherwise
        """
        return mod_name in self._enabled_mods

    def get_rewrite_files(self, mod_name: str) -> List[str]:
        """
        Get list of files that need tag rewriting for a mod.

        Args:
            mod_name: Name of the mod

        Returns:
            List of relative file paths that need tag processing
        """
        return self._rewrite_files.get(mod_name, []).copy()

    def get_all_rewrite_files(self) -> Set[str]:
        """
        Get all rewrite files from all enabled mods.

        Returns:
            Set of relative file paths that need tag processing
        """
        all_files: Set[str] = set()
        for mod_name in self._enabled_mods:
            all_files.update(self._rewrite_files.get(mod_name, []))
        return all_files

    def __repr__(self) -> str:
        return f"ModConfig({self.config_path}, enabled_mods={self._enabled_mods})"
