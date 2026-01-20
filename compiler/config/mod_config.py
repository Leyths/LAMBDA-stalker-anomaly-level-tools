#!/usr/bin/env python3
"""
Mod Configuration Parser

Parses INI files (anomaly.ini, gamma.ini) that define which mods are enabled
for a given basemod configuration.

INI Format:
    [config]
    ; Global configuration options (reserved for future use)

    [mods]
    Dynamic Items And Anomalies Anomaly = true
    _New Game Start Locations Anomaly = true
"""

import configparser
from pathlib import Path
from typing import List, Optional

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
        self.config = configparser.ConfigParser()
        self._enabled_mods: List[str] = []

        self._load()

    def _load(self) -> None:
        """Load and parse the INI file."""
        if not self.config_path.exists():
            logWarning(f"Mod config not found: {self.config_path}")
            return

        # Read with UTF-8 encoding
        self.config.read(self.config_path, encoding='utf-8')

        # Parse [mods] section
        if 'mods' in self.config:
            for mod_name, value in self.config['mods'].items():
                # ConfigParser lowercases keys, so we need to preserve original case
                # Re-read the file to get original case
                pass

        # Re-read to preserve case (ConfigParser lowercases keys by default)
        self._parse_mods_section()

    def _parse_mods_section(self) -> None:
        """Parse [mods] section preserving original key case."""
        if not self.config_path.exists():
            return

        content = self.config_path.read_text(encoding='utf-8')
        in_mods_section = False

        for line in content.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith(';') or line.startswith('#') or not line:
                continue

            # Section header
            if line.startswith('[') and line.endswith(']'):
                section_name = line[1:-1].strip().lower()
                in_mods_section = (section_name == 'mods')
                continue

            # Parse mod entries in [mods] section
            if in_mods_section and '=' in line:
                parts = line.split('=', 1)
                mod_name = parts[0].strip()
                value = parts[1].strip().lower()

                if value in ('true', '1', 'yes', 'on'):
                    self._enabled_mods.append(mod_name)

    def get_enabled_mods(self) -> List[str]:
        """
        Get list of enabled mod names in order.

        Returns:
            List of mod names where value is true
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

    def get_config_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a value from the [config] section.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if 'config' in self.config:
            return self.config['config'].get(key, default)
        return default

    def __repr__(self) -> str:
        return f"ModConfig({self.config_path}, enabled_mods={self._enabled_mods})"
