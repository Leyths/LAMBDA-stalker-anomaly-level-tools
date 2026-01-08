#!/usr/bin/env python3
"""
Level Changer Config Loader

Loads level changer destination configuration from an INI config file.
Config is authoritative - level changers not in config are removed from spawns.

Config format (level_changers.ini):
    [source_level_name]
    entity_name.dest = dest_level_name
    entity_name.pos = x, y, z
    entity_name.dir = dx, dy, dz

Example:
    [l01_escape]
    esc_level_changer_marsh.dest = k00_marsh
    esc_level_changer_marsh.pos = 123.5, 0.5, 456.2
    esc_level_changer_marsh.dir = 0.0, 0.0, 1.0
"""

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils import log, logDebug, logWarning


@dataclass
class LevelChangerDestination:
    """Destination configuration for a level changer."""
    dest_level_name: str
    position: Tuple[float, float, float]      # (x, y, z)
    direction: Tuple[float, float, float]     # (dir_x, dir_y, dir_z)


class LevelChangerConfig:
    """
    Config loader for level changer destination overrides.

    Provides lookup of override destinations by (source_level, entity_name).
    """

    def __init__(self, config_path: Path):
        """
        Load level changer config from INI file.

        Args:
            config_path: Path to level_changers.ini
        """
        # Dict: (source_level, entity_name) -> LevelChangerDestination
        self._overrides: Dict[Tuple[str, str], LevelChangerDestination] = {}
        self._config_path = config_path
        self._load_config(config_path)

    def _load_config(self, config_path: Path):
        """Parse INI file and populate overrides dict."""
        if not config_path.exists():
            logWarning(f"Level changer config not found: {config_path}")
            return

        config = configparser.ConfigParser()
        # Preserve case sensitivity for section names and keys
        config.optionxform = str

        try:
            config.read(config_path, encoding='utf-8')
        except Exception as e:
            logWarning(f"Failed to parse level changer config: {e}")
            return

        error_count = 0

        for section in config.sections():
            source_level = section

            # Group keys by entity name prefix
            # Format: entity_name.dest, entity_name.pos, entity_name.dir
            entity_keys: Dict[str, Dict[str, str]] = {}

            for key, value in config.items(section):
                if not value.strip():
                    continue

                # Split key into entity_name and suffix
                if '.' not in key:
                    logWarning(f"Invalid key format in [{section}] {key}: expected 'entity_name.dest/pos/dir'")
                    error_count += 1
                    continue

                entity_name, suffix = key.rsplit('.', 1)
                if suffix not in ('dest', 'pos', 'dir'):
                    logWarning(f"Unknown suffix in [{section}] {key}: expected .dest, .pos, or .dir")
                    error_count += 1
                    continue

                if entity_name not in entity_keys:
                    entity_keys[entity_name] = {}
                entity_keys[entity_name][suffix] = value.strip()

            # Process each entity's grouped keys
            for entity_name, keys in entity_keys.items():
                # Require all 3 keys present
                if 'dest' not in keys:
                    logWarning(f"Missing .dest for [{section}] {entity_name}")
                    error_count += 1
                    continue
                if 'pos' not in keys:
                    logWarning(f"Missing .pos for [{section}] {entity_name}")
                    error_count += 1
                    continue
                if 'dir' not in keys:
                    logWarning(f"Missing .dir for [{section}] {entity_name}")
                    error_count += 1
                    continue

                dest_level = keys['dest']

                # Parse position as comma-separated floats
                try:
                    pos_parts = [float(x.strip()) for x in keys['pos'].split(',')]
                    if len(pos_parts) != 3:
                        raise ValueError("expected 3 values")
                    position = (pos_parts[0], pos_parts[1], pos_parts[2])
                except (ValueError, IndexError) as e:
                    logWarning(f"Invalid position format in [{section}] {entity_name}.pos: {e}")
                    error_count += 1
                    continue

                # Parse direction as comma-separated floats
                try:
                    dir_parts = [float(x.strip()) for x in keys['dir'].split(',')]
                    if len(dir_parts) != 3:
                        raise ValueError("expected 3 values")
                    direction = (dir_parts[0], dir_parts[1], dir_parts[2])
                except (ValueError, IndexError) as e:
                    logWarning(f"Invalid direction format in [{section}] {entity_name}.dir: {e}")
                    error_count += 1
                    continue

                config_key = (source_level, entity_name)
                self._overrides[config_key] = LevelChangerDestination(
                    dest_level_name=dest_level,
                    position=position,
                    direction=direction
                )

        if error_count > 0:
            logWarning(f"  {error_count} config errors")

    def get_override(self, source_level: str, entity_name: str) -> Optional[LevelChangerDestination]:
        """
        Get destination override for a level changer.

        Args:
            source_level: Level where the level changer is located
            entity_name: Name of the level changer entity

        Returns:
            LevelChangerDestination if override exists, None otherwise
        """
        return self._overrides.get((source_level, entity_name))

    def has_override(self, source_level: str, entity_name: str) -> bool:
        """Check if an override exists for the given level changer."""
        return (source_level, entity_name) in self._overrides

    @property
    def override_count(self) -> int:
        """Total number of overrides loaded."""
        return len(self._overrides)

    def get_overrides_for_level(self, source_level: str) -> Dict[str, LevelChangerDestination]:
        """
        Get all overrides for a specific source level.

        Args:
            source_level: Level name

        Returns:
            Dict mapping entity_name -> LevelChangerDestination
        """
        result = {}
        for (level, entity), dest in self._overrides.items():
            if level == source_level:
                result[entity] = dest
        return result
