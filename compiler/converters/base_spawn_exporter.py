"""
Base Spawn Exporter

Shared logic for exporters that collect entity positions during the build
and write them back to an INI-style config file. Subclasses define the
config path, entity prefix, log label, and how to format collected entries.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import log, logError


class BaseSpawnExporter:
    """
    Base class for dynamic spawn exporters (items and anomalies).

    Handles:
      - Loading/preserving config structure from source (mods/) path
      - Writing updated config with collected positions
      - Preserving data for levels without collected entities

    Subclasses must set class attributes:
        config_rel: str         e.g. "configs/zones/dynamic_anomaly_locations.ltx"
        entity_prefix: str      e.g. "dynamic_anomaly_spawn_"
        log_label: str          e.g. "dynamic anomaly spawn"
    """

    config_rel: str
    entity_prefix: str
    log_label: str

    def __init__(self, source_config_path: Path = None, output_dir: Path = None):
        # Collected data: level_name -> list of (name, type_or_category, x, y, z)
        self._level_entries: Dict[str, List[Tuple[str, str, float, float, float]]] = {}

        if not output_dir:
            raise ValueError(f"output_dir is required for {self.log_label}")

        # Output config path (user-configured gamedata/ build output, NOT mods/ source)
        self._config_path = output_dir / self.config_rel

        # Source config path — read from mods/ to get the full structure
        self._source_config_path = source_config_path

        # Preserved config structure
        self._header_lines: List[str] = []
        self._levels_list: List[str] = []
        self._trailing_sections: List[str] = []

        if not source_config_path or not source_config_path.exists():
            logError(f"{self.log_label} source config not found: {source_config_path}")
            return

        self._load_existing_config()

    def _load_existing_config(self):
        """Load existing config to preserve structure. Reads from source (mods/) path."""
        if not self._source_config_path.exists():
            return

        content = self._source_config_path.read_text(encoding='utf-8', errors='ignore')

        # First pass: collect [levels] list so we know which sections are level data
        current_section = None
        for line in content.split('\n'):
            stripped = line.strip()
            section_match = re.match(r'^\[([^\]]+)\]', stripped)
            if section_match:
                current_section = section_match.group(1)
                continue
            if current_section == 'levels' and stripped and not stripped.startswith(';') and not stripped.startswith('--'):
                self._levels_list.append(stripped)

        # Build set of known level names for fast lookup
        level_names = set(self._levels_list)

        # Second pass: collect header, and trailing non-level sections
        current_section = None
        in_header = True
        in_trailing = False

        for line in content.split('\n'):
            stripped = line.strip()

            section_match = re.match(r'^\[([^\]]+)\]', stripped)
            if section_match:
                current_section = section_match.group(1)
                if current_section == 'levels':
                    in_header = False
                elif current_section not in level_names and not in_header:
                    in_trailing = True

                if in_trailing:
                    self._trailing_sections.append(line.rstrip())
                continue

            if in_header:
                self._header_lines.append(line.rstrip())
            elif in_trailing:
                self._trailing_sections.append(line.rstrip())

    def _add_entry(self, level_name: str, name: str, type_or_category: str,
                   x: float, y: float, z: float):
        """Add a collected entry for a level."""
        if level_name not in self._level_entries:
            self._level_entries[level_name] = []
        self._level_entries[level_name].append((name, type_or_category, x, y, z))

    def _format_entry(self, name: str, type_or_category: str,
                      x: float, y: float, z: float) -> str:
        """Format a single entry line. Override if needed."""
        return f'{name} = {type_or_category}, {x}, {y}, {z}'

    def write_config(self):
        """
        Write updated config file with collected positions.

        Preserves header comments and [levels] section. For levels with
        collected entities, writes updated positions. For levels without
        collected entities, preserves original data.
        """
        if not self._level_entries and not self._levels_list:
            return

        # Load original per-level data for levels we didn't collect
        original_level_data = self._load_original_level_data()

        # Build output
        lines = []

        # Header comments
        for header_line in self._header_lines:
            lines.append(header_line)

        # [levels] section
        lines.append('')
        lines.append('[levels]')

        # Determine final level list — preserve order, add new levels at end
        final_levels = list(self._levels_list)
        for level in sorted(self._level_entries.keys()):
            if level not in final_levels:
                final_levels.append(level)

        for level in final_levels:
            lines.append(level)

        # Per-level sections
        for level in final_levels:
            lines.append('')
            lines.append(f'[{level}]')

            if level in self._level_entries:
                # Write updated entries sorted by name
                entries = sorted(self._level_entries[level], key=lambda t: t[0])
                for name, type_or_cat, x, y, z in entries:
                    lines.append(self._format_entry(name, type_or_cat, x, y, z))
            elif level in original_level_data:
                # Preserve original data
                for item_line in original_level_data[level]:
                    lines.append(item_line)

        # Append trailing non-level sections
        if self._trailing_sections:
            lines.append('')
            lines.extend(self._trailing_sections)

        # Write file
        lines.append('')  # Trailing newline
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text('\n'.join(lines), encoding='utf-8')

        # Log summary
        total = sum(len(entries) for entries in self._level_entries.values())
        log(f"  Wrote {total} {self.log_label} positions to {self._config_path}")
        for level in sorted(self._level_entries.keys()):
            log(f"    {level}: {len(self._level_entries[level])} entries")

    def _load_original_level_data(self) -> Dict[str, List[str]]:
        """Load original per-level data from source config."""
        data: Dict[str, List[str]] = {}

        if not self._source_config_path.exists():
            return data

        level_names = set(self._levels_list)
        content = self._source_config_path.read_text(encoding='utf-8', errors='ignore')
        current_section = None

        for line in content.split('\n'):
            stripped = line.strip()

            section_match = re.match(r'^\[([^\]]+)\]', stripped)
            if section_match:
                current_section = section_match.group(1)
                continue

            if current_section and current_section in level_names:
                if stripped and not stripped.startswith(';') and not stripped.startswith('--'):
                    if current_section not in data:
                        data[current_section] = []
                    data[current_section].append(stripped)

        return data

    @property
    def collected_count(self) -> int:
        """Total number of collected entities across all levels."""
        return sum(len(entries) for entries in self._level_entries.values())
