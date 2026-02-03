"""
Dynamic Anomaly Spawn Exporter

During the build, extracts positions and types from 'dynamic_anomaly_spawn_*'
anomalous zone entities found in level.spawn, and writes them back to the
config file (build output only — mods/ source is untouched).

These entities are created by the editor import script
(editorscripts/import_dynamic_anomaly_spawns.py) and should NOT be included
in the final all.spawn — they exist only for SDK visualization.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import log, logError


# Reverse mapping: section name → anomaly type
SECTION_TO_TYPE = {
    'zone_field_radioactive_weak': 'radioactive',
    'zone_mine_gravitational_weak': 'gravitational',
    'zone_mine_thermal_weak': 'thermal',
    'zone_mine_electric_weak': 'electric',
    'zone_mine_acidic_weak': 'chemical',
}


class DynamicAnomalySpawnExporter:
    """
    Collects dynamic_anomaly_spawn_* entity positions during the build and
    writes an updated dynamic_anomaly_locations.ltx config file.
    """

    def __init__(self, base_mod: str = None, source_config_path: Path = None):
        """
        Initialize the exporter.

        Args:
            base_mod: Base mod variant (anomaly, gamma) — for future use
            source_config_path: Path to the mods/ source config to read structure from.
                                Must be provided and must exist.
        """
        self.base_mod = base_mod

        # Collected data: level_name -> list of (ano_name, type, x, y, z)
        self._level_anomalies: Dict[str, List[Tuple[str, str, float, float, float]]] = {}

        # Output config path (gamedata/ build output, NOT mods/ source)
        self._config_path = (
            Path(__file__).parent.parent.parent
            / "gamedata/configs/zones/dynamic_anomaly_locations.ltx"
        )

        # Source config path — read from mods/ to get the full structure
        # including trailing sections (categories, type-specific sections, etc.)
        self._source_config_path = source_config_path

        # Preserved config structure
        self._header_lines: List[str] = []
        self._levels_list: List[str] = []
        self._trailing_sections: List[str] = []

        if not source_config_path or not source_config_path.exists():
            logError(f"Dynamic anomaly spawn source config not found: {source_config_path}")
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
                    # This is a non-level section after [levels] — start of trailing content
                    in_trailing = True

                if in_trailing:
                    self._trailing_sections.append(line.rstrip())
                continue

            if in_header:
                self._header_lines.append(line.rstrip())
            elif in_trailing:
                self._trailing_sections.append(line.rstrip())

    def collect_entity(self, level_name: str, entity_name: str,
                       section_name: str, position: Tuple[float, float, float]):
        """
        Called during build for each dynamic_anomaly_spawn_* entity found.

        Args:
            level_name: Level the entity was found in
            entity_name: Full entity name (e.g. 'dynamic_anomaly_spawn_ano_jup_0')
            section_name: Entity section name (e.g. 'zone_mine_electric_weak')
            position: (x, y, z) position tuple
        """
        # Strip prefix to get anomaly name
        ano_name = entity_name
        if ano_name.startswith('dynamic_anomaly_spawn_'):
            ano_name = ano_name[len('dynamic_anomaly_spawn_'):]

        # Parse type from section name (reverse mapping)
        anomaly_type = SECTION_TO_TYPE.get(section_name, 'radioactive')

        if level_name not in self._level_anomalies:
            self._level_anomalies[level_name] = []

        self._level_anomalies[level_name].append((
            ano_name, anomaly_type, position[0], position[1], position[2]
        ))

    def write_config(self):
        """
        Write updated config file with collected positions.

        Called after all levels have been processed. Preserves header comments
        and [levels] section. For levels with collected entities, writes updated
        positions. For levels without collected entities, preserves original data.

        Always writes the file if the source config was loaded, even when no
        new entities were collected — ensures gamedata/ has the file on clean builds.
        """
        if not self._level_anomalies and not self._levels_list:
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
        for level in sorted(self._level_anomalies.keys()):
            if level not in final_levels:
                final_levels.append(level)

        for level in final_levels:
            lines.append(level)

        # Per-level sections
        for level in final_levels:
            lines.append('')
            lines.append(f'[{level}]')

            if level in self._level_anomalies:
                # Write updated anomalies sorted by name
                anomalies = sorted(self._level_anomalies[level], key=lambda t: t[0])
                for ano_name, anomaly_type, x, y, z in anomalies:
                    lines.append(f'{ano_name} = {anomaly_type}, {x}, {y}, {z}')
            elif level in original_level_data:
                # Preserve original data
                for item_line in original_level_data[level]:
                    lines.append(item_line)

        # Append trailing non-level sections (categories, type-specific sections, etc.)
        if self._trailing_sections:
            lines.append('')
            lines.extend(self._trailing_sections)

        # Write file
        lines.append('')  # Trailing newline
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text('\n'.join(lines), encoding='utf-8')

        # Log summary
        total = sum(len(anomalies) for anomalies in self._level_anomalies.values())
        log(f"  Wrote {total} dynamic anomaly spawn positions to {self._config_path}")
        for level in sorted(self._level_anomalies.keys()):
            log(f"    {level}: {len(self._level_anomalies[level])} anomalies")

    def _load_original_level_data(self) -> Dict[str, List[str]]:
        """Load original per-level anomaly data from source config."""
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

            # Only collect data from level sections (not [levels], [categories], etc.)
            if current_section and current_section in level_names:
                if stripped and not stripped.startswith(';') and not stripped.startswith('--'):
                    if current_section not in data:
                        data[current_section] = []
                    data[current_section].append(stripped)

        return data

    @property
    def collected_count(self) -> int:
        """Total number of collected entities across all levels."""
        return sum(len(anomalies) for anomalies in self._level_anomalies.values())
