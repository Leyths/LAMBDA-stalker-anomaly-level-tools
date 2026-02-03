"""
Dynamic Item Spawn Exporter

During the build, extracts positions and categories from 'dynamic_item_spawn_*'
space_restrictor entities found in level.spawn, and writes them back to the
config file (build output only — mods/ source is untouched).

These entities are created by the editor import script
(editorscripts/import_dynamic_item_spawns.py) and should NOT be included
in the final all.spawn — they exist only for SDK visualization.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import log, logError


class DynamicItemSpawnExporter:
    """
    Collects dynamic_item_spawn_* entity positions during the build and
    writes an updated dynamic_item_spawn_locations.ltx config file.
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

        # Collected data: level_name -> list of (item_name, category, x, y, z)
        self._level_items: Dict[str, List[Tuple[str, str, float, float, float]]] = {}

        # Output config path (gamedata/ build output, NOT mods/ source)
        self._config_path = (
            Path(__file__).parent.parent.parent
            / "gamedata/configs/items/settings/dynamic_item_spawn_locations.ltx"
        )

        # Source config path — read from mods/ to get the full structure
        # including trailing sections (categories, possible_uses, item lists)
        self._source_config_path = source_config_path

        # Preserved config structure
        self._header_lines: List[str] = []
        self._levels_list: List[str] = []
        self._trailing_sections: List[str] = []

        if not source_config_path or not source_config_path.exists():
            logError(f"Dynamic item spawn source config not found: {source_config_path}")
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
                       position: Tuple[float, float, float],
                       custom_data: str):
        """
        Called during build for each dynamic_item_spawn_* entity found.

        Args:
            level_name: Level the entity was found in
            entity_name: Full entity name (e.g. 'dynamic_item_spawn_itm_jup_0')
            position: (x, y, z) position tuple
            custom_data: Custom data string (e.g. '[item_spawn]\\ntype = ammo\\n')
        """
        # Strip prefix to get item name
        item_name = entity_name
        if item_name.startswith('dynamic_item_spawn_'):
            item_name = item_name[len('dynamic_item_spawn_'):]

        # Parse category from custom_data
        category = self._parse_category(custom_data)

        if level_name not in self._level_items:
            self._level_items[level_name] = []

        self._level_items[level_name].append((
            item_name, category, position[0], position[1], position[2]
        ))

    def _parse_category(self, custom_data: str) -> str:
        """
        Parse item category from custom_data string.

        The custom_data format is:
            [item_spawn]
            type = ammo

        Or in escaped form: [item_spawn]\\ntype = ammo\\n
        """
        if not custom_data:
            return 'misc'

        # Handle both real newlines and escaped newlines
        normalized = custom_data.replace('\\n', '\n')

        for line in normalized.split('\n'):
            line = line.strip()
            if line.startswith('type') and '=' in line:
                _, value = line.split('=', 1)
                return value.strip()

        return 'misc'

    def write_config(self):
        """
        Write updated config file with collected positions.

        Called after all levels have been processed. Preserves header comments
        and [levels] section. For levels with collected entities, writes updated
        positions. For levels without collected entities, preserves original data.

        Always writes the file if the source config was loaded, even when no
        new entities were collected — ensures gamedata/ has the file on clean builds.
        """
        if not self._level_items and not self._levels_list:
            return

        # Load original per-level data for levels we didn't collect
        original_level_data = self._load_original_level_data()

        # Merge: use collected data where available, original otherwise
        all_levels = set(self._levels_list)
        for level in self._level_items:
            all_levels.add(level)

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
        for level in sorted(self._level_items.keys()):
            if level not in final_levels:
                final_levels.append(level)

        for level in final_levels:
            lines.append(level)

        # Per-level sections
        for level in final_levels:
            lines.append('')
            lines.append(f'[{level}]')

            if level in self._level_items:
                # Write updated items sorted by name
                items = sorted(self._level_items[level], key=lambda t: t[0])
                for item_name, category, x, y, z in items:
                    lines.append(f'{item_name} = {category}, {x}, {y}, {z}')
            elif level in original_level_data:
                # Preserve original data
                for item_line in original_level_data[level]:
                    lines.append(item_line)

        # Append trailing non-level sections (categories, possible_uses, item lists)
        if self._trailing_sections:
            lines.append('')
            lines.extend(self._trailing_sections)

        # Write file
        lines.append('')  # Trailing newline
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text('\n'.join(lines), encoding='utf-8')

        # Log summary
        total = sum(len(items) for items in self._level_items.values())
        log(f"  Wrote {total} dynamic item spawn positions to {self._config_path}")
        for level in sorted(self._level_items.keys()):
            log(f"    {level}: {len(self._level_items[level])} items")

    def _load_original_level_data(self) -> Dict[str, List[str]]:
        """Load original per-level item data from source config."""
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
        return sum(len(items) for items in self._level_items.values())
