"""
Dynamic Item Spawn Exporter

During the build, extracts positions and categories from 'dynamic_item_spawn_*'
space_restrictor entities found in level.spawn, and writes them back to the
config file (build output only — mods/ source is untouched).

These entities are created by the editor import script
(editorscripts/import_dynamic_item_spawns.py) and should NOT be included
in the final all.spawn — they exist only for SDK visualization.
"""

from typing import Tuple

from .base_spawn_exporter import BaseSpawnExporter


class DynamicItemSpawnExporter(BaseSpawnExporter):
    config_rel = "configs/items/settings/dynamic_item_spawn_locations.ltx"
    entity_prefix = "dynamic_item_spawn_"
    log_label = "dynamic item spawn"

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
        item_name = entity_name
        if item_name.startswith(self.entity_prefix):
            item_name = item_name[len(self.entity_prefix):]

        category = self._parse_category(custom_data)
        self._add_entry(level_name, item_name, category,
                        position[0], position[1], position[2])

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

        normalized = custom_data.replace('\\n', '\n')
        for line in normalized.split('\n'):
            line = line.strip()
            if line.startswith('type') and '=' in line:
                _, value = line.split('=', 1)
                return value.strip()

        return 'misc'
