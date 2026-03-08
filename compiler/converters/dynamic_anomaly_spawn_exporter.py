"""
Dynamic Anomaly Spawn Exporter

During the build, extracts positions and types from 'dynamic_anomaly_spawn_*'
anomalous zone entities found in level.spawn, and writes them back to the
config file (build output only — mods/ source is untouched).

These entities are created by the editor import script
(editorscripts/import_dynamic_anomaly_spawns.py) and should NOT be included
in the final all.spawn — they exist only for SDK visualization.
"""

from typing import Tuple

from .base_spawn_exporter import BaseSpawnExporter


# Reverse mapping: section name → anomaly type
SECTION_TO_TYPE = {
    'zone_field_radioactive_weak': 'radioactive',
    'zone_mine_gravitational_weak': 'gravitational',
    'zone_mine_thermal_weak': 'thermal',
    'zone_mine_electric_weak': 'electric',
    'zone_mine_acidic_weak': 'chemical',
}


class DynamicAnomalySpawnExporter(BaseSpawnExporter):
    config_rel = "configs/zones/dynamic_anomaly_locations.ltx"
    entity_prefix = "dynamic_anomaly_spawn_"
    log_label = "dynamic anomaly spawn"

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
        ano_name = entity_name
        if ano_name.startswith(self.entity_prefix):
            ano_name = ano_name[len(self.entity_prefix):]

        anomaly_type = SECTION_TO_TYPE.get(section_name, 'radioactive')
        self._add_entry(level_name, ano_name, anomaly_type,
                        position[0], position[1], position[2])
