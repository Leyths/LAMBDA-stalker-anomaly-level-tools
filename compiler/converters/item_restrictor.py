"""
Item Restrictor Converter

Creates ONE space_restrictor per level for dynamic item spawning.
The restrictor triggers the Lua script which reads all spawn locations
from dynamic_item_spawn_locations.ltx and spawns items at runtime.
"""

from .base_restrictor import BaseRestrictorConverter, create_restrictors_for_level


class ItemRestrictorConverter(BaseRestrictorConverter):
    entity_prefix = "sr_item_spawner_"
    logic_section = "sr_item_spawner"
    fallback_config_rel = "configs/items/settings/dynamic_item_spawn_locations.ltx"
    log_label = "item restrictors"


def create_item_restrictors_for_level(converter, level_name):
    return create_restrictors_for_level(converter, level_name)
