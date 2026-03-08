"""
Anomaly Restrictor Converter

Creates ONE space_restrictor per level for dynamic anomaly spawning.
The restrictor triggers the Lua script which reads all spawn locations
from dynamic_anomaly_locations.ltx and spawns anomalies at runtime.
"""

from .base_restrictor import BaseRestrictorConverter, create_restrictors_for_level


class AnomalyRestrictorConverter(BaseRestrictorConverter):
    entity_prefix = "sr_dynamic_anomaly_"
    logic_section = "sr_dynamic_anomaly"
    fallback_config_rel = "configs/zones/dynamic_anomaly_locations.ltx"
    log_label = "anomaly restrictors"


def create_anomaly_restrictors_for_level(converter, level_name):
    return create_restrictors_for_level(converter, level_name)
