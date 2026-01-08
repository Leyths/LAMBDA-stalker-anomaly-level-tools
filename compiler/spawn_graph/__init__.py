"""
Spawn Graph Builder Module

Builds chunk 1 (spawn graph) from level.spawn files with support for:
- Merging old spawn data to preserve custom_data/ini_string
- Blacklist support to exclude specific entities
- Smart merging to preserve M_UPDATE packets
- Automatic fix for Anomaly Zones missing M_SPAWN data
- Fix for m_tSpawnID (must equal vertex_id or engine crashes)

This module is organized into submodules:
- entity_categories: Entity section category definitions and helpers
- update_packets: M_UPDATE packet creation and GVID remapping
- builder: Main SpawnGraphBuilder class
"""

# Main builder
from .builder import SpawnGraphBuilder, build_spawn_graph

# Entity categories (for external use if needed)
from .entity_categories import (
    W_GL_SECTIONS,
    SHOTGUN_SECTIONS,
    SIMPLE_WEAPON_SECTIONS,
    OUTFIT_SECTIONS,
    HELMET_SECTIONS,
    ST_MGUN_SECTIONS,
    PHYSICS_SECTIONS,
    PHYSICS_PREFIXES,
    STATIC_SECTIONS,
    STATIC_PREFIXES,
    MONSTER_SECTIONS,
    MONSTER_PREFIXES,
    ANOMALY_PREFIXES,
    is_monster_section,
    is_physics_section,
    is_static_section,
    is_anomaly_section,
)

# Update packet utilities
from .update_packets import (
    create_update_packet,
    get_expected_update_size,
    remap_update_packet_gvids,
)

__all__ = [
    # Main API
    'SpawnGraphBuilder',
    'build_spawn_graph',
    # Entity categories
    'W_GL_SECTIONS',
    'SHOTGUN_SECTIONS',
    'SIMPLE_WEAPON_SECTIONS',
    'OUTFIT_SECTIONS',
    'HELMET_SECTIONS',
    'ST_MGUN_SECTIONS',
    'PHYSICS_SECTIONS',
    'PHYSICS_PREFIXES',
    'STATIC_SECTIONS',
    'STATIC_PREFIXES',
    'MONSTER_SECTIONS',
    'MONSTER_PREFIXES',
    'ANOMALY_PREFIXES',
    'is_monster_section',
    'is_physics_section',
    'is_static_section',
    'is_anomaly_section',
    # Update packets
    'create_update_packet',
    'get_expected_update_size',
    'remap_update_packet_gvids',
]
