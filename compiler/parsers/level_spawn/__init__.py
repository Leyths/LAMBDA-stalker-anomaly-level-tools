"""
Level Spawn File Parser

Parses level.spawn files containing entity definitions.
Supports both formats:
- NEW format: Direct M_SPAWN packets (from editor)
- OLD format: Nested chunks with M_SPAWN + M_UPDATE (from extract_spawns.py)

Also provides AllSpawnSpawnIterator for parsing spawn entities from all.spawn.

This module is organized into submodules:
- data_types: Shared data structures (Shape, ALifeObject, LevelChangerData, SpawnEntity, GraphPoint)
- entity_parser: Main spawn packet parsing
- shape_parser: CSE_Shape parsing for space_restrictor entities
- alife_parser: CSE_ALifeObject parsing
- level_changer: CSE_ALifeLevelChanger parsing
- parsers: Main parser classes (LevelSpawnParser, AllSpawnSpawnIterator)
"""

# Data types
from .data_types import (
    Shape,
    ALifeObject,
    LevelChangerData,
    SpawnEntity,
    GraphPoint,
)

# Entity parsing
from .entity_parser import parse_spawn_packet, M_SPAWN

# Shape parsing
from .shape_parser import parse_entity_shapes

# ALife object parsing
from .alife_parser import parse_alife_object

# Level changer parsing
from .level_changer import parse_level_changer_data, parse_level_changer_packet

# Main parser classes
from .parsers import (
    SpawnFormat,
    LevelSpawnParser,
    AllSpawnSpawnIterator,
    detect_spawn_format,
)

# String utilities (internal, but exposed for backward compatibility)
from .string_utils import read_stringz_cp1251

__all__ = [
    # Data types
    'Shape',
    'ALifeObject',
    'LevelChangerData',
    'SpawnEntity',
    'GraphPoint',
    # Entity parsing
    'parse_spawn_packet',
    'M_SPAWN',
    # Shape parsing
    'parse_entity_shapes',
    # ALife object parsing
    'parse_alife_object',
    # Level changer parsing
    'parse_level_changer_data',
    'parse_level_changer_packet',
    # Parser classes
    'SpawnFormat',
    'LevelSpawnParser',
    'AllSpawnSpawnIterator',
    'detect_spawn_format',
    # String utilities
    'read_stringz_cp1251',
]
