"""
X-Ray Engine Binary File Parsers

This package provides clean parser classes for various X-Ray Engine binary file formats:

- base: Shared utilities (read_stringz, ChunkReader)
- cross_table: CrossTableParser for .gct files
- level_ai: LevelAIParser for .ai files
- level_spawn: LevelSpawnParser for .spawn files
- game_graph: GameGraphParser for all.spawn chunk 4
- patrol_paths: PatrolPathParser for all.spawn chunk 3 and .patrols files

Usage:
    from parsers import CrossTableParser, LevelAIParser, LevelSpawnParser

    # Parse cross table
    ct = CrossTableParser("level.gct")
    game_vertex = ct.get_game_vertex_id(level_vertex_id=1234)

    # Parse level AI
    ai = LevelAIParser("level.ai")
    pos = ai.get_vertex_position(vertex_id=1234)
    nearest = ai.find_nearest_vertex((100.0, 0.0, 200.0))

    # Parse spawn file
    sp = LevelSpawnParser("level.spawn")
    for entity in sp.get_entities():
        print(entity.section_name, entity.entity_name)
"""

# Base utilities
from .base import (
    read_stringz,
    read_chunk_header,
    Chunk,
    ChunkReader,
)

# Cross table parser
from .cross_table import (
    CrossTableParser,
    CrossTableHeader,
    find_game_vertex_from_cross_table,  # Legacy wrapper
)

# Level AI parser
from .level_ai import (
    LevelAIParser,
    LevelAIHeader,
    load_level_ai_positions,  # Legacy wrapper
    find_nearest_level_vertex,  # Legacy wrapper
)

# Level spawn parser
from .level_spawn import (
    LevelSpawnParser,
    SpawnFormat,
    SpawnEntity,
    Shape,
    ALifeObject,
    LevelChangerData,
    GraphPoint,
    detect_spawn_format,  # Legacy wrapper
    parse_spawn_packet,  # Shared utility
    parse_entity_shapes,  # Parse shapes from space_restrictor entities
    parse_alife_object,  # Parse ALifeObject data from entities
    parse_level_changer_data,  # Parse level_changer destination data from SpawnEntity
    parse_level_changer_packet,  # Parse level_changer destination data from raw packet
    AllSpawnSpawnIterator,  # For parsing all.spawn spawn chunk
)

# Game graph parser
from .game_graph import (
    GameGraphParser,
    GameGraphHeader,
    GameGraphLevel,
    GameGraphVertex,
    GameGraphEdge,
)

# Patrol path parser
from .patrol_paths import (
    PatrolPathParser,
    PatrolPath,
    PatrolPoint,
    PatrolEdge,
    read_extracted_patrols,  # Legacy wrapper
)

__all__ = [
    # Base
    'read_stringz',
    'read_chunk_header',
    'Chunk',
    'ChunkReader',
    # Cross table
    'CrossTableParser',
    'CrossTableHeader',
    'find_game_vertex_from_cross_table',
    # Level AI
    'LevelAIParser',
    'LevelAIHeader',
    'load_level_ai_positions',
    'find_nearest_level_vertex',
    # Level spawn
    'LevelSpawnParser',
    'SpawnFormat',
    'SpawnEntity',
    'Shape',
    'ALifeObject',
    'LevelChangerData',
    'GraphPoint',
    'detect_spawn_format',
    'parse_spawn_packet',
    'parse_entity_shapes',
    'parse_alife_object',
    'parse_level_changer_data',
    'parse_level_changer_packet',
    'AllSpawnSpawnIterator',
    # Game graph
    'GameGraphParser',
    'GameGraphHeader',
    'GameGraphLevel',
    'GameGraphVertex',
    'GameGraphEdge',
    # Patrol paths
    'PatrolPathParser',
    'PatrolPath',
    'PatrolPoint',
    'PatrolEdge',
    'read_extracted_patrols',
]
