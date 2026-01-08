"""
Serialization Package

Handles binary serialization of game data for the all.spawn file.
This is Phase 5 of the build pipeline - final output generation.

Chunks in all.spawn:
- Chunk 0: Header (header_serializer)
- Chunk 1: Spawn graph (spawn_graph_builder)
- Chunk 2: Artefacts (empty)
- Chunk 3: Patrols (patrol_path_extractor)
- Chunk 4: Game graph (game_graph_serializer)
"""

from .game_graph_serializer import GameGraphSerializer
from .header_serializer import create_header
from .all_spawn_writer import build_all_spawn
