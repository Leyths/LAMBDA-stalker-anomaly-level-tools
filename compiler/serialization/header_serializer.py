#!/usr/bin/env python3
"""
Spawn Header Builder

Creates the header chunk (chunk 0) for all.spawn files.
"""

import struct
import io

from utils import generate_guid


def create_header(game_graph_guid: bytes, spawn_count: int, level_count: int) -> bytes:
    """
    Create all.spawn header (chunk 0)

    Structure (from alife_spawn_registry_header.cpp):
    - u32 version
    - GUID spawn_guid (16 bytes)
    - GUID graph_guid (16 bytes)
    - u32 count (spawn point count)
    - u32 level_count
    """
    buffer = io.BytesIO()

    # Version
    buffer.write(struct.pack('<I', 10))  # XRAI_CURRENT_VERSION

    # Spawn GUID (deterministic based on content)
    spawn_guid = generate_guid(
        "all_spawn",
        game_graph_guid,
        spawn_count,
        level_count
    )
    buffer.write(spawn_guid)

    # Graph GUID (from game graph)
    buffer.write(game_graph_guid)

    # Counts
    buffer.write(struct.pack('<I', spawn_count))
    buffer.write(struct.pack('<I', level_count))

    return buffer.getvalue()
