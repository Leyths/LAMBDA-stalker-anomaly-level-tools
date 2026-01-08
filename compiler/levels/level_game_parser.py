#!/usr/bin/env python3
"""
Level.Game Parser

Parses level.game files to extract waypoint/patrol path data.

Format (from BuilderGame.cpp):
- Chunk 0x1000 + waytype: Waypoints of that type
  - Sub-chunk i: Individual waypoint data
"""

import struct
import io
from pathlib import Path
from typing import Dict, List, Tuple

from parsers import ChunkReader


def parse_level_game(level_game_path: Path) -> Dict[int, List[bytes]]:
    """
    Parse level.game file

    Args:
        level_game_path: Path to level.game file

    Returns:
        Dictionary of {chunk_id: [waypoint_data_list]}
    """
    waypoints = {}

    with open(level_game_path, 'rb') as f:
        data = f.read()

    for chunk in ChunkReader(data):
        # Parse waypoint chunks (0x1000+)
        if chunk.chunk_id >= 0x1000:
            waypoint_list = _parse_waypoint_chunk(chunk.data)
            waypoints[chunk.chunk_id] = waypoint_list

    return waypoints


def _parse_waypoint_chunk(data: bytes) -> List[bytes]:
    """
    Parse a waypoint chunk containing sub-chunks

    Returns:
        List of raw waypoint data (one per sub-chunk)
    """
    waypoints = []
    for sub_chunk in ChunkReader(data):
        waypoints.append(sub_chunk.data)
    return waypoints


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python level_game_parser.py level.game")
        sys.exit(1)
    
    level_game_path = Path(sys.argv[1])
    
    waypoints = parse_level_game(level_game_path)
    
    print(f"Parsed {len(waypoints)} waypoint chunks:")
    for chunk_id, wp_list in waypoints.items():
        print(f"  Chunk 0x{chunk_id:04X}: {len(wp_list)} waypoints")