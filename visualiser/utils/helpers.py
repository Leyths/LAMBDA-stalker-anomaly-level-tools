"""
Utility functions for node graph visualization.
"""
import math
import sys
from pathlib import Path
from typing import List

# Add compiler to path for shared parser access
compiler_path = str(Path(__file__).parent.parent.parent / "compiler")
if compiler_path not in sys.path:
    sys.path.append(compiler_path)
from parsers import parse_alife_object, parse_level_changer_data, GameGraphVertex


def format_node_info(idx: int, point, cover_score: float, gvid: int = None) -> str:
    """Format node information for display.

    Args:
        idx: Vertex index
        point: 3D point array (with Z already mirrored for display)
        cover_score: Cover score value
        gvid: Game vertex ID from cross-table (optional)

    Returns:
        Formatted string for display
    """
    # Display original coordinates (unflip Z for display)
    original_z = -point[2]

    info_text = f"""VERTEX INDEX: {idx}

Position:
  X: {point[0]:.2f}
  Y: {point[1]:.2f}
  Z: {original_z:.2f}

Cover Score: {cover_score:.0f}
"""

    if gvid is not None:
        info_text += f"\nGame Vertex ID: {gvid}\n"

    return info_text


def format_spawn_info(entity) -> str:
    """Format spawn entity information for display.

    Args:
        entity: SpawnEntity object with spawn data

    Returns:
        Formatted string for display
    """
    # Convert angles from radians to degrees for display
    angle_x = math.degrees(entity.angle[0])
    angle_y = math.degrees(entity.angle[1])
    angle_z = math.degrees(entity.angle[2])

    info_text = f"""SPAWN OBJECT
Entity: {entity.entity_name or '(unnamed)'}
Type: {entity.section_name}

Position:
  X: {entity.position[0]:.2f}
  Y: {entity.position[1]:.2f}
  Z: {entity.position[2]:.2f}

Angle:
  {angle_x:.1f}, {angle_y:.1f}, {angle_z:.1f}

Game Vertex ID: {entity.game_vertex_id}
Level Vertex ID: {entity.level_vertex_id}
Spawn ID: {entity.spawn_id}
"""

    # Add alife_object data if the entity has one
    alife = parse_alife_object(entity)
    if alife:
        info_text += f"""
ALife Object:
  Game Graph ID: {alife.game_graph_id}
  Distance: {alife.distance:.2f}
  Direct Control: {alife.direct_control}
  Node ID: {alife.node_id}
  Object Flags: {alife.object_flags}
  Story ID: {alife.story_id}
  Spawn Story ID: {alife.spawn_story_id}
"""
        if alife.ini_string:
            # Format ini_string nicely (replace \r\n with actual newlines)
            ini_display = alife.ini_string.replace('\r\n', '\n  ').replace('\r', '\n  ')
            info_text += f"""
INI Config:
  {ini_display}
"""

    # Add level_changer destination data if this is a level_changer
    lc_data = parse_level_changer_data(entity)
    if lc_data:
        info_text += f"""
Level Changer Destination:
  Dest GVID: {lc_data.dest_game_vertex_id}
  Dest Level Vertex: {lc_data.dest_level_vertex_id}
  Dest Position:
    X: {lc_data.dest_position[0]:.2f}
    Y: {lc_data.dest_position[1]:.2f}
    Z: {lc_data.dest_position[2]:.2f}
  Dest Level: {lc_data.dest_level_name or '(empty)'}
  Dest Graph Point: {lc_data.dest_graph_point or '(empty)'}
  Silent Mode: {lc_data.silent_mode}
"""

    return info_text


def format_graph_vertex_info(vertex: GameGraphVertex, edges_info: List[dict], local_idx: int = None) -> str:
    """Format game graph vertex information for display.

    Args:
        vertex: GameGraphVertex object with vertex data
        edges_info: List of edge info dicts from GraphData.get_edges_info()
        local_idx: Local index within this level (optional)

    Returns:
        Formatted string for display
    """
    local_idx_text = f" (local #{local_idx})" if local_idx is not None else ""
    info_text = f"""GAME GRAPH VERTEX
Global Vertex ID (GVID): {vertex.vertex_id}{local_idx_text}

Local Position:
  X: {vertex.local_point[0]:.2f}
  Y: {vertex.local_point[1]:.2f}
  Z: {vertex.local_point[2]:.2f}

Global Position:
  X: {vertex.global_point[0]:.2f}
  Y: {vertex.global_point[1]:.2f}
  Z: {vertex.global_point[2]:.2f}

Level ID: {vertex.level_id}
Level Vertex ID: {vertex.level_vertex_id}
Neighbours: {vertex.neighbour_count}
Death Points: {vertex.death_point_count}
"""

    # Add edges section
    if edges_info:
        info_text += "\nEdges:\n"
        for edge in edges_info:
            inter_marker = " [INTER]" if edge['is_inter_level'] else ""
            info_text += f"  -> {edge['target_vertex_id']} ({edge['level_name']}) d={edge['distance']:.1f}{inter_marker}\n"
    else:
        info_text += "\nNo edges\n"

    return info_text
