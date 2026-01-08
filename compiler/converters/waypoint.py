#!/usr/bin/env python3
"""
Waypoint to Patrol Path Converter

Converts waypoints from level.game to CPatrolPath format for all.spawn.

Format (from WayPoint.cpp Export method):
- Chunk WAYOBJECT_CHUNK_VERSION (0x0001): u16 version (0x0012)
- Chunk WAYOBJECT_CHUNK_NAME (0x0005): stringZ name
- Chunk WAYOBJECT_CHUNK_TYPE (0x0004): u32 type
- Chunk WAYOBJECT_CHUNK_POINTS (0x0002):
    - u16 count
    - For each point: Fvector position (3f), u32 flags
- Chunk WAYOBJECT_CHUNK_LINKS (0x0003):
    - u16 link_count
    - For each link: u16 from, u16 to
"""

import struct
import io
from pathlib import Path
from typing import Dict, List, Tuple

from utils import write_chunk


class WayObject:
    """Represents a parsed waypoint object"""

    def __init__(self):
        self.name = ""
        self.way_type = 0  # EWayType
        self.points = []   # List of (position: tuple, flags: int, name: str)
        self.links = []    # List of (from_idx: int, to_idx: int)

    def to_patrol_path(self, level_vertex_cache=None, cross_table_path: Path = None) -> bytes:
        """
        Convert to CPatrolPath (CGraphAbstractSerialize) format

        Args:
            level_vertex_cache: Pre-loaded numpy array of vertex positions (or None)
            cross_table_path: Path to cross table for game vertex lookup

        Format (from graph_abstract_inline.h lines 227-274):
        - Chunk 0: vertex_count (u32)
        - Chunk 1: Vertices
          - For each vertex i:
            - Sub-chunk 0: vertex_id (u32)
            - Sub-chunk 1: CPatrolPoint data
        - Chunk 2: Edges
          - For each vertex with edges:
            - vertex_id (u32)
            - edge_count (u32)
            - For each edge: target_id (u32), weight (f32)
        """
        buffer = io.BytesIO()

        # Chunk 0: vertex count
        count_buffer = io.BytesIO()
        count_buffer.write(struct.pack('<I', len(self.points)))
        write_chunk(buffer, 0, count_buffer.getvalue())

        # Chunk 1: Vertices
        vertices_buffer = io.BytesIO()
        for i, point_data in enumerate(self.points):
            if len(point_data) == 3:
                position, flags, point_name = point_data
            else:
                # Old format without name
                position, flags = point_data
                point_name = None

            vertex_buffer = io.BytesIO()

            # Sub-chunk 0: vertex_id
            id_buffer = io.BytesIO()
            id_buffer.write(struct.pack('<I', i))
            write_chunk(vertex_buffer, 0, id_buffer.getvalue())

            # Sub-chunk 1: CPatrolPoint data
            point_buffer = io.BytesIO()

            # CPatrolPoint format (from patrol_point.cpp):
            # 1. shared_str name (stringZ)
            # 2. Fvector position (3 floats)
            # 3. u32 flags
            # 4. u32 level_vertex_id (set to 0 - game will recalculate)
            # 5. u16 game_vertex_id (set to 0 - game will recalculate)
            #
            # Note: level_vertex_id and game_vertex_id are normally calculated
            # by finding the nearest vertex in level.ai and game graph.
            # Setting them to 0 allows the game to recalculate at runtime.

            # Name - use from waypoint if available, otherwise generate
            if point_name:
                # Strip triple quotes if present (SDK format uses """name""")
                name = point_name.strip('"')
            else:
                name = f'wp{i:02d}'
            point_buffer.write(name.encode('utf-8') + b'\x00')

            # Position
            point_buffer.write(struct.pack('<3f', *position))

            # Flags
            point_buffer.write(struct.pack('<I', flags))

            # Level vertex ID - find nearest vertex in level.ai
            if level_vertex_cache is not None:
                level_vertex_id = find_nearest_vertex_cached(position, level_vertex_cache)
            else:
                level_vertex_id = 0xFFFFFFFF  # -1 / invalid
            point_buffer.write(struct.pack('<I', level_vertex_id))

            # Game vertex ID - find in cross table if available
            if cross_table_path and cross_table_path.exists() and level_vertex_id != 0xFFFFFFFF:
                game_vertex_id = find_game_vertex_from_cross_table(level_vertex_id, cross_table_path)
            else:
                game_vertex_id = 0  # Will be calculated at runtime
            point_buffer.write(struct.pack('<H', game_vertex_id))

            write_chunk(vertex_buffer, 1, point_buffer.getvalue())

            write_chunk(vertices_buffer, i, vertex_buffer.getvalue())

        write_chunk(buffer, 1, vertices_buffer.getvalue())

        # Chunk 2: Edges
        edges_buffer = io.BytesIO()

        # Build adjacency list
        adjacency = {}
        for from_idx, to_idx in self.links:
            if from_idx not in adjacency:
                adjacency[from_idx] = []
            adjacency[from_idx].append(to_idx)

        # Write edges
        for vertex_id in sorted(adjacency.keys()):
            edges_buffer.write(struct.pack('<I', vertex_id))  # vertex_id
            targets = adjacency[vertex_id]
            edges_buffer.write(struct.pack('<I', len(targets)))  # edge_count

            for target_id in targets:
                edges_buffer.write(struct.pack('<I', target_id))  # target vertex_id

                # Calculate distance (weight)
                from_point = self.points[vertex_id]
                to_point = self.points[target_id]

                # Handle both 2-tuple and 3-tuple formats
                from_pos = from_point[0] if len(from_point) > 0 else from_point
                to_pos = to_point[0] if len(to_point) > 0 else to_point

                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                dz = to_pos[2] - from_pos[2]
                distance = (dx*dx + dy*dy + dz*dz) ** 0.5

                edges_buffer.write(struct.pack('<f', distance))  # weight

        write_chunk(buffer, 2, edges_buffer.getvalue())

        return buffer.getvalue()


def parse_wayobject(data: bytes) -> WayObject:
    """
    Parse a wayobject from level.game chunk data

    Args:
        data: Raw wayobject data

    Returns:
        Parsed WayObject
    """
    obj = WayObject()
    offset = 0

    while offset < len(data):
        if offset + 8 > len(data):
            break

        # Read chunk header
        chunk_id = struct.unpack('<I', data[offset:offset+4])[0]
        chunk_size = struct.unpack('<I', data[offset+4:offset+8])[0]
        offset += 8

        if offset + chunk_size > len(data):
            break

        chunk_data = data[offset:offset+chunk_size]
        offset += chunk_size

        # Parse chunks
        if chunk_id == 0x0001:  # WAYOBJECT_CHUNK_VERSION
            version = struct.unpack('<H', chunk_data[:2])[0]

        elif chunk_id == 0x0005:  # WAYOBJECT_CHUNK_NAME
            # StringZ - null-terminated
            null_pos = chunk_data.find(b'\x00')
            if null_pos >= 0:
                obj.name = chunk_data[:null_pos].decode('utf-8', errors='ignore')

        elif chunk_id == 0x0004:  # WAYOBJECT_CHUNK_TYPE
            obj.way_type = struct.unpack('<I', chunk_data[:4])[0]

        elif chunk_id == 0x0002:  # WAYOBJECT_CHUNK_POINTS
            count = struct.unpack('<H', chunk_data[:2])[0]
            pos = 2
            for i in range(count):
                if pos + 12 > len(chunk_data):  # Need at least position (12 bytes)
                    break

                # Fvector (3 floats)
                x, y, z = struct.unpack('<3f', chunk_data[pos:pos+12])
                pos += 12

                # u32 flags
                if pos + 4 > len(chunk_data):
                    break
                flags = struct.unpack('<I', chunk_data[pos:pos+4])[0]
                pos += 4

                # Check if there's a name (stringZ) after flags
                # Some waypoint exports include point names
                point_name = None
                if pos < len(chunk_data):
                    # Try to read stringZ
                    null_pos = chunk_data.find(b'\x00', pos)
                    if null_pos > pos and null_pos - pos < 100:  # Reasonable name length
                        try:
                            point_name = chunk_data[pos:null_pos].decode('utf-8', errors='ignore')
                            pos = null_pos + 1
                        except Exception:
                            pass

                obj.points.append(((x, y, z), flags, point_name))


        elif chunk_id == 0x0003:  # WAYOBJECT_CHUNK_LINKS
            link_count = struct.unpack('<H', chunk_data[:2])[0]
            pos = 2
            for i in range(link_count):
                if pos + 4 > len(chunk_data):
                    break
                from_idx, to_idx = struct.unpack('<HH', chunk_data[pos:pos+4])
                obj.links.append((from_idx, to_idx))
                pos += 4

    return obj


def convert_wayobjects_to_patrol_paths(wayobjects: List[bytes],
                                        level_ai_path: Path = None,
                                        cross_table_path: Path = None) -> Dict[str, bytes]:
    """
    Convert list of wayobject data to patrol paths

    Args:
        wayobjects: List of raw wayobject data
        level_ai_path: Path to level.ai for vertex lookup
        cross_table_path: Path to cross table for game vertex lookup

    Returns:
        Dictionary of {patrol_name: patrol_path_data}
    """
    patrol_paths = {}

    print(f"      Converting {len(wayobjects)} wayobjects...")

    # Load level.ai cache ONCE for all wayobjects
    level_vertex_cache = None
    if level_ai_path and level_ai_path.exists():
        level_vertex_cache = load_level_ai_positions(level_ai_path)

    for idx, wayobject_data in enumerate(wayobjects):
        if idx % 100 == 0:
            print(f"        Progress: {idx}/{len(wayobjects)}")

        try:
            obj = parse_wayobject(wayobject_data)

            if obj.name and obj.points:
                # Validate links
                valid_links = []
                for from_idx, to_idx in obj.links:
                    if from_idx < len(obj.points) and to_idx < len(obj.points):
                        valid_links.append((from_idx, to_idx))
                obj.links = valid_links

                # Only include patrol paths (wtPatrolPath = 0)
                if obj.way_type == 0:
                    # Pass the pre-loaded cache instead of the path
                    patrol_data = obj.to_patrol_path(level_vertex_cache, cross_table_path)
                    patrol_paths[obj.name] = patrol_data

        except Exception as e:
            if idx < 10:  # Only print first few errors
                print(f"        Warning: Failed to parse wayobject {idx}: {e}")
            continue

    print(f"      Converted {len(patrol_paths)} patrol paths")
    return patrol_paths


def load_level_ai_positions(level_ai_path: Path):
    """
    Load all vertex positions from level.ai once

    Returns numpy array of positions for fast nearest-neighbor lookups

    Positions are quantized in level.ai format:
    - X/Z: 12 bits each, packed into 24-bit value at offset 18
    - Y: 16 bits at offset 21
    """
    try:
        import numpy as np

        print(f"          Loading level.ai vertex cache...")

        with open(level_ai_path, 'rb') as f:
            # Header
            version = struct.unpack('<I', f.read(4))[0]
            vertex_count = struct.unpack('<I', f.read(4))[0]

            if vertex_count == 0:
                return None

            # Read cell sizes
            cell_size = struct.unpack('<f', f.read(4))[0]
            cell_size_y = struct.unpack('<f', f.read(4))[0]

            # Read bounds
            min_x, min_y, min_z = struct.unpack('<3f', f.read(12))
            max_x, max_y, max_z = struct.unpack('<3f', f.read(12))

            print(f"          Header: vertices={vertex_count}, cell_size={cell_size:.2f}")
            print(f"          Bounds: ({min_x:.2f},{min_y:.2f},{min_z:.2f}) to ({max_x:.2f},{max_y:.2f},{max_z:.2f})")

            # Skip GUID
            f.read(16)

            # Read all vertex data (23 bytes per vertex = NODE_SIZE)
            NODE_SIZE = 23
            HEADER_SIZE = 56

            print(f"          Reading {vertex_count} vertices...")

            positions = np.zeros((vertex_count, 3), dtype=np.float32)

            for i in range(vertex_count):
                # Read full node
                node_data = f.read(NODE_SIZE)
                if len(node_data) != NODE_SIZE:
                    print(f"          ERROR: Could only read {len(node_data)} bytes for vertex {i}")
                    return None

                # Parse quantized position from bytes 18-22
                # XZ packed: bytes 18-20 (24 bits)
                # Y: bytes 21-22 (16 bits)
                xz_packed = struct.unpack_from("<I", node_data, 18)[0] & 0x00FFFFFF
                y_quant = struct.unpack_from("<H", node_data, 21)[0]

                # Dequantize XZ (12 bits each)
                x_idx = xz_packed & 0xFFF  # Lower 12 bits
                z_idx = (xz_packed >> 12) & 0xFFF  # Upper 12 bits

                x = x_idx * cell_size + min_x
                z = z_idx * cell_size + min_z

                # Dequantize Y (16 bits, normalized)
                y_normalized = y_quant / 65535.0
                y_range = max_y - min_y
                y = y_normalized * y_range + min_y

                positions[i] = [x, y, z]

            print(f"          Cached {vertex_count} vertices successfully")
            return positions

    except Exception as e:
        print(f"          Warning: Could not cache level.ai: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_nearest_vertex_cached(position: Tuple[float, float, float],
                                positions_cache) -> int:
    """
    Find nearest vertex using pre-loaded position cache

    Much faster than loading level.ai each time
    """
    try:
        import numpy as np

        if positions_cache is None:
            return 0xFFFFFFFF

        px, py, pz = position
        vertex_count = len(positions_cache)

        min_dist_sq = float('inf')
        nearest_idx = 0

        # Process in chunks to avoid overflow
        chunk_size = 5000
        for chunk_start in range(0, vertex_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, vertex_count)
            chunk = positions_cache[chunk_start:chunk_end]

            # Calculate distances for this chunk
            dx = chunk[:, 0] - np.float32(px)
            dy = chunk[:, 1] - np.float32(py)
            dz = chunk[:, 2] - np.float32(pz)

            # Use float64 for intermediate calc to avoid overflow
            dist_sq = dx.astype(np.float64)**2 + dy.astype(np.float64)**2 + dz.astype(np.float64)**2

            # Find minimum in chunk
            chunk_min_idx = np.argmin(dist_sq)
            chunk_min_dist = float(dist_sq[chunk_min_idx])

            if chunk_min_dist < min_dist_sq:
                min_dist_sq = chunk_min_dist
                nearest_idx = chunk_start + chunk_min_idx

        return int(nearest_idx)

    except Exception as e:
        print(f"          Warning: Cached lookup failed: {e}")
        import traceback
        traceback.print_exc()
        return 0xFFFFFFFF


def find_nearest_level_vertex(position: Tuple[float, float, float],
                               level_ai_path: Path) -> int:
    """
    Find nearest vertex in level.ai for a given position

    Uses the same approach as level_graph_navigator.py

    Args:
        position: 3D position (x, y, z)
        level_ai_path: Path to level.ai file

    Returns:
        Vertex ID (u32), or 0xFFFFFFFF if not found
    """
    try:
        import numpy as np

        # Read level.ai to get vertex positions
        with open(level_ai_path, 'rb') as f:
            # Header
            version = struct.unpack('<I', f.read(4))[0]
            vertex_count = struct.unpack('<I', f.read(4))[0]

            if vertex_count == 0:
                return 0xFFFFFFFF

            # Skip cell sizes and bounds
            f.read(8 + 24)  # 2 floats + 2 vec3

            # Read GUID (skip)
            f.read(16)

            # Now at vertex data - each vertex is 20 bytes:
            # - 3 x float32 (position) = 12 bytes
            # - 2 x uint32 (indices) = 8 bytes

            # Read all vertex data at once
            print(f"          Reading {vertex_count} vertices from level.ai...")
            vertex_data = f.read(vertex_count * 20)

            if len(vertex_data) != vertex_count * 20:
                return 0xFFFFFFFF

            # Extract positions - read manually since it's mixed types
            print(f"          Extracting positions...")
            positions = np.zeros((vertex_count, 3), dtype=np.float32)

            for i in range(vertex_count):
                offset = i * 20
                # Read 3 floats for position
                positions[i] = struct.unpack_from('<3f', vertex_data, offset)

            # Find nearest using vectorized operations on chunks
            print(f"          Finding nearest vertex to {position}...")
            px, py, pz = position

            min_dist_sq = float('inf')
            nearest_idx = 0

            # Process in chunks to avoid overflow
            chunk_size = 5000
            for chunk_start in range(0, vertex_count, chunk_size):
                chunk_end = min(chunk_start + chunk_size, vertex_count)
                chunk = positions[chunk_start:chunk_end]

                # Calculate distances for this chunk
                dx = chunk[:, 0] - np.float32(px)
                dy = chunk[:, 1] - np.float32(py)
                dz = chunk[:, 2] - np.float32(pz)

                # Use float64 for intermediate calc to avoid overflow
                dist_sq = dx.astype(np.float64)**2 + dy.astype(np.float64)**2 + dz.astype(np.float64)**2

                # Find minimum in chunk
                chunk_min_idx = np.argmin(dist_sq)
                chunk_min_dist = dist_sq[chunk_min_idx]

                if chunk_min_dist < min_dist_sq:
                    min_dist_sq = chunk_min_dist
                    nearest_idx = chunk_start + chunk_min_idx

            print(f"          Found vertex {nearest_idx}")
            return int(nearest_idx)

    except Exception as e:
        print(f"          Warning: Could not find level vertex: {e}")
        import traceback
        traceback.print_exc()
        return 0xFFFFFFFF


def find_game_vertex_from_cross_table(level_vertex_id: int,
                                       cross_table_path: Path) -> int:
    """
    Find game vertex ID from cross table

    Cross table format (chunked):
    - Chunk 0xFFFF: Header
      - version (u32)
      - level_vertex_count (u32)
      - game_vertex_count (u32)
      - level_guid (16 bytes)
      - game_guid (16 bytes)
    - Chunk 1: Data (level_vertex_count entries)
      - game_vertex_id (u16) + distance (f32) per entry

    Args:
        level_vertex_id: Level vertex ID (u32)
        cross_table_path: Path to level.gct file

    Returns:
        Game vertex ID (u16), or 0 if not found
    """
    try:
        # Cache to avoid repeated warnings and parsing
        if not hasattr(find_game_vertex_from_cross_table, '_warned_paths'):
            find_game_vertex_from_cross_table._warned_paths = set()
        if not hasattr(find_game_vertex_from_cross_table, '_cache'):
            find_game_vertex_from_cross_table._cache = {}

        cache_key = str(cross_table_path)
        if cache_key in find_game_vertex_from_cross_table._cache:
            header_size, level_vertex_count = find_game_vertex_from_cross_table._cache[cache_key]
        else:
            # Read chunked format
            with open(cross_table_path, 'rb') as f:
                # Read chunk 0xFFFF header
                chunk_id = struct.unpack('<I', f.read(4))[0]
                chunk_size = struct.unpack('<I', f.read(4))[0]

                if chunk_id != 0xFFFF:
                    print(f"          Warning: Invalid cross table format (chunk_id={chunk_id:#x})")
                    return 0

                # Read header
                version = struct.unpack('<I', f.read(4))[0]
                level_vertex_count = struct.unpack('<I', f.read(4))[0]
                game_vertex_count = struct.unpack('<I', f.read(4))[0]
                f.read(32)  # Skip GUIDs

                # Remember where data chunk starts
                header_size = f.tell()

                # Cache it
                find_game_vertex_from_cross_table._cache[cache_key] = (header_size, level_vertex_count)

        # Validate vertex ID
        if level_vertex_id >= level_vertex_count:
            if cross_table_path not in find_game_vertex_from_cross_table._warned_paths:
                print(f"          Warning: level_vertex_id {level_vertex_id} >= {level_vertex_count}")
                find_game_vertex_from_cross_table._warned_paths.add(cross_table_path)
            return 0

        # Read from data chunk
        with open(cross_table_path, 'rb') as f:
            # Skip to data chunk header
            f.seek(header_size)

            # Read data chunk header
            chunk_id = struct.unpack('<I', f.read(4))[0]
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id != 1:
                print(f"          Warning: Expected data chunk 1, got {chunk_id}")
                return 0

            # Seek to the specific entry (6 bytes per entry: u16 + f32)
            data_start = f.tell()
            f.seek(data_start + level_vertex_id * 6)

            game_vertex_id = struct.unpack('<H', f.read(2))[0]

            return game_vertex_id

    except Exception as e:
        print(f"          Warning: Could not read cross table: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == '__main__':
    # Test parsing
    import sys
    from levels import parse_level_game

    if len(sys.argv) < 2:
        print("Usage: python waypoint.py level.game")
        sys.exit(1)

    level_game_path = Path(sys.argv[1])
    waypoints = parse_level_game(level_game_path)

    # Get patrol path waypoints (chunk 0x1000)
    if 0x1000 in waypoints:
        patrol_paths = convert_wayobjects_to_patrol_paths(waypoints[0x1000])

        print(f"Converted {len(patrol_paths)} patrol paths:")
        for name in sorted(patrol_paths.keys()):
            print(f"  {name}: {len(patrol_paths[name])} bytes")
