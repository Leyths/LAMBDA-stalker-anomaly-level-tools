"""
Data loading for level.ai files and spawn data using shared parsers.
"""
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

# Add compiler to path for shared parser access (append to avoid shadowing visualiser modules)
compiler_path = str(Path(__file__).parent.parent.parent / "compiler")
if compiler_path not in sys.path:
    sys.path.append(compiler_path)
from parsers import LevelAIParser, GameGraphParser, GameGraphVertex, SpawnEntity, AllSpawnSpawnIterator, PatrolPathParser, PatrolPoint, PatrolEdge


class LevelData:
    """Manages level.ai data for visualization (read-only)."""

    # Invalid link marker constant
    INVALID_LINK = 0x7FFFFF  # 8388607

    def __init__(self, filepath: str, all_spawn_path: str = None, level_id: int = None):
        self.filepath = filepath
        self._parser = LevelAIParser(filepath, build_adjacency=True)

        # Cache frequently accessed data
        self.vertex_count = self._parser.vertex_count
        self.points = self._parser.get_all_positions()
        self.points[:, 2] = -self.points[:, 2]  # Mirror Z for display
        self.colors = self._parser.get_all_cover_scores()

        # Load cross-table mapping if all_spawn_path and level_id are provided
        self._cross_table_gvids: Optional[np.ndarray] = None
        self._gvid_to_level_vertices: dict = {}  # Reverse mapping: gvid -> set of level vertex ids
        if all_spawn_path and level_id is not None:
            self._load_cross_table(all_spawn_path, level_id)

    def _load_cross_table(self, all_spawn_path: str, level_id: int):
        """Load cross-table mapping from all.spawn."""
        filepath = Path(all_spawn_path)
        if not filepath.exists():
            return

        try:
            game_graph = GameGraphParser.from_all_spawn(filepath)
            cross_table = game_graph.get_cross_table_for_level(level_id)
            if cross_table:
                self._cross_table_gvids = np.array(cross_table['gvids'], dtype=np.uint16)
                # Build reverse mapping: gvid -> set of level vertex ids
                for lvid, gvid in enumerate(self._cross_table_gvids):
                    gvid_int = int(gvid)
                    if gvid_int not in self._gvid_to_level_vertices:
                        self._gvid_to_level_vertices[gvid_int] = set()
                    self._gvid_to_level_vertices[gvid_int].add(lvid)
        except Exception:
            pass  # Failed to load cross-table, leave as None

    def has_cross_table(self) -> bool:
        """Check if cross-table data is available."""
        return self._cross_table_gvids is not None

    def get_gvid(self, level_vertex_id: int) -> Optional[int]:
        """Get the game vertex ID for a level vertex from cross-table."""
        if self._cross_table_gvids is None:
            return None
        if level_vertex_id < 0 or level_vertex_id >= len(self._cross_table_gvids):
            return None
        return int(self._cross_table_gvids[level_vertex_id])

    def get_level_vertices_for_gvid(self, gvid: int) -> set:
        """Get all level vertex IDs that map to the given game graph vertex ID.

        Args:
            gvid: Game graph vertex ID

        Returns:
            Set of level vertex IDs, or empty set if none found
        """
        return self._gvid_to_level_vertices.get(gvid, set())

    def get_point(self, idx: int) -> np.ndarray:
        """Get the 3D point for a vertex."""
        if 0 <= idx < self.vertex_count:
            return self.points[idx]
        return None

    def get_color_score(self, idx: int) -> float:
        """Get the cover score for a vertex."""
        if 0 <= idx < self.vertex_count:
            return self.colors[idx]
        return None

    def get_links(self, idx: int) -> list:
        """Get the raw link values for a vertex (4 values, including invalid markers)."""
        return self._parser.get_vertex_raw_links(idx)

    def get_all_links(self) -> np.ndarray:
        """Get all vertex links as (vertex_count, 4) array in single read.

        Returns:
            (vertex_count, 4) int32 array where each row contains 4 link values.
            Invalid links are marked with INVALID_LINK (0x7FFFFF).
        """
        return self._parser.get_all_links()

    def find_nearest_node(self, x: float, y: float, z: float) -> tuple:
        """Find the nearest vertex to given coordinates.

        Args:
            x, y, z: World coordinates (Z will be unmirrored internally)

        Returns:
            (vertex_index, distance) tuple
        """
        # Unmirror Z to match the parser's coordinate system
        nearest_idx = self._parser.find_nearest_vertex((x, y, -z))

        if nearest_idx is not None:
            pos = self._parser.get_vertex_position(nearest_idx)
            target = np.array([x, y, -z])
            parser_pos = np.array(pos)
            distance = np.linalg.norm(parser_pos - target)
            return nearest_idx, distance

        return None, float('inf')

    def __len__(self) -> int:
        """Return the number of vertices."""
        return self.vertex_count


class SpawnData:
    """Manages spawn entity data from all.spawn for visualization."""

    def __init__(self, all_spawn_path: str, level_id: int):
        """
        Load and filter spawn entities for a specific level.

        Args:
            all_spawn_path: Path to all.spawn file
            level_id: Level ID to filter entities by
        """
        self.filepath = all_spawn_path
        self.level_id = level_id
        self._entities: List[SpawnEntity] = []
        self._positions: Optional[np.ndarray] = None

        self._load_spawns()

    def _load_spawns(self):
        """Load spawn data from all.spawn, filtering by level_id."""
        filepath = Path(self.filepath)
        if not filepath.exists():
            return

        # Parse game graph to build game_vertex_id -> level_id mapping
        try:
            game_graph = GameGraphParser.from_all_spawn(filepath)
        except Exception:
            return

        # Build vertex_id -> level_id mapping
        vertex_to_level = {}
        for vertex_id in range(game_graph.vertex_count):
            vertex_to_level[vertex_id] = game_graph.get_level_id_for_vertex(vertex_id)

        # Parse spawn entities using shared parser
        try:
            spawn_iterator = AllSpawnSpawnIterator.from_all_spawn(filepath)
        except Exception:
            return

        # Filter entities by level_id
        for entity in spawn_iterator:
            game_vertex_id = entity.game_vertex_id
            if game_vertex_id in vertex_to_level:
                if vertex_to_level[game_vertex_id] == self.level_id:
                    self._entities.append(entity)

        # Build positions array
        if self._entities:
            positions = []
            for entity in self._entities:
                x, y, z = entity.position
                # Mirror Z for display (same as LevelData)
                positions.append([x, y, -z])
            self._positions = np.array(positions, dtype=np.float64)
        else:
            self._positions = np.zeros((0, 3), dtype=np.float64)

    def __len__(self) -> int:
        """Return the number of spawn entities."""
        return len(self._entities)

    def get_entity(self, idx: int) -> Optional[SpawnEntity]:
        """Get a spawn entity by index."""
        if 0 <= idx < len(self._entities):
            return self._entities[idx]
        return None

    def get_position(self, idx: int) -> Optional[np.ndarray]:
        """Get the position of a spawn entity (Z mirrored for display)."""
        if 0 <= idx < len(self._entities):
            return self._positions[idx]
        return None

    @property
    def positions(self) -> np.ndarray:
        """Get all positions as numpy array (Z mirrored for display)."""
        return self._positions

    def find_nearest_spawn(self, x: float, y: float, z: float) -> tuple:
        """Find the nearest spawn to given coordinates.

        Args:
            x, y, z: World coordinates (Z should be mirrored to match display)

        Returns:
            (spawn_index, distance) tuple
        """
        if len(self._entities) == 0:
            return None, float('inf')

        target = np.array([x, y, z])
        distances = np.linalg.norm(self._positions - target, axis=1)
        nearest_idx = np.argmin(distances)
        return int(nearest_idx), float(distances[nearest_idx])

    def find_by_name(self, name: str) -> List[Tuple[int, str]]:
        """Find spawn entities by name (case-insensitive substring match).

        Args:
            name: Search string to match against entity names

        Returns:
            List of (index, entity_name) tuples for matching entities
        """
        if len(self._entities) == 0:
            return []

        matches = []
        search_lower = name.lower()
        for idx, entity in enumerate(self._entities):
            if search_lower in entity.entity_name.lower():
                matches.append((idx, entity.entity_name))

        return matches


class GraphData:
    """Manages game graph vertex data from all.spawn for visualization."""

    def __init__(self, all_spawn_path: str, level_id: int):
        """
        Load and filter game graph vertices for a specific level.

        Args:
            all_spawn_path: Path to all.spawn file
            level_id: Level ID to filter vertices by
        """
        self.filepath = all_spawn_path
        self.level_id = level_id
        self._parser: Optional[GameGraphParser] = None
        self._vertices: List[GameGraphVertex] = []
        self._positions: Optional[np.ndarray] = None
        self._inter_level_flags: List[bool] = []
        self._intra_level_edges: List[Tuple[int, int]] = []
        self._global_to_local: dict = {}  # Maps global vertex_id to local index

        self._load_graph()

    def _load_graph(self):
        """Load game graph data from all.spawn, filtering by level_id."""
        filepath = Path(self.filepath)
        if not filepath.exists():
            return

        try:
            self._parser = GameGraphParser.from_all_spawn(filepath)
        except Exception:
            return

        # Get all vertices for this level
        all_vertices = self._parser.get_vertices_for_level(self.level_id)

        # Sort by vertex_id for consistent ordering
        all_vertices.sort(key=lambda v: v.vertex_id)
        self._vertices = all_vertices

        # Build global to local index mapping
        for local_idx, vertex in enumerate(self._vertices):
            self._global_to_local[vertex.vertex_id] = local_idx

        # Build positions array (Z mirrored for display)
        if self._vertices:
            positions = []
            for vertex in self._vertices:
                x, y, z = vertex.local_point
                positions.append([x, y, -z])  # Mirror Z for display
            self._positions = np.array(positions, dtype=np.float64)
        else:
            self._positions = np.zeros((0, 3), dtype=np.float64)

        # Determine inter-level flags and intra-level edges
        self._compute_edge_data()

    def _compute_edge_data(self):
        """Compute inter-level flags and intra-level edges."""
        if not self._vertices or self._parser is None:
            return

        self._inter_level_flags = []
        self._intra_level_edges = []

        for local_idx, vertex in enumerate(self._vertices):
            has_inter_level_edge = False
            edges = self._parser.get_edges_for_vertex(vertex.vertex_id)

            for edge in edges:
                target_level = self._parser.get_level_id_for_vertex(edge.target_vertex_id)

                if target_level != self.level_id:
                    has_inter_level_edge = True
                else:
                    # Intra-level edge - only add if target is in our local vertex list
                    # and only add one direction (lower -> higher) to avoid duplicates
                    if edge.target_vertex_id in self._global_to_local:
                        target_local_idx = self._global_to_local[edge.target_vertex_id]
                        if local_idx < target_local_idx:
                            self._intra_level_edges.append((local_idx, target_local_idx))

            self._inter_level_flags.append(has_inter_level_edge)

    def __len__(self) -> int:
        """Return the number of graph vertices."""
        return len(self._vertices)

    def get_vertex(self, idx: int) -> Optional[GameGraphVertex]:
        """Get a graph vertex by local index."""
        if 0 <= idx < len(self._vertices):
            return self._vertices[idx]
        return None

    def get_position(self, idx: int) -> Optional[np.ndarray]:
        """Get the position of a graph vertex (Z mirrored for display)."""
        if 0 <= idx < len(self._vertices):
            return self._positions[idx]
        return None

    @property
    def positions(self) -> np.ndarray:
        """Get all positions as numpy array (Z mirrored for display)."""
        return self._positions

    @property
    def inter_level_flags(self) -> List[bool]:
        """Get list of flags indicating which vertices have inter-level edges."""
        return self._inter_level_flags

    @property
    def intra_level_edges(self) -> List[Tuple[int, int]]:
        """Get list of edge pairs (local indices) for intra-level connections."""
        return self._intra_level_edges

    def find_nearest_vertex(self, x: float, y: float, z: float) -> tuple:
        """Find the nearest graph vertex to given coordinates.

        Args:
            x, y, z: World coordinates (Z should be mirrored to match display)

        Returns:
            (vertex_index, distance) tuple
        """
        if len(self._vertices) == 0:
            return None, float('inf')

        target = np.array([x, y, z])
        distances = np.linalg.norm(self._positions - target, axis=1)
        nearest_idx = np.argmin(distances)
        return int(nearest_idx), float(distances[nearest_idx])

    def get_edges_info(self, idx: int) -> List[dict]:
        """Get edge information for a vertex for sidebar display.

        Args:
            idx: Local vertex index

        Returns:
            List of dicts with edge info (target_vertex_id, level_name, distance, is_inter_level)
        """
        if self._parser is None or not (0 <= idx < len(self._vertices)):
            return []

        vertex = self._vertices[idx]
        edges = self._parser.get_edges_for_vertex(vertex.vertex_id)
        levels = self._parser.get_levels()

        result = []
        for edge in edges:
            target_level_id = self._parser.get_level_id_for_vertex(edge.target_vertex_id)
            level_info = levels.get(target_level_id)
            level_name = level_info.name if level_info else f"level_{target_level_id}"
            is_inter_level = target_level_id != self.level_id

            result.append({
                'target_vertex_id': edge.target_vertex_id,
                'level_name': level_name,
                'distance': edge.distance,
                'is_inter_level': is_inter_level
            })

        return result


class PatrolData:
    """Manages patrol path data from all.spawn for visualization."""

    def __init__(self, all_spawn_path: str, level_id: int):
        """
        Load and filter patrol points for a specific level.

        Args:
            all_spawn_path: Path to all.spawn file
            level_id: Level ID to filter points by
        """
        self.filepath = all_spawn_path
        self.level_id = level_id
        self._points: List[PatrolPoint] = []
        self._positions: Optional[np.ndarray] = None
        self._edges: List[Tuple[int, int]] = []
        self._point_to_patrol: dict = {}  # Maps point index to patrol name
        self._point_edges: dict = {}  # Maps point index to list of connected point indices

        self._load_patrols()

    def _load_patrols(self):
        """Load patrol data from all.spawn, filtering by level_id."""
        filepath = Path(self.filepath)
        if not filepath.exists():
            return

        # Parse game graph to build game_vertex_id -> level_id mapping
        try:
            game_graph = GameGraphParser.from_all_spawn(filepath)
        except Exception:
            return

        # Build vertex_id -> level_id mapping
        vertex_to_level = {}
        for vertex_id in range(game_graph.vertex_count):
            vertex_to_level[vertex_id] = game_graph.get_level_id_for_vertex(vertex_id)

        # Parse patrol paths using shared parser
        try:
            patrol_parser = PatrolPathParser.from_all_spawn(filepath)
        except Exception:
            return

        # Flatten all patrol points, filtering by level_id
        # We need to track the original point IDs within each patrol for edge mapping
        patrol_point_mapping = {}  # (patrol_name, original_id) -> local_index

        for patrol in patrol_parser.get_patrols():
            for point in patrol.points:
                game_vertex_id = point.game_vertex_id
                if game_vertex_id in vertex_to_level:
                    if vertex_to_level[game_vertex_id] == self.level_id:
                        local_idx = len(self._points)
                        self._points.append(point)
                        self._point_to_patrol[local_idx] = patrol.name
                        patrol_point_mapping[(patrol.name, point.id)] = local_idx

        # Build positions array (Z mirrored for display)
        if self._points:
            positions = []
            for point in self._points:
                x, y, z = point.position
                positions.append([x, y, -z])  # Mirror Z for display
            self._positions = np.array(positions, dtype=np.float64)
        else:
            self._positions = np.zeros((0, 3), dtype=np.float64)

        # Build edges list using local indices
        # First, try explicit edges from the parser
        for patrol in patrol_parser.get_patrols():
            has_explicit_edges = sum(len(e) for e in patrol.edges.values()) > 0

            if has_explicit_edges:
                # Use explicit edges
                for source_id, edges in patrol.edges.items():
                    source_key = (patrol.name, source_id)
                    if source_key not in patrol_point_mapping:
                        continue
                    source_idx = patrol_point_mapping[source_key]

                    for edge in edges:
                        target_key = (patrol.name, edge.target_id)
                        if target_key not in patrol_point_mapping:
                            continue
                        target_idx = patrol_point_mapping[target_key]

                        # Add edge (source -> target)
                        self._edges.append((source_idx, target_idx))

                        # Track connections for each point
                        if source_idx not in self._point_edges:
                            self._point_edges[source_idx] = []
                        self._point_edges[source_idx].append(target_idx)
            else:
                # No explicit edges - create sequential connections
                # Collect points from this patrol that are on this level, in order
                patrol_local_indices = []
                for point in patrol.points:
                    key = (patrol.name, point.id)
                    if key in patrol_point_mapping:
                        patrol_local_indices.append(patrol_point_mapping[key])

                # Connect sequential points
                for i in range(len(patrol_local_indices) - 1):
                    source_idx = patrol_local_indices[i]
                    target_idx = patrol_local_indices[i + 1]

                    self._edges.append((source_idx, target_idx))

                    if source_idx not in self._point_edges:
                        self._point_edges[source_idx] = []
                    self._point_edges[source_idx].append(target_idx)

    def __len__(self) -> int:
        """Return the number of patrol points."""
        return len(self._points)

    def get_point(self, idx: int) -> Optional[PatrolPoint]:
        """Get a patrol point by index."""
        if 0 <= idx < len(self._points):
            return self._points[idx]
        return None

    def get_position(self, idx: int) -> Optional[np.ndarray]:
        """Get the position of a patrol point (Z mirrored for display)."""
        if 0 <= idx < len(self._points):
            return self._positions[idx]
        return None

    @property
    def positions(self) -> np.ndarray:
        """Get all positions as numpy array (Z mirrored for display)."""
        return self._positions

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Get all edges as list of (source_idx, target_idx) tuples."""
        return self._edges

    def get_patrol_name(self, idx: int) -> Optional[str]:
        """Get the patrol name for a point index."""
        return self._point_to_patrol.get(idx)

    def get_connected_points(self, idx: int) -> List[int]:
        """Get indices of points connected to the given point."""
        return self._point_edges.get(idx, [])

    def get_patrol_point_indices(self, patrol_name: str) -> List[int]:
        """Get all point indices belonging to a patrol path.

        Args:
            patrol_name: Name of the patrol path

        Returns:
            List of point indices in this patrol
        """
        indices = []
        for idx, name in self._point_to_patrol.items():
            if name == patrol_name:
                indices.append(idx)
        return sorted(indices)

    def get_patrol_edges(self, patrol_name: str) -> List[Tuple[int, int]]:
        """Get all edges belonging to a patrol path.

        Args:
            patrol_name: Name of the patrol path

        Returns:
            List of (source_idx, target_idx) tuples for edges in this patrol
        """
        patrol_indices = set(self.get_patrol_point_indices(patrol_name))
        return [(src, tgt) for src, tgt in self._edges
                if src in patrol_indices and tgt in patrol_indices]

    def find_nearest_point(self, x: float, y: float, z: float) -> Tuple[Optional[int], float]:
        """Find the nearest patrol point to given coordinates.

        Args:
            x, y, z: World coordinates (Z should be mirrored to match display)

        Returns:
            (point_index, distance) tuple
        """
        if len(self._points) == 0:
            return None, float('inf')

        target = np.array([x, y, z])
        distances = np.linalg.norm(self._positions - target, axis=1)
        nearest_idx = np.argmin(distances)
        return int(nearest_idx), float(distances[nearest_idx])

