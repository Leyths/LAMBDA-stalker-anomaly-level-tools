"""
Data loading for level.ai files and spawn data using shared parsers.
"""
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

# Add compiler to path for shared parser access (append to avoid shadowing visualiser modules)
compiler_path = str(Path(__file__).parent.parent.parent / "compiler")
if compiler_path not in sys.path:
    sys.path.append(compiler_path)
from parsers import LevelAIParser, GameGraphParser, GameGraphVertex, SpawnEntity, AllSpawnSpawnIterator


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

