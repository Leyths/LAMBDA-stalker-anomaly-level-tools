"""
World graph data loader - loads ALL game graph vertices in world space.
"""
import colorsys
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

# Add compiler to path for shared parser access
compiler_path = str(Path(__file__).parent.parent.parent / "compiler")
if compiler_path not in sys.path:
    sys.path.append(compiler_path)
from parsers import GameGraphParser, GameGraphVertex, GameGraphLevel


class WorldGraphData:
    """Manages world-space game graph data for all levels."""

    def __init__(self, all_spawn_path: str):
        self.filepath = all_spawn_path
        self._parser: Optional[GameGraphParser] = None
        self._vertices: List[GameGraphVertex] = []
        self._positions: Optional[np.ndarray] = None
        self._levels: Dict[int, GameGraphLevel] = {}
        self._vertex_level_ids: List[int] = []
        self._intra_level_edges: List[Tuple[int, int]] = []
        self._inter_level_edges: List[Tuple[int, int]] = []
        self._level_colors: Dict[int, List[float]] = {}

        self._load()

    def _load(self):
        """Load all game graph data from all.spawn."""
        filepath = Path(self.filepath)
        if not filepath.exists():
            self._positions = np.zeros((0, 3), dtype=np.float64)
            return

        self._parser = GameGraphParser.from_all_spawn(filepath)
        self._levels = self._parser.get_levels()

        # Load all vertices
        for vid in range(self._parser.vertex_count):
            self._vertices.append(self._parser.get_vertex(vid))

        # Build positions array using global_point (world space), Z mirrored
        if self._vertices:
            positions = []
            for v in self._vertices:
                x, y, z = v.global_point
                positions.append([x, y, -z])
            self._positions = np.array(positions, dtype=np.float64)
        else:
            self._positions = np.zeros((0, 3), dtype=np.float64)

        # Build level_id array
        self._vertex_level_ids = [v.level_id for v in self._vertices]

        # Generate per-level colors
        self._generate_level_colors()

        # Build edge lists
        self._compute_edges()

    def _generate_level_colors(self):
        """Generate distinct colors for each level using HSV palette."""
        level_ids = sorted(self._levels.keys())
        n = max(len(level_ids), 1)
        for i, lid in enumerate(level_ids):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            self._level_colors[lid] = [r, g, b]

    def _compute_edges(self):
        """Classify edges as intra-level or inter-level, deduplicated."""
        if not self._vertices or self._parser is None:
            return

        seen = set()
        for vid, vertex in enumerate(self._vertices):
            edges = self._parser.get_edges_for_vertex(vid)
            for edge in edges:
                target_vid = edge.target_vertex_id
                if target_vid >= len(self._vertices):
                    continue

                # Deduplicate: only store (min, max)
                pair = (min(vid, target_vid), max(vid, target_vid))
                if pair in seen:
                    continue
                seen.add(pair)

                if vertex.level_id == self._vertices[target_vid].level_id:
                    self._intra_level_edges.append(pair)
                else:
                    self._inter_level_edges.append(pair)

    def __len__(self) -> int:
        return len(self._vertices)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def levels(self) -> Dict[int, GameGraphLevel]:
        return self._levels

    @property
    def level_colors(self) -> Dict[int, List[float]]:
        return self._level_colors

    @property
    def intra_level_edges(self) -> List[Tuple[int, int]]:
        return self._intra_level_edges

    @property
    def inter_level_edges(self) -> List[Tuple[int, int]]:
        return self._inter_level_edges

    def get_vertex(self, idx: int) -> Optional[GameGraphVertex]:
        if 0 <= idx < len(self._vertices):
            return self._vertices[idx]
        return None

    def get_position(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < len(self._vertices):
            return self._positions[idx]
        return None

    def get_vertex_level_id(self, idx: int) -> int:
        if 0 <= idx < len(self._vertex_level_ids):
            return self._vertex_level_ids[idx]
        return -1

    def get_vertex_color(self, idx: int) -> List[float]:
        lid = self.get_vertex_level_id(idx)
        return self._level_colors.get(lid, [0.5, 0.5, 0.5])

    def find_nearest_vertex(self, x: float, y: float, z: float) -> Tuple[Optional[int], float]:
        if len(self._vertices) == 0:
            return None, float('inf')
        target = np.array([x, y, z])
        distances = np.linalg.norm(self._positions - target, axis=1)
        nearest_idx = np.argmin(distances)
        return int(nearest_idx), float(distances[nearest_idx])

    def get_edges_info(self, idx: int) -> List[dict]:
        """Get edge information for a vertex for display."""
        if self._parser is None or not (0 <= idx < len(self._vertices)):
            return []

        vertex = self._vertices[idx]
        edges = self._parser.get_edges_for_vertex(vertex.vertex_id)

        result = []
        for edge in edges:
            target_level_id = self._parser.get_level_id_for_vertex(edge.target_vertex_id)
            level_info = self._levels.get(target_level_id)
            level_name = level_info.name if level_info else f"level_{target_level_id}"
            is_inter_level = target_level_id != vertex.level_id

            result.append({
                'target_vertex_id': edge.target_vertex_id,
                'level_name': level_name,
                'distance': edge.distance,
                'is_inter_level': is_inter_level
            })

        return result
