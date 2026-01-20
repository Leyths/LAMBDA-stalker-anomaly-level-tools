#!/usr/bin/env python3
"""
Game Graph Merger

Merges multiple level game graphs into a single global game graph.
Generates death points using the 10% sampling algorithm.

Based on xr_graph_merge.cpp from X-Ray SDK.
"""

import struct
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random

from utils import logDebug, log, logWarning, logError
from constants import VERTEX_MATCH_TOLERANCE
# Import parsers
from parsers import CrossTableParser, LevelAIParser
# Import GameGraph
from graph import GameGraph
# Import LevelGraphNavigator for precise level_vertex_id computation
from crosstables import LevelGraphNavigator
# Import DeathPointGenerator from generation module
from generation import DeathPointGenerator
# Import LevelsConfig
from levels import LevelsConfig, LevelConfig


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GameVertex:
    """CVertex structure from game graph (42 bytes packed)"""
    local_point: Tuple[float, float, float]  # Position in level space
    global_point: Tuple[float, float, float]  # Position in world space
    level_id: int  # u8 (0-255)
    level_vertex_id: int  # u24 (0-16777215)
    vertex_types: bytes  # 4 bytes (location flags)

    # Runtime data (not serialized directly)
    edges: List['GameEdge']  # Edges to other game vertices
    death_points: List['DeathPoint']  # Death/spawn points

    # Offsets (filled during serialization)
    edge_offset: int = 0
    death_point_offset: int = 0

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def death_point_count(self) -> int:
        return len(self.death_points)


@dataclass
class GameEdge:
    """CEdge structure (6 bytes packed)"""
    vertex_id: int  # u16 - target game vertex ID
    distance: float  # f32 - path distance in meters


@dataclass
class DeathPoint:
    """CLevelPoint structure (20 bytes)"""
    position: Tuple[float, float, float]  # 3D position
    level_vertex_id: int  # u32 - which level.ai vertex
    distance: float  # f32 - distance from game vertex


# ============================================================================
# Cross Table Reader (uses CrossTableParser from parsers module)
# ============================================================================

class CrossTableReader:
    """
    Read level.gct cross table files.

    This is a thin wrapper around CrossTableParser for backward compatibility.
    New code should use CrossTableParser directly.
    """

    def __init__(self, filepath: Path):
        self._parser = CrossTableParser(filepath)
        # Expose properties for backward compatibility
        self.version = self._parser.header.version
        self.level_vertex_count = self._parser.header.level_vertex_count
        self.game_vertex_count = self._parser.header.game_vertex_count
        self.level_guid = self._parser.header.level_guid
        self.game_guid = self._parser.header.game_guid
        self.cells = self._parser.get_all_cells()

    def get_game_vertex_id(self, level_vertex_id: int) -> int:
        """Get which game vertex this level vertex maps to"""
        return self._parser.get_game_vertex_id(level_vertex_id)

    def get_distance(self, level_vertex_id: int) -> float:
        """Get distance from level vertex to its game vertex"""
        return self._parser.get_distance(level_vertex_id)

    def find_level_vertices_for_game_vertex(self, game_vertex_id: int) -> np.ndarray:
        """Find all level vertices mapping to this game vertex"""
        return self._parser.find_level_vertices_for_game_vertex(game_vertex_id)


# ============================================================================
# Level Graph Reader (uses LevelAIParser from parsers module)
# ============================================================================

class LevelGraphReader:
    """
    Lightweight reader for level.ai positions only.

    This is a thin wrapper around LevelAIParser for backward compatibility.
    New code should use LevelAIParser directly.
    """

    def __init__(self, filepath: Path):
        # Use build_adjacency=False for lightweight loading (positions only)
        self._parser = LevelAIParser(filepath, build_adjacency=False)
        # Expose header for backward compatibility
        self.header = {
            'vertex_count': self._parser.header.vertex_count,
            'cell_size': self._parser.header.cell_size,
            'cell_size_y': self._parser.header.cell_size_y,
            'min': self._parser.header.min,
            'max': self._parser.header.max,
            'guid': self._parser.header.guid
        }

    def get_vertex_position(self, vertex_id: int) -> Tuple[float, float, float]:
        """Get world position of a level vertex"""
        return self._parser.get_vertex_position(vertex_id)


# ============================================================================
# Game Graph Merger
# ============================================================================

class GameGraphMerger:
    """
    Merge multiple level game graphs into global game graph

    Based on xrMergeGraphs() from xr_graph_merge.cpp
    """

    def __init__(self,
                 levels_config: LevelsConfig,
                 graph_points_by_level: Dict[int, List[dict]],
                 random_seed: Optional[int] = None,
                 base_mod: str = None,
                 mod_config=None):
        """
        Initialize merger.

        Args:
            levels_config: Level configurations (provides paths, lookups)
            graph_points_by_level: Map of level_id -> extracted graph points
            random_seed: Optional seed for death point randomization
            base_mod: Base mod name (anomaly, gamma)
            mod_config: ModConfig instance for enabled mods
        """
        self.levels_config = levels_config
        self.graph_points_by_level = graph_points_by_level
        self.base_mod = base_mod
        self.mod_config = mod_config

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Merged data
        self.vertices: List[GameVertex] = []
        self.edges: List[GameEdge] = []
        self.death_points: List[DeathPoint] = []

        # Mapping: (level_id, local_game_vertex_id) -> global_game_vertex_id
        self.vertex_mapping: Dict[Tuple[int, int], int] = {}

        # Mapping: graph_point_name -> global_game_vertex_id
        # Used for resolving level_changer destinations
        self.name_to_gvid: Dict[str, int] = {}

        # Level offsets: level_name -> cumulative GVID offset
        # The offset for a level is the sum of all previous levels' vertex counts
        self.level_offsets: Dict[str, int] = {}

        # Pre-extracted edges from original game graph (per level)
        # Dict[level_id, List[edge_data]]
        self.preloaded_edges_by_level: Dict[int, List[dict]] = {}
        self._load_level_edges()

    def _load_level_edges(self):
        """Load pre-extracted edges from per-level JSON files"""
        total_edges = 0
        levels_with_edges = 0

        for level in self.levels_config.levels:
            edges_path = self.levels_config.get_edges_path(level)
            if edges_path and edges_path.exists():
                with open(edges_path, 'r') as f:
                    data = json.load(f)

                edges = data.get('edges', [])
                self.preloaded_edges_by_level[level.id] = edges
                total_edges += len(edges)
                levels_with_edges += 1

        if total_edges > 0:
            logDebug(f"  Loaded {total_edges} pre-extracted edges from {levels_with_edges} levels")

    def merge(self) -> GameGraph:
        """
        Perform full merge

        Returns:
            GameGraph object containing all merged data
        """
        log("\n" + "=" * 70)
        log("MERGING GAME GRAPHS")
        log("=" * 70)

        # Step 1: Merge vertices from all levels
        self._merge_vertices()

        # Step 2: Add inter-level connection edges
        self._add_level_connections()

        # Step 2.5: Connect orphan nodes
        self._connect_orphan_nodes()

        # Step 3: Generate death points
        self._generate_death_points()

        # Step 4: Build flat edge and death point lists for serialization
        self._build_flat_lists()

        log(f"\nMerge complete:")
        log(f"  Total game vertices: {len(self.vertices)}")
        log(f"  Total edges: {len(self.edges)}")
        log(f"  Total death points: {len(self.death_points)}")
        log(f"  Average death points per vertex: {len(self.death_points) / len(self.vertices):.1f}")
        log(f"  Graph point name mappings: {len(self.name_to_gvid)}")

        # Build and return GameGraph object
        return GameGraph(
            vertices=self.vertices,
            edges=self.edges,
            death_points=self.death_points,
            levels=self.levels_config.levels,
            level_offsets=self.level_offsets,
            name_to_gvid=self.name_to_gvid,
            vertex_mapping=self.vertex_mapping,
            levels_config=self.levels_config,
            base_mod=self.base_mod,
            mod_config=self.mod_config
        )

    def _merge_vertices(self):
        """Merge game vertices from all levels"""
        log("\n[1/4] Merging vertices...")

        # Use LevelGraphNavigator for precise level_vertex_id computation
        # CRITICAL: Must use the same algorithm as build_cross_table.py to ensure
        # the cross-table GVID mappings match the merged game graph vertex ordering.

        global_vertex_id = 0

        for level in self.levels_config.levels:
            graph_points = self.graph_points_by_level.get(level.id, [])
            log(f"  Level {level.id}: {level.name}")
            log(f"    Graph points: {len(graph_points)}")

            # Record the GVID offset for this level (sum of all previous levels' vertex counts)
            self.level_offsets[level.name] = global_vertex_id

            # Load level graph for computing level_vertex_ids
            # Use LevelGraphNavigator (same as cross-table builder) for precise spatial search
            level_graph = None
            level_ai_path = self.levels_config.get_level_ai_path(level)
            if level_ai_path and level_ai_path.exists():
                try:
                    level_graph = LevelGraphNavigator(str(level_ai_path))
                    log(f"    Loaded level.ai: {level_graph.header['vertex_count']:,} vertices")
                except Exception as e:
                    logError(f"    Warning: Could not load level.ai: {e}")

            for local_id, graph_point in enumerate(graph_points):
                # Create game vertex
                pos = graph_point['position']
                local_point = (pos['x'], pos['y'], pos['z'])

                # Apply level offset for global position
                global_point = (
                    local_point[0] + level.offset[0],
                    local_point[1] + level.offset[1],
                    local_point[2] + level.offset[2]
                )

                # Compute level_vertex_id by finding nearest vertex in level.ai
                # CRITICAL: Use the SAME algorithm as build_cross_table.py!
                # This ensures the cross-table GVID mappings match our vertex ordering.
                level_vertex_id = -1

                if level_graph is not None:
                    level_vertex_id = level_graph.find_nearest_vertex(local_point)

                # Fallback: if still negative (no level.ai), use 0
                if level_vertex_id < 0:
                    level_vertex_id = 0

                # Get location types
                graph_point_data = graph_point.get('graph_point_data', {})
                locations = graph_point_data.get('locations', [0, 0, 0, 0])
                location_bytes = bytes(locations[:4])

                vertex = GameVertex(
                    local_point=local_point,
                    global_point=global_point,
                    level_id=level.id,
                    level_vertex_id=level_vertex_id,
                    vertex_types=location_bytes,
                    edges=[],
                    death_points=[]
                )

                self.vertices.append(vertex)

                # Map (level_id, local_id) -> global_id
                self.vertex_mapping[(level.id, local_id)] = global_vertex_id

                # Map graph point name -> global_id (for level_changer resolution)
                # Use qualified key "level_name:graph_point_name" since multiple levels
                # can have graph points with the same name (e.g., "start_actor_02")
                graph_point_name = graph_point.get('name_replace') or graph_point.get('original_name') or graph_point.get('entity_name')
                if graph_point_name:
                    qualified_key = f"{level.name}:{graph_point_name}"
                    self.name_to_gvid[qualified_key] = global_vertex_id

                global_vertex_id += 1

            log(f"    Added {len(graph_points)} vertices (total: {global_vertex_id})")

    def _add_level_connections(self):
        """Add edges between levels based on connection points or preloaded edges"""
        log("\n[2/4] Adding level connections...")

        # Use preloaded edges for levels that have them
        if self.preloaded_edges_by_level:
            self._add_preloaded_edges()

        # Also use symbolic connections for levels WITHOUT preloaded edges
        # This handles levels like k01_darkscape that have connection data
        # in their level.spawn but no .edges.json file
        self._add_symbolic_connections()

    def _add_preloaded_edges(self):
        """Add edges from preloaded per-level edge files, matching by position"""
        log("  Using preloaded edges from original game graph...")
        logDebug("  Matching edges to vertices by position...")

        # Build a spatial index of our vertices for fast lookup
        # Group vertices by level for efficiency
        vertices_by_level: Dict[int, List[Tuple[int, Tuple[float, float, float]]]] = {}

        for global_id, vertex in enumerate(self.vertices):
            level_id = vertex.level_id
            if level_id not in vertices_by_level:
                vertices_by_level[level_id] = []
            vertices_by_level[level_id].append((global_id, vertex.local_point))

        # Build level name -> level_id mapping (use cached property)
        level_name_to_id = self.levels_config.name_to_id

        def find_nearest_vertex(level_id: int, x: float, y: float, z: float,
                                  tolerance: float = VERTEX_MATCH_TOLERANCE) -> int:
            """Find the vertex closest to the given position within a level"""
            if level_id not in vertices_by_level:
                return -1

            best_id = -1
            best_dist_sq = tolerance * tolerance

            for global_id, pos in vertices_by_level[level_id]:
                dx = pos[0] - x
                dy = pos[1] - y
                dz = pos[2] - z
                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_id = global_id

            return best_id

        edges_added = 0
        edges_skipped_no_source = 0
        edges_skipped_no_target = 0
        edges_skipped_no_target_level = 0

        # Process edges from each level
        for source_level_id, edges in self.preloaded_edges_by_level.items():
            for edge_data in edges:
                target_level_name = edge_data['target_level']
                distance = edge_data['distance']

                # Look up target level ID by name
                target_level_id = level_name_to_id.get(target_level_name)
                if target_level_id is None:
                    logWarning(
                        f"Game graph merger: target level ID not set: source - {source_level_id} [{edge_data['source_x']}{edge_data['source_y']}{edge_data['source_z']}]]")
                    edges_skipped_no_target_level += 1
                    continue

                # Find source vertex by position
                source_id = find_nearest_vertex(
                    source_level_id,
                    edge_data['source_x'],
                    edge_data['source_y'],
                    edge_data['source_z']
                )

                if source_id < 0:
                    logWarning(
                        f"Game graph merger: source edge not found: source - {source_level_id} [{edge_data['source_x']}{edge_data['source_y']}{edge_data['source_z']}], target - {target_level_name} [{edge_data['target_x']}{edge_data['target_y']}{edge_data['target_z']}]")
                    edges_skipped_no_source += 1
                    continue

                # Find target vertex by position
                target_id = find_nearest_vertex(
                    target_level_id,
                    edge_data['target_x'],
                    edge_data['target_y'],
                    edge_data['target_z']
                )

                if target_id < 0:
                    logWarning(f"Game graph merger: target edge not found: source - {source_level_id} [{edge_data['source_x']}{edge_data['source_y']}{edge_data['source_z']}], target - {target_level_name} [{edge_data['target_x']}{edge_data['target_y']}{edge_data['target_z']}]")
                    edges_skipped_no_target += 1
                    continue

                # Add edge
                edge = GameEdge(vertex_id=target_id, distance=distance)
                self.vertices[source_id].edges.append(edge)
                edges_added += 1

        # Analyze what we added
        if edges_added > 0:
            # Count intra vs inter level
            intra_level_count = 0
            inter_level_count = 0
            for vertex in self.vertices:
                for edge in vertex.edges:
                    if self.vertices[edge.vertex_id].level_id == vertex.level_id:
                        intra_level_count += 1
                    else:
                        inter_level_count += 1

            log(f"  Added {edges_added} edges")
            log(f"    Intra-level: {intra_level_count}")
            log(f"    Inter-level: {inter_level_count}")

        if edges_skipped_no_target_level > 0:
            logWarning(f"  Game graph merger skipped {edges_skipped_no_target_level} edges (target level name not found)")
        if edges_skipped_no_source > 0:
            logWarning(f"  Game graph merger skipped {edges_skipped_no_source} edges (source vertex not found)")
        if edges_skipped_no_target > 0:
            logWarning(f"  Game graph merger skipped {edges_skipped_no_target} edges (target vertex not found)")

    def _has_edge(self, source_id: int, target_id: int) -> bool:
        """Check if an edge from source to target already exists"""
        for edge in self.vertices[source_id].edges:
            if edge.vertex_id == target_id:
                return True
        return False

    def _add_symbolic_connections(self):
        """Add edges using symbolic connection names from level.spawn files"""
        logDebug("  Using symbolic connection lookup...")

        connections_added = 0
        connection_points_found = 0
        target_level_not_found = 0
        target_point_not_found = 0
        duplicates_skipped = 0

        for level in self.levels_config.levels:
            graph_points = self.graph_points_by_level.get(level.id, [])
            for local_id, graph_point in enumerate(graph_points):
                graph_point_data = graph_point.get('graph_point_data', {})
                target_level_name = graph_point_data.get('connection_level_name', '')
                target_point_name = graph_point_data.get('connection_point_name', '')

                if not target_level_name or not target_point_name:
                    continue  # Not a connection point

                connection_points_found += 1

                # Find source global ID
                source_global_id = self.vertex_mapping[(level.id, local_id)]

                # Find target level
                target_level = self.levels_config.get_level_by_name(target_level_name)

                if not target_level:
                    logError(f"    Error: Target level not found: {target_level_name}")
                    target_level_not_found += 1
                    continue

                # Find target graph point by name (check both name_replace and original_name)
                target_graph_points = self.graph_points_by_level.get(target_level.id, [])
                target_local_id = None
                for t_local_id, target_gp in enumerate(target_graph_points):
                    # Check current name
                    if target_gp.get('name_replace') == target_point_name:
                        target_local_id = t_local_id
                        break
                    # Also check original name (for merged graph points)
                    if target_gp.get('original_name') == target_point_name:
                        target_local_id = t_local_id
                        break

                if target_local_id is None:
                    logWarning(f"    Warning: Target point not found: Source {graph_point['name_replace']} in {level.name} TO {target_point_name} in {target_level_name}")
                    target_point_not_found += 1
                    continue

                # Get target global ID
                target_global_id = self.vertex_mapping[(target_level.id, target_local_id)]

                # Calculate distance (simple Euclidean for now)
                src_pos = self.vertices[source_global_id].global_point
                tgt_pos = self.vertices[target_global_id].global_point

                dx = tgt_pos[0] - src_pos[0]
                dy = tgt_pos[1] - src_pos[1]
                dz = tgt_pos[2] - src_pos[2]
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5

                # Add bidirectional edges (with deduplication)
                forward_added = False
                backward_added = False

                # Add forward edge if it doesn't exist
                if not self._has_edge(source_global_id, target_global_id):
                    edge_forward = GameEdge(vertex_id=target_global_id, distance=distance)
                    self.vertices[source_global_id].edges.append(edge_forward)
                    forward_added = True
                else:
                    duplicates_skipped += 1

                # Add backward edge if it doesn't exist
                if not self._has_edge(target_global_id, source_global_id):
                    edge_backward = GameEdge(vertex_id=source_global_id, distance=distance)
                    self.vertices[target_global_id].edges.append(edge_backward)
                    backward_added = True
                else:
                    duplicates_skipped += 1

                if forward_added or backward_added:
                    connections_added += 1
                    log(f"    {graph_point['name_replace']} <-> {target_point_name} ({distance:.1f}m)")

        log(f"  Connection points found: {connection_points_found}")
        log(f"  Target level not found: {target_level_not_found}")
        log(f"  Target point not found: {target_point_not_found}")
        log(f"  Duplicates skipped: {duplicates_skipped}")
        log(f"  Connections made: {connections_added}")

    def _connect_orphan_nodes(self):
        """Connect orphan vertices to nearest non-orphan neighbors"""
        log("\n[2.5/4] Connecting orphan nodes...")

        from graph.orphan_connector import OrphanConnector

        # Check if any levels have orphan connection enabled
        levels_with_flag = [
            level for level in self.levels_config.levels
            if level.connect_orphans_automatically
        ]

        if not levels_with_flag:
            log("  No levels have connect_orphans_automatically enabled")
            return

        connector = OrphanConnector(
            vertices=self.vertices,
            level_ai_cache=None,  # Could add level.ai cache for reachability checks
            require_reachability=False
        )

        total_connections = 0
        total_pruned = 0

        for level in levels_with_flag:
            # Load level.ai for complexity-based pruning
            level_ai = None
            level_ai_path = self.levels_config.get_level_ai_path(level)
            if level_ai_path and level_ai_path.exists():
                try:
                    level_ai = LevelGraphNavigator(str(level_ai_path))
                except Exception as e:
                    logWarning(f"    Could not load level.ai for {level.name}: {e}")

            result = connector.connect_level(level.id, level.name, level_ai=level_ai)

            if result.connections_made > 0 or result.connections_pruned > 0:
                log(f"    {level.name}: made {result.connections_made} connections "
                    f"({result.orphan_count} orphans found)")
                total_connections += result.connections_made
                total_pruned += result.connections_pruned

            for error in result.errors:
                logWarning(f"    {level.name}: {error}")

        log(f"  Total orphan connections made: {total_connections}")
        if total_pruned > 0:
            log(f"  Total high-complexity connections pruned: {total_pruned}")

    def _generate_death_points(self):
        """Generate death points for all game vertices"""
        log("\n[3/4] Generating death points...")

        total_death_points = 0

        for level in self.levels_config.levels:
            graph_points = self.graph_points_by_level.get(level.id, [])
            log(f"  Level {level.id}: {level.name}")

            # Load cross table and level graph
            cross_table_path = self.levels_config.get_cross_table_path(level)
            level_ai_path = self.levels_config.get_level_ai_path(level)

            if not cross_table_path or not cross_table_path.exists():
                logError("ERROR: No cross table found - skipping")
                continue

            cross_table = CrossTableReader(cross_table_path)
            level_graph = LevelGraphReader(level_ai_path)

            # Create generator
            generator = DeathPointGenerator(cross_table, level_graph)

            # Generate for each game vertex in this level
            for local_id in range(len(graph_points)):
                global_id = self.vertex_mapping[(level.id, local_id)]

                # Generate death points
                death_points = generator.generate_for_vertex(local_id)

                # Apply level offset to positions
                for dp in death_points:
                    dp.position = (
                        dp.position[0] + level.offset[0],
                        dp.position[1] + level.offset[1],
                        dp.position[2] + level.offset[2]
                    )

                self.vertices[global_id].death_points = death_points
                total_death_points += len(death_points)

            log(f"    Generated {total_death_points} death points so far")

        log(f"  Total death points: {total_death_points}")

    def _build_flat_lists(self):
        """
        Build flat edge and death point lists from vertices.

        These flat lists are used for:
        - Statistics in the merge summary
        - Serialization (offset calculation happens in GameGraphSerializer)
        """
        log("\n[4/4] Building flat edge and death point lists...")

        self.edges = []
        self.death_points = []

        for vertex in self.vertices:
            self.edges.extend(vertex.edges)
            self.death_points.extend(vertex.death_points)

        log(f"  Edges: {len(self.edges)}")
        log(f"  Death points: {len(self.death_points)}")