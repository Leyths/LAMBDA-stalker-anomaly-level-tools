#!/usr/bin/env python3
"""
Orphan Connector

Automatically connects orphaned game graph vertices to ensure all level vertices
form a single connected component. Uses a greedy nearest-neighbor approach to
minimize total edge distance added.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import math

from utils import log, logDebug, logWarning, logError


@dataclass
class ConnectionResult:
    """Result of connecting orphan nodes for a level"""
    level_id: int
    level_name: str
    total_vertices: int
    orphan_count: int
    connections_made: int
    connections_pruned: int
    errors: List[str]


@dataclass
class GeneratedEdge:
    """Tracks a generated edge for complexity analysis"""
    source_gvid: int
    target_gvid: int
    euclidean_distance: float
    # Filled in during pruning phase:
    path_distance: Optional[float] = None
    complexity: Optional[float] = None


class OrphanConnector:
    """
    Ensures all level vertices form a single connected component.

    Detects orphan vertices (no edges) and disconnected components (edges exist
    but isolated from main graph), then connects them using greedy nearest-neighbor.
    """

    def __init__(self, vertices: List, level_ai_cache: Optional[Dict] = None,
                 require_reachability: bool = False, distance_tolerance_pct: float = 0.9,
                 complexity_threshold: float = 1.4, min_angular_separation: float = 30.0):
        """
        Initialize connector.

        Args:
            vertices: List of GameVertex objects (from GameGraphMerger)
            level_ai_cache: Optional cache of loaded LevelGraphNavigator objects
            require_reachability: If True, verify vertices are connected via level.ai
                                  before adding edges (prevents connections across gaps)
            distance_tolerance_pct: When connecting an orphan to its nearest neighbor,
                                    also connect to other nodes within this percentage
                                    of the nearest distance. E.g., 0.2 means 20% tolerance:
                                    nearest=100m → connect up to 120m (default: 0.2)
            complexity_threshold: Maximum path/straight ratio for "low complexity"
                                  connections. Connections with higher ratios may
                                  be pruned if better alternatives exist (default: 1.2)
            min_angular_separation: Minimum angle (degrees) between connections
                                    from an orphan to different targets. Targets
                                    closer in angle to a nearer target are skipped.
                                    (default: 30.0°)
        """
        self.vertices = vertices
        self.level_ai_cache = level_ai_cache or {}
        self.require_reachability = require_reachability
        self.distance_tolerance_pct = distance_tolerance_pct
        self.complexity_threshold = complexity_threshold
        self.min_angular_separation = min_angular_separation
        # Track generated edges for complexity-based pruning
        self.generated_edges: List[GeneratedEdge] = []

    def connect_level(self, level_id: int, level_name: str,
                      level_ai=None) -> ConnectionResult:
        """
        Connect orphan vertices for a specific level.

        Uses greedy nearest-neighbor to ensure all level vertices form a single
        connected component:
        1. Find all vertices for this level
        2. Build adjacency map from existing edges
        3. Find vertices with inter-level edges (world-connected)
        4. BFS from seed to find reachable set
        5. Greedy nearest-neighbor to connect unreachable vertices
        6. Prune high-complexity connections using level.ai path analysis

        Args:
            level_id: Level ID to process
            level_name: Level name (for logging)
            level_ai: Optional LevelGraphNavigator for complexity-based pruning

        Returns:
            ConnectionResult with statistics
        """
        errors = []

        # Clear generated edges from previous calls
        self.generated_edges = []

        # Step 1: Get all vertices for this level
        level_vertices = self._get_level_vertices(level_id)

        if not level_vertices:
            return ConnectionResult(
                level_id=level_id,
                level_name=level_name,
                total_vertices=0,
                orphan_count=0,
                connections_made=0,
                connections_pruned=0,
                errors=[]
            )

        # Step 2: Build adjacency map for this level's vertices
        adjacency = self._build_adjacency_map(level_vertices, level_id)

        # Step 3: Find vertices with inter-level edges (these anchor us to world graph)
        world_connected = self._find_world_connected_vertices(level_vertices, level_id)

        # Step 4: Determine seed for BFS
        if world_connected:
            seed_gvid = world_connected[0]
            logDebug(f"    Using world-connected vertex {seed_gvid} as seed")
        else:
            # No inter-level edges - level is isolated from world graph
            seed_gvid = level_vertices[0]
            errors.append(f"Level {level_name} has no inter-level connections - "
                         "needs edges to other levels!")
            logError(f"    Level {level_name} has no inter-level connections!")

        # Step 5: BFS to find all vertices reachable from seed within this level
        reachable = self._bfs_reachable(seed_gvid, adjacency, set(level_vertices))
        unreachable = set(level_vertices) - reachable

        if not unreachable:
            logDebug(f"    Level {level_name}: all {len(level_vertices)} vertices already connected")
            return ConnectionResult(
                level_id=level_id,
                level_name=level_name,
                total_vertices=len(level_vertices),
                orphan_count=0,
                connections_made=0,
                connections_pruned=0,
                errors=errors
            )

        orphan_count = len(unreachable)
        log(f"    {level_name}: {orphan_count} vertices need connection")

        # Step 6: Greedy nearest-neighbor to connect unreachable vertices
        connections_made = self._greedy_connect(reachable, unreachable, level_name, errors)

        # Step 7: Prune high-complexity connections using level.ai path analysis
        connections_pruned = 0
        if level_ai and self.generated_edges:
            connections_pruned = self._prune_complex_connections(level_ai, level_name)
            if connections_pruned > 0:
                log(f"    Pruned {connections_pruned} high-complexity connections")

        return ConnectionResult(
            level_id=level_id,
            level_name=level_name,
            total_vertices=len(level_vertices),
            orphan_count=orphan_count,
            connections_made=connections_made,
            connections_pruned=connections_pruned,
            errors=errors
        )

    def _get_level_vertices(self, level_id: int) -> List[int]:
        """Get all GVIDs for vertices belonging to this level"""
        return [
            gvid for gvid, vertex in enumerate(self.vertices)
            if vertex.level_id == level_id
        ]

    def _build_adjacency_map(self, level_vertices: List[int],
                             level_id: int) -> Dict[int, Set[int]]:
        """
        Build adjacency map for level vertices.

        Only includes edges within the same level (for component detection).
        """
        level_vertex_set = set(level_vertices)
        adjacency: Dict[int, Set[int]] = {gvid: set() for gvid in level_vertices}

        for gvid in level_vertices:
            vertex = self.vertices[gvid]
            for edge in vertex.edges:
                target_gvid = edge.vertex_id
                # Only include intra-level edges for component detection
                if target_gvid in level_vertex_set:
                    adjacency[gvid].add(target_gvid)
                    # Ensure bidirectional (if edge exists target->source, mark it)
                    if target_gvid in adjacency:
                        adjacency[target_gvid].add(gvid)

        return adjacency

    def _find_world_connected_vertices(self, level_vertices: List[int],
                                       level_id: int) -> List[int]:
        """
        Find vertices that have inter-level edges (connected to world graph).

        These vertices anchor the level to the rest of the game world.
        """
        world_connected = []

        for gvid in level_vertices:
            vertex = self.vertices[gvid]
            for edge in vertex.edges:
                target_vertex = self.vertices[edge.vertex_id]
                if target_vertex.level_id != level_id:
                    world_connected.append(gvid)
                    break

        return world_connected

    def _bfs_reachable(self, seed_gvid: int, adjacency: Dict[int, Set[int]],
                       level_vertices: Set[int]) -> Set[int]:
        """
        BFS from seed to find all reachable vertices within level.

        Only follows edges within the level (as recorded in adjacency map).
        """
        reachable = set()
        queue = deque([seed_gvid])
        reachable.add(seed_gvid)

        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in reachable and neighbor in level_vertices:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        return reachable

    def _greedy_connect(self, reachable: Set[int], unreachable: Set[int],
                        level_name: str, errors: List[str]) -> int:
        """
        Greedy nearest-neighbor algorithm to connect unreachable vertices.

        Always connects the closest unreachable vertex to any reachable vertex,
        plus any other reachable vertices within distance_tolerance of that
        closest distance. Then moves the orphan to the reachable set.
        """
        connections_made = 0

        while unreachable:
            # Find the (unreachable, reachable) pair with minimum distance
            best_unreach = None
            best_dist = float('inf')

            # Also track all distances from each unreachable to all reachable
            # so we can find approximately equidistant nodes
            distances_from_orphan: Dict[int, List[Tuple[int, float]]] = {}

            for unreach_gvid in unreachable:
                unreach_pos = self.vertices[unreach_gvid].global_point
                distances_from_orphan[unreach_gvid] = []

                for reach_gvid in reachable:
                    reach_pos = self.vertices[reach_gvid].global_point
                    dist = self._euclidean_distance(unreach_pos, reach_pos)
                    distances_from_orphan[unreach_gvid].append((reach_gvid, dist))

                    if dist < best_dist:
                        best_dist = dist
                        best_unreach = unreach_gvid

            if best_unreach is None:
                # Should not happen unless unreachable is empty
                break

            # Get the minimum distance for this orphan
            orphan_distances = distances_from_orphan[best_unreach]
            min_dist = min(d for _, d in orphan_distances)

            # Find all reachable nodes within tolerance percentage of the minimum distance
            threshold = min_dist * (1 + self.distance_tolerance_pct)
            targets_to_connect = [
                (reach_gvid, dist) for reach_gvid, dist in orphan_distances
                if dist <= threshold
            ]

            # Filter targets by angular separation
            pre_filter_count = len(targets_to_connect)
            targets_to_connect = self._filter_by_angular_separation(
                best_unreach, targets_to_connect
            )
            if len(targets_to_connect) < pre_filter_count:
                logDebug(f"    Orphan {best_unreach}: {pre_filter_count} candidates within distance tolerance, "
                        f"filtered to {len(targets_to_connect)} by angular separation ({self.min_angular_separation}°)")

            # Optional: verify level.ai reachability for the primary target
            if self.require_reachability:
                primary_target = min(targets_to_connect, key=lambda x: x[1])[0]
                if not self._verify_reachability(best_unreach, primary_target, level_name):
                    logWarning(f"    Vertex {best_unreach} unreachable from {primary_target} via level.ai, skipping")
                    unreachable.remove(best_unreach)
                    errors.append(f"Vertex {best_unreach} unreachable via level.ai")
                    continue

            # Connect to all targets within tolerance
            for target_gvid, dist in targets_to_connect:
                if not self._has_edge(best_unreach, target_gvid):
                    self._add_edge(best_unreach, target_gvid, dist)
                    self._add_edge(target_gvid, best_unreach, dist)
                    connections_made += 1
                    logDebug(f"    Connected vertex {best_unreach} to {target_gvid} ({dist:.1f}m)")

                    # Track generated edge for complexity-based pruning
                    # Only track from orphan's perspective (source=orphan)
                    self.generated_edges.append(GeneratedEdge(
                        source_gvid=best_unreach,
                        target_gvid=target_gvid,
                        euclidean_distance=dist
                    ))

            if len(targets_to_connect) > 1:
                logDebug(f"    Orphan {best_unreach}: connected to {len(targets_to_connect)} nodes "
                        f"(nearest: {min_dist:.1f}m, +{self.distance_tolerance_pct*100:.0f}% → {threshold:.1f}m)")

            # Move vertex to reachable set
            unreachable.remove(best_unreach)
            reachable.add(best_unreach)

        return connections_made

    def _euclidean_distance(self, pos1: Tuple[float, float, float],
                            pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _filter_by_angular_separation(
        self, orphan_gvid: int, targets: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Filter targets to ensure minimum angular separation.

        Keeps the closest target in each angular sector. Targets too close
        in angle to a closer target are filtered out.

        Args:
            orphan_gvid: The orphan vertex being connected
            targets: List of (target_gvid, distance) tuples

        Returns:
            Filtered list with angular separation enforced
        """
        if len(targets) <= 1:
            return targets

        orphan_pos = self.vertices[orphan_gvid].global_point

        # Sort by distance (closest first)
        sorted_targets = sorted(targets, key=lambda x: x[1])

        # Calculate angle from orphan to target (XZ plane)
        def calc_angle(target_gvid: int) -> float:
            target_pos = self.vertices[target_gvid].global_point
            dx = target_pos[0] - orphan_pos[0]
            dz = target_pos[2] - orphan_pos[2]
            return math.atan2(dz, dx)  # radians

        accepted = []
        accepted_angles = []
        min_sep_rad = math.radians(self.min_angular_separation)

        for target_gvid, dist in sorted_targets:
            angle = calc_angle(target_gvid)

            # Check if too close to any already-accepted target
            too_close = False
            for accepted_angle in accepted_angles:
                # Angular difference (handle wraparound)
                diff = abs(angle - accepted_angle)
                if diff > math.pi:
                    diff = 2 * math.pi - diff

                if diff < min_sep_rad:
                    too_close = True
                    break

            if not too_close:
                accepted.append((target_gvid, dist))
                accepted_angles.append(angle)

        return accepted

    def _has_edge(self, source_gvid: int, target_gvid: int) -> bool:
        """Check if an edge from source to target already exists"""
        for edge in self.vertices[source_gvid].edges:
            if edge.vertex_id == target_gvid:
                return True
        return False

    def _add_edge(self, source_gvid: int, target_gvid: int, distance: float):
        """Add an edge from source to target"""
        # Import here to avoid circular dependency
        from game_graph_merger import GameEdge

        edge = GameEdge(vertex_id=target_gvid, distance=distance)
        self.vertices[source_gvid].edges.append(edge)

    def _remove_edge(self, source_gvid: int, target_gvid: int) -> bool:
        """Remove an edge from source to target. Returns True if removed."""
        edges = self.vertices[source_gvid].edges
        for i, edge in enumerate(edges):
            if edge.vertex_id == target_gvid:
                edges.pop(i)
                return True
        return False

    def _prune_complex_connections(self, level_ai, level_name: str) -> int:
        """
        Prune generated connections with high complexity scores.

        Complexity is calculated as: path_distance / euclidean_distance
        where path_distance is the actual walking distance through level.ai.

        For each orphan vertex:
        - Calculate complexity for all its generated connections
        - If ANY connection has low complexity (< threshold): prune all high-complexity ones
        - If ALL connections have high complexity: keep only the lowest one

        Args:
            level_ai: LevelGraphNavigator for path distance calculation
            level_name: Level name (for logging)

        Returns:
            Number of connections pruned
        """
        # Group generated edges by orphan vertex (source)
        edges_by_orphan: Dict[int, List[GeneratedEdge]] = {}
        for edge in self.generated_edges:
            if edge.source_gvid not in edges_by_orphan:
                edges_by_orphan[edge.source_gvid] = []
            edges_by_orphan[edge.source_gvid].append(edge)

        pruned_count = 0

        for orphan_gvid, edges in edges_by_orphan.items():
            # Calculate complexity for each edge
            for edge in edges:
                source_lvid = self.vertices[edge.source_gvid].level_vertex_id
                target_lvid = self.vertices[edge.target_gvid].level_vertex_id

                path_dist, _ = level_ai.bfs_path_distance(source_lvid, target_lvid)
                edge.path_distance = path_dist

                if path_dist == float('inf'):
                    edge.complexity = float('inf')
                else:
                    # Avoid division by zero
                    edge.complexity = path_dist / max(edge.euclidean_distance, 0.001)

            # Determine which edges to keep
            low_complexity = [e for e in edges if e.complexity <= self.complexity_threshold]

            if low_complexity:
                # Has good options - prune the bad ones
                to_prune = [e for e in edges if e.complexity > self.complexity_threshold]
            else:
                # No good options - keep only the single best one
                edges.sort(key=lambda e: e.complexity)
                to_prune = edges[1:]  # Prune all except the best

            # Remove pruned edges from vertex.edges (bidirectional)
            for edge in to_prune:
                self._remove_edge(edge.source_gvid, edge.target_gvid)
                self._remove_edge(edge.target_gvid, edge.source_gvid)
                pruned_count += 1
                logDebug(f"    Pruned edge {edge.source_gvid}<->{edge.target_gvid} "
                         f"(complexity: {edge.complexity:.2f})")

        return pruned_count

    def _verify_reachability(self, source_gvid: int, target_gvid: int,
                             level_name: str) -> bool:
        """
        Verify two vertices are connected via level.ai walkable mesh.

        Uses BFS on the level.ai graph to check if there's a path.
        This prevents connecting nodes across cliffs, walls, etc.
        """
        if level_name not in self.level_ai_cache:
            # No level.ai available, assume reachable
            return True

        level_ai = self.level_ai_cache[level_name]

        source_lvid = self.vertices[source_gvid].level_vertex_id
        target_lvid = self.vertices[target_gvid].level_vertex_id

        try:
            # Use BFS to check connectivity
            distances = level_ai.bfs_distances(source_lvid, max_distance=10000)
            # INFINITY is max uint32
            INFINITY = 0xFFFFFFFF
            return distances[target_lvid] != INFINITY
        except Exception as e:
            logWarning(f"    Could not verify reachability: {e}")
            return True  # Assume reachable on error
