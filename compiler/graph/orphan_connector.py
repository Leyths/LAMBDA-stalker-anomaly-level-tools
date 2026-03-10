#!/usr/bin/env python3
"""
Orphan Connector

Connects orphaned game graph vertices to ensure all level vertices form a
single connected component. Uses level.ai walkability validation to only
create edges between vertices NPCs can actually walk between.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from collections import deque
import math

from utils import log, logDebug, logError


@dataclass
class ConnectionResult:
    """Result of connecting orphan nodes for a level"""
    level_id: int
    level_name: str
    total_vertices: int
    orphan_count: int
    connections_made: int
    errors: List[str]


class OrphanConnector:
    """
    Ensures all level vertices form a single connected component.

    Builds walkability-validated edges using level.ai, then verifies
    full connectivity and force-connects any remaining gaps.
    """

    def __init__(self, vertices: List,
                 complexity_threshold: float = 1.3,
                 min_angular_separation: float = 30.0,
                 max_neighbours: int = 10,
                 max_edges_per_vertex: int = 3):
        """
        Args:
            vertices: List of GameVertex objects (from GameGraphMerger)
            complexity_threshold: Maximum path_distance/manhattan_xz_distance ratio.
                                  Measures obstruction independent of direction. (default: 1.3)
            min_angular_separation: Minimum angle (degrees) between connections
                                    from a vertex to different targets. (default: 30.0)
            max_neighbours: Max candidate neighbours to evaluate per vertex (default: 10)
            max_edges_per_vertex: Max edges to keep per vertex after pruning (default: 3)
        """
        self.vertices = vertices
        self.complexity_threshold = complexity_threshold
        self.min_angular_separation = min_angular_separation
        self.max_neighbours = max_neighbours
        self.max_edges_per_vertex = max_edges_per_vertex

    def connect_level(self, level_id: int, level_name: str,
                      level_ai) -> ConnectionResult:
        """
        Connect orphan vertices for a specific level.

        Requires level.ai (LevelGraphNavigator) for walkability validation.
        Raises RuntimeError if level_ai is None.

        Args:
            level_id: Level ID to process
            level_name: Level name (for logging)
            level_ai: LevelGraphNavigator for walkability validation (required)

        Returns:
            ConnectionResult with statistics
        """
        if level_ai is None:
            raise RuntimeError(
                f"level.ai is required for orphan connection but was not "
                f"loaded for {level_name}. Cannot proceed."
            )

        errors = []

        # Step 1: Get all vertices for this level
        level_vertices = self._get_level_vertices(level_id)

        if not level_vertices:
            return ConnectionResult(
                level_id=level_id, level_name=level_name,
                total_vertices=0, orphan_count=0,
                connections_made=0, errors=[]
            )

        # Step 2: Build adjacency map from existing edges
        adjacency = self._build_adjacency_map(level_vertices, level_id)

        # Step 3: Find seed vertex (world-connected preferred)
        world_connected = self._find_world_connected_vertices(level_vertices, level_id)
        if world_connected:
            seed_gvid = world_connected[0]
            logDebug(f"    Using world-connected vertex {seed_gvid} as seed")
        else:
            seed_gvid = level_vertices[0]
            errors.append(f"Level {level_name} has no inter-level connections")
            logError(f"    Level {level_name} has no inter-level connections!")

        # Step 4: Find unreachable vertices
        level_vertex_set = set(level_vertices)
        reachable = self._bfs_reachable(seed_gvid, adjacency, level_vertex_set)
        unreachable = level_vertex_set - reachable

        if not unreachable:
            logDebug(f"    Level {level_name}: all {len(level_vertices)} vertices already connected")
            return ConnectionResult(
                level_id=level_id, level_name=level_name,
                total_vertices=len(level_vertices), orphan_count=0,
                connections_made=0, errors=errors
            )

        orphan_count = len(unreachable)
        log(f"    {level_name}: {orphan_count} vertices need connection")

        # Step 5: Build walkability-validated edge network
        connections_made = self._build_edge_network(
            level_vertices, level_id, level_name, level_ai,
            seed_gvid, errors
        )

        return ConnectionResult(
            level_id=level_id, level_name=level_name,
            total_vertices=len(level_vertices), orphan_count=orphan_count,
            connections_made=connections_made, errors=errors
        )

    # =========================================================================
    # Graph queries
    # =========================================================================

    def _get_level_vertices(self, level_id: int) -> List[int]:
        """Get all GVIDs belonging to this level."""
        return [
            gvid for gvid, vertex in enumerate(self.vertices)
            if vertex.level_id == level_id
        ]

    def _build_adjacency_map(self, level_vertices: List[int],
                             level_id: int) -> Dict[int, Set[int]]:
        """Build bidirectional adjacency map for intra-level edges only."""
        level_vertex_set = set(level_vertices)
        adjacency: Dict[int, Set[int]] = {gvid: set() for gvid in level_vertices}

        for gvid in level_vertices:
            for edge in self.vertices[gvid].edges:
                target_gvid = edge.vertex_id
                if target_gvid in level_vertex_set:
                    adjacency[gvid].add(target_gvid)
                    adjacency[target_gvid].add(gvid)

        return adjacency

    def _find_world_connected_vertices(self, level_vertices: List[int],
                                       level_id: int) -> List[int]:
        """Find vertices with inter-level edges (anchors to world graph)."""
        world_connected = []
        for gvid in level_vertices:
            for edge in self.vertices[gvid].edges:
                if self.vertices[edge.vertex_id].level_id != level_id:
                    world_connected.append(gvid)
                    break
        return world_connected

    def _bfs_reachable(self, seed_gvid: int, adjacency: Dict[int, Set[int]],
                       level_vertices: Set[int]) -> Set[int]:
        """BFS from seed to find all reachable vertices within level."""
        reachable = {seed_gvid}
        queue = deque([seed_gvid])

        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in reachable and neighbor in level_vertices:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        return reachable

    # =========================================================================
    # Edge operations
    # =========================================================================

    def _has_edge(self, source_gvid: int, target_gvid: int) -> bool:
        for edge in self.vertices[source_gvid].edges:
            if edge.vertex_id == target_gvid:
                return True
        return False

    def _add_edge(self, source_gvid: int, target_gvid: int, distance: float):
        from game_graph_merger import GameEdge
        self.vertices[source_gvid].edges.append(
            GameEdge(vertex_id=target_gvid, distance=distance)
        )

    def _add_bidi_edge(self, gvid_a: int, gvid_b: int, distance: float):
        """Add bidirectional edge if it doesn't already exist. Returns True if added."""
        if self._has_edge(gvid_a, gvid_b):
            return False
        self._add_edge(gvid_a, gvid_b, distance)
        self._add_edge(gvid_b, gvid_a, distance)
        return True

    def _remove_bidi_edge(self, gvid_a: int, gvid_b: int):
        """Remove bidirectional edge between two vertices."""
        self.vertices[gvid_a].edges = [
            e for e in self.vertices[gvid_a].edges if e.vertex_id != gvid_b
        ]
        self.vertices[gvid_b].edges = [
            e for e in self.vertices[gvid_b].edges if e.vertex_id != gvid_a
        ]

    # =========================================================================
    # Geometry helpers
    # =========================================================================

    def _euclidean_distance(self, pos1: Tuple[float, float, float],
                            pos2: Tuple[float, float, float]) -> float:
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _segments_cross_xz(p1, p2, p3, p4) -> bool:
        """Check if segments p1-p2 and p3-p4 cross in XZ plane (ignoring Y)."""
        def cross_2d(ox, oz, ax, az, bx, bz):
            return (ax - ox) * (bz - oz) - (az - oz) * (bx - ox)

        d1 = cross_2d(p3[0], p3[2], p4[0], p4[2], p1[0], p1[2])
        d2 = cross_2d(p3[0], p3[2], p4[0], p4[2], p2[0], p2[2])
        d3 = cross_2d(p1[0], p1[2], p2[0], p2[2], p3[0], p3[2])
        d4 = cross_2d(p1[0], p1[2], p2[0], p2[2], p4[0], p4[2])

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def _remove_crossing_edges(self, level_vertices: List[int],
                               level_id: int, seed_gvid: int) -> int:
        """
        Remove edges that cross in XZ projection (top-down view).

        For each crossing pair, removes the longer edge — but only if
        both its endpoints would retain at least one other intra-level
        connection. After all removals, verifies the graph stays fully
        connected via BFS from the seed; skips any removal that would
        break connectivity.
        """
        level_vertex_set = set(level_vertices)

        # Collect all intra-level edge pairs (deduplicated)
        edge_list: List[Tuple[int, int, float]] = []
        seen: Set[Tuple[int, int]] = set()
        for gvid in level_vertices:
            for edge in self.vertices[gvid].edges:
                tgt = edge.vertex_id
                if tgt not in level_vertex_set:
                    continue
                pair = (min(gvid, tgt), max(gvid, tgt))
                if pair not in seen:
                    seen.add(pair)
                    edge_list.append((pair[0], pair[1], edge.distance))

        log(f"    Phase 4: Checking {len(edge_list)} edges for crossings...")

        if len(edge_list) < 2:
            return 0

        # Find all crossing pairs (only if edges are at similar height)
        max_y_separation = 5.0
        crossings: List[Tuple[int, int]] = []  # indices into edge_list
        for i in range(len(edge_list)):
            a, b, _ = edge_list[i]
            pa = self.vertices[a].global_point
            pb = self.vertices[b].global_point
            avg_y_i = (pa[1] + pb[1]) / 2.0
            for j in range(i + 1, len(edge_list)):
                c, d, _ = edge_list[j]
                # Skip if edges share a vertex
                if a == c or a == d or b == c or b == d:
                    continue
                pc = self.vertices[c].global_point
                pd = self.vertices[d].global_point
                # Skip if edges are at very different heights (e.g. bridge over road)
                avg_y_j = (pc[1] + pd[1]) / 2.0
                if abs(avg_y_i - avg_y_j) > max_y_separation:
                    continue
                if self._segments_cross_xz(pa, pb, pc, pd):
                    crossings.append((i, j))

        log(f"    Phase 4: Found {len(crossings)} crossing pairs")

        if not crossings:
            return 0

        # For each crossing, mark the longer edge for removal
        # Sort so we process the longest offending edges first
        removal_candidates: List[Tuple[float, int]] = []
        for i, j in crossings:
            dist_i = edge_list[i][2]
            dist_j = edge_list[j][2]
            if dist_i >= dist_j:
                removal_candidates.append((dist_i, i))
            else:
                removal_candidates.append((dist_j, j))

        # Deduplicate and sort longest-first
        removal_candidates = sorted(set(removal_candidates), reverse=True)

        # Build adjacency for connectivity checks
        adjacency = self._build_adjacency_map(level_vertices, level_id)
        removed = set()
        removed_count = 0

        for _, edge_idx in removal_candidates:
            if edge_idx in removed:
                continue
            a, b, dist = edge_list[edge_idx]

            # Don't remove if either vertex would be left with no intra-level edges
            if len(adjacency[a]) <= 1 or len(adjacency[b]) <= 1:
                continue

            # Trial removal: check connectivity is preserved
            adjacency[a].discard(b)
            adjacency[b].discard(a)

            reachable = self._bfs_reachable(seed_gvid, adjacency, level_vertex_set)
            if len(reachable) == len(level_vertex_set):
                # Safe to remove
                self._remove_bidi_edge(a, b)
                removed.add(edge_idx)
                removed_count += 1
                logDebug(f"    Removed crossing edge {a} <-> {b} ({dist:.1f}m)")
            else:
                # Restore — removal would disconnect the graph
                adjacency[a].add(b)
                adjacency[b].add(a)

        return removed_count

    def _filter_by_angular_separation(
        self, source_gvid: int, targets: List[Tuple[int, float]],
        guaranteed_count: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Filter targets to ensure minimum angular separation.
        Keeps the closest target in each angular sector (XZ plane).
        Input must be sorted by distance (closest first).

        The first `guaranteed_count` candidates are always accepted
        (their angles are still recorded for filtering subsequent ones).
        This prevents nearby vertices from being filtered out in favour
        of distant ones in a different direction.
        """
        if len(targets) <= 1:
            return targets

        source_pos = self.vertices[source_gvid].global_point
        min_sep_rad = math.radians(self.min_angular_separation)

        def calc_angle(target_gvid: int) -> float:
            target_pos = self.vertices[target_gvid].global_point
            dx = target_pos[0] - source_pos[0]
            dz = target_pos[2] - source_pos[2]
            return math.atan2(dz, dx)

        accepted = []
        accepted_angles = []

        for i, (target_gvid, dist) in enumerate(targets):
            angle = calc_angle(target_gvid)

            if i < guaranteed_count:
                # Always accept the closest candidates
                accepted.append((target_gvid, dist))
                accepted_angles.append(angle)
                continue

            too_close = False
            for accepted_angle in accepted_angles:
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

    # =========================================================================
    # Core algorithm
    # =========================================================================

    def _build_edge_network(self, level_vertices: List[int], level_id: int,
                            level_name: str, level_ai,
                            seed_gvid: int, errors: List[str]) -> int:
        """
        Build walkability-validated edges for orphan vertices.

        Only adds edges for vertices that lack intra-level connections.
        Existing edges from pre-extracted files are preserved.

        Phase 1: Find nearby candidates, validate walkability via level.ai
        Phase 2: Select edges (angular separation + edge count cap)
        Phase 3: Verify full connectivity, force-connect remaining gaps
        Phase 4: Remove crossing edges (preserving connectivity)
        """
        connections_made = 0
        level_vertex_set = set(level_vertices)

        # Find vertices without intra-level edges
        vertices_needing_edges = []
        for gvid in level_vertices:
            has_intra = any(
                edge.vertex_id in level_vertex_set
                for edge in self.vertices[gvid].edges
            )
            if not has_intra:
                vertices_needing_edges.append(gvid)

        if not vertices_needing_edges:
            log(f"    All vertices already have intra-level edges")
            return 0

        # Phase 1: Find walkable neighbours for orphan vertices
        log(f"    Phase 1: Validating walkability for {len(vertices_needing_edges)} vertices...")
        candidates_by_vertex: Dict[int, List[Tuple[int, float]]] = {}

        for source_gvid in vertices_needing_edges:
            source_pos = self.vertices[source_gvid].global_point
            source_lvid = self.vertices[source_gvid].level_vertex_id

            # Find nearest N level vertices by euclidean distance
            all_distances = []
            for target_gvid in level_vertices:
                if target_gvid == source_gvid:
                    continue
                if self._has_edge(source_gvid, target_gvid):
                    continue
                target_pos = self.vertices[target_gvid].global_point
                dist = self._euclidean_distance(source_pos, target_pos)
                all_distances.append((target_gvid, dist))

            all_distances.sort(key=lambda x: x[1])
            nearest = all_distances[:self.max_neighbours]

            # Batch walkability validation via level.ai (one BFS per orphan)
            target_lvids = [self.vertices[tg].level_vertex_id for tg, _ in nearest]
            lvid_to_gvid = {
                self.vertices[tg].level_vertex_id: (tg, ed)
                for tg, ed in nearest
            }

            try:
                path_results = level_ai.bfs_path_distances_to_targets(
                    source_lvid, target_lvids
                )
            except Exception:
                path_results = {}

            valid = []
            for target_lvid, (path_dist, _) in path_results.items():
                target_gvid, euclidean_dist = lvid_to_gvid[target_lvid]
                # Use Manhattan distance in XZ for complexity to avoid
                # penalizing diagonal paths on the 4-connected grid
                target_pos = self.vertices[target_gvid].global_point
                dx = abs(target_pos[0] - source_pos[0])
                dz = abs(target_pos[2] - source_pos[2])
                manhattan_xz = dx + dz
                complexity = path_dist / max(manhattan_xz, 0.001)
                if complexity > self.complexity_threshold:
                    continue
                valid.append((target_gvid, path_dist))

            if valid:
                candidates_by_vertex[source_gvid] = valid

        # Phase 2: Select edges — angular separation + edge count cap
        log(f"    Phase 2: Selecting edges...")
        edges_to_add: Set[Tuple[int, int]] = set()
        edge_distances: Dict[Tuple[int, int], float] = {}

        for source_gvid, candidates in candidates_by_vertex.items():
            # Sort by distance, apply angular filter, cap count
            candidates.sort(key=lambda x: x[1])
            filtered = self._filter_by_angular_separation(source_gvid, candidates)
            selected = filtered[:self.max_edges_per_vertex]

            for target_gvid, dist in selected:
                pair = (min(source_gvid, target_gvid), max(source_gvid, target_gvid))
                if pair not in edges_to_add:
                    edges_to_add.add(pair)
                    edge_distances[pair] = dist

        for (src, tgt) in edges_to_add:
            dist = edge_distances[(src, tgt)]
            if self._add_bidi_edge(src, tgt, dist):
                connections_made += 1
                logDebug(f"    Edge {src} <-> {tgt} ({dist:.1f}m)")

        log(f"    Phase 2: {connections_made} edges added")

        # Phase 3: Verify full connectivity
        adjacency = self._build_adjacency_map(level_vertices, level_id)
        reachable = self._bfs_reachable(seed_gvid, adjacency, level_vertex_set)
        unreachable = level_vertex_set - reachable

        if unreachable:
            log(f"    Phase 3: {len(unreachable)} vertices still unreachable, force-connecting...")
            forced = self._force_connect_components(
                reachable, unreachable, adjacency, level_name
            )
            connections_made += forced

        # Phase 4: Remove crossing edges (runs after full connectivity established)
        crossings_removed = self._remove_crossing_edges(
            level_vertices, level_id, seed_gvid
        )
        if crossings_removed:
            connections_made -= crossings_removed

        log(f"    Total connections: {connections_made}")
        return connections_made

    def _force_connect_components(self, reachable: Set[int], unreachable: Set[int],
                                  adjacency: Dict[int, Set[int]],
                                  level_name: str) -> int:
        """
        Force-connect remaining disconnected components to the reachable set.

        Greedy nearest-neighbour with component absorption.
        """
        connections_made = 0

        while unreachable:
            best_unreach = None
            best_reach = None
            best_dist = float('inf')

            for u_gvid in unreachable:
                u_pos = self.vertices[u_gvid].global_point
                for r_gvid in reachable:
                    r_pos = self.vertices[r_gvid].global_point
                    dist = self._euclidean_distance(u_pos, r_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_unreach = u_gvid
                        best_reach = r_gvid

            if best_unreach is None:
                break

            if self._add_bidi_edge(best_unreach, best_reach, best_dist):
                connections_made += 1
                adjacency[best_unreach].add(best_reach)
                adjacency[best_reach].add(best_unreach)
                log(f"    Forced connection: {best_unreach} <-> {best_reach} ({best_dist:.1f}m)")

            # Absorb entire component via BFS
            queue = deque([best_unreach])
            unreachable.discard(best_unreach)
            reachable.add(best_unreach)

            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, set()):
                    if neighbor in unreachable:
                        unreachable.discard(neighbor)
                        reachable.add(neighbor)
                        queue.append(neighbor)

        return connections_made
