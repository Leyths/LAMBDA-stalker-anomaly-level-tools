#!/usr/bin/env python3
"""
Death Point Generator

Generates death points using 10% sampling algorithm.
Based on xr_graph_merge.cpp vfGenerateDeathPoints()
"""

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from game_graph_merger import CrossTableReader, LevelGraphReader, DeathPoint


class DeathPointGenerator:
    """
    Generate death points using 10% sampling algorithm

    From xr_graph_merge.cpp vfGenerateDeathPoints()
    """

    def __init__(self, cross_table: 'CrossTableReader', level_graph: 'LevelGraphReader'):
        self.cross_table = cross_table
        self.level_graph = level_graph

    def generate_for_vertex(self, game_vertex_id: int) -> List['DeathPoint']:
        """
        Generate death points for a single game vertex

        Algorithm (from vfGenerateDeathPoints):
        1. Find all level vertices mapping to this game vertex
        2. Shuffle them randomly
        3. Take 10% (or all if <=10, max 255)
        4. Create DeathPoint for each selected vertex

        Args:
            game_vertex_id: Game vertex ID (local to this level)

        Returns:
            List of death points
        """
        # Import here to avoid circular dependency
        from game_graph_merger import DeathPoint

        # Step 1: Find all candidate level vertices
        candidates = self.cross_table.find_level_vertices_for_game_vertex(game_vertex_id)

        if len(candidates) == 0:
            # No vertices map to this game vertex
            return []

        # Step 2: Shuffle for randomness
        candidates = candidates.copy()  # Don't modify original
        np.random.shuffle(candidates)

        # Step 3: Select subset
        # If > 10 vertices: take 10% (max 255 due to u8 storage)
        # If <= 10 vertices: take all
        if len(candidates) > 10:
            count = min(int(0.1 * len(candidates)), 255)
        else:
            count = len(candidates)

        selected = candidates[:count]

        # Step 4: Create death points
        death_points = []
        for level_vertex_id in selected:
            position = self.level_graph.get_vertex_position(int(level_vertex_id))
            distance = self.cross_table.get_distance(int(level_vertex_id))

            dp = DeathPoint(
                position=position,
                level_vertex_id=int(level_vertex_id),
                distance=distance
            )
            death_points.append(dp)

        return death_points
