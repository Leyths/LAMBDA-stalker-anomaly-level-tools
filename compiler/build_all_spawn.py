#!/usr/bin/env python3
"""
Build All Spawn

Master build script for compiling all.spawn from all levels.

Pipeline:
1. Load levels configuration (levels.ini)
2. For each level:
   a. Parse level.spawn to JSON (if needed)
   b. Build cross table (level.gct)
3. Merge all level graphs into game.graph
4. Generate death points
5. Write final all.spawn

Usage:
    python build_all_spawn.py --config ../levels.ini --output ../gamedata/spawns/all.spawn
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional
import time

from levels import LevelsConfig, LevelConfig
from graph import GameGraph
from utils import log, logWarning, logError, init_logging, print_summary
from config import ModConfig, ModCopier


class GameGraphBuilder:
    """
    Orchestrates the full game graph build process
    """

    def __init__(self, config: LevelsConfig, base_path: str = ".",
                 blacklist_path: str = None, output_path: str = None, base_mod: str = "anomaly"):
        """
        Initialize builder

        Args:
            config: Levels configuration
            base_path: Base path for resolving relative paths
            blacklist_path: Path to spawn_blacklist.ini file
            output_path: Path to output all.spawn file
            base_mod: Base mod name (anomaly, gamma)
        """
        self.config = config
        self.base_path = Path(base_path)
        self.build_dir = self.base_path / ".." / ".tmp"
        self.build_dir.mkdir(exist_ok=True)
        self.base_mod = base_mod

        # Set paths on LevelsConfig for centralized path resolution
        self.config.set_paths(self.base_path, self.build_dir)

        # Output path
        self.output_path = Path(output_path) if output_path else self.base_path / ".." / "gamedata" / "spawns" / "all.spawn"

        # Blacklist path
        self.blacklist_path = Path(blacklist_path) if blacklist_path else None

        # Load mod configuration from {basemod}.ini
        mod_config_path = self.base_path / ".." / f"{base_mod}.ini"
        self.mod_config = None
        if mod_config_path.exists():
            self.mod_config = ModConfig(mod_config_path)
            log(f"Mod config: {mod_config_path}")
            log(f"  Enabled mods: {self.mod_config.get_enabled_mods()}")
        else:
            logWarning(f"Mod config not found: {mod_config_path}")

        # Initialize mod copier
        self.mods_dir = self.base_path / ".." / "mods"
        self.gamedata_dir = self.base_path / ".." / "gamedata"
        self.mod_copier = ModCopier(self.mods_dir, self.gamedata_dir)

        # Initialize dependency tracker
        from crosstables import DependencyTracker
        self.dep_tracker = DependencyTracker(self.build_dir)

        log(f"Build directory: {self.build_dir.absolute()}")
        if self.blacklist_path:
            log(f"Blacklist: {self.blacklist_path}")
        log()

    def build_all(self, force_rebuild: bool = False):
        """
        Build complete game graph

        Args:
            force_rebuild: Force rebuild of all cross tables
        """
        log("=" * 70)
        log("GAME GRAPH BUILDER")
        log("=" * 70)
        log()

        self.config.print_summary()
        log()

        start_time = time.time()

        # Step 1: Build cross tables for all levels
        log("\n" + "=" * 70)
        log("STEP 1: Building Cross Tables")
        log("=" * 70)
        log(f"Output directory: {self.build_dir.absolute()}")
        log()

        cross_table_paths = []
        successful_levels = []  # Track which levels were successfully built
        total_vertices = 0
        total_game_vertices = 0

        for i, level in enumerate(self.config.levels, 1):
            log(f"[{i}/{len(self.config.levels)}] {level.name} ({level.caption})")

            cross_table_path = self._build_level_cross_table(level, force_rebuild)

            if cross_table_path is None:
                # Level was skipped due to invalid spawn
                continue

            cross_table_paths.append(cross_table_path)
            successful_levels.append(level)  # Track successful level

            # Read stats from cross table
            stats = self._read_cross_table_stats(cross_table_path)
            if stats:
                total_vertices += stats['level_vertices']
                total_game_vertices += stats['game_vertices']
                log(f"    Level vertices: {stats['level_vertices']:,}")
                log(f"    Game vertices: {stats['game_vertices']}")

            log()

        # Print summary
        log("=" * 70)
        log("CROSS TABLE BUILD SUMMARY")
        log("=" * 70)
        log(f"Levels processed: {len(self.config.levels)}")
        log(f"Total level vertices: {total_vertices:,}")
        log(f"Total game vertices: {total_game_vertices}")
        log(f"Output directory: {self.build_dir.absolute()}")
        log()

        # Step 2: Merge game graphs
        log("\n" + "=" * 70)
        log("STEP 2: Merging Game Graphs")
        log("=" * 70)

        game_graph = self._merge_game_graphs(cross_table_paths, successful_levels)

        # Step 3: Generate death points (done in merge step)
        log("\n" + "=" * 70)
        log("STEP 3: Death Points")
        log("=" * 70)
        log("  Death points generated during merge")
        log(f"  Total: {game_graph.get_death_point_count():,}")

        # Step 4: Write output
        log("\n" + "=" * 70)
        log("STEP 4: Writing Output")
        log("=" * 70)

        self._write_game_graph(game_graph, self.output_path)

        # Step 5: Copy mod variant files (with tag rewriting)
        log("\n" + "=" * 70)
        log("STEP 5: Copying Mod Variant Files")
        log("=" * 70)
        self._copy_mod_variant_files(game_graph)

        elapsed = time.time() - start_time
        log("\n" + "=" * 70)
        log(f"BUILD COMPLETE in {elapsed:.1f} seconds")
        log(f"All files needed are in ./gamedata")
        log("=" * 70)

        # Print warning/error summary
        print_summary()

    def _build_level_cross_table(self, level: LevelConfig, force: bool) -> Path:
        """
        Build cross table for a single level

        Args:
            level: Level configuration
            force: Force rebuild even if file exists

        Returns:
            Path to generated cross table file
        """
        # Resolve paths
        level_path = self.base_path / level.path
        level_ai = level_path / "level.ai"
        level_spawn = level_path / "level.spawn"

        # Original spawn path (if configured)
        original_spawn = None
        if level.original_spawn:
            original_spawn = self.base_path / level.original_spawn

        # Output path
        cross_table = self.build_dir / f"{level.name}.gct"

        # Check if rebuild needed using dependency tracker
        if not force:
            needs_rebuild, reason = self.dep_tracker.needs_rebuild(
                level.name,
                level_ai,
                level_spawn,
                cross_table,
                original_spawn=original_spawn
            )

            if not needs_rebuild:
                log(f"    Up to date: {cross_table.name}")
                return cross_table
            else:
                log(f"    Rebuilding: {reason}")
        else:
            log(f"    Force rebuild enabled")

        # Validate inputs exist
        if not level_ai.exists():
            raise FileNotFoundError(f"Missing level.ai: {level_ai}")

        if not level_spawn.exists():
            raise FileNotFoundError(f"Missing level.spawn: {level_spawn}")

        # Build cross table directly from binary spawn files
        self._build_cross_table(level_ai, level_spawn, cross_table, original_spawn)

        # Update dependencies
        self.dep_tracker.update(
            level.name,
            level_ai,
            level_spawn,
            cross_table,
            original_spawn=original_spawn
        )

        return cross_table

    def _build_cross_table(self, level_ai: Path, level_spawn: Path, output: Path,
                           original_spawn: Optional[Path] = None):
        """Build cross table from binary spawn files"""
        from crosstables import build_cross_table_for_level

        # Call the builder directly instead of via subprocess
        success = build_cross_table_for_level(
            level_ai_path=level_ai,
            level_spawn_path=level_spawn,
            output_path=output,
            original_spawn_path=original_spawn if original_spawn and original_spawn.exists() else None
        )

        if not success:
            raise RuntimeError(f"Failed to build cross table for {output.stem}")

    def _merge_game_graphs(self, cross_table_paths: List[Path], successful_levels: List) -> GameGraph:
        """
        Merge all level game graphs into global graph

        Args:
            cross_table_paths: List of cross table paths (only for successful levels)
            successful_levels: List of LevelConfig objects that were successfully built

        Returns:
            GameGraph object containing all merged data
        """
        from game_graph_merger import GameGraphMerger
        from crosstables import extract_and_merge_graph_points
        from typing import Dict

        # Build graph_points_by_level map
        graph_points_by_level: Dict[int, List[dict]] = {}

        for level_config in successful_levels:
            level_path = self.base_path / level_config.path
            level_spawn_path = level_path / "level.spawn"

            # Use extract_and_merge_graph_points to ensure SAME ordering as cross-table builder
            # CRITICAL: The cross-table assigns local GVIDs based on this order.
            # If we use a different order here, the GVID remapping will be wrong.
            original_spawn_path = None
            if level_config.original_spawn:
                original_spawn_path = self.base_path / level_config.original_spawn

            graph_points_list = extract_and_merge_graph_points(level_spawn_path, original_spawn_path)

            # Convert GraphPoint objects to dict format expected by merger
            # IMPORTANT: Preserve the exact order from extract_and_merge_graph_points
            graph_points = []
            for gp in graph_points_list:
                gp_dict = {
                    'name_replace': gp.name,
                    'original_name': gp.name,  # Track original name for lookups
                    'position': {'x': gp.position[0], 'y': gp.position[1], 'z': gp.position[2]},
                    'level_vertex_id': gp.level_vertex_id,
                    'graph_point_data': {
                        'connection_point_name': gp.connection_point_name,
                        'connection_level_name': gp.connection_level_name,
                        'locations': list(gp.location_types) if isinstance(gp.location_types,
                                                                           bytes) else gp.location_types
                    }
                }
                graph_points.append(gp_dict)

            if not graph_points:
                logWarning(f"No graph points found for {level_config.name}")

            # Log edges file if configured
            edges_path = self.config.get_edges_path(level_config)
            if edges_path and edges_path.exists():
                log(f"    Edges file: {level_config.original_edges}")
            elif level_config.original_edges:
                logWarning(f"Edges file not found: {level_config.original_edges}")

            graph_points_by_level[level_config.id] = graph_points

        # Merge graphs using new API
        merger = GameGraphMerger(
            levels_config=self.config,
            graph_points_by_level=graph_points_by_level,
            random_seed=42,
            base_mod=self.base_mod,
            mod_config=self.mod_config
        )
        game_graph = merger.merge()

        # Set paths for cross table and level AI caching
        game_graph.set_paths(
            base_path=self.base_path,
            cross_table_dir=self.build_dir
        )

        return game_graph

    def _write_game_graph(self, game_graph: GameGraph, output_path: Path):
        """
        Write merged graph to all.spawn file

        Args:
            game_graph: GameGraph object containing merged data
            output_path: Output file path
        """
        from serialization import GameGraphSerializer, build_all_spawn
        from crosstables import CrossTableRemapper

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build game graph binary data
        log("\nBuilding game graph (chunk 4)...")

        # Populate level GUIDs on game_graph
        game_graph.level_guids = self._read_level_guids()

        # Create serializer (computes graph GUID needed for cross table remapping)
        serializer = GameGraphSerializer(game_graph)

        # Remap cross tables with the graph GUID
        remapper = CrossTableRemapper(
            levels_config=self.config,
            vertices=game_graph.vertices,
            graph_guid=serializer.guid
        )

        # Populate cross_tables on game_graph (serializer accesses this at serialize time)
        game_graph.cross_tables = remapper.remap_all()

        # Serialize (uses game_graph.cross_tables which is now populated)
        game_graph_data = serializer.serialize()
        game_graph_guid = serializer.guid

        # Collect level.spawn files (binary, not JSON)
        level_spawn_paths = []
        for level in self.config.levels:
            level_path = self.base_path / level.path
            level_spawn = level_path / "level.spawn"
            if level_spawn.exists():
                level_spawn_paths.append(level_spawn)

        # Build complete all.spawn with per-level original spawn merging and blacklist
        build_all_spawn(
            game_graph_data=game_graph_data,
            game_graph_guid=game_graph_guid,
            level_spawn_paths=level_spawn_paths,
            level_count=len(self.config.levels),
            output_path=output_path,
            level_configs=self.config.levels,
            base_path=self.base_path,
            blacklist_path=self.blacklist_path,
            game_graph=game_graph
        )

        log(f"\nAll.spawn written to: {output_path}")

    def _read_cross_table_stats(self, cross_table_path: Path) -> dict:
        """Read statistics from cross table file"""
        import struct

        try:
            with open(cross_table_path, 'rb') as f:
                # Read chunk 0 (header)
                chunk_id, chunk_size = struct.unpack('<II', f.read(8))
                if chunk_id != 0xFFFF:
                    return None

                header = f.read(chunk_size)
                version, level_verts, game_verts = struct.unpack('<III', header[:12])

                return {
                    'level_vertices': level_verts,
                    'game_vertices': game_verts,
                    'version': version
                }
        except Exception:
            return None

    def _read_level_guids(self) -> dict:
        """
        Read GUIDs from level.ai files for all levels.

        Returns:
            Dict mapping level_id -> 16-byte GUID
        """
        import struct

        level_guids = {}

        for level in self.config.levels:
            level_ai_path = self.base_path / level.path / "level.ai"

            if not level_ai_path.exists():
                logWarning(f"level.ai not found at {level_ai_path}, using zero GUID")
                level_guids[level.id] = b'\x00' * 16
                continue

            try:
                with open(level_ai_path, 'rb') as f:
                    # GUID is at offset 40 in level.ai header
                    # Header: version(4) + vertex_count(4) + cell_size(4) + cell_size_y(4)
                    #         + min(12) + max(12) = 40 bytes, then GUID(16)
                    f.seek(40)
                    guid = f.read(16)

                    if len(guid) != 16:
                        logWarning(f"Could not read GUID from {level.name}, using zero GUID")
                        level_guids[level.id] = b'\x00' * 16
                    else:
                        level_guids[level.id] = guid
            except Exception as e:
                logWarning(f"Error reading GUID from {level.name}: {e}")
                level_guids[level.id] = b'\x00' * 16

        return level_guids

    def _copy_mod_variant_files(self, game_graph: GameGraph):
        """
        Copy enabled mod files from mods/ to gamedata/ using ModCopier.
        Files listed in rewrite_files are processed by TagRewriter.

        Args:
            game_graph: GameGraph for tag rewriting (LVID/GVID lookups)
        """
        if not self.mod_config:
            log(f"  No mod configuration loaded")
            return

        log(f"  Mods directory: {self.mods_dir}")
        log(f"  Destination: {self.gamedata_dir}")

        # Create ModCopier with game_graph for tag rewriting
        mod_copier = ModCopier(self.mods_dir, self.gamedata_dir, game_graph)

        # Copy all enabled mods (files in rewrite_files will be processed by TagRewriter)
        copied_count = mod_copier.copy_all_enabled_mods(self.mod_config)

        log(f"  Total files processed: {copied_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Build all.spawn from all levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_all_spawn.py --config ../levels.ini --output ../gamedata/spawns/all.spawn

    # With blacklist:
    python build_all_spawn.py --config ../levels.ini --blacklist ../spawn_blacklist.ini

    # Force rebuild all cross tables:
    python build_all_spawn.py --config ../levels.ini --force

Note: Per-level original_spawn paths are configured in levels.ini:
    [level27]
    name = zaton
    path = levels/zaton
    id = 27
    original_spawn = tmp/oldspawnfiles/zaton.spawn
        """
    )

    parser.add_argument('--config', default='../levels.ini',
                        help='Path to levels.ini configuration file')
    parser.add_argument('--output', default='../gamedata/spawns/all.spawn',
                        help='Output path for game.graph / all.spawn')
    parser.add_argument('--base-path', default='.',
                        help='Base path for resolving relative paths')
    parser.add_argument('--blacklist', default='../spawn_blacklist.ini',
                        help='Path to spawn_blacklist.ini file')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild of all cross tables')
    parser.add_argument('--basemod', default='anomaly',
                        help='The base mod you are targeting for this build')
    args = parser.parse_args()

    # Initialize logging
    init_logging()

    try:
        # Load configuration
        config = LevelsConfig(args.config)

        # Resolve blacklist
        blacklist_path = args.blacklist if args.blacklist else None

        # Check if blacklist exists
        if blacklist_path and not Path(blacklist_path).exists():
            logWarning(f"Blacklist file not found: {blacklist_path}, continuing without blacklist")
            blacklist_path = None

        # Build
        builder = GameGraphBuilder(
            config,
            args.base_path,
            blacklist_path=blacklist_path,
            output_path=args.output,
            base_mod=args.basemod,
        )
        builder.build_all(force_rebuild=args.force)

    except Exception as e:
        logError(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()