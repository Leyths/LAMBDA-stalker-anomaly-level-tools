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
    python build_all_spawn.py --project-root ..
"""

import sys
import argparse
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional
import time

from levels import LevelsConfig, LevelConfig
from graph import GameGraph
from project_paths import ProjectPaths
from utils import log, logWarning, logError, init_logging, print_summary
from config import ModConfig, ModCopier


class GameGraphBuilder:
    """
    Orchestrates the full game graph build process
    """

    def __init__(self, config: LevelsConfig, paths: ProjectPaths, base_mod: str = "anomaly"):
        """
        Initialize builder

        Args:
            config: Levels configuration
            paths: Resolved project paths
            base_mod: Base mod name (anomaly, gamma)
        """
        self.config = config
        self.paths = paths
        self.build_dir = paths.build_dir
        self.base_mod = base_mod

        # Output path
        self.output_path = paths.output_spawn

        # Blacklist path
        self.blacklist_path = paths.get_blacklist_ini()

        # Load mod configuration from {basemod}.ini
        mod_config_path = paths.get_base_mod_ini(base_mod)
        self.mod_config = None
        if mod_config_path.exists():
            self.mod_config = ModConfig(mod_config_path)
            log(f"Mod config: {mod_config_path}")
            log(f"  Enabled mods: {self.mod_config.get_enabled_mods()}")
        else:
            logWarning(f"Mod config not found: {mod_config_path}")

        # Initialize mod copier
        self.mod_copier = ModCopier(paths.mods_dir, paths.output_dir)

        # Initialize dependency tracker
        from crosstables import DependencyTracker
        self.dep_tracker = DependencyTracker(self.build_dir)

        log(f"Build directory: {self.build_dir}")
        if self.blacklist_path:
            log(f"Blacklist: {self.blacklist_path}")
        log()

    def build_all(self, force_rebuild: bool = False, deploy_only: bool = False):
        """
        Build complete game graph

        Args:
            force_rebuild: Force rebuild of all cross tables
            deploy_only: Skip build, only deploy existing output
        """
        if deploy_only:
            self._deploy_only()
            return
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
        log(f"Output directory: {self.build_dir}")
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
        log(f"Output directory: {self.build_dir}")
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

        deferred_exporters = self._write_game_graph(game_graph, self.output_path)

        # Step 5: Copy mod variant files (with tag rewriting)
        log("\n" + "=" * 70)
        log("STEP 5: Copying Mod Variant Files")
        log("=" * 70)
        self._copy_mod_variant_files(game_graph)

        # Step 5b: Run deferred exporters (must happen AFTER tag rewriter
        # so their output isn't overwritten by mod file copies)
        if deferred_exporters:
            log("\n" + "=" * 70)
            log("STEP 5b: Writing Deferred Exporter Configs")
            log("=" * 70)
            for name, (exporter, count) in deferred_exporters.items():
                log(f"  {name}: writing config ({count} entities)")
                exporter.write_config()

        # Step 6: Copy modified level files to gamedata
        log("\n" + "=" * 70)
        log("STEP 6: Checking Level Files For Changes")
        log("=" * 70)
        self._copy_modified_level_files()

        elapsed = time.time() - start_time
        log("\n" + "=" * 70)
        log(f"BUILD COMPLETE in {elapsed:.1f} seconds")
        log(f"All files needed are in {self.paths.output_dir}")
        log("=" * 70)

        # Print level file source table (only when overrides are active)
        self.config.print_override_summary()

        # Print warning/error summary
        print_summary()

    def _deploy_only(self):
        """
        Fast deploy: copy mod files to gamedata without rebuilding spawn.
        Files requiring tag rewriting (rewrite_files) are skipped.
        """
        log("=" * 70)
        log("FAST DEPLOY (mod files only, no spawn rebuild)")
        log("=" * 70)
        log()

        start_time = time.time()

        if not self.mod_config:
            log("No mod configuration loaded, nothing to deploy.")
            return

        log(f"  Mods directory: {self.paths.mods_dir}")
        log(f"  Destination: {self.paths.output_dir}")
        log()

        # Copy mod files, skipping rewrite_files (they need a full build)
        mod_copier = ModCopier(self.paths.mods_dir, self.paths.output_dir)
        copied_count = mod_copier.copy_all_enabled_mods(self.mod_config, skip_rewrite=True)

        log(f"\n  Total files deployed: {copied_count}")

        # Also check modified level files (cheap)
        log("\n" + "=" * 70)
        log("Checking Level Files For Changes")
        log("=" * 70)
        self._copy_modified_level_files()

        elapsed = time.time() - start_time
        log("\n" + "=" * 70)
        log(f"FAST DEPLOY COMPLETE in {elapsed:.1f} seconds")
        log(f"All files needed are in {self.paths.output_dir}")
        log("=" * 70)

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
        # level.path is absolute
        level_ai = level.path / "level.ai"
        level_spawn = level.path / "level.spawn"

        # Original spawn path (if configured) — already absolute
        original_spawn = level.original_spawn

        # Output path
        cross_table = self.build_dir / f"{level.name}.gct"

        # Check if rebuild needed using dependency tracker
        if not force:
            needs_rebuild, reason = self.dep_tracker.needs_rebuild(
                level.name,
                level_ai,
                level_spawn,
                cross_table,
                original_spawn=original_spawn,
                original_only=level.base_anomaly_spawns_only
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
        self._build_cross_table(level_ai, level_spawn, cross_table, original_spawn,
                                original_only=level.base_anomaly_spawns_only)

        # Update dependencies
        self.dep_tracker.update(
            level.name,
            level_ai,
            level_spawn,
            cross_table,
            original_spawn=original_spawn,
            original_only=level.base_anomaly_spawns_only
        )

        return cross_table

    def _build_cross_table(self, level_ai: Path, level_spawn: Path, output: Path,
                           original_spawn: Optional[Path] = None,
                           original_only: bool = False):
        """Build cross table from binary spawn files"""
        from crosstables import build_cross_table_for_level

        # Call the builder directly instead of via subprocess
        success = build_cross_table_for_level(
            level_ai_path=level_ai,
            level_spawn_path=level_spawn,
            output_path=output,
            original_spawn_path=original_spawn if original_spawn and original_spawn.exists() else None,
            original_only=original_only
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
            # level_config.path is absolute
            level_spawn_path = level_config.path / "level.spawn"

            # original_spawn is already absolute (or None)
            original_spawn_path = level_config.original_spawn

            graph_points_list = extract_and_merge_graph_points(
                level_spawn_path, original_spawn_path,
                original_only=level_config.base_anomaly_spawns_only
            )

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

        # Set cross_table_dir for caching
        game_graph.cross_table_dir = self.build_dir

        return game_graph

    def _write_game_graph(self, game_graph: GameGraph, output_path: Path):
        """
        Write merged graph to all.spawn file

        Args:
            game_graph: GameGraph object containing merged data
            output_path: Output file path

        Returns:
            Dict of deferred exporters that must run after the mod copier/tag rewriter
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
            level_spawn = level.path / "level.spawn"
            if level_spawn.exists():
                level_spawn_paths.append(level_spawn)

        # Build complete all.spawn with per-level original spawn merging and blacklist
        deferred_exporters = build_all_spawn(
            game_graph_data=game_graph_data,
            game_graph_guid=game_graph_guid,
            level_spawn_paths=level_spawn_paths,
            level_count=len(self.config.levels),
            output_path=output_path,
            level_configs=self.config.levels,
            paths=self.paths,
            game_graph=game_graph,
        )

        log(f"\nAll.spawn written to: {output_path}")

        return deferred_exporters or {}

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
            # level.path is absolute
            level_ai_path = level.path / "level.ai"

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

        log(f"  Mods directory: {self.paths.mods_dir}")
        log(f"  Destination: {self.paths.output_dir}")

        # Create ModCopier with game_graph for tag rewriting
        mod_copier = ModCopier(self.paths.mods_dir, self.paths.output_dir, game_graph)

        # Copy all enabled mods (files in rewrite_files will be processed by TagRewriter)
        copied_count = mod_copier.copy_all_enabled_mods(self.mod_config)

        log(f"  Total files processed: {copied_count}")

    def _copy_modified_level_files(self):
        """
        Compare level.ai, level.spawn, and level.game against vanilla hashes.
        If a file differs from vanilla, copy it to gamedata/levels/<name>/.
        """
        copied = 0
        checked = 0

        for level in self.config.levels:
            # level.path is absolute
            level_dir = level.path

            files_to_check = [
                ("level.ai", level.vanilla_hash_ai),
                ("level.spawn", level.vanilla_hash_spawn),
                ("level.game", level.vanilla_hash_game),
            ]

            for filename, vanilla_hash in files_to_check:
                if not vanilla_hash:
                    continue

                filepath = level_dir / filename
                if not filepath.exists():
                    continue

                checked += 1
                current_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:16]

                if current_hash != vanilla_hash:
                    dest = self.paths.output_dir / "levels" / level.name / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(filepath, dest)
                    copied += 1
                    log(f"  {level.name}/{filename} modified -> copied to gamedata")

        if copied == 0:
            log(f"  No modified level files detected ({checked} files checked)")
        else:
            log(f"  Copied {copied} modified file(s) to gamedata/levels/ ({checked} checked)")


def main():
    parser = argparse.ArgumentParser(
        description='Build all.spawn from all levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_all_spawn.py --project-root ..

    # Force rebuild all cross tables:
    python build_all_spawn.py --project-root .. --force

    # Custom output directory:
    python build_all_spawn.py --project-root .. --output-dir /path/to/gamedata
        """
    )

    parser.add_argument('--project-root', default='..',
                        help='Project root directory (default: ..)')
    parser.add_argument('--levels-dir', default=None,
                        help='Override levels directory')
    parser.add_argument('--levels-override-dir', default=None,
                        help='Directory with partial level file overrides (per-file fallback to levels-dir)')
    parser.add_argument('--output-dir', default=None,
                        help='Override output gamedata directory')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild of all cross tables')
    parser.add_argument('--deploy-only', action='store_true',
                        help='Skip build, only deploy existing output')
    parser.add_argument('--basemod', default='anomaly',
                        help='The base mod you are targeting for this build')
    args = parser.parse_args()

    # Build ProjectPaths
    paths = ProjectPaths.from_root(
        project_root=Path(args.project_root),
        levels_dir=Path(args.levels_dir) if args.levels_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        levels_override_dir=Path(args.levels_override_dir) if args.levels_override_dir else None,
    )

    # Initialize logging
    init_logging(log_path=paths.output_dir / "build.log")

    try:
        # Load configuration
        config = LevelsConfig(
            config_path=str(paths.get_levels_ini()),
            levels_dir=paths.levels_dir,
            cross_table_dir=paths.build_dir,
            resolve_root=paths.compiler_dir,
            levels_override_dir=paths.levels_override_dir,
            build_dir=paths.build_dir,
        )

        # Build
        builder = GameGraphBuilder(config, paths, base_mod=args.basemod)
        builder.build_all(force_rebuild=args.force, deploy_only=args.deploy_only)

    except Exception as e:
        logError(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
