"""
RF Stash Exporter

During the build, extracts positions and metadata from 'tb_rf_stash_*'
space_restrictor entities found in level.spawn, and writes them back to
TB_RF_Receiver_Packages.script (build output only — mods/ source is untouched).

These entities are created by the editor import script
(editorscripts/import_rf_stash_spawns.py) and should NOT be included
in the final all.spawn — they exist only for SDK visualization.

The exporter only regenerates the tb_package_coords table. All code before
and after it (header, functions, loot tables, etc.) is preserved verbatim.

IMPORTANT: write_config() runs AFTER the tag rewriter / mod copier, so
it reads the already-rewritten gamedata/ file as its base and writes
fully-resolved values (no {w%...} or {r%...} tags).
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils import log, logError


class RFStashExporter:
    """
    Collects tb_rf_stash_* entity positions during the build and
    writes an updated TB_RF_Receiver_Packages.script file.
    """

    def __init__(self, source_config_path: Path = None,
                 game_graph=None, output_dir: Path = None):
        self.game_graph = game_graph

        # Collected data: level_name -> list of (x, y, z, direction, level_short)
        self._level_stashes: Dict[str, List[Tuple[float, float, float, str, str]]] = {}

        if not output_dir:
            raise ValueError("output_dir is required for RFStashExporter")

        # Output path (user-configured gamedata/ build output, NOT mods/ source)
        self._config_path = output_dir / "scripts" / "TB_RF_Receiver_Packages.script"

        # Source config path — kept for reference but not needed at init time
        self._source_config_path = source_config_path

        if not source_config_path or not source_config_path.exists():
            logError(f"RF stash source config not found: {source_config_path}")

    def collect_entity(self, level_name: str, entity_name: str,
                       position: Tuple[float, float, float],
                       custom_data: str):
        """
        Called during build for each tb_rf_stash_* entity found.

        Args:
            level_name: Level the entity was found in
            entity_name: Full entity name (e.g. 'tb_rf_stash_k00_marsh_03')
            position: (x, y, z) position tuple
            custom_data: Custom data string containing direction and level_short
        """
        direction, level_short = self._parse_custom_data(custom_data)

        if level_name not in self._level_stashes:
            self._level_stashes[level_name] = []

        self._level_stashes[level_name].append((
            position[0], position[1], position[2],
            direction, level_short
        ))

    def _parse_custom_data(self, custom_data: str) -> Tuple[str, str]:
        """Parse direction and level_short from custom_data string."""
        direction = "tb_loc_w_r"
        level_short = "tb_loc_dbg"

        if not custom_data:
            return direction, level_short

        normalized = custom_data.replace('\\n', '\n')
        for line in normalized.split('\n'):
            line = line.strip()
            if line.startswith('direction') and '=' in line:
                _, value = line.split('=', 1)
                direction = value.strip()
            elif line.startswith('level_short') and '=' in line:
                _, value = line.split('=', 1)
                level_short = value.strip()

        return direction, level_short

    def write_config(self):
        """
        Write updated script file with collected positions.

        Reads the tag-rewritten gamedata/ file (post mod copier) as the base,
        then replaces level sections for collected levels with entity data.
        Writes fully-resolved values (no tags) since the tag rewriter has
        already run.
        """
        if not self._level_stashes:
            return

        # Read the tag-rewritten file that the mod copier already wrote
        if not self._config_path.exists():
            logError(f"RF stash: Cannot find tag-rewritten file at {self._config_path}")
            return

        content = self._config_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')

        # Find tb_package_coords table boundaries
        table_start = None
        for i, line in enumerate(lines):
            if re.match(r'^tb_package_coords\s*=\s*\{', line.strip()):
                table_start = i
                break

        if table_start is None:
            logError("RF stash: Could not find tb_package_coords table in tag-rewritten file")
            return

        table_end = None
        for i in range(table_start + 1, len(lines)):
            if lines[i].strip() == '}':
                table_end = i
                break

        if table_end is None:
            logError("RF stash: Could not find closing } for tb_package_coords table")
            return

        # Extract existing level data from the tag-rewritten file
        existing_level_data = self._load_level_data_from_resolved(lines, table_start, table_end)

        # Extract level order from the tag-rewritten file
        # (tag-rewritten file has {r%level:X} stripped, so we use ["level"] = { markers)
        resolved_level_order = []
        level_key_re = re.compile(r'^\s*\["([^"]+)"\]\s*=\s*\{')
        for i in range(table_start + 1, table_end):
            m = level_key_re.match(lines[i])
            if m:
                resolved_level_order.append(m.group(1))

        # Determine final level order — use resolved file's order, add new at end
        final_levels = list(resolved_level_order)
        for level in sorted(self._level_stashes.keys()):
            if level not in final_levels:
                final_levels.append(level)

        # Build the tb_package_coords table body
        table_lines = []
        for level in final_levels:
            table_lines.append(f'\t["{level}"] = {{')

            if level in self._level_stashes:
                # Write updated positions with resolved values (no tags)
                stashes = self._level_stashes[level]
                for i, (x, y, z, direction, level_short) in enumerate(stashes):
                    lvid = self._resolve_lvid(level, x, y, z)
                    gvid = self._resolve_gvid(level, x, y, z)
                    trailing = "," if i < len(stashes) - 1 else ""
                    table_lines.append(
                        f'\t\t{{{x}, {y}, {z}, '
                        f'{lvid}, {gvid}, '
                        f'"{direction}", "{level_short}"}}{trailing}'
                    )
            elif level in existing_level_data:
                # Preserve existing resolved data from tag-rewritten file
                for raw_line in existing_level_data[level]:
                    table_lines.append(raw_line)

            table_lines.append('\t},')

        # Assemble: header + new table body + trailing
        header = '\n'.join(lines[:table_start + 1])
        trailing = '\n'.join(lines[table_end:])
        output = header + '\n' + '\n'.join(table_lines) + '\n' + trailing

        # Write file
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(output, encoding='utf-8')

        # Log summary
        total = sum(len(stashes) for stashes in self._level_stashes.values())
        log(f"  Wrote {total} RF stash positions to {self._config_path}")
        for level in sorted(self._level_stashes.keys()):
            log(f"    {level}: {len(self._level_stashes[level])} stashes")

    def _load_level_data_from_resolved(self, lines: List[str],
                                        table_start: int, table_end: int) -> Dict[str, List[str]]:
        """Load per-level stash data from the tag-rewritten (resolved) file."""
        data: Dict[str, List[str]] = {}
        current_level = None

        level_key_re = re.compile(r'^\s*\["([^"]+)"\]\s*=\s*\{')

        for i in range(table_start + 1, table_end):
            line = lines[i]
            stripped = line.strip()

            # Level section start: ["level_name"] = {
            m = level_key_re.match(stripped)
            if m:
                current_level = m.group(1)
                continue

            # Level section end: },
            if stripped == '},':
                current_level = None
                continue

            # Stash entry line (resolved: no tags, just values)
            if current_level and stripped.startswith('{') and ',' in stripped:
                if current_level not in data:
                    data[current_level] = []
                data[current_level].append(line.rstrip())

        return data

    def _resolve_lvid(self, level_name: str, x: float, y: float, z: float) -> int:
        """Resolve LVID from position using game graph."""
        if self.game_graph:
            lvid = self.game_graph.get_level_vertex_for_position(
                level_name, (x, y, z))
            if lvid is not None:
                return lvid
        return 0

    def _resolve_gvid(self, level_name: str, x: float, y: float, z: float) -> int:
        """Resolve GVID from position using game graph."""
        if self.game_graph:
            gvid = self.game_graph.get_gvid_for_position(
                level_name, (x, y, z))
            if gvid is not None:
                return gvid
        return 0

    @property
    def collected_count(self) -> int:
        """Total number of collected entities across all levels."""
        return sum(len(stashes) for stashes in self._level_stashes.values())
