#!/usr/bin/env python3
"""
Start Location Remapper

Updates LVID and GVID values in new_game_start_locations.ltx based on position coordinates.
Uses the GameGraph for spatial lookup to determine correct vertex IDs.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING

from utils import log, logWarning, logError

if TYPE_CHECKING:
    from graph import GameGraph


def _parse_start_locations_file(source_path: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[str]]:
    """
    Parse new_game_start_locations.ltx file.

    Returns:
        Tuple of:
        - location_to_level: mapping of location_name -> level_name
        - location_data: mapping of location_name -> {key: value} for coordinate sections
        - lines: original file lines for preserving structure
    """
    location_to_level: Dict[str, str] = {}
    location_data: Dict[str, Dict[str, str]] = {}
    lines: List[str] = []

    current_section = None
    is_index_section = False

    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith(';'):
            continue

        # Check for section header
        section_match = re.match(r'^\[([^\]]+)\]$', stripped)
        if section_match:
            current_section = section_match.group(1)
            # Index sections end with _start_locations
            is_index_section = current_section.endswith('_start_locations')
            if not is_index_section:
                # Initialize data dict for location section
                location_data[current_section] = {}
            continue

        # Skip if no section yet
        if current_section is None:
            continue

        # Parse key = value pairs
        # Handle both "key = value" and "key\t=\tvalue" formats
        kv_match = re.match(r'^([^\s=]+)\s*=\s*(.*)$', stripped)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()

            if is_index_section:
                # Index section: location_name = level_name,boolean
                # Extract just the level name (before comma if present)
                level_name = value.split(',')[0].strip()
                if level_name:  # Only add if there's a level name
                    location_to_level[key] = level_name
            else:
                # Location section: store all key-value pairs
                location_data[current_section][key] = value

    return location_to_level, location_data, lines


def _extract_position(data: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    """
    Extract position from location data.

    Args:
        data: Dictionary containing x, y, z keys

    Returns:
        (x, y, z) tuple or None if missing coordinates
    """
    try:
        # Handle values that may have comments (e.g., "142.41\t;\t248.76")
        def parse_coord(val: str) -> float:
            # Take first value if there's a semicolon (comment separator)
            val = val.split(';')[0].strip()
            return float(val)

        x = parse_coord(data.get('x', ''))
        y = parse_coord(data.get('y', ''))
        z = parse_coord(data.get('z', ''))
        return (x, y, z)
    except (ValueError, KeyError):
        return None


def remap_start_locations(source_path: Path, dest_path: Path, game_graph: 'GameGraph'):
    """
    Remap start locations file with updated LVID and GVID values.

    Args:
        source_path: Path to source new_game_start_locations.ltx
        dest_path: Path to write remapped file
        game_graph: GameGraph for position-based lookups
    """
    log(f"  Source: {source_path}")
    log(f"  Destination: {dest_path}")

    # Parse the file
    location_to_level, location_data, original_lines = _parse_start_locations_file(source_path)

    log(f"  Found {len(location_to_level)} location->level mappings")
    log(f"  Found {len(location_data)} location sections to remap")

    # Build remapped values for each location
    remapped_values: Dict[str, Dict[str, str]] = {}
    remapped_count = 0
    error_count = 0

    for location_name, data in location_data.items():
        # Get level name for this location
        level_name = location_to_level.get(location_name)
        if not level_name:
            logWarning(f"  Start location remapper: location '{location_name}' not found in any index section, keeping original values")
            remapped_values[location_name] = data.copy()
            continue

        # Extract position
        position = _extract_position(data)
        if not position:
            logWarning(f"  Start location remapper: location '{location_name}' has no valid position, keeping original values")
            remapped_values[location_name] = data.copy()
            continue

        # Look up new LVID
        new_lvid = game_graph.get_level_vertex_for_position(level_name, position)
        if new_lvid is None:
            logError(f"  Start location remapper: location '{location_name}': cannot resolve position {position} on '{level_name}' to LVID")
            remapped_values[location_name] = data.copy()
            error_count += 1
            continue

        # Look up new GVID
        new_gvid = game_graph.get_gvid_for_position(level_name, position)
        if new_gvid is None:
            logError(f"  Start location remapper: location '{location_name}': cannot resolve position {position} on '{level_name}' to GVID")
            remapped_values[location_name] = data.copy()
            error_count += 1
            continue

        # Store remapped values
        remapped_values[location_name] = data.copy()
        remapped_values[location_name]['lvid'] = str(new_lvid)
        remapped_values[location_name]['gvid'] = str(new_gvid)
        remapped_count += 1

    log(f"  Remapped {remapped_count} locations")
    if error_count > 0:
        logWarning(f"  Start location remapper: Failed to remap {error_count} locations")

    # Write output file, preserving structure
    _write_remapped_file(original_lines, remapped_values, dest_path)


def _write_remapped_file(original_lines: List[str], remapped_values: Dict[str, Dict[str, str]], dest_path: Path):
    """
    Write remapped file preserving original structure and comments.

    Args:
        original_lines: Original file lines
        remapped_values: Remapped values for each location section
        dest_path: Output path
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: List[str] = []
    current_section = None
    is_index_section = False

    for line in original_lines:
        stripped = line.strip()

        # Preserve empty lines and comments as-is
        if not stripped or stripped.startswith(';'):
            output_lines.append(line)
            continue

        # Check for section header
        section_match = re.match(r'^\[([^\]]+)\]$', stripped)
        if section_match:
            current_section = section_match.group(1)
            is_index_section = current_section.endswith('_start_locations')
            output_lines.append(line)
            continue

        # For index sections, preserve as-is
        if is_index_section:
            output_lines.append(line)
            continue

        # For location sections, check if we need to remap lvid/gvid
        if current_section and current_section in remapped_values:
            kv_match = re.match(r'^([^\s=]+)(\s*=\s*)(.*)$', stripped)
            if kv_match:
                key = kv_match.group(1).strip()

                # Only remap lvid and gvid keys
                if key in ('lvid', 'gvid'):
                    new_value = remapped_values[current_section].get(key)
                    if new_value is not None:
                        # Preserve the original formatting style
                        # Detect if original used tabs
                        if '\t' in line:
                            output_lines.append(f"{key}\t=\t{new_value}\n")
                        else:
                            output_lines.append(f"{key} = {new_value}\n")
                        continue

        # Preserve line as-is
        output_lines.append(line)

    with open(dest_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
