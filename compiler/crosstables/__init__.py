"""
Cross Tables Package

Handles building, remapping, and dependency tracking for cross tables
that map level-local vertex IDs to global game graph vertex IDs.

Submodules:
- data_types: GraphPoint and CrossTableCell dataclasses
- level_graph_navigator: LevelGraphNavigator class for level.ai navigation
- graph_point_parser: Graph point extraction from spawn files
- builder: CrossTableBuilder and build_cross_table_for_level
- cross_table_remapper: CrossTableRemapper for GVID remapping
- cross_table_deps: DependencyTracker for build caching
"""

from .cross_table_remapper import CrossTableRemapper, RemappedCrossTable
from .cross_table_deps import DependencyTracker

# Main builder API
from .builder import (
    CrossTableBuilder,
    build_cross_table_for_level,
    write_cross_table_gct,
)

# Data types
from .data_types import GraphPoint, CrossTableCell

# Navigation
from .level_graph_navigator import LevelGraphNavigator

# Graph point extraction
from .graph_point_parser import (
    extract_graph_points_from_binary,
    parse_graph_point_packet,
    extract_and_merge_graph_points,
)

__all__ = [
    # Remapper
    'CrossTableRemapper',
    'RemappedCrossTable',
    # Dependencies
    'DependencyTracker',
    # Builder
    'CrossTableBuilder',
    'build_cross_table_for_level',
    'write_cross_table_gct',
    # Data types
    'GraphPoint',
    'CrossTableCell',
    # Navigation
    'LevelGraphNavigator',
    # Extraction
    'extract_graph_points_from_binary',
    'parse_graph_point_packet',
    'extract_and_merge_graph_points',
]
