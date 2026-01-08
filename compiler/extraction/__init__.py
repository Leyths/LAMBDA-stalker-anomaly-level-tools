"""
Extraction Package

Handles read-only extraction of game data from level files.
Extraction is Phase 1 of the build pipeline - no remapping occurs here.
"""

from .spawn_entity_extractor import (
    load_blacklist,
    is_blacklisted,
    extract_entity_name,
    extract_section_name,
    load_entities_from_spawn_file,
    collect_level_entities,
)
