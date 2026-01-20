#!/usr/bin/env python3
"""
Tag Rewriter

Processes files with special tags ({r%...} for read-context, {w%...} for write/rewrite)
and computes new LVID/GVID values based on position coordinates.

Tag Format Specification:
    {r%key:value} - Read context tags (stripped from output, sets context state)
    {w%key:value} - Write/rewrite tags (tag stripped, value computed or preserved)

Context Tags:
    {r%level:level_name} - Sets current level context for subsequent LVID/GVID lookups

Rewrite Tags:
    {w%lvid:default} - Level Vertex ID, computed from position using GameGraph
    {w%gvid:default} - Global Vertex ID, computed from position using GameGraph
    {w%x:value}, {w%y:value}, {w%z:value} - Position markers (value preserved, tag stripped)
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from dataclasses import dataclass, field

from utils import log, logWarning, logError

if TYPE_CHECKING:
    from graph import GameGraph


# Tag patterns
READ_TAG_PATTERN = re.compile(r'\{r%([^:}]+):([^}]*)\}')
WRITE_TAG_PATTERN = re.compile(r'\{w%([^:}]+):([^}]*)\}')
ANY_TAG_PATTERN = re.compile(r'\{[rw]%[^}]+\}')


@dataclass
class TagLocation:
    """Represents a tag location in the file with its context."""
    start: int  # Start position in file
    end: int  # End position in file
    tag_type: str  # 'r' or 'w'
    key: str  # Tag key (level, lvid, gvid, x, y, z)
    value: str  # Original value in tag
    level_context: Optional[str] = None  # Level name at this point
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) for this tag


class TagRewriter:
    """
    Processes files with {r%...} and {w%...} tags.

    Uses a two-pass approach:
    1. Parse & Collect: Scan file, collect all tags
    2. Resolve & Replace: Resolve positions for lvid/gvid, generate output
    """

    def __init__(self, game_graph: 'GameGraph'):
        """
        Initialize tag rewriter.

        Args:
            game_graph: GameGraph for position-based LVID/GVID lookups
        """
        self.game_graph = game_graph

    def rewrite_file(self, source_path: Path, dest_path: Path) -> bool:
        """
        Process a file with tags and write the result.

        Args:
            source_path: Path to source file with tags
            dest_path: Path to write processed file

        Returns:
            True if successful, False on error
        """
        try:
            content = source_path.read_text(encoding='utf-8')
        except Exception as e:
            logError(f"Tag rewriter: Cannot read {source_path}: {e}")
            return False

        # Check if file has any tags
        if not ANY_TAG_PATTERN.search(content):
            # No tags, just copy the file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(content, encoding='utf-8')
            return True

        # Pass 1: Parse and collect all tags
        tags = self._collect_tags(content)
        log(f"  Tag rewriter: {len(tags)} tags found in file")
        # Pass 2: Resolve level context and positions for lvid/gvid tags
        self._resolve_contexts(tags)

        # Pass 3: Generate output with computed values
        output = self._generate_output(content, tags)

        # Write output
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(output, encoding='utf-8')

        return True

    def _collect_tags(self, content: str) -> List[TagLocation]:
        """
        Pass 1: Scan file and collect all tags.

        Args:
            content: File content

        Returns:
            List of TagLocation objects (without context resolved yet)
        """
        tags: List[TagLocation] = []

        # Find all tags in order
        pos = 0
        while pos < len(content):
            # Find next tag
            read_match = READ_TAG_PATTERN.search(content, pos)
            write_match = WRITE_TAG_PATTERN.search(content, pos)

            # Determine which match comes first
            next_match = None
            tag_type = None

            if read_match and write_match:
                if read_match.start() <= write_match.start():
                    next_match = read_match
                    tag_type = 'r'
                else:
                    next_match = write_match
                    tag_type = 'w'
            elif read_match:
                next_match = read_match
                tag_type = 'r'
            elif write_match:
                next_match = write_match
                tag_type = 'w'
            else:
                break  # No more tags

            key = next_match.group(1).lower()
            value = next_match.group(2)

            # Record tag location
            tags.append(TagLocation(
                start=next_match.start(),
                end=next_match.end(),
                tag_type=tag_type,
                key=key,
                value=value
            ))

            pos = next_match.end()

        return tags

    def _resolve_contexts(self, tags: List[TagLocation]) -> None:
        """
        Pass 2: Resolve level context and positions for all tags.

        Optimized O(n) single-pass algorithm:
        - Track x, y, z values as we scan each level section
        - When x,y,z are complete, assign to any pending lvid/gvid tags
        - Reset position when a new x is encountered (new entry)

        Args:
            tags: List of TagLocation objects (modified in place)
        """
        if not tags:
            return

        current_level: Optional[str] = None

        # First, set level context for all tags
        for tag in tags:
            if tag.tag_type == 'r' and tag.key == 'level':
                current_level = tag.value
            tag.level_context = current_level

        # Process each level section separately
        section_start = 0
        for i, tag in enumerate(tags):
            if tag.tag_type == 'r' and tag.key == 'level':
                # Process section before this level tag (if any)
                if i > section_start:
                    self._process_section_tags(tags, section_start, i)
                section_start = i + 1

        # Process final section (from last level tag to end)
        if section_start < len(tags):
            self._process_section_tags(tags, section_start, len(tags))

    def _process_section_tags(self, tags: List[TagLocation], start: int, end: int) -> None:
        """
        Process a level section to assign positions to lvid/gvid tags.

        Algorithm:
        - Track x, y, z values as we scan forward
        - lvid/gvid tags get the current position if complete, otherwise queued
        - When position becomes complete, update all queued tags
        - New x value resets position tracking (new entry)

        Args:
            tags: List of all tags
            start: Start index of section (inclusive)
            end: End index of section (exclusive)
        """
        x: Optional[float] = None
        y: Optional[float] = None
        z: Optional[float] = None
        pending: List[TagLocation] = []

        for i in range(start, end):
            tag = tags[i]
            if tag.tag_type != 'w':
                continue

            if tag.key == 'x':
                x = self._parse_float(tag.value)
                y = z = None  # New entry, reset position
            elif tag.key == 'y':
                y = self._parse_float(tag.value)
            elif tag.key == 'z':
                z = self._parse_float(tag.value)
            elif tag.key in ('lvid', 'gvid'):
                if x is not None and y is not None and z is not None:
                    tag.position = (x, y, z)
                else:
                    pending.append(tag)

            # Update pending tags when position becomes complete
            if x is not None and y is not None and z is not None and pending:
                for ptag in pending:
                    ptag.position = (x, y, z)
                pending.clear()

    def _generate_output(self, content: str, tags: List[TagLocation]) -> str:
        """
        Pass 3: Generate output with computed values and stripped tags.

        Args:
            content: Original file content
            tags: List of tag locations with context

        Returns:
            Processed file content
        """
        if not tags:
            return content

        # Build output by processing content with tag replacements
        output_parts: List[str] = []
        last_end = 0

        for tag in tags:
            # Add content before this tag
            output_parts.append(content[last_end:tag.start])

            # Determine replacement value
            replacement = self._compute_replacement(tag)
            output_parts.append(replacement)

            last_end = tag.end

        # Add remaining content after last tag
        output_parts.append(content[last_end:])

        return ''.join(output_parts)

    def _compute_replacement(self, tag: TagLocation) -> str:
        """
        Compute the replacement value for a tag.

        Args:
            tag: TagLocation with context

        Returns:
            Replacement string (empty for read tags, computed/preserved value for write tags)
        """
        # Read tags are stripped entirely
        if tag.tag_type == 'r':
            return ''

        # Write tags
        if tag.key in ('x', 'y', 'z'):
            # Position values are preserved, tag is stripped
            return tag.value

        if tag.key == 'lvid':
            return self._compute_lvid(tag)

        if tag.key == 'gvid':
            return self._compute_gvid(tag)

        # Unknown write tag, preserve original value
        logWarning(f"Tag rewriter: Unknown write tag key '{tag.key}', preserving value")
        return tag.value

    def _compute_lvid(self, tag: TagLocation) -> str:
        """
        Compute LVID for a tag location.

        Args:
            tag: TagLocation with level context and position

        Returns:
            Computed LVID as string, or original value on error
        """
        if not tag.level_context:
            logError(f"Tag rewriter: Missing level context for lvid tag, using default '{tag.value}'")
            return tag.value

        if not tag.position:
            logError(f"Tag rewriter: Missing position for lvid tag on level '{tag.level_context}', using default '{tag.value}'")
            return tag.value

        lvid = self.game_graph.get_level_vertex_for_position(tag.level_context, tag.position)
        if lvid is None:
            logError(f"Tag rewriter: Cannot resolve position {tag.position} on '{tag.level_context}' to LVID, using default '{tag.value}'")
            return tag.value

        return str(lvid)

    def _compute_gvid(self, tag: TagLocation) -> str:
        """
        Compute GVID for a tag location.

        Args:
            tag: TagLocation with level context and position

        Returns:
            Computed GVID as string, or original value on error
        """
        if not tag.level_context:
            logError(f"Tag rewriter: Missing level context for gvid tag, using default '{tag.value}'")
            return tag.value

        if not tag.position:
            logError(f"Tag rewriter: Missing position for gvid tag on level '{tag.level_context}', using default '{tag.value}'")
            return tag.value

        gvid = self.game_graph.get_gvid_for_position(tag.level_context, tag.position)
        if gvid is None:
            logError(f"Tag rewriter: Cannot resolve position {tag.position} on '{tag.level_context}' to GVID, using default '{tag.value}'")
            return tag.value

        return str(gvid)

    def _parse_float(self, value: str) -> Optional[float]:
        """
        Parse a float value, handling comments.

        Args:
            value: String value that may contain comments (e.g., "1.5 ; old value")

        Returns:
            Float value or None on parse error
        """
        try:
            # Take first value if there's a semicolon (comment separator)
            clean_value = value.split(';')[0].strip()
            return float(clean_value)
        except ValueError:
            return None
