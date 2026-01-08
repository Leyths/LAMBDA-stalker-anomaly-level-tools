"""
Main parser classes for level spawn files.

Contains LevelSpawnParser and AllSpawnSpawnIterator.
"""

import struct
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Union
from enum import Enum

from ..base import read_stringz, ChunkReader
from .data_types import SpawnEntity, GraphPoint
from .entity_parser import parse_spawn_packet


class SpawnFormat(Enum):
    """Detected spawn file format."""
    NEW = "new"  # Direct M_SPAWN packets (from editor)
    OLD = "old"  # Nested chunks (from extract_spawns.py / all.spawn extraction)


class LevelSpawnParser:
    """
    Parser for level.spawn files.

    Handles two distinct formats:
    - NEW format: Editor exports with direct M_SPAWN packets
    - OLD format: Extracted spawns with nested chunk structure

    Usage:
        parser = LevelSpawnParser(Path("level.spawn"))
        for entity in parser.get_entities():
            print(f"{entity.section_name}: {entity.entity_name}")

        for gp in parser.get_graph_points():
            print(f"Graph point: {gp.name} at {gp.position}")
    """

    M_SPAWN = 1

    def __init__(self, filepath: Union[str, Path]):
        """
        Load and parse a level.spawn file.

        Args:
            filepath: Path to the .spawn file
        """
        self.filepath = Path(filepath)
        self._data: bytes = b""
        self._format: Optional[SpawnFormat] = None
        self._entities: List[SpawnEntity] = []
        self._graph_points: List[GraphPoint] = []
        self._entities_by_name: Dict[str, SpawnEntity] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Load file if not already loaded."""
        if self._loaded:
            return

        if not self.filepath.exists():
            self._loaded = True
            return

        with open(self.filepath, 'rb') as f:
            self._data = f.read()

        self._format = self._detect_format()
        self._parse()
        self._loaded = True

    def _detect_format(self) -> SpawnFormat:
        """
        Detect spawn file format by examining first chunk structure.

        Returns:
            SpawnFormat.NEW or SpawnFormat.OLD
        """
        if len(self._data) < 16:
            return SpawnFormat.NEW

        # Read first chunk
        chunk_id, chunk_size = struct.unpack_from('<II', self._data, 0)
        if chunk_size > len(self._data) - 8:
            return SpawnFormat.NEW

        chunk_data = self._data[8:8 + chunk_size]

        # NEW format: chunk_data starts with M_SPAWN (0x0001)
        if len(chunk_data) >= 2:
            first_word = struct.unpack_from('<H', chunk_data, 0)[0]
            if first_word == self.M_SPAWN:
                return SpawnFormat.NEW

        # OLD format: chunk_data contains nested sub-chunks
        if len(chunk_data) >= 12:
            sub_id, sub_size = struct.unpack_from('<II', chunk_data, 0)
            if sub_id in [0, 1] and sub_size > 0 and sub_size < chunk_size:
                return SpawnFormat.OLD

        return SpawnFormat.NEW

    def _parse(self):
        """Parse the spawn file based on detected format."""
        if self._format == SpawnFormat.NEW:
            self._parse_new_format()
        else:
            self._parse_old_format()

    def _parse_new_format(self):
        """Parse NEW format spawn file (direct M_SPAWN packets)."""
        offset = 0
        graph_point_index = 0

        while offset < len(self._data) - 8:
            chunk_id, chunk_size = struct.unpack_from('<II', self._data, offset)
            chunk_data = self._data[offset + 8:offset + 8 + chunk_size]

            if len(chunk_data) != chunk_size:
                break

            # Wrap packet with size prefix
            spawn_packet = struct.pack('<H', len(chunk_data)) + chunk_data

            entity = self._parse_spawn_packet(spawn_packet, None)
            if entity:
                if entity.section_name == 'graph_point':
                    gp = self._entity_to_graph_point(entity, graph_point_index)
                    if gp:
                        self._graph_points.append(gp)
                        graph_point_index += 1
                else:
                    self._entities.append(entity)
                    if entity.entity_name:
                        self._entities_by_name[entity.entity_name] = entity

            offset += 8 + chunk_size

    def _parse_old_format(self):
        """Parse OLD format spawn file (nested chunks from extract_spawns.py)."""
        offset = 0
        graph_point_index = 0

        while offset < len(self._data) - 8:
            chunk_id, chunk_size = struct.unpack_from('<II', self._data, offset)
            chunk_data = self._data[offset + 8:offset + 8 + chunk_size]

            if len(chunk_data) != chunk_size:
                break

            # Parse sub-chunks
            sub_offset = 0
            spawn_packet = None
            update_packet = None

            while sub_offset < len(chunk_data) - 8:
                sub_id, sub_size = struct.unpack_from('<II', chunk_data, sub_offset)
                sub_data = chunk_data[sub_offset + 8:sub_offset + 8 + sub_size]

                if sub_id == 0:  # M_SPAWN packet
                    spawn_packet = sub_data
                elif sub_id == 1:  # M_UPDATE packet
                    update_packet = sub_data

                sub_offset += 8 + sub_size

            if spawn_packet:
                entity = self._parse_spawn_packet(spawn_packet, update_packet)
                if entity:
                    if entity.section_name == 'graph_point':
                        gp = self._entity_to_graph_point(entity, graph_point_index)
                        if gp:
                            self._graph_points.append(gp)
                            graph_point_index += 1
                    else:
                        self._entities.append(entity)
                        if entity.entity_name:
                            self._entities_by_name[entity.entity_name] = entity

            offset += 8 + chunk_size

    def _parse_spawn_packet(self, spawn_packet: bytes, update_packet: Optional[bytes]) -> Optional[SpawnEntity]:
        """Parse a spawn packet into a SpawnEntity."""
        return parse_spawn_packet(spawn_packet, update_packet, has_size_prefix=True)

    def _entity_to_graph_point(self, entity: SpawnEntity, index: int) -> Optional[GraphPoint]:
        """Convert a graph_point entity to a GraphPoint dataclass."""
        try:
            spawn_packet = entity.spawn_packet
            offset = 2  # Skip size prefix
            offset += 2  # Skip M_SPAWN type

            # Skip section and entity names
            _, offset = read_stringz(spawn_packet, offset)
            _, offset = read_stringz(spawn_packet, offset)

            # Skip to position
            offset += 2  # gameid, rp

            position = struct.unpack_from('<3f', spawn_packet, offset)
            offset += 12 + 12  # position + angle
            offset += 8  # respawn, id, parent, phantom

            s_flags = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

            version = 0
            if s_flags & 0x20:
                version = struct.unpack_from('<H', spawn_packet, offset)[0]
                offset += 2

            if version > 120:
                offset += 2
            if version > 69:
                offset += 2
            if version > 70:
                client_size = struct.unpack_from('<H', spawn_packet, offset)[0]
                offset += 2 + client_size
            if version > 79:
                offset += 2

            # data_size
            data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

            state_data = spawn_packet[offset:offset + data_size]

            # Parse STATE data based on content, not size
            # OLD format: CSE_ALifeObject fields followed by CSE_ALifeGraphPoint
            # NEW format: Only CSE_ALifeGraphPoint (connection strings + location types)
            #
            # Detection: OLD format starts with binary values (game_vertex_id as u16),
            # NEW format starts with connection_point_name string (printable ASCII or null)
            level_vertex_id = -1
            connection_point_name = ""
            connection_level_name = ""
            locations = bytes([0, 0, 0, 0])

            # Detect format by checking if first bytes look like a string or binary data
            # If first byte is printable ASCII (0x20-0x7E) or null (0x00), it's NEW format
            is_new_format = True
            if data_size >= 20 and len(state_data) >= 2:
                first_byte = state_data[0]
                # Binary data typically has non-printable first byte (game_vertex_id low byte)
                # String data starts with printable char or null terminator
                if first_byte != 0 and (first_byte < 0x20 or first_byte > 0x7E):
                    is_new_format = False

            if not is_new_format:
                # OLD format with CSE_ALifeObject STATE
                state_offset = 10  # Skip game_vertex_id(2) + distance(4) + direct_control(4)
                level_vertex_id = struct.unpack_from('<I', state_data, state_offset)[0]
                state_offset += 4 + 4  # level_vertex_id + flags

                # custom_data (stringZ)
                if state_offset < len(state_data):
                    _, state_offset = read_stringz(state_data, state_offset)

                # story_id, spawn_story_id
                state_offset += 8

                # CSE_ALifeGraphPoint STATE
                if state_offset < len(state_data):
                    connection_point_name, state_offset = read_stringz(state_data, state_offset)
                if state_offset < len(state_data):
                    connection_level_name, state_offset = read_stringz(state_data, state_offset)
                if state_offset + 4 <= len(state_data):
                    locations = state_data[state_offset:state_offset + 4]
            else:
                # NEW format: Only CSE_ALifeGraphPoint STATE
                state_offset = 0
                if state_offset < len(state_data):
                    connection_point_name, state_offset = read_stringz(state_data, state_offset)
                if state_offset < len(state_data):
                    connection_level_name, state_offset = read_stringz(state_data, state_offset)
                if state_offset + 4 <= len(state_data):
                    locations = state_data[state_offset:state_offset + 4]

            return GraphPoint(
                index=index,
                name=entity.entity_name,
                position=position,
                level_vertex_id=level_vertex_id,
                location_types=locations,
                connection_point_name=connection_point_name,
                connection_level_name=connection_level_name
            )

        except Exception:
            return None

    @property
    def format(self) -> SpawnFormat:
        """Get detected spawn file format."""
        self._ensure_loaded()
        return self._format or SpawnFormat.NEW

    def get_entities(self) -> Iterator[SpawnEntity]:
        """
        Iterate over all non-graph_point entities.

        Yields:
            SpawnEntity objects
        """
        self._ensure_loaded()
        return iter(self._entities)

    def get_graph_points(self) -> Iterator[GraphPoint]:
        """
        Iterate over all graph_point entities.

        Yields:
            GraphPoint objects
        """
        self._ensure_loaded()
        return iter(self._graph_points)

    def get_entity_by_name(self, name: str) -> Optional[SpawnEntity]:
        """
        Get an entity by its name.

        Args:
            name: Entity name to find

        Returns:
            SpawnEntity if found, None otherwise
        """
        self._ensure_loaded()
        return self._entities_by_name.get(name)

    def get_all_entities_by_name(self) -> Dict[str, SpawnEntity]:
        """
        Get all entities indexed by name.

        Returns:
            Dict mapping entity names to SpawnEntity objects
        """
        self._ensure_loaded()
        return self._entities_by_name.copy()

    @property
    def entity_count(self) -> int:
        """Number of non-graph_point entities."""
        self._ensure_loaded()
        return len(self._entities)

    @property
    def graph_point_count(self) -> int:
        """Number of graph_point entities."""
        self._ensure_loaded()
        return len(self._graph_points)


class AllSpawnSpawnIterator:
    """
    Iterator for spawn entities from all.spawn's spawn chunk (chunk 1).

    This parses the spawn graph structure from all.spawn, which has a different
    format than level.spawn files. The structure is:
    - Chunk 1 (spawn chunk)
      - Sub-chunk 0: vertex count
      - Sub-chunk 1: vertices data
        - Per-vertex chunks containing CServerEntityWrapper
          - Sub-chunk 0: M_SPAWN packet
          - Sub-chunk 1: M_UPDATE packet

    Usage:
        iterator = AllSpawnSpawnIterator.from_all_spawn(Path("all.spawn"))
        for entity in iterator:
            print(f"{entity.section_name}: {entity.entity_name}")

    Note: This does NOT filter by level - use game_vertex_id with GameGraphParser
    to filter entities by level.
    """

    SPAWN_CHUNK_ID = 1

    def __init__(self, spawn_chunk_data: bytes):
        """
        Initialize with spawn chunk data.

        Args:
            spawn_chunk_data: Raw data from all.spawn chunk 1
        """
        self._data = spawn_chunk_data
        self._entities: List[SpawnEntity] = []
        self._parse()

    @classmethod
    def from_all_spawn(cls, filepath: Union[str, Path]) -> 'AllSpawnSpawnIterator':
        """
        Create iterator from all.spawn file.

        Args:
            filepath: Path to all.spawn file

        Returns:
            AllSpawnSpawnIterator instance
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = f.read()

        # Find spawn chunk (chunk 1)
        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == cls.SPAWN_CHUNK_ID:
                return cls(chunk.data)

        raise ValueError(f"Spawn chunk ({cls.SPAWN_CHUNK_ID}) not found in {filepath}")

    def _parse(self):
        """Parse the spawn chunk structure."""
        # Parse spawn graph structure
        # Sub-chunk 0: vertex count
        # Sub-chunk 1: vertices data
        sg_offset = 0
        vertices_data = None

        while sg_offset < len(self._data) - 8:
            sub_id, sub_size = struct.unpack_from('<II', self._data, sg_offset)
            sub_data = self._data[sg_offset + 8:sg_offset + 8 + sub_size]

            if sub_id == 1:  # Vertices data
                vertices_data = sub_data

            sg_offset += 8 + sub_size

        if vertices_data is None:
            return

        # Parse each spawn vertex
        v_offset = 0

        while v_offset < len(vertices_data) - 8:
            v_chunk_id, v_chunk_size = struct.unpack_from('<II', vertices_data, v_offset)
            v_chunk_data = vertices_data[v_offset + 8:v_offset + 8 + v_chunk_size]

            # Parse vertex sub-chunks
            vv_offset = 0
            spawn_packet = None
            update_packet = None

            while vv_offset < len(v_chunk_data) - 8:
                vv_id, vv_size = struct.unpack_from('<II', v_chunk_data, vv_offset)
                vv_data = v_chunk_data[vv_offset + 8:vv_offset + 8 + vv_size]

                if vv_id == 1:  # CServerEntityWrapper - contains M_SPAWN and M_UPDATE
                    wrapper_offset = 0
                    while wrapper_offset < len(vv_data) - 8:
                        w_id, w_size = struct.unpack_from('<II', vv_data, wrapper_offset)
                        w_data = vv_data[wrapper_offset + 8:wrapper_offset + 8 + w_size]

                        if w_id == 0:  # M_SPAWN packet
                            spawn_packet = w_data
                        elif w_id == 1:  # M_UPDATE packet
                            update_packet = w_data

                        wrapper_offset += 8 + w_size

                vv_offset += 8 + vv_size

            if spawn_packet:
                entity = parse_spawn_packet(spawn_packet, update_packet, has_size_prefix=False)
                if entity and entity.section_name != 'graph_point':
                    self._entities.append(entity)

            v_offset += 8 + v_chunk_size

    def __iter__(self) -> Iterator[SpawnEntity]:
        """Iterate over all spawn entities."""
        return iter(self._entities)

    def __len__(self) -> int:
        """Return the number of spawn entities."""
        return len(self._entities)

    @property
    def entities(self) -> List[SpawnEntity]:
        """Get all parsed spawn entities."""
        return self._entities


def detect_spawn_format(data: bytes) -> str:
    """
    Legacy function: Detect spawn file format.

    This is a compatibility wrapper for code that hasn't migrated to LevelSpawnParser.

    Args:
        data: Binary spawn file data

    Returns:
        'NEW' or 'OLD'
    """
    if len(data) < 16:
        return 'NEW'

    chunk_id, chunk_size = struct.unpack_from('<II', data, 0)
    if chunk_size > len(data) - 8:
        return 'NEW'

    chunk_data = data[8:8 + chunk_size]

    if len(chunk_data) >= 2:
        first_word = struct.unpack_from('<H', chunk_data, 0)[0]
        if first_word == 1:  # M_SPAWN
            return 'NEW'

    if len(chunk_data) >= 12:
        sub_id, sub_size = struct.unpack_from('<II', chunk_data, 0)
        if sub_id in [0, 1] and sub_size > 0 and sub_size < chunk_size:
            return 'OLD'

    return 'NEW'
