"""
Spawn Graph Builder

Builds chunk 1 (spawn graph) from level.spawn files.
Now with support for merging old spawn data to preserve custom_data/ini_string
AND blacklist support to exclude specific entities.
AND Smart Merging to preserve M_UPDATE packets.
AND Automatic fix for Anomaly Zones missing M_SPAWN data.
AND FIX for m_tSpawnID (must equal vertex_id or engine crashes).

Format (from graph_abstract_inline.h lines 227-274):
- Sub-chunk 0: vertex_count (u32)
- Sub-chunk 1: Vertices (nested chunks)
  - For each vertex i:
    - Sub-sub-chunk 0: vertex_id (u16)
    - Sub-sub-chunk 1: CServerEntityWrapper data
      - Sub-sub-sub-chunk 0: M_SPAWN packet (u16 size + data)
      - Sub-sub-sub-chunk 1: M_UPDATE packet (u16 size + data)
- Sub-chunk 2: Edges
  - For each vertex with edges:
    - vertex_id (u16)
    - edge_count (u32)
    - For each edge: target_vertex_id (u16), weight (f32)
"""

import struct
import io
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, TYPE_CHECKING

# Import entity GVID remapper and level changer config
from remapping import remap_entity_gvids, LevelChangerConfig

# Import extraction functions
from extraction import (
    load_blacklist,
    extract_entity_name,
    extract_section_name,
    collect_level_entities,
)

# Import spawn packet parser
from parsers import parse_spawn_packet

# Unified logging and utilities
from utils import log, logWarning, logError, logDebug, write_chunk

# Import update packet functions
from .update_packets import (
    create_update_packet,
    get_expected_update_size,
    remap_update_packet_gvids,
)

# Import category checking
from .entity_categories import is_anomaly_section

if TYPE_CHECKING:
    from graph import GameGraph

# Import item restrictor converter (optional - only if config exists)
try:
    from converters import ItemRestrictorConverter, create_item_restrictors_for_level
    ITEM_RESTRICTORS_AVAILABLE = True
except ImportError:
    ITEM_RESTRICTORS_AVAILABLE = False

# Import anomaly restrictor converter (optional - only if config exists)
try:
    from converters import AnomalyRestrictorConverter, create_anomaly_restrictors_for_level
    ANOMALY_RESTRICTORS_AVAILABLE = True
except ImportError:
    ANOMALY_RESTRICTORS_AVAILABLE = False


class SpawnGraphBuilder:
    """
    Build spawn graph (chunk 1) from level.spawn files with two-pass approach.
    """

    def __init__(self, blacklist_exact: Set[str] = None, blacklist_patterns: List[str] = None,
                 game_graph: 'GameGraph' = None, enable_item_restrictors: bool = True,
                 base_path: Path = None):
        self.entities = []  # List of (spawn_id, spawn_packet, update_packet)
        self.edges = []  # List of (parent_id, child_id, weight)
        self.next_spawn_id = 0
        self.blacklist_exact = blacklist_exact or set()
        self.blacklist_patterns = blacklist_patterns or []
        self.blacklisted_count = 0
        self.base_path = base_path or Path('.')

        # GameGraph object for GVID resolution (contains name_to_gvid, level offsets, cached data)
        self.game_graph = game_graph
        # For backwards compatibility
        self.name_to_gvid = game_graph.name_to_gvid if game_graph else {}

        # Level changer config (authoritative source for level changers)
        # Config is in project root (same location as levels.ini)
        self.lc_config = None
        self.level_changers_removed = 0
        config_path = self.base_path / '..' / 'level_changers.ini'
        if config_path.exists():
            self.lc_config = LevelChangerConfig(config_path)
            if self.lc_config.override_count > 0:
                log(f"  Loaded level changer config: {self.lc_config.override_count} entries")

        # Two-pass data structures
        # Values are now Tuples: (spawn_packet_bytes, optional_update_packet_bytes)
        self.level_entities: Dict[str, List[Tuple[bytes, Optional[bytes]]]] = {}
        self.level_graph_point_counts: Dict[str, int] = {}

        # Item restrictor converter for dynamic item spawn locations
        self.item_restrictor_converter = None
        self.item_restrictor_count = 0
        if enable_item_restrictors and ITEM_RESTRICTORS_AVAILABLE:
            try:
                self.item_restrictor_converter = ItemRestrictorConverter(base_mod = game_graph.base_mod)
                log("  Item restrictor converter initialized")
            except Exception as e:
                logWarning(f"  Warning: Could not initialize dynamic item restrictor converter: {e}")

        # Anomaly restrictor converter for dynamic anomaly spawn locations
        self.anomaly_restrictor_converter = None
        self.anomaly_restrictor_count = 0
        if ANOMALY_RESTRICTORS_AVAILABLE:
            try:
                self.anomaly_restrictor_converter = AnomalyRestrictorConverter(base_mod = game_graph.base_mod if game_graph else None)
                log("  Anomaly restrictor converter initialized")
            except Exception as e:
                logWarning(f"  Warning: Could not initialize dynamic anomaly restrictor converter: {e}")

    def _read_cross_table_game_vertex_count(self, cross_table_path: Path) -> int:
        """Read game_vertex_count from cross table header"""
        with open(cross_table_path, 'rb') as f:
            chunk_id = struct.unpack('<I', f.read(4))[0]
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id != 0xFFFF:
                return 0
            f.read(4)  # version
            f.read(4)  # level_vertex_count
            game_vertex_count = struct.unpack('<I', f.read(4))[0]
            return game_vertex_count

    def collect_level_entities_for_level(self, level_name: str, level_spawn_path: Path,
                                          old_spawn_path=None, base_path: Path = Path('.')):
        """
        Collect and merge entities for a level (Pass 1).
        Delegates to extraction module, then stores results.
        Also adds item spawn restrictors if converter is available.
        """
        # Convert string path to Path if needed
        if old_spawn_path and isinstance(old_spawn_path, str):
            old_spawn_path = Path(old_spawn_path)

        # Use extraction module to collect entities
        merged_packets, graph_point_count, blacklisted = collect_level_entities(
            level_name=level_name,
            level_spawn_path=level_spawn_path,
            old_spawn_path=old_spawn_path,
            blacklist_exact=self.blacklist_exact,
            blacklist_patterns=self.blacklist_patterns
        )

        # Add item spawn restrictors for this level
        if self.item_restrictor_converter:
            restrictor_packets = create_item_restrictors_for_level(
                self.item_restrictor_converter, level_name
            )
            if restrictor_packets:
                merged_packets.extend(restrictor_packets)
                self.item_restrictor_count += len(restrictor_packets)
                log(f"    Added {len(restrictor_packets)} item spawn restrictors")

        # Add anomaly spawn restrictors for this level
        if self.anomaly_restrictor_converter:
            anomaly_packets = create_anomaly_restrictors_for_level(
                self.anomaly_restrictor_converter, level_name
            )
            if anomaly_packets:
                merged_packets.extend(anomaly_packets)
                self.anomaly_restrictor_count += len(anomaly_packets)
                log(f"    Added {len(anomaly_packets)} anomaly spawn restrictors")

        self.blacklisted_count += blacklisted
        self.level_entities[level_name] = merged_packets
        self.level_graph_point_counts[level_name] = graph_point_count

    def build_with_resolution(self, level_configs: List, base_path: Path) -> bytes:
        """
        Build spawn graph with proper graph ID resolution.
        """
        log("\n" + "=" * 70)
        log("BUILDING SPAWN GRAPH (Two-Pass with Level Offsets)")
        log("=" * 70)

        # === PASS 1: Collect entities ===
        log("\n[Pass 1] Collecting and merging entities...")

        for level in level_configs:
            level_name = level.name if hasattr(level, 'name') else str(level)
            level_path = Path(level.path) if hasattr(level, 'path') else Path(f"levels/{level_name}")

            level_spawn_path = base_path / level_path / "level.spawn"
            old_spawn_path = getattr(level, 'original_spawn', None)

            if level_spawn_path.exists():
                self.collect_level_entities_for_level(level_name, level_spawn_path, old_spawn_path, base_path)
            else:
                logWarning(f"{level_spawn_path} not found")

            # === PASS 1.5: Get offsets from GameGraph ===
            log("\n[Pass 1.5] Using GameGraph for level offsets...")
            if self.game_graph:
                log(f"  Total game vertices: {len(self.game_graph.vertices)}")

        # === PASS 2: Resolve graph IDs ===
        log("\n[Pass 2] Resolving graph IDs...")

        for level in level_configs:
            level_name = level.name if hasattr(level, 'name') else str(level)

            # Check if GameGraph has resolution data for this level
            has_resolution = False
            if self.game_graph:
                level_ai = self.game_graph.get_level_ai_for_level(level_name)
                cross_table = self.game_graph.get_cross_table_for_level(level_name)
                has_resolution = level_ai is not None and cross_table is not None

            if not has_resolution and self.game_graph:
                missing = []
                if self.game_graph.get_level_ai_for_level(level_name) is None:
                    missing.append("level.ai")
                if self.game_graph.get_cross_table_for_level(level_name) is None:
                    missing.append(f"{level_name}.gct")
                if missing:
                    logError(f"  {level_name}: Missing {', '.join(missing)} - GVIDs cannot be resolved!")

            # Retrieve tuple list
            packets = self.level_entities.get(level_name, [])
            if not packets:
                continue

            game_vertex_offset = self.game_graph.get_level_offset(level_name) if self.game_graph else 0
            log(f"  Processing {level_name}: {len(packets)} entities, offset={game_vertex_offset}")

            resolved_count = 0
            level_changers_removed_this_level = 0
            for spawn_packet, existing_update_packet in packets:
                # Check entity section and name for level_changer filtering
                entity_name = extract_entity_name(spawn_packet)
                section_name = extract_section_name(spawn_packet)

                # Filter out level_changers not in config
                if section_name == 'level_changer' and self.lc_config:
                    if not self.lc_config.has_override(level_name, entity_name):
                        # Level changer not in config - skip it
                        level_changers_removed_this_level += 1
                        self.level_changers_removed += 1
                        continue

                spawn_id = self.next_spawn_id
                self.next_spawn_id += 1

                # Resolve graph IDs (and level_changer destinations) using GameGraph
                if has_resolution and self.game_graph:
                    try:
                        resolved_packet = remap_entity_gvids(
                            spawn_packet, self.game_graph, level_name,
                            spawn_id, name_to_gvid=self.name_to_gvid,
                            lc_config=self.lc_config
                        )
                        if resolved_packet != spawn_packet:
                            spawn_packet = resolved_packet
                            resolved_count += 1
                    except Exception as e:
                        logWarning(f"Failed to resolve IDs for entity {spawn_id}: {e}")

                # === FIX FOR ANOMALY SPAWN PACKETS ===
                # Always check if the spawn packet needs the 'last_spawn_time' fix
                # for se_zone_anom classes (applies to raw level.spawn packets)
                spawn_packet = self._fix_anomaly_spawn_packet(spawn_packet)

                # === SMART UPDATE PACKET LOGIC ===
                # If we have a valid update packet from the old file, USE IT.
                # But remap any GVIDs in the packet to match the new game graph.
                # Note: entity_name and section_name already extracted above

                if existing_update_packet is not None and len(existing_update_packet) > 2:
                    # Remap GVIDs in M_UPDATE packet if needed (for monsters/stalkers)
                    existing_update_packet = remap_update_packet_gvids(
                        existing_update_packet, section_name, spawn_packet
                    )
                    # Validate the existing packet against expected format
                    expected_size = get_expected_update_size(section_name)
                    actual_size = len(existing_update_packet)

                    if expected_size is not None and actual_size != expected_size:
                        logDebug(f"M_UPDATE size mismatch for '{entity_name}' ({section_name}): "
                            f"expected {expected_size} bytes, got {actual_size} bytes")
                        # Parse basic structure: u16 size, u16 msg_type, then data
                        if actual_size >= 4:
                            pkt_size = struct.unpack_from('<H', existing_update_packet, 0)[0]
                            msg_type = struct.unpack_from('<H', existing_update_packet, 2)[0]
                            logDebug(f"  Parsed: size_field={pkt_size}, msg_type={msg_type}, payload={actual_size - 4} bytes")

                    update_packet = existing_update_packet
                else:
                    # Otherwise, generate a fresh minimal packet
                    update_packet = create_update_packet(spawn_packet)

                self.entities.append((spawn_id, spawn_packet, update_packet))

            if resolved_count > 0:
                log(f"    Resolved {resolved_count} graph IDs")
            if level_changers_removed_this_level > 0:
                log(f"    Removed {level_changers_removed_this_level} level changers (not in config)")

        return self.build()

    def _fix_anomaly_spawn_packet(self, spawn_packet: bytes) -> bytes:
        """
        Check if the entity is an Anomaly Zone and append 'last_spawn_time' (0x00)
        if it's missing. This fixes crashes for se_zone_anom/visual/torrid classes.
        """
        section = extract_section_name(spawn_packet)

        if is_anomaly_section(section):
            # Strip current size prefix (first 2 bytes)
            data = spawn_packet[2:]

            # Append 0x00 (empty complex_time)
            data += b'\x00'

            # Re-pack with new size
            return struct.pack('<H', len(data)) + data

        return spawn_packet

    def _fix_spawn_id_in_packet(self, spawn_packet: bytes, vertex_id: int) -> bytes:
        """
        Update m_tSpawnID in spawn packet to match vertex_id.

        CRITICAL: The engine uses m_tSpawnID to look up entities in the spawn graph.
        From alife_spawn_registry_spawn.cpp:
            spawns.push_back(vertex->data()->object().m_tSpawnID);
        Then from alife_surge_manager.cpp:
            spawns().spawns().vertex(*I)->data()->object()  // Uses m_tSpawnID as vertex ID!

        If m_tSpawnID != vertex_id, the engine crashes trying to access non-existent vertex.
        """
        M_SPAWN_VERSION = 0x0020

        try:
            offset = 2  # Skip size prefix (u16)

            # M_SPAWN message type (u16)
            if offset + 2 > len(spawn_packet):
                return spawn_packet
            msg_type = struct.unpack_from('<H', spawn_packet, offset)[0]
            if msg_type != 1:  # Not M_SPAWN
                return spawn_packet
            offset += 2

            # Skip section name (stringZ)
            end = spawn_packet.find(b'\x00', offset)
            if end == -1:
                return spawn_packet
            offset = end + 1

            # Skip entity name (stringZ)
            end = spawn_packet.find(b'\x00', offset)
            if end == -1:
                return spawn_packet
            offset = end + 1

            # Skip fixed fields:
            # gameid(1) + rp(1) + position(12) + angle(12) + respawn(2) + id(2) + parent(2) + phantom(2)
            offset += 1 + 1 + 12 + 12 + 2 + 2 + 2 + 2

            # s_flags (u16)
            if offset + 2 > len(spawn_packet):
                return spawn_packet
            s_flags = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

            # Version (u16) - only if M_SPAWN_VERSION flag is set
            version = 0
            if s_flags & M_SPAWN_VERSION:
                if offset + 2 > len(spawn_packet):
                    return spawn_packet
                version = struct.unpack_from('<H', spawn_packet, offset)[0]
                offset += 2

            # game_type (u16) - if version > 120
            if version > 120:
                offset += 2

            # script_version (u16) - if version > 69
            if version > 69:
                offset += 2

            # client_data - if version > 70
            if version > 70:
                if offset + 2 > len(spawn_packet):
                    return spawn_packet
                client_data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
                offset += 2 + client_data_size

            # m_tSpawnID (u16) - if version > 79
            if version > 79:
                if offset + 2 > len(spawn_packet):
                    return spawn_packet

                # Update m_tSpawnID to vertex_id
                updated = bytearray(spawn_packet)
                struct.pack_into('<H', updated, offset, vertex_id)
                return bytes(updated)

            return spawn_packet

        except Exception:
            return spawn_packet

    def _build_vertex_count(self) -> bytes:
        buffer = io.BytesIO()
        buffer.write(struct.pack('<I', len(self.entities)))
        return buffer.getvalue()

    def _build_vertices(self) -> bytes:
        buffer = io.BytesIO()
        for vertex_id, (_, spawn_packet, update_packet) in enumerate(self.entities):
            # FIX: Update m_tSpawnID in packet to match vertex_id
            # Without this, the engine crashes when looking up entities
            spawn_packet = self._fix_spawn_id_in_packet(spawn_packet, vertex_id)

            vertex_buffer = io.BytesIO()
            id_buffer = io.BytesIO()
            id_buffer.write(struct.pack('<H', vertex_id))
            write_chunk(vertex_buffer, 0, id_buffer.getvalue())
            wrapper_buffer = io.BytesIO()
            write_chunk(wrapper_buffer, 0, spawn_packet)
            write_chunk(wrapper_buffer, 1, update_packet)
            write_chunk(vertex_buffer, 1, wrapper_buffer.getvalue())
            write_chunk(buffer, 0, vertex_buffer.getvalue())
        return buffer.getvalue()

    def _build_edges(self) -> bytes:
        buffer = io.BytesIO()
        edges_by_vertex = {}
        for parent_id, child_id, weight in self.edges:
            if parent_id not in edges_by_vertex:
                edges_by_vertex[parent_id] = []
            edges_by_vertex[parent_id].append((child_id, weight))
        for vertex_id, edges in sorted(edges_by_vertex.items()):
            buffer.write(struct.pack('<H', vertex_id))
            buffer.write(struct.pack('<I', len(edges)))
            for target_id, weight in edges:
                buffer.write(struct.pack('<H', target_id))
                buffer.write(struct.pack('<f', weight))
        return buffer.getvalue()

    def build(self) -> bytes:
        """Build the spawn graph binary data"""
        log(f"\n  Building spawn graph...")
        log(f"    Vertices: {len(self.entities)}")
        log(f"    Edges: {len(self.edges)}")
        if self.blacklisted_count > 0:
            log(f"    Total blacklisted: {self.blacklisted_count}")
        if self.level_changers_removed > 0:
            log(f"    Level changers removed (not in config): {self.level_changers_removed}")
        if self.item_restrictor_count > 0:
            log(f"    Item spawn restrictors: {self.item_restrictor_count}")
        if self.anomaly_restrictor_count > 0:
            log(f"    Anomaly spawn restrictors: {self.anomaly_restrictor_count}")
        buffer = io.BytesIO()
        write_chunk(buffer, 0, self._build_vertex_count())
        write_chunk(buffer, 1, self._build_vertices())
        write_chunk(buffer, 2, self._build_edges())
        return buffer.getvalue()


def build_spawn_graph(level_configs: List,
                      base_path: Path = Path('.'), blacklist_path: Optional[Path] = None,
                      game_graph: 'GameGraph' = None) -> Tuple[bytes, int]:
    """Build spawn graph from multiple level.spawn files"""
    blacklist_exact, blacklist_patterns = set(), []
    if blacklist_path and blacklist_path.exists():
        blacklist_exact, blacklist_patterns = load_blacklist(blacklist_path)
        total_rules = len(blacklist_exact) + len(blacklist_patterns)
        if total_rules > 0:
            log(f"\nLoaded blacklist from {blacklist_path}")
            log(f"  Exact matches: {len(blacklist_exact)}")
            log(f"  Wildcard patterns: {len(blacklist_patterns)}")

    builder = SpawnGraphBuilder(blacklist_exact, blacklist_patterns, game_graph, base_path=base_path)
    spawn_graph = builder.build_with_resolution(level_configs, base_path)
    spawn_count = len(builder.entities)
    return (spawn_graph, spawn_count)
