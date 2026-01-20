#!/usr/bin/env python3
"""
Anomaly Restrictor Converter

Creates ONE space_restrictor per level for dynamic anomaly spawning.
The restrictor triggers the Lua script which reads all spawn locations
from dynamic_anomaly_locations.ltx and spawns anomalies at runtime.

The converter:
1. Reads level list from dynamic_anomaly_locations.ltx
2. Creates ONE space_restrictor per level (name: sr_dynamic_anomaly_<level>)
3. Lua script handles runtime LVID/GVID lookups and spawning
"""

import struct
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import log

from .base import LVID_UNSET, GVID_UNSET, LevelRestrictorInfo


class AnomalyRestrictorConverter:
    """
    Creates one space_restrictor per level for dynamic anomaly spawning.
    """

    def __init__(self, base_mod: str = None):
        """
        Load level list from merged config file.

        The config file should already be in gamedata/ (copied by ModCopier).

        Args:
            base_mod: Base mod variant (anomaly, gamma) - kept for backwards compatibility
        """
        # Config path - look in gamedata/ where ModCopier places it
        config_path = Path(__file__).parent.parent.parent / "gamedata/configs/zones/dynamic_anomaly_locations.ltx"

        self.levels: List[str] = []
        self.locations: Dict[str, LevelRestrictorInfo] = {}

        # Load level list
        self._load_level_list(config_path)

    def _load_level_list(self, path: Path) -> None:
        """Load level list from dynamic_anomaly_locations.ltx."""
        if not path.exists():
            log(f"Warning: {path} not found, no anomaly restrictors will be created")
            return

        content = path.read_text(encoding='utf-8', errors='ignore')
        current_section = None

        for line in content.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith(';') or line.startswith('--') or not line:
                continue

            # Section header
            section_match = re.match(r'^\[([^\]]+)\]', line)
            if section_match:
                current_section = section_match.group(1)
                continue

            # Read level names from [levels] section
            if current_section == 'levels' and line:
                level_name = line.strip()
                if level_name:
                    self.levels.append(level_name)
                    # Create a restrictor info for this level at origin
                    self.locations[level_name] = LevelRestrictorInfo(
                        level_name=level_name,
                        x=0.0, y=0.0, z=0.0
                    )

        log(f"  Loaded {len(self.levels)} levels for anomaly restrictors")

    def get_levels(self) -> List[str]:
        """Get list of all levels that need anomaly restrictors."""
        return self.levels

    def get_restrictor_for_level(self, level_name: str) -> Optional[LevelRestrictorInfo]:
        """Get the restrictor info for a level."""
        return self.locations.get(level_name)

    def create_restrictor_packet(self, level_name: str, location: LevelRestrictorInfo) -> Tuple[bytes, bytes]:
        """
        Create M_SPAWN and M_UPDATE packets for a space_restrictor entity.

        Args:
            level_name: Name of the level
            location: LevelRestrictorInfo with coordinates

        Returns:
            Tuple of (spawn_packet, update_packet)
        """
        spawn_packet = self._create_spawn_packet(level_name, location)
        update_packet = self._create_update_packet()
        return spawn_packet, update_packet

    def _create_spawn_packet(self, level_name: str, location: LevelRestrictorInfo) -> bytes:
        """
        Create an M_SPAWN packet for a space_restrictor entity.

        Format based on CSE_Abstract::Spawn_Write, CSE_ALifeObject::STATE_Write,
        and CSE_ALifeSpaceRestrictor::STATE_Write.
        """
        buffer = io.BytesIO()

        # Leave space for size prefix (filled at end)
        size_pos = buffer.tell()
        buffer.write(struct.pack('<H', 0))

        # M_SPAWN message type
        buffer.write(struct.pack('<H', 1))  # M_SPAWN = 1

        # Section name (stringZ)
        buffer.write(b'space_restrictor\x00')

        # Entity name (stringZ) - one per level: sr_dynamic_anomaly_<level>
        entity_name = f"sr_dynamic_anomaly_{level_name}"
        buffer.write(entity_name.encode('utf-8') + b'\x00')

        # gameid (u8), s_RP (u8) - 0xFE = use supplied coords
        buffer.write(struct.pack('<BB', 0, 0xFE))

        # Position (Fvector) - at origin
        buffer.write(struct.pack('<3f', location.x, location.y, location.z))

        # Angle (Fvector) - zeros
        buffer.write(struct.pack('<3f', 0.0, 0.0, 0.0))

        # RespawnTime, ID, ID_Parent, ID_Phantom (all u16)
        buffer.write(struct.pack('<HHHH', 0, 0xFFFF, 0xFFFF, 0xFFFF))

        # s_flags (u16) - M_SPAWN_VERSION = 0x20
        buffer.write(struct.pack('<H', 0x20))

        # Version (u16) - SPAWN_VERSION = 128
        buffer.write(struct.pack('<H', 128))

        # game_type (u16) - since version > 120
        buffer.write(struct.pack('<H', 1))  # GAME_SINGLE

        # script_version (u16) - since version > 69
        buffer.write(struct.pack('<H', 8))

        # client_data_size (u16) - since version > 70
        buffer.write(struct.pack('<H', 0))

        # spawn_id (u16) - since version > 79 (will be fixed by builder)
        buffer.write(struct.pack('<H', 0))

        # data_size placeholder
        data_start = buffer.tell()
        buffer.write(struct.pack('<H', 0))

        # === CSE_ALifeObject fields ===
        buffer.write(struct.pack('<H', location.game_vertex_id))  # game_vertex_id
        buffer.write(struct.pack('<f', 0.0))  # distance
        buffer.write(struct.pack('<I', 1))  # direct_control = true (required for ALife registration)
        buffer.write(struct.pack('<I', location.level_vertex_id))  # level_vertex_id
        buffer.write(struct.pack('<I', 0x202))  # flags = flSwitchOnline | flCanSave (persist across level transitions)

        # custom_data (stringZ format: null-terminated, no length prefix)
        custom_data = self._create_custom_data(level_name)
        buffer.write(custom_data.encode('utf-8') + b'\x00')

        buffer.write(struct.pack('<II', 0xFFFFFFFF, 0xFFFFFFFF))  # story_id, spawn_story_id

        # === CSE_Shape fields (for space_restrictor) ===
        # Shape count
        buffer.write(struct.pack('<B', 1))  # 1 shape

        # Shape type 0 = Sphere
        buffer.write(struct.pack('<B', 0))

        # Sphere center (local offset from position)
        buffer.write(struct.pack('<3f', 0.0, 0.0, 0.0))

        # Sphere radius - small radius, we don't need a real restrictor zone
        buffer.write(struct.pack('<f', 1.0))

        # === CSE_ALifeSpaceRestrictor fields ===
        # restrictor_type (u8)
        buffer.write(struct.pack('<B', 0))  # default restrictor type

        # Go back and write data_size
        data_end = buffer.tell()
        data_size = data_end - data_start - 2
        buffer.seek(data_start)
        buffer.write(struct.pack('<H', data_size))

        # Write size prefix
        packet_data = buffer.getvalue()[2:]  # Exclude size prefix placeholder
        final_packet = struct.pack('<H', len(packet_data)) + packet_data

        return final_packet

    def _create_custom_data(self, level_name: str) -> str:
        """
        Create custom_data string for the space_restrictor.
        This is an ini-format string embedded in the spawn packet.
        """
        return "[logic]\nactive = sr_dynamic_anomaly\n\n[sr_dynamic_anomaly]"

    def _create_update_packet(self) -> bytes:
        """
        Create an M_UPDATE packet for space_restrictor.
        Static objects use empty update packets (just msg type, no data).

        Format: [u16 size][u16 msg_type=M_UPDATE]
        M_UPDATE = 0 (from xrMessages.h)
        """
        buffer = io.BytesIO()

        # M_UPDATE message type (M_UPDATE = 0)
        buffer.write(struct.pack('<H', 0))

        packet_data = buffer.getvalue()

        # Final packet with size prefix
        final_buffer = io.BytesIO()
        final_buffer.write(struct.pack('<H', len(packet_data)))  # Size = 2
        final_buffer.write(packet_data)  # [00 00] = M_UPDATE

        return final_buffer.getvalue()


def create_anomaly_restrictors_for_level(
    converter: AnomalyRestrictorConverter,
    level_name: str
) -> List[Tuple[bytes, Optional[bytes]]]:
    """
    Create anomaly restrictor packet for a given level.

    Args:
        converter: AnomalyRestrictorConverter instance
        level_name: Name of the level

    Returns:
        List of (spawn_packet, update_packet) tuples (will be a single-item list)
    """
    location = converter.get_restrictor_for_level(level_name)
    if not location:
        return []

    spawn_pkt, update_pkt = converter.create_restrictor_packet(level_name, location)
    return [(spawn_pkt, update_pkt)]
