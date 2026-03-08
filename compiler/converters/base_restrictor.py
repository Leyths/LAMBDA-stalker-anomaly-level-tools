"""
Base Restrictor Converter

Shared logic for creating space_restrictor entities per level.
Subclasses define only the entity prefix, custom_data section,
config path, and log label.
"""

import struct
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import log

from .base import LVID_UNSET, GVID_UNSET, LevelRestrictorInfo


class BaseRestrictorConverter:
    """
    Creates one space_restrictor per level.
    Subclasses must set class attributes:
        entity_prefix: str         e.g. "sr_dynamic_anomaly_"
        logic_section: str         e.g. "sr_dynamic_anomaly"
        fallback_config_rel: str   e.g. "configs/zones/dynamic_anomaly_locations.ltx"
        log_label: str             e.g. "anomaly restrictors"
    """

    entity_prefix: str
    logic_section: str
    fallback_config_rel: str
    log_label: str

    def __init__(self, source_config_path: Path = None, output_dir: Path = None):
        if source_config_path and source_config_path.exists():
            config_path = source_config_path
        elif output_dir:
            config_path = output_dir / self.fallback_config_rel
        else:
            raise ValueError(f"No source_config_path or output_dir provided for {self.log_label}")

        self.levels: List[str] = []
        self.locations: Dict[str, LevelRestrictorInfo] = {}
        self._load_level_list(config_path)

    def _load_level_list(self, path: Path) -> None:
        if not path.exists():
            log(f"Warning: {path} not found, no {self.log_label} will be created")
            return

        content = path.read_text(encoding='utf-8', errors='ignore')
        current_section = None

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(';') or line.startswith('--') or not line:
                continue
            section_match = re.match(r'^\[([^\]]+)\]', line)
            if section_match:
                current_section = section_match.group(1)
                continue
            if current_section == 'levels' and line:
                level_name = line.strip()
                if level_name:
                    self.levels.append(level_name)
                    self.locations[level_name] = LevelRestrictorInfo(
                        level_name=level_name, x=0.0, y=0.0, z=0.0
                    )

        log(f"  Loaded {len(self.levels)} levels for {self.log_label}")

    def get_levels(self) -> List[str]:
        return self.levels

    def get_restrictor_for_level(self, level_name: str) -> Optional[LevelRestrictorInfo]:
        return self.locations.get(level_name)

    def create_restrictor_packet(self, level_name: str, location: LevelRestrictorInfo) -> Tuple[bytes, bytes]:
        spawn_packet = self._create_spawn_packet(level_name, location)
        update_packet = self._create_update_packet()
        return spawn_packet, update_packet

    def _create_spawn_packet(self, level_name: str, location: LevelRestrictorInfo) -> bytes:
        buffer = io.BytesIO()

        # Size prefix placeholder
        buffer.write(struct.pack('<H', 0))
        # M_SPAWN message type
        buffer.write(struct.pack('<H', 1))
        # Section name
        buffer.write(b'space_restrictor\x00')
        # Entity name
        entity_name = f"{self.entity_prefix}{level_name}"
        buffer.write(entity_name.encode('utf-8') + b'\x00')
        # gameid, s_RP
        buffer.write(struct.pack('<BB', 0, 0xFE))
        # Position
        buffer.write(struct.pack('<3f', location.x, location.y, location.z))
        # Angle
        buffer.write(struct.pack('<3f', 0.0, 0.0, 0.0))
        # RespawnTime, ID, ID_Parent, ID_Phantom
        buffer.write(struct.pack('<HHHH', 0, 0xFFFF, 0xFFFF, 0xFFFF))
        # s_flags
        buffer.write(struct.pack('<H', 0x20))
        # Version
        buffer.write(struct.pack('<H', 128))
        # game_type
        buffer.write(struct.pack('<H', 1))
        # script_version
        buffer.write(struct.pack('<H', 8))
        # client_data_size
        buffer.write(struct.pack('<H', 0))
        # spawn_id
        buffer.write(struct.pack('<H', 0))

        # data_size placeholder
        data_start = buffer.tell()
        buffer.write(struct.pack('<H', 0))

        # CSE_ALifeObject fields
        buffer.write(struct.pack('<H', location.game_vertex_id))
        buffer.write(struct.pack('<f', 0.0))
        buffer.write(struct.pack('<I', 1))  # direct_control
        buffer.write(struct.pack('<I', location.level_vertex_id))
        buffer.write(struct.pack('<I', 0x202))  # flags

        # custom_data
        custom_data = self._create_custom_data(level_name)
        buffer.write(custom_data.encode('utf-8') + b'\x00')

        buffer.write(struct.pack('<II', 0xFFFFFFFF, 0xFFFFFFFF))  # story_id, spawn_story_id

        # CSE_Shape fields
        buffer.write(struct.pack('<B', 1))  # 1 shape
        buffer.write(struct.pack('<B', 0))  # Sphere
        buffer.write(struct.pack('<3f', 0.0, 0.0, 0.0))  # center
        buffer.write(struct.pack('<f', 1.0))  # radius

        # CSE_ALifeSpaceRestrictor fields
        buffer.write(struct.pack('<B', 0))  # restrictor_type

        # Write data_size
        data_end = buffer.tell()
        data_size = data_end - data_start - 2
        buffer.seek(data_start)
        buffer.write(struct.pack('<H', data_size))

        # Write size prefix
        packet_data = buffer.getvalue()[2:]
        final_packet = struct.pack('<H', len(packet_data)) + packet_data
        return final_packet

    def _create_custom_data(self, level_name: str) -> str:
        return f"[logic]\nactive = {self.logic_section}\n\n[{self.logic_section}]"

    def _create_update_packet(self) -> bytes:
        buffer = io.BytesIO()
        buffer.write(struct.pack('<H', 0))  # M_UPDATE
        packet_data = buffer.getvalue()
        final_buffer = io.BytesIO()
        final_buffer.write(struct.pack('<H', len(packet_data)))
        final_buffer.write(packet_data)
        return final_buffer.getvalue()


def create_restrictors_for_level(
    converter: BaseRestrictorConverter,
    level_name: str
) -> List[Tuple[bytes, Optional[bytes]]]:
    """
    Create restrictor packet for a given level.

    Returns:
        List of (spawn_packet, update_packet) tuples (single-item list or empty)
    """
    location = converter.get_restrictor_for_level(level_name)
    if not location:
        return []
    spawn_pkt, update_pkt = converter.create_restrictor_packet(level_name, location)
    return [(spawn_pkt, update_pkt)]
