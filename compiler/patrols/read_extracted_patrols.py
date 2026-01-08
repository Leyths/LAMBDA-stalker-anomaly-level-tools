#!/usr/bin/env python3
"""
Extracted Patrols Reader

Reads .patrols files (binary intermediate format) created by extract_patrols_from_spawn.py
"""

import struct
from pathlib import Path
from typing import Dict


def read_extracted_patrols(patrols_file: Path) -> Dict[str, bytes]:
    """
    Read patrol paths from extracted .patrols file

    Format:
    - u32 patrol_count
    - For each patrol:
      - u16 name_length
      - name (bytes)
      - u32 data_length
      - data (bytes - CPatrolPath binary)

    Args:
        patrols_file: Path to .patrols file

    Returns:
        Dict of {patrol_name: patrol_binary_data}
    """
    if not patrols_file.exists():
        return {}

    patrols = {}

    with open(patrols_file, 'rb') as f:
        # Read count
        patrol_count = struct.unpack('<I', f.read(4))[0]

        for _ in range(patrol_count):
            # Read name
            name_length = struct.unpack('<H', f.read(2))[0]
            name = f.read(name_length).decode('utf-8')

            # Read data
            data_length = struct.unpack('<I', f.read(4))[0]
            data = f.read(data_length)

            patrols[name] = data

    return patrols


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_extracted_patrols.py <patrols_file>")
        sys.exit(1)

    patrols_file = Path(sys.argv[1])
    patrols = read_extracted_patrols(patrols_file)

    print(f"Read {len(patrols)} patrol paths from {patrols_file}:")
    for name in sorted(patrols.keys()):
        print(f"  {name}: {len(patrols[name])} bytes")