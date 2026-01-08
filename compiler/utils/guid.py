"""
Deterministic GUID Generation

Generates reproducible GUIDs based on input data using SHA-256 hashing.
This ensures build reproducibility - the same inputs always produce the same outputs.
"""

import hashlib
from typing import Union, List


def generate_guid(*args: Union[str, bytes, int, float]) -> bytes:
    """
    Generate a deterministic 16-byte GUID from input arguments.

    Uses SHA-256 hash truncated to 16 bytes. The same inputs will
    always produce the same GUID.

    Args:
        *args: Any combination of strings, bytes, integers, or floats.
               These are concatenated and hashed to produce the GUID.

    Returns:
        16-byte GUID suitable for X-Ray Engine binary files.

    Example:
        # Game graph GUID based on level data
        guid = generate_guid("game_graph", level_count, total_vertices)

        # Cross table GUID based on level name and vertex count
        guid = generate_guid("cross_table", level_name, vertex_count)

        # Spawn GUID based on spawn count
        guid = generate_guid("spawn", spawn_count, level_count)
    """
    hasher = hashlib.sha256()

    for arg in args:
        if isinstance(arg, str):
            hasher.update(arg.encode('utf-8'))
        elif isinstance(arg, bytes):
            hasher.update(arg)
        elif isinstance(arg, (int, float)):
            hasher.update(str(arg).encode('utf-8'))
        else:
            # Fallback: convert to string
            hasher.update(str(arg).encode('utf-8'))

        # Add separator to prevent collisions like ("ab", "c") vs ("a", "bc")
        hasher.update(b'\x00')

    # Return first 16 bytes of SHA-256 hash
    return hasher.digest()[:16]


def generate_guid_from_bytes(data: bytes, prefix: str = "") -> bytes:
    """
    Generate a deterministic GUID from raw binary data.

    Useful for hashing file contents or large binary structures.

    Args:
        data: Binary data to hash
        prefix: Optional prefix string for namespacing

    Returns:
        16-byte GUID
    """
    hasher = hashlib.sha256()
    if prefix:
        hasher.update(prefix.encode('utf-8'))
        hasher.update(b'\x00')
    hasher.update(data)
    return hasher.digest()[:16]
