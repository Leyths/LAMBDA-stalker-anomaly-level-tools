"""
String utilities for level spawn parsing.

Provides cp1251-aware string reading for Russian text in STALKER game files.
"""

from typing import Tuple


def read_stringz_cp1251(data: bytes, offset: int) -> Tuple[str, int]:
    """
    Read null-terminated string from data using cp1251 encoding.

    Uses cp1251 (Windows Cyrillic) as primary encoding since S.T.A.L.K.E.R.
    game files use this encoding for Russian text. Falls back to UTF-8 if
    cp1251 decoding fails.

    Args:
        data: Binary data to read from
        offset: Starting offset

    Returns:
        Tuple of (string, new_offset after null terminator)
    """
    end = offset
    while end < len(data) and data[end] != 0:
        end += 1
    try:
        s = data[offset:end].decode('cp1251')
    except UnicodeDecodeError:
        s = data[offset:end].decode('utf-8', errors='replace')
    return s, end + 1
