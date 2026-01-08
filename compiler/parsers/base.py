"""
Base utilities for X-Ray Engine binary file parsing.

This module provides shared utilities used by all parser classes:
- read_stringz: Read null-terminated strings
- ChunkReader: Iterator for chunked binary data formats
"""

import struct
from typing import Tuple, Iterator, Optional
from dataclasses import dataclass


def read_stringz(data: bytes, offset: int) -> Tuple[str, int]:
    """
    Read a null-terminated string from binary data.

    Args:
        data: Binary data to read from
        offset: Starting offset in the data

    Returns:
        Tuple of (string, new_offset after null terminator)
    """
    end = data.find(b'\x00', offset)
    if end == -1:
        return "", len(data)
    return data[offset:end].decode('utf-8', errors='replace'), end + 1


@dataclass
class Chunk:
    """A single chunk from a chunked binary file."""
    chunk_id: int
    size: int
    data: bytes
    offset: int  # Offset in the original data where this chunk starts


class ChunkReader:
    """
    Iterator for X-Ray Engine chunked binary data formats.

    X-Ray uses a consistent chunked format across many file types:
    - u32 chunk_id
    - u32 chunk_size
    - [chunk_size bytes of data]

    This class provides a convenient iterator interface for reading these chunks.

    Usage:
        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == 0:
                process_header(chunk.data)
            elif chunk.chunk_id == 1:
                process_content(chunk.data)
    """

    HEADER_SIZE = 8  # u32 chunk_id + u32 chunk_size

    def __init__(self, data: bytes, start_offset: int = 0):
        """
        Initialize chunk reader.

        Args:
            data: Binary data containing chunks
            start_offset: Offset to start reading from (default 0)
        """
        self.data = data
        self.offset = start_offset

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over all chunks in the data."""
        return self

    def __next__(self) -> Chunk:
        """Read the next chunk."""
        if self.offset >= len(self.data) - self.HEADER_SIZE + 1:
            raise StopIteration

        chunk_id, chunk_size = struct.unpack_from('<II', self.data, self.offset)
        data_start = self.offset + self.HEADER_SIZE
        data_end = data_start + chunk_size

        if data_end > len(self.data):
            raise StopIteration

        chunk = Chunk(
            chunk_id=chunk_id,
            size=chunk_size,
            data=self.data[data_start:data_end],
            offset=self.offset
        )

        self.offset = data_end
        return chunk

    def read_chunk(self) -> Optional[Chunk]:
        """
        Read the next chunk, returning None if no more chunks.

        This is an alternative to iteration for cases where you need
        more control over the reading process.
        """
        try:
            return next(self)
        except StopIteration:
            return None

    def peek_chunk_id(self) -> Optional[int]:
        """
        Peek at the next chunk's ID without consuming it.

        Returns:
            Chunk ID or None if no more chunks
        """
        if self.offset >= len(self.data) - self.HEADER_SIZE + 1:
            return None
        return struct.unpack_from('<I', self.data, self.offset)[0]

    def skip_chunk(self) -> bool:
        """
        Skip the next chunk without reading its data.

        Returns:
            True if a chunk was skipped, False if no more chunks
        """
        if self.offset >= len(self.data) - self.HEADER_SIZE + 1:
            return False

        chunk_size = struct.unpack_from('<I', self.data, self.offset + 4)[0]
        self.offset += self.HEADER_SIZE + chunk_size
        return True

    def find_chunk(self, chunk_id: int) -> Optional[Chunk]:
        """
        Find and return a specific chunk by ID.

        Note: This advances the reader to after the found chunk.

        Args:
            chunk_id: The chunk ID to find

        Returns:
            The chunk if found, None otherwise
        """
        for chunk in self:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    @property
    def remaining_bytes(self) -> int:
        """Number of bytes remaining to be read."""
        return max(0, len(self.data) - self.offset)

    @property
    def has_more(self) -> bool:
        """Check if there are more chunks to read."""
        return self.offset < len(self.data) - self.HEADER_SIZE + 1


def read_chunk_header(data: bytes, offset: int) -> Tuple[int, int, int]:
    """
    Read a chunk header from binary data.

    Args:
        data: Binary data
        offset: Offset to read from

    Returns:
        Tuple of (chunk_id, chunk_size, new_offset after header)
    """
    if offset + 8 > len(data):
        raise ValueError(f"Not enough data for chunk header at offset {offset}")
    chunk_id, chunk_size = struct.unpack_from('<II', data, offset)
    return chunk_id, chunk_size, offset + 8
