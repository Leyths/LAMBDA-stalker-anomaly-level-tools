"""
Binary File Utilities

Common utilities for reading and writing X-Ray Engine binary file formats.
"""

import struct
import io
from typing import BinaryIO, Union


def write_chunk(buffer: Union[BinaryIO, io.BytesIO], chunk_id: int, data: bytes):
    """
    Write a chunked data block to a buffer.

    X-Ray Engine chunk format:
    - u32 chunk_id
    - u32 chunk_size
    - [chunk_size bytes of data]

    Args:
        buffer: Output buffer (file or BytesIO)
        chunk_id: Chunk identifier
        data: Chunk data bytes
    """
    buffer.write(struct.pack('<I', chunk_id))
    buffer.write(struct.pack('<I', len(data)))
    buffer.write(data)


def write_chunk_header(buffer: Union[BinaryIO, io.BytesIO], chunk_id: int, size: int):
    """
    Write just a chunk header (useful when streaming data).

    Args:
        buffer: Output buffer
        chunk_id: Chunk identifier
        size: Size of data that will follow
    """
    buffer.write(struct.pack('<II', chunk_id, size))
