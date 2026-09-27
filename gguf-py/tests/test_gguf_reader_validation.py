import struct
import numpy as np
import pytest

from gguf.gguf_reader import GGUFReader
from gguf.scripts.gguf_pq2_bonsai_validate import _first_q3


def _write_gguf(path, n_dims_field, dims):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)  # version 3, 1 tensor, 0 kv
    name = b'bad_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', n_dims_field)
    for d in dims:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', 0)  # dtype F32
    buf += struct.pack('<Q', 0)  # tensor offset
    buf += b'\x00' * 64
    path.write_bytes(buf)


def test_n_dims_upper_bound(tmp_path):
    # crafted file claims 1_000_000 dims; must be rejected, not read past EOF
    p = tmp_path / 'evil_ndims.gguf'
    _write_gguf(p, 1_000_000, [1] * 8)
    with pytest.raises(ValueError, match='exceeds GGML_MAX_DIMS'):
        GGUFReader(p)


def test_dims_product_no_uint64_wraparound(tmp_path):
    # dims whose true product overflows uint64; np.prod would wrap to 4 and
    # silently pass an undersized read. The reader must not accept it.
    dims = [4194305, 4194305, 211106198978564]
    assert int(np.prod(np.array(dims, dtype=np.uint64))) == 4  # the wrap bug
    p = tmp_path / 'evil_overflow.gguf'
    _write_gguf(p, len(dims), dims)
    with pytest.raises(ValueError):
        GGUFReader(p)


@pytest.mark.parametrize("packed_byte", [0x00, 0x55, 0xAA])
def test_bonsai_validator_accepts_ternary_codes(packed_byte):
    quants = np.full((2, 32), packed_byte, dtype=np.uint8)
    assert _first_q3(quants) is None


@pytest.mark.parametrize(
    ("packed_byte", "shift"),
    [(0x03, 0), (0x0C, 2), (0x30, 4), (0xC0, 6)],
)
def test_bonsai_validator_finds_q3_in_each_packed_lane(packed_byte, shift):
    quants = np.zeros((3, 32), dtype=np.uint8)
    quants[2, 7] = packed_byte
    assert _first_q3(quants) == (2, 7, shift)
