"""Malformed archives must fail before reaching PyTorch or native allocation."""
from dataclasses import replace

import numpy as np
import pytest

from needle2.archive import Archive, CQ, FP32, HEADER, RECORD, TAG


@pytest.fixture
def tiny_archive():
    # A hand-built one-layer SAN with one Engram. Keep tensors small, and use
    # CQ for embedding so alternate row/column factorizations share byte size.
    layer_shapes = [(8,), (8, 8), (4, 8), (4, 8), (4,), (4,), (8, 8),
                    (8, 8), (8,), (1,), (8,), (8,), (8,), (8,)]
    mhc_shapes = [(1,), (1,), (1,), (1, 2), (1, 2), (1, 2, 2), (2, 16), (2, 16), (4, 16)]
    shapes = [(16, 8)] + layer_shapes + mhc_shapes + [(8, 4), (8, 8), (8, 8), (4, 8), (8,)]
    header = [TAG, len(shapes), 28, 5, 8, 16, 8, 2, 1, 1, 4, 64,
              8, 2, 4, 4, 2, 4, 3, 2, 2, 3, 0, 0, 1, 0, 0, 0, 0, 100000.]
    prefix = HEADER.pack(*header) + np.linspace(-1, 1, 28, dtype="<f4").tobytes()
    directory, payload = bytearray(), bytearray()
    position = len(prefix) + len(shapes) * RECORD.size
    for i, shape in enumerate(shapes):
        dtype, group, bits = (CQ, 8, 2) if i == 0 else (FP32, 0, 0)
        blob = bytes(64) if i == 0 else np.zeros(shape, dtype="<f4").tobytes()
        aligned = (position + 63) & ~63
        payload.extend(bytes(aligned - position))
        directory.extend(RECORD.pack(dtype, len(shape), 0, *(list(shape) + [0] * 4)[:4],
                                     aligned, len(blob), group, bits))
        payload.extend(blob)
        position = aligned + len(blob)
    return Archive(prefix + directory + payload)


def test_header_and_named_shapes_must_agree(tiny_archive):
    raw = bytearray(tiny_archive.raw)
    offset = HEADER.size + 28 * 4
    record = list(RECORD.unpack_from(raw, offset))
    # Both [16,8] and [8,16] are valid CQ records with exactly 64 bytes.
    record[3:5] = [8, 16]
    RECORD.pack_into(raw, offset, *record)
    with pytest.raises(ValueError, match="embedding: expected"):
        Archive(raw)


def test_rebuild_checks_metadata_even_when_payload_is_unchanged(tiny_archive):
    embedding = tiny_archive.tensors["embedding"]
    altered = replace(embedding, shape=(8, 16))
    with pytest.raises(ValueError, match="embedding: expected"):
        tiny_archive.rebuild({"embedding": altered})
    with pytest.raises(ValueError, match="replacement name mismatch"):
        tiny_archive.rebuild({"embedding": replace(embedding, name="final_norm")})
    assert tiny_archive.rebuild({"embedding": replace(embedding)}) == tiny_archive.raw


@pytest.mark.parametrize("field,value,message", [
    (10, 3, "RoPE/Hadamard"),
    (12, 16, "RoPE/Hadamard"),
    (29, float("nan"), "rope_theta"),
    (25, 1, "engram layer"),
    (9, 0xFFFFFFFF, "tensor count"),
])
def test_invalid_geometry_is_rejected_before_allocation(tiny_archive, field, value, message):
    raw = bytearray(tiny_archive.raw)
    header = list(HEADER.unpack_from(raw))
    header[field] = value
    HEADER.pack_into(raw, 0, *header)
    with pytest.raises(ValueError, match=message):
        Archive(raw)
