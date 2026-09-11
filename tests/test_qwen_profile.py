#!/usr/bin/env python3
"""Минимальные тесты W3 на искусственном Safetensors-наборе."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen_profile


def bf16_bytes(values: list[float]) -> bytes:
    result = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        result.extend(struct.pack("<H", bits >> 16))
    return bytes(result)


def make_file(path: Path) -> None:
    names = ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
    payloads = [bf16_bytes([1.0, -1.0, 0.0, 2.0]), bf16_bytes([0.5, -0.5])]
    offset = 0
    header: dict[str, object] = {}
    for name, payload in zip(names, payloads):
        header[name] = {
            "dtype": "BF16",
            "shape": [len(payload) // 2],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + b"".join(payloads))


def test_bf16_conversion_exact_for_representable_values() -> None:
    values = qwen_profile.bf16_view(bf16_bytes([1.0, -1.0, 0.0, 2.0]))
    assert np.array_equal(values, np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32))


def test_profile_tensor(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    make_file(path)
    header, data_start = qwen_profile.read_header(path)
    records = qwen_profile.tensor_records(header)
    with path.open("rb") as handle:
        profile = qwen_profile.profile_tensor(handle, data_start, records[0])

    assert profile["name"] == "model.layers.0.input_layernorm.weight"
    assert profile["elements"] == 2
    assert profile["min"] == -0.5
    assert profile["max"] == 0.5
    assert profile["positive"] == 1
    assert profile["negative"] == 1
    assert profile["zero"] == 0


def test_layer_aggregate() -> None:
    profiles = [
        {"name": "model.layers.0.a.weight", "elements": 2, "mean": 1.0, "mean_abs": 1.0, "max_abs": 2.0, "positive": 1, "negative": 1, "zero": 0},
        {"name": "model.layers.0.b.weight", "elements": 2, "mean": 3.0, "mean_abs": 3.0, "max_abs": 4.0, "positive": 2, "negative": 0, "zero": 0},
    ]
    result = qwen_profile.aggregate_layers(profiles)
    assert len(result) == 1
    assert result[0]["layer"] == 0
    assert result[0]["elements"] == 4
    assert result[0]["weighted_mean"] == 2.0
    assert result[0]["max_abs"] == 4.0
