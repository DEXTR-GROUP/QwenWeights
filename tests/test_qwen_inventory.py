#!/usr/bin/env python3
"""Минимальные тесты формата инвентаризатора Qwen."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qwen_inventory import validate_safetensors  # noqa: E402


class SafetensorsHeaderTests(unittest.TestCase):
    def make_file(self, header: dict, data: bytes = b"\0" * 8) -> Path:
        directory = Path(self.tmp.name)
        path = directory / "test.safetensors"
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        return path

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_header(self) -> None:
        path = self.make_file(
            {
                "weight": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [0, 8],
                }
            }
        )
        report = validate_safetensors(path)
        self.assertEqual(report["tensor_count"], 1)
        self.assertEqual(report["total_parameters"], 4)
        self.assertEqual(report["tensors"][0]["shape"], [2, 2])

    def test_overlap_is_rejected(self) -> None:
        path = self.make_file(
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
            }
        )
        with self.assertRaises(RuntimeError):
            validate_safetensors(path)

    def test_out_of_bounds_is_rejected(self) -> None:
        path = self.make_file(
            {
                "weight": {
                    "dtype": "F32",
                    "shape": [3],
                    "data_offsets": [0, 12],
                }
            }
        )
        with self.assertRaises(RuntimeError):
            validate_safetensors(path)


if __name__ == "__main__":
    unittest.main()
