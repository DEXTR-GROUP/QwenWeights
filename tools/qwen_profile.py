#!/usr/bin/env python3
"""Воспроизводимый числовой профиль тензоров Qwen2.5-0.5B.

Инструмент читает реальные данные Safetensors и работает с BF16 без
Transformers/PyTorch. NumPy используется только как вычислительный инструмент
для потоковой обработки отдельных тензоров. Каждый тензор обрабатывается
отдельно, поэтому весь файл весов не загружается в память одновременно.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECTED_REVISION = "060db64"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            fail("Safetensors-файл короче 8 байт.")
        header_length = struct.unpack("<Q", raw)[0]
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            fail("Safetensors-заголовок обрезан.")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"Некорректный заголовок Safetensors: {exc}")
    if not isinstance(header, dict):
        fail("Заголовок Safetensors должен быть объектом JSON.")
    return header, 8 + header_length


def tensor_records(header: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            fail(f"Некорректное описание тензора {name!r}.")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype != "BF16":
            fail(f"W3 ожидает BF16, но {name!r} имеет dtype={dtype!r}.")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            fail(f"Некорректная metadata у {name!r}.")
        records.append({"name": name, "shape": shape, "data_offsets": offsets})
    return sorted(records, key=lambda item: item["name"])


def bf16_view(raw: bytes) -> np.ndarray:
    """Преобразовать little-endian BF16 bytes в float32 без потери BF16-значения."""
    words = np.frombuffer(raw, dtype="<u2")
    bits = words.astype(np.uint32) << 16
    return bits.view("<f4")


def profile_tensor(handle: Any, data_start: int, record: dict[str, Any]) -> dict[str, Any]:
    start, end = record["data_offsets"]
    byte_size = end - start
    if byte_size % 2:
        fail(f"Нечётный размер BF16-тензора {record['name']!r}.")
    handle.seek(data_start + start)
    raw = handle.read(byte_size)
    if len(raw) != byte_size:
        fail(f"Не удалось полностью прочитать {record['name']!r}.")

    values = bf16_view(raw)
    if values.size == 0:
        fail(f"Пустой тензор {record['name']!r} не поддерживается в W3.")
    if not np.all(np.isfinite(values)):
        fail(f"Тензор {record['name']!r} содержит NaN или Inf.")

    abs_values = np.abs(values)
    positive = int(np.count_nonzero(values > 0))
    negative = int(np.count_nonzero(values < 0))
    zero = int(np.count_nonzero(values == 0))
    nonzero = positive + negative
    mean = float(np.mean(values, dtype=np.float64))
    variance = float(np.var(values, dtype=np.float64))
    l2_norm = float(np.linalg.norm(values.astype(np.float64)))
    max_abs = float(np.max(abs_values))
    mean_abs = float(np.mean(abs_values, dtype=np.float64))

    return {
        "name": record["name"],
        "shape": record["shape"],
        "dtype": "BF16",
        "elements": int(values.size),
        "byte_size": byte_size,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": mean,
        "variance": variance,
        "stddev": float(np.sqrt(variance)),
        "l2_norm": l2_norm,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "max_abs_over_mean_abs": float(max_abs / mean_abs) if mean_abs else None,
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "positive_fraction": positive / values.size,
        "negative_fraction": negative / values.size,
        "zero_fraction": zero / values.size,
        "quantiles": {
            "p01": float(np.quantile(values, 0.01)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        },
    }


def layer_index(name: str) -> int | None:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    token = rest.split(".", 1)[0]
    return int(token) if token.isdigit() else None


def aggregate_layers(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for profile in profiles:
        index = layer_index(profile["name"])
        if index is not None:
            groups.setdefault(index, []).append(profile)

    result = []
    for index in sorted(groups):
        items = groups[index]
        total = sum(item["elements"] for item in items)
        weighted_mean = sum(item["mean"] * item["elements"] for item in items) / total
        weighted_abs = sum(item["mean_abs"] * item["elements"] for item in items) / total
        result.append(
            {
                "layer": index,
                "tensor_count": len(items),
                "elements": total,
                "weighted_mean": weighted_mean,
                "weighted_mean_abs": weighted_abs,
                "max_abs": max(item["max_abs"] for item in items),
                "positive_fraction": sum(item["positive"] for item in items) / total,
                "negative_fraction": sum(item["negative"] for item in items) / total,
                "zero_fraction": sum(item["zero"] for item in items) / total,
            }
        )
    return result


def build_report(model_dir: Path, expected_sha256: str) -> dict[str, Any]:
    weights = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    if not weights.is_file() or not config_path.is_file():
        fail("Для W3 нужны config.json и model.safetensors.")

    actual_sha256 = sha256_file(weights)
    if actual_sha256 != expected_sha256:
        fail(f"SHA-256 не совпадает: actual={actual_sha256} expected={expected_sha256}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    header, data_start = read_header(weights)
    records = tensor_records(header)
    profiles = []
    with weights.open("rb") as handle:
        for number, record in enumerate(records, start=1):
            profile = profile_tensor(handle, data_start, record)
            profiles.append(profile)
            print(f"W3 {number}/{len(records)} {record['name']}")

    return {
        "stage": "W3",
        "status": "NUMERIC_PROFILE",
        "control_object": "Qwen2.5-0.5B Base",
        "source": "Qwen/Qwen2.5-0.5B",
        "source_revision": EXPECTED_REVISION,
        "sha256": actual_sha256,
        "sha256_verified": True,
        "numpy_version": np.__version__,
        "config": {
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
        },
        "tensor_count": len(profiles),
        "total_elements": sum(item["elements"] for item in profiles),
        "tensors": profiles,
        "layer_aggregates": aggregate_layers(profiles),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()

    try:
        report = build_report(args.model_dir, args.expected_sha256)
    except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
        print(f"W3 FAIL: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("W3 PASS: числовой профиль построен")
    print(f"Тензоров: {report['tensor_count']}")
    print(f"Параметров: {report['total_elements']:,}")
    print(f"Слоёв: {len(report['layer_aggregates'])}")
    print(f"Отчёт: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
