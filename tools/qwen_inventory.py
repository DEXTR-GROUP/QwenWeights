#!/usr/bin/env python3
"""Проверка и структурная инвентаризация Qwen2.5-0.5B Base.

Инструмент не зависит от Transformers, PyTorch или safetensors-пакета.
Он читает только заголовок Safetensors, поэтому может проверить структуру
файла весов без загрузки ~1 ГБ тензорных данных в память.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

EXPECTED_MODEL_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECTED_SOURCE_REVISION = "060db64"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    file_size = path.stat().st_size
    if file_size < 8:
        fail("Safetensors-файл короче 8 байт.")

    with path.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            fail("Длина заголовка Safetensors выходит за пределы файла.")
        header_raw = handle.read(header_length)

    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"Заголовок Safetensors не является корректным JSON: {exc}")

    if not isinstance(header, dict):
        fail("Заголовок Safetensors должен быть JSON-объектом.")

    data_start = 8 + header_length
    return header, data_start


def validate_safetensors(path: Path) -> dict[str, Any]:
    header, data_start = read_safetensors_header(path)
    file_size = path.stat().st_size
    data_size = file_size - data_start
    tensors: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []

    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                fail("Поле __metadata__ должно быть объектом.")
            continue

        if not isinstance(descriptor, dict):
            fail(f"Описание тензора {name!r} должно быть объектом.")

        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            fail(f"Неполное описание тензора {name!r}.")
        if len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
            fail(f"Некорректные data_offsets у {name!r}.")

        start, end = offsets
        if start < 0 or end < start or end > data_size:
            fail(f"data_offsets у {name!r} выходят за границы data section.")

        element_count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                fail(f"Некорректная shape у {name!r}: {shape!r}")
            element_count *= dimension

        intervals.append((start, end, name))
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
                "byte_size": end - start,
                "element_count": element_count,
            }
        )

    intervals.sort()
    previous_end = 0
    for start, end, name in intervals:
        if start < previous_end:
            fail(f"Перекрытие data_offsets около тензора {name!r}.")
        previous_end = max(previous_end, end)

    metadata = header.get("__metadata__", {})
    total_parameters = sum(item["element_count"] for item in tensors)
    dtype_counts: dict[str, int] = {}
    for item in tensors:
        dtype_counts[item["dtype"]] = dtype_counts.get(item["dtype"], 0) + item["element_count"]

    return {
        "file_size": file_size,
        "header_length": data_start - 8,
        "data_start": data_start,
        "data_size": data_size,
        "tensor_count": len(tensors),
        "total_parameters": total_parameters,
        "dtype_parameter_counts": dtype_counts,
        "metadata": metadata,
        "tensors": sorted(tensors, key=lambda item: item["name"]),
    }


def tensor_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in report["tensors"]}


def cross_check_config(config: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> list[str]:
    checks: list[str] = []

    required_config = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
    )
    missing = [key for key in required_config if key not in config]
    if missing:
        fail(f"В config.json отсутствуют обязательные поля: {', '.join(missing)}")

    if config["model_type"] != "qwen2":
        fail(f"Ожидался model_type=qwen2, получено {config['model_type']!r}.")

    embed = tensors.get("model.embed_tokens.weight")
    norm = tensors.get("model.norm.weight")
    lm_head = tensors.get("lm_head.weight")
    if embed is None or norm is None:
        fail("Не найдены обязательные model.embed_tokens.weight/model.norm.weight.")

    if embed["shape"] != [config["vocab_size"], config["hidden_size"]]:
        fail(
            "Несоответствие embedding: "
            f"tensor={embed['shape']} config={[config['vocab_size'], config['hidden_size']]}"
        )
    if norm["shape"] != [config["hidden_size"]]:
        fail(f"Несоответствие model.norm.weight: {norm['shape']!r}.")

    if config.get("tie_word_embeddings"):
        if lm_head is not None and lm_head["shape"] != [config["vocab_size"], config["hidden_size"]]:
            fail(f"Несоответствие lm_head.weight: {lm_head['shape']!r}.")

    layer_count = config["num_hidden_layers"]
    for layer_index in range(layer_count):
        prefix = f"model.layers.{layer_index}."
        expected = {
            f"{prefix}input_layernorm.weight": [config["hidden_size"]],
            f"{prefix}self_attn.q_proj.weight": [config["hidden_size"], config["hidden_size"]],
            f"{prefix}self_attn.k_proj.weight": [
                config["num_key_value_heads"] * (config["hidden_size"] // config["num_attention_heads"]),
                config["hidden_size"],
            ],
            f"{prefix}self_attn.v_proj.weight": [
                config["num_key_value_heads"] * (config["hidden_size"] // config["num_attention_heads"]),
                config["hidden_size"],
            ],
            f"{prefix}self_attn.o_proj.weight": [config["hidden_size"], config["hidden_size"]],
            f"{prefix}post_attention_layernorm.weight": [config["hidden_size"]],
            f"{prefix}mlp.gate_proj.weight": [config["intermediate_size"], config["hidden_size"]],
            f"{prefix}mlp.up_proj.weight": [config["intermediate_size"], config["hidden_size"]],
            f"{prefix}mlp.down_proj.weight": [config["hidden_size"], config["intermediate_size"]],
        }
        for name, shape in expected.items():
            tensor = tensors.get(name)
            if tensor is None:
                fail(f"Не найден обязательный тензор: {name}")
            if tensor["shape"] != shape:
                fail(f"Несоответствие {name}: tensor={tensor['shape']} expected={shape}")

    checks.append(f"24/24 слоёв проверены" if layer_count == 24 else f"{layer_count}/{layer_count} слоёв проверены")
    checks.append("embedding соответствует vocab_size × hidden_size")
    checks.append("attention projection shapes соответствуют head/KV-head конфигурации")
    checks.append("MLP projection shapes соответствуют intermediate_size")
    return checks


def build_report(model_dir: Path, expected_sha256: str | None) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        fail("Отсутствуют файлы контрольного набора: " + ", ".join(missing))

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weights = model_dir / "model.safetensors"
    actual_sha256 = sha256_file(weights)
    expected = expected_sha256 or EXPECTED_MODEL_SHA256

    safetensors = validate_safetensors(weights)
    tensors = tensor_map(safetensors)
    checks = cross_check_config(config, tensors)

    if actual_sha256 != expected:
        fail(f"SHA-256 не совпадает: actual={actual_sha256} expected={expected}")

    return {
        "control_object": "Qwen2.5-0.5B Base",
        "source": "Qwen/Qwen2.5-0.5B",
        "source_revision": EXPECTED_SOURCE_REVISION,
        "sha256": actual_sha256,
        "sha256_verified": True,
        "required_files_verified": True,
        "config": config,
        "safetensors": {key: value for key, value in safetensors.items() if key != "tensors"},
        "config_weight_checks": checks,
        "tensors": safetensors["tensors"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Каталог с файлами Qwen2.5-0.5B Base")
    parser.add_argument("--output", type=Path, help="Путь для JSON-отчёта")
    parser.add_argument("--expected-sha256", help="Ожидаемый SHA-256 model.safetensors")
    args = parser.parse_args()

    try:
        report = build_report(args.model_dir, args.expected_sha256)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"W1 FAIL: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print("W1 PASS")
    print(f"Модель: {report['control_object']}")
    print(f"Ревизия источника: {report['source_revision']}")
    print(f"SHA-256: {report['sha256']}")
    print(f"Тензоров: {report['safetensors']['tensor_count']}")
    print(f"Параметров: {report['safetensors']['total_parameters']:,}")
    print(f"Dtype: {report['safetensors']['dtype_parameter_counts']}")
    for check in report["config_weight_checks"]:
        print(f"CHECK: {check}")
    if args.output:
        print(f"Отчёт: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
