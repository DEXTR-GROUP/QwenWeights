#!/usr/bin/env python3
"""Получение точного контрольного набора Qwen2.5-0.5B Base.

Загружает файлы напрямую из Hugging Face по зафиксированной ревизии,
не используя Transformers. Основной файл model.safetensors проверяется
по SHA-256 после полной загрузки.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db64"
EXPECTED_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
FILES = (
    ".gitattributes",
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
CHUNK_SIZE = 8 * 1024 * 1024


def download_file(filename: str, destination: Path) -> str:
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}?download=true"
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "QwenWeights/1.0"})

    print(f"Загрузка: {filename}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось загрузить {filename}: {exc}") from exc

    partial.replace(destination)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Каталог для контрольного набора")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename in FILES:
            digest = download_file(filename, args.output_dir / filename)
            if filename == "model.safetensors":
                print(f"SHA-256: {digest}")
                if digest != EXPECTED_SHA256:
                    raise RuntimeError(
                        "SHA-256 model.safetensors не совпал с контрольным значением: "
                        f"{digest} != {EXPECTED_SHA256}"
                    )
                print("SHA-256 model.safetensors: PASS")
    except RuntimeError as exc:
        print(f"W1 FAIL: {exc}", file=sys.stderr)
        return 1

    print("Контрольный набор загружен.")
    print(f"Ревизия: {REVISION}")
    print(f"Каталог: {args.output_dir}")
    print("Следующий шаг: tools/qwen_inventory.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
