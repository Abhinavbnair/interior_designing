"""Shared save-image-and-metadata helpers.

Extracted out of Phase 1's text_to_image.py so Phase 2 (image-to-image) can reuse it
without duplicating file-writing logic. No torch dependency.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image


def resolve_output_path(output_dir: Path, filename: str) -> Path:
    """Ensure output_dir exists and return the full path for `filename`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def save_image(image: Image.Image, image_path: Path) -> None:
    image.save(image_path)


def write_metadata(result: Any, metadata_path: Path) -> None:
    """Serialize `result` (a dataclass instance) to `metadata_path` as JSON."""
    payload = asdict(result) if is_dataclass(result) else dict(result)
    metadata_path.write_text(json.dumps(payload, indent=2, default=str))
