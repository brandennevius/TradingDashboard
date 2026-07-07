from __future__ import annotations

import csv
import hashlib
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Chart Setups"
OUTPUT_FILE = ROOT / "data" / "canslim_reference" / "chart_setup_index.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

FIELDNAMES = [
    "image_id",
    "source_image_path",
    "filename",
    "sha256",
    "width",
    "height",
    "source_page",
    "duplicate_of",
    "setup_type",
    "ticker",
    "timeframe",
    "base_quality",
    "volume_notes",
    "relative_strength_notes",
    "buy_point_notes",
    "failure_warnings",
    "model_lesson",
    "outcome_note",
    "confidence",
    "review_status",
]


def png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width, height = struct.unpack(">II", header[16:24])
        return width, height
    return None, None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    images = sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, image in enumerate(images, start=1):
            width, height = png_size(image)
            writer.writerow(
                {
                    "image_id": f"canslim-{index:03d}",
                    "source_image_path": str(image.relative_to(ROOT)),
                    "filename": image.name,
                    "sha256": sha256(image),
                    "width": width or "",
                    "height": height or "",
                    "source_page": "",
                    "duplicate_of": "",
                    "setup_type": "",
                    "ticker": "",
                    "timeframe": "",
                    "base_quality": "",
                    "volume_notes": "",
                    "relative_strength_notes": "",
                    "buy_point_notes": "",
                    "failure_warnings": "",
                    "model_lesson": "",
                    "outcome_note": "",
                    "confidence": "",
                    "review_status": "needs_labeling",
                }
            )

    print(f"Wrote {len(images)} rows to {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
