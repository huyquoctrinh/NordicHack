#!/usr/bin/env python3
"""Build a tiled YOLO dataset by merging multiple sources.

Each source image is split into a grid (default: 2 columns x 1 row, turning
960x540 into two 480x540 tiles). Bounding boxes are clipped to each tile and
kept when enough of the original box is visible. Sources that already have a
train/val split keep it; sources with only train are split randomly.

Supports both YOLO-format sources and the Helsinki JSON annotation format.
Helsinki (3840x2160) gets its own grid (default 8x4) to produce tiles matching
the 480x540 size from the other sources.

Usage:
    python prepare_merged_data.py \
        --sources map-drone-data/yolo data/inference_yolo \
        --helsinki Nordic-AI-Cup-2026/drone-flyby/src/helsinki \
        --output full-data-tile-2 \
        --cols 2 --rows 1 \
        --val-ratio 0.15 --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="YOLO dataset directories to merge (each must have images/ and labels/)",
    )
    parser.add_argument(
        "--helsinki", type=str, default=None,
        help="Helsinki source dir (images/ + annotations/ with JSON). "
             "Gets its own grid (--helsinki-cols x --helsinki-rows).",
    )
    parser.add_argument("--helsinki-cols", type=int, default=8,
                        help="Grid columns for Helsinki 3840x2160 images (default: 8)")
    parser.add_argument("--helsinki-rows", type=int, default=4,
                        help="Grid rows for Helsinki 3840x2160 images (default: 4)")
    parser.add_argument("--helsinki-train", type=int, default=15,
                        help="Number of Helsinki frames for training (rest go to val)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=2, help="Grid columns (default: 2)")
    parser.add_argument("--rows", type=int, default=1, help="Grid rows (default: 1)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Val fraction for sources without a val split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible-fraction", type=float, default=0.25,
                        help="Min fraction of a box that must remain inside a tile")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output directory")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_classes(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data.get("names")
    if isinstance(raw, list):
        return [str(n) for n in raw]
    elif isinstance(raw, dict):
        ids = sorted(int(k) for k in raw)
        return [str(raw.get(i, raw.get(str(i)))) for i in ids]
    raise ValueError(f"Missing or invalid 'names' in {data_yaml}")


def load_helsinki_boxes(annotation_path: Path, class_to_id: dict[str, int]) -> list[Box]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    boxes: list[Box] = []
    for ann in data.get("annotations", []):
        name = ann["object_id"]
        if name not in class_to_id:
            continue
        x1, y1, x2, y2 = map(float, ann["bbox"])
        boxes.append(Box(class_to_id[name], x1, y1, x2, y2))
    return boxes


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0])
        cx, cy, bw, bh = map(float, fields[1:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append(Box(class_id, x1, y1, x2, y2))
    return boxes


def clipped_label(
    box: Box,
    tile: tuple[int, int, int, int],
    min_visible_fraction: float,
) -> str | None:
    left, top, right, bottom = tile
    bx1, by1 = max(0.0, box.x1), max(0.0, box.y1)
    bx2, by2 = max(bx1, box.x2), max(by1, box.y2)
    original_area = (bx2 - bx1) * (by2 - by1)
    if original_area <= 0:
        return None

    x1 = max(bx1, left)
    y1 = max(by1, top)
    x2 = min(bx2, right)
    y2 = min(by2, bottom)
    vw, vh = x2 - x1, y2 - y1
    if vw <= 0 or vh <= 0:
        return None
    if vw * vh / original_area < min_visible_fraction:
        return None

    tw, th = right - left, bottom - top
    cx = ((x1 + x2) / 2 - left) / tw
    cy = ((y1 + y2) / 2 - top) / th
    w = vw / tw
    h = vh / th
    return f"{box.class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def tile_image(
    image_path: Path,
    boxes: list[Box],
    dest: Path,
    split: str,
    prefix: str,
    cols: int,
    rows: int,
    min_visible_fraction: float,
    jpeg_quality: int,
) -> tuple[int, Counter[int]]:
    img_dir = dest / "images" / split
    lbl_dir = dest / "labels" / split
    label_counts: Counter[int] = Counter()

    with Image.open(image_path) as image:
        width, height = image.size
        x_edges = [c * width // cols for c in range(cols + 1)]
        y_edges = [r * height // rows for r in range(rows + 1)]
        for row in range(rows):
            for col in range(cols):
                tile = (x_edges[col], y_edges[row],
                        x_edges[col + 1], y_edges[row + 1])
                stem = f"{prefix}_{image_path.stem}_r{row}_c{col}"
                out_img = img_dir / f"{stem}{image_path.suffix.lower()}"
                out_lbl = lbl_dir / f"{stem}.txt"

                labels = [
                    lbl for box in boxes
                    if (lbl := clipped_label(box, tile, min_visible_fraction)) is not None
                ]
                for lbl in labels:
                    label_counts[int(lbl.split(maxsplit=1)[0])] += 1

                crop = image.crop(tile)
                if out_img.suffix.lower() in {".jpg", ".jpeg"}:
                    if crop.mode not in {"RGB", "L"}:
                        crop = crop.convert("RGB")
                    crop.save(out_img, quality=jpeg_quality, optimize=False)
                else:
                    crop.save(out_img)

                out_lbl.write_text(
                    "\n".join(labels) + ("\n" if labels else ""),
                    encoding="utf-8",
                )
    return cols * rows, label_counts


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.building")

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace")
    if staging.exists():
        shutil.rmtree(staging)
    for split in ("train", "val"):
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_names: list[str] | None = None
    tile_counts = {"train": 0, "val": 0}
    annotation_counts: Counter[int] = Counter()
    rng = random.Random(args.seed)

    for src_idx, src_path_str in enumerate(args.sources, 1):
        src = Path(src_path_str).resolve()
        prefix = f"s{src_idx}"
        print(f"\n=== Source {src_idx}: {src.name} ===")

        data_yaml = src / "data.yaml"
        if data_yaml.exists():
            src_classes = load_classes(data_yaml)
            if class_names is None:
                class_names = src_classes
            elif src_classes != class_names:
                raise ValueError(
                    f"Class mismatch in {data_yaml}: {src_classes} != {class_names}"
                )

        has_splits = (src / "images" / "train").is_dir()

        if has_splits:
            for split in ("train", "val"):
                images = image_files(src / "images" / split)
                if not images:
                    continue
                print(f"  {split}: {len(images)} images")
                for i, img_path in enumerate(images, 1):
                    with Image.open(img_path) as im:
                        w, h = im.size
                    lbl_path = src / "labels" / split / f"{img_path.stem}.txt"
                    boxes = load_yolo_boxes(lbl_path, w, h)
                    n, lc = tile_image(
                        img_path, boxes, staging, split, prefix,
                        args.cols, args.rows,
                        args.min_visible_fraction, args.jpeg_quality,
                    )
                    tile_counts[split] += n
                    annotation_counts.update(lc)
                    if i % 500 == 0 or i == len(images):
                        print(f"    {split}: {i}/{len(images)} tiled", flush=True)
        else:
            img_dir = src / "images"
            if not img_dir.is_dir():
                print(f"  SKIP: no images/ directory in {src}")
                continue
            images = image_files(img_dir)
            if not images:
                print(f"  SKIP: no images found in {img_dir}")
                continue

            shuffled = images.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * args.val_ratio))
            val_set = set(p.stem for p in shuffled[:n_val])
            print(f"  {len(images)} images -> {len(images) - n_val} train, {n_val} val")

            lbl_dir = src / "labels"
            for i, img_path in enumerate(images, 1):
                with Image.open(img_path) as im:
                    w, h = im.size
                split = "val" if img_path.stem in val_set else "train"
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                boxes = load_yolo_boxes(lbl_path, w, h)
                n, lc = tile_image(
                    img_path, boxes, staging, split, prefix,
                    args.cols, args.rows,
                    args.min_visible_fraction, args.jpeg_quality,
                )
                tile_counts[split] += n
                annotation_counts.update(lc)
                if i % 200 == 0 or i == len(images):
                    print(f"    {i}/{len(images)} tiled", flush=True)

    # --- Helsinki source (JSON annotations, separate grid) ---
    if args.helsinki:
        helsinki = Path(args.helsinki).resolve()
        helsinki_imgs = image_files(helsinki / "images")
        if not helsinki_imgs:
            print(f"\nWARN: No images in {helsinki / 'images'}")
        else:
            if class_names is None:
                raise ValueError("Helsinki requires class_names from a YOLO source first")
            class_to_id = {name: i for i, name in enumerate(class_names)}

            shuffled_h = helsinki_imgs.copy()
            random.Random(args.seed).shuffle(shuffled_h)
            n_train_h = min(args.helsinki_train, len(shuffled_h))
            helsinki_splits = {
                "train": sorted(shuffled_h[:n_train_h]),
                "val": sorted(shuffled_h[n_train_h:]),
            }

            h_cols, h_rows = args.helsinki_cols, args.helsinki_rows
            print(f"\n=== Helsinki: {len(helsinki_imgs)} images, "
                  f"grid {h_cols}x{h_rows}, "
                  f"{n_train_h} train / {len(helsinki_imgs) - n_train_h} val ===")

            for split in ("train", "val"):
                for img_path in helsinki_splits[split]:
                    ann_path = helsinki / "annotations" / f"{img_path.stem}.json"
                    if not ann_path.exists():
                        print(f"  SKIP {img_path.name}: no annotation")
                        continue
                    boxes = load_helsinki_boxes(ann_path, class_to_id)
                    n, lc = tile_image(
                        img_path, boxes, staging, split, "helsinki",
                        h_cols, h_rows,
                        args.min_visible_fraction, args.jpeg_quality,
                    )
                    tile_counts[split] += n
                    annotation_counts.update(lc)
                print(f"  {split}: {len(helsinki_splits[split])} images "
                      f"-> {len(helsinki_splits[split]) * h_cols * h_rows} tiles")

    if class_names is None:
        raise ValueError("No data.yaml found in any source -- cannot determine classes")

    data_yaml_content = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(data_yaml_content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for split in ("train", "val"):
        n_img = len(image_files(staging / "images" / split))
        n_lbl = len(list((staging / "labels" / split).glob("*.txt")))
        assert n_img == tile_counts[split], f"{split}: {n_img} images != {tile_counts[split]} expected"
        assert n_lbl == tile_counts[split], f"{split}: {n_lbl} labels != {tile_counts[split]} expected"

    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)

    print(f"\n{'='*50}")
    print(f"Output: {output}")
    print(f"Grid: {args.cols} cols x {args.rows} rows")
    print(f"Tiles: train={tile_counts['train']}, val={tile_counts['val']}")
    print(f"Total annotations: {sum(annotation_counts.values())}")
    for cid in sorted(annotation_counts):
        print(f"  {class_names[cid]}: {annotation_counts[cid]}")


if __name__ == "__main__":
    main()
