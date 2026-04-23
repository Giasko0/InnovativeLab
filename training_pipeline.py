#!/usr/bin/env python3
"""
Prepare a Hong Kong-focused YOLO dataset from TACO.

Features:
- Downloads TACO annotations and images
- Remaps 60 original classes to a simplified HK-practical taxonomy
- Applies webcam-like train augmentation (brightness/contrast + mild blur)
- Exports YOLO-format labels and data.yaml (YOLO26-compatible layout)
- Validates annotation consistency
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageEnhance


ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"

# Small, demo-friendly taxonomy built around common items a person actually shows to a webcam.
TARGET_CLASSES = [
    "Clear plastic bottle",
    "Other plastic bottle",
    "Drink can",
    "Food Can",
    "Glass bottle",
    "Drink carton",
    "Corrugated carton",
    "Paper bag",
    "Plastic film",
    "Other plastic",
    "Disposable plastic cup",
    "Food waste",
    "Cigarette",
    "Battery",
    "some sort of general waste item",
]

CLASS_REMAP = {
    "Aluminium foil": "some sort of general waste item",
    "Battery": "Battery",
    "Aluminium blister pack": "some sort of general waste item",
    "Carded blister pack": "some sort of general waste item",
    "Other plastic bottle": "Other plastic bottle",
    "Clear plastic bottle": "Clear plastic bottle",
    "Glass bottle": "Glass bottle",
    "Metal bottle cap": "some sort of general waste item",
    "Broken glass": "some sort of general waste item",
    "Food Can": "Food Can",
    "Aerosol": "some sort of general waste item",
    "Drink can": "Drink can",
    "Toilet tube": "Corrugated carton",
    "Other carton": "Corrugated carton",
    "Egg carton": "Corrugated carton",
    "Drink carton": "Drink carton",
    "Corrugated carton": "Corrugated carton",
    "Meal carton": "some sort of general waste item",
    "Pizza box": "some sort of general waste item",
    "Paper cup": "some sort of general waste item",
    "Disposable plastic cup": "Disposable plastic cup",
    "Foam cup": "some sort of general waste item",
    "Glass cup": "Glass bottle",
    "Other plastic cup": "Disposable plastic cup",
    "Food waste": "Food waste",
    "Glass jar": "Glass bottle",
    "Plastic lid": "some sort of general waste item",
    "Metal lid": "some sort of general waste item",
    "Other plastic": "Other plastic",
    "Magazine paper": "Corrugated carton",
    "Tissues": "some sort of general waste item",
    "Wrapping paper": "Corrugated carton",
    "Normal paper": "Corrugated carton",
    "Paper bag": "Paper bag",
    "Plastified paper bag": "some sort of general waste item",
    "Plastic film": "Plastic film",
    "Six pack rings": "Plastic film",
    "Garbage bag": "Plastic film",
    "Other plastic wrapper": "Plastic film",
    "Single-use carrier bag": "Plastic film",
    "Polypropylene bag": "Plastic film",
    "Crisp packet": "Plastic film",
    "Spread tub": "Disposable plastic cup",
    "Tupperware": "Disposable plastic cup",
    "Disposable food container": "Disposable plastic cup",
    "Foam food container": "some sort of general waste item",
    "Other plastic container": "Disposable plastic cup",
    "Plastic glooves": "some sort of general waste item",
    "Plastic utensils": "some sort of general waste item",
    "Pop tab": "some sort of general waste item",
    "Rope & strings": "some sort of general waste item",
    "Scrap metal": "some sort of general waste item",
    "Shoe": "some sort of general waste item",
    "Squeezable tube": "Other plastic",
    "Plastic straw": "some sort of general waste item",
    "Paper straw": "some sort of general waste item",
    "Styrofoam piece": "some sort of general waste item",
    "Unlabeled litter": "some sort of general waste item",
    "Cigarette": "Cigarette",
}


@dataclass(slots=True)
class ExportStats:
    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0
    augmented_images: int = 0
    skipped_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HK-focused YOLO26 dataset from TACO")
    parser.add_argument("--output-dir", type=Path, default=Path("Code/datasets/taco_hk_yolo26"))
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw TACO cache dir (default: <output-dir>/raw)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    return parser.parse_args()


def download_annotations(raw_dir: Path, timeout_s: float) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = raw_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Downloading annotations -> {annotations_path}")
        response = requests.get(ANNOTATIONS_URL, timeout=timeout_s)
        response.raise_for_status()
        annotations_path.write_text(response.text, encoding="utf-8")
    return json.loads(annotations_path.read_text(encoding="utf-8"))


def _download_one_image(
    image_entry: dict[str, Any],
    images_root: Path,
    timeout_s: float,
) -> tuple[int, bool]:
    image_id = int(image_entry["id"])
    rel_path = image_entry["file_name"]
    dest = images_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return image_id, True

    candidates = [image_entry.get("flickr_url"), image_entry.get("flickr_640_url")]
    for url in candidates:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code != 200 or not r.content:
                continue
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(dest, quality=95)
            return image_id, True
        except Exception:
            continue

    return image_id, False


def download_images(images: list[dict[str, Any]], images_root: Path, timeout_s: float) -> set[int]:
    images_root.mkdir(parents=True, exist_ok=True)
    ok_ids: set[int] = set()
    total = len(images)
    failures = 0
    for i, image_entry in enumerate(images, start=1):
        image_id, ok = _download_one_image(image_entry, images_root, timeout_s)
        if ok:
            ok_ids.add(image_id)
        else:
            failures += 1
        if i % 100 == 0 or i == total:
            print(f"Downloaded {i}/{total} images (failed: {failures})")

    return ok_ids


def clip_bbox_to_image(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None

    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None

    x1 = max(0.0, min(float(img_w), float(x)))
    y1 = max(0.0, min(float(img_h), float(y)))
    x2 = max(0.0, min(float(img_w), float(x + w)))
    y2 = max(0.0, min(float(img_h), float(y + h)))

    cw = x2 - x1
    ch = y2 - y1
    if cw <= 1.0 or ch <= 1.0:
        return None
    return x1, y1, cw, ch


def to_yolo_bbox(x1: float, y1: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xc = (x1 + w / 2.0) / img_w
    yc = (y1 + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn


def build_remapped_annotations(data: dict[str, Any]) -> tuple[dict[int, list[tuple[int, float, float, float, float]]], dict[int, dict[str, Any]], Counter]:
    class_to_idx = {name: i for i, name in enumerate(TARGET_CLASSES)}
    categories = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    images_by_id = {int(img["id"]): img for img in data["images"]}
    labels_by_image: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    class_counter: Counter = Counter()

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        img = images_by_id.get(image_id)
        if img is None:
            continue

        original_name = categories.get(int(ann["category_id"]))
        mapped_name = CLASS_REMAP.get(original_name)
        if not mapped_name:
            continue

        bbox = clip_bbox_to_image(ann.get("bbox", []), int(img["width"]), int(img["height"]))
        if bbox is None:
            continue

        class_idx = class_to_idx[mapped_name]
        yolo_bbox = to_yolo_bbox(*bbox, int(img["width"]), int(img["height"]))
        labels_by_image[image_id].append((class_idx, *yolo_bbox))
        class_counter[mapped_name] += 1

    return labels_by_image, images_by_id, class_counter


def split_ids(
    ids: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def sanitize_stem(image_id: int, rel_path: str) -> str:
    clean = rel_path.replace("\\", "/").replace("/", "__")
    stem = Path(clean).stem
    return f"{image_id}_{stem}"


def write_label_file(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def apply_webcam_augmentation(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    return ImageEnhance.Contrast(
        ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.20))
    ).enhance(rng.uniform(0.90, 1.10))


def export_split(
    split_name: str,
    split_ids: list[int],
    images_by_id: dict[int, dict[str, Any]],
    labels_by_image: dict[int, list[tuple[int, float, float, float, float]]],
    raw_images_root: Path,
    output_dir: Path,
    stats: ExportStats,
    rng: random.Random,
) -> None:
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_id in split_ids:
        image_meta = images_by_id[image_id]
        labels = labels_by_image[image_id]
        src = raw_images_root / image_meta["file_name"]
        if not src.exists() or not labels:
            stats.skipped_images += 1
            continue

        stem = sanitize_stem(image_id, image_meta["file_name"])
        image_out = images_out / f"{stem}.jpg"
        label_out = labels_out / f"{stem}.txt"

        img = Image.open(src).convert("RGB")
        img.save(image_out, quality=95)
        write_label_file(label_out, labels)

        if split_name == "train":
            stats.train_images += 1
            stats.train_labels += len(labels)
        else:
            stats.val_images += 1
            stats.val_labels += len(labels)

        if split_name != "train":
            continue

        aug_img = apply_webcam_augmentation(image=img, rng=rng)
        aug_image_out = images_out / f"{stem}__aug.jpg"
        aug_label_out = labels_out / f"{stem}__aug.txt"
        aug_img.save(aug_image_out, quality=95)
        write_label_file(aug_label_out, labels)
        stats.augmented_images += 1
        stats.train_images += 1
        stats.train_labels += len(labels)


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(TARGET_CLASSES)}",
        "names:",
    ]
    lines.extend([f"  - {name}" for name in TARGET_CLASSES])
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_consistency_checks(output_dir: Path) -> dict[str, Any]:
    problems: list[str] = []
    counts: dict[str, dict[str, int]] = {}

    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split

        image_files = sorted(image_dir.glob("*.jpg"))
        label_files = sorted(label_dir.glob("*.txt"))
        counts[split] = {"images": len(image_files), "labels": len(label_files), "boxes": 0}

        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        if missing_labels:
            problems.append(f"{split}: {len(missing_labels)} images missing labels")
        if missing_images:
            problems.append(f"{split}: {len(missing_images)} labels missing images")

        for label_path in label_files:
            for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{label_path}:{line_idx} does not have 5 columns")
                    continue
                try:
                    cls_idx = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    problems.append(f"{label_path}:{line_idx} has invalid number format")
                    continue

                if cls_idx < 0 or cls_idx >= len(TARGET_CLASSES):
                    problems.append(f"{label_path}:{line_idx} class index out of range ({cls_idx})")
                xc, yc, w, h = coords
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    problems.append(f"{label_path}:{line_idx} invalid normalized bbox ({coords})")
                counts[split]["boxes"] += 1

    report = {"ok": not problems, "counts": counts, "problems": problems}
    (output_dir / "consistency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    t0 = time.time()

    output_dir = args.output_dir.resolve()
    raw_dir = (args.raw_dir if args.raw_dir is not None else output_dir / "raw").resolve()
    raw_images_root = raw_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/6 - Download/load annotations")
    data = download_annotations(raw_dir=raw_dir, timeout_s=args.request_timeout)

    print("Step 2/6 - Download TACO images")
    image_entries = list(data["images"])
    downloaded_ids = download_images(images=image_entries, images_root=raw_images_root, timeout_s=args.request_timeout)
    print(f"Images available locally: {len(downloaded_ids)}")

    print("Step 3/6 - Remap classes and convert boxes")
    labels_by_image, images_by_id, class_counter = build_remapped_annotations(data)
    kept_ids = [image_id for image_id, labels in labels_by_image.items() if labels and image_id in downloaded_ids]
    if not kept_ids:
        raise RuntimeError("No images left after remapping/filtering. Check connectivity and mapping rules.")
    print(f"Images with mapped annotations: {len(kept_ids)}")

    print("Step 4/6 - Train/val split")
    train_ids, val_ids = split_ids(ids=kept_ids, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

    print("Step 5/6 - Export YOLO dataset with augmentation")
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    if images_root.exists():
        shutil.rmtree(images_root)
    if labels_root.exists():
        shutil.rmtree(labels_root)

    stats = ExportStats()
    rng = random.Random(args.seed)
    export_split(
        split_name="train",
        split_ids=train_ids,
        images_by_id=images_by_id,
        labels_by_image=labels_by_image,
        raw_images_root=raw_images_root,
        output_dir=output_dir,
        stats=stats,
        rng=rng,
    )
    export_split(
        split_name="val",
        split_ids=val_ids,
        images_by_id=images_by_id,
        labels_by_image=labels_by_image,
        raw_images_root=raw_images_root,
        output_dir=output_dir,
        stats=stats,
        rng=rng,
    )

    write_data_yaml(output_dir)
    (output_dir / "class_distribution.json").write_text(
        json.dumps(dict(sorted(class_counter.items(), key=lambda kv: kv[0])), indent=2),
        encoding="utf-8",
    )

    print("Step 6/6 - Consistency checks")
    report = run_consistency_checks(output_dir)
    if not report["ok"]:
        print("Consistency issues found:")
        for p in report["problems"][:20]:
            print(f" - {p}")
        raise RuntimeError("Consistency checks failed. See consistency_report.json for details.")

    elapsed = time.time() - t0
    print("\nDone.")
    print(f"Output dataset: {output_dir}")
    print(f"Train images: {stats.train_images} (augmented: {stats.augmented_images})")
    print(f"Val images:   {stats.val_images}")
    print(f"Train boxes:  {stats.train_labels}")
    print(f"Val boxes:    {stats.val_labels}")
    print(f"Classes kept: {len([c for c in TARGET_CLASSES if class_counter.get(c, 0) > 0])}/{len(TARGET_CLASSES)}")
    print(f"Elapsed:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
