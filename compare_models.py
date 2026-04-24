#!/usr/bin/env python3
"""
Compare two YOLO models on the same validation split.

Default comparison:
- Notebook model: Alessandro/train5/weights/best.pt
- Python pipeline model: datasets/taco_hk_yolo26/runs/train_py/weights/best.pt
- Dataset: datasets/taco_hk_yolo26/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

DEFAULT_REFERENCE_WEIGHTS = Path("Alessandro/train5/weights/best.pt")
DEFAULT_CANDIDATE_WEIGHTS = Path("datasets/taco_hk_yolo26/runs/train_py/weights/best.pt")
DEFAULT_DATA_YAML = Path("datasets/taco_hk_yolo26/data.yaml")
DEFAULT_OUTPUT_JSON = Path("model_comparison.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO model quality on the same validation set.")
    parser.add_argument("--reference-weights", type=Path, default=DEFAULT_REFERENCE_WEIGHTS)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="YOLO data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0", help="e.g. 0, cpu")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def must_exist(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def run_val(weights: Path, data_yaml: Path, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
    )
    rd = metrics.results_dict
    return {
        "weights": str(weights),
        "classes": len(model.names),
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
        "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(getattr(metrics, "fitness", 0.0)),
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }


def print_table(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    print("\nModel quality comparison (same validation set)")
    print("| Model | Classes | Precision | Recall | mAP50 | mAP50-95 |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        f"| reference | {reference['classes']} | {reference['precision']:.4f} | "
        f"{reference['recall']:.4f} | {reference['map50']:.4f} | {reference['map50_95']:.4f} |"
    )
    print(
        f"| candidate | {candidate['classes']} | {candidate['precision']:.4f} | "
        f"{candidate['recall']:.4f} | {candidate['map50']:.4f} | {candidate['map50_95']:.4f} |"
    )


def main() -> None:
    args = parse_args()

    reference_weights = must_exist(args.reference_weights, "reference weights")
    candidate_weights = must_exist(args.candidate_weights, "candidate weights")
    data_yaml = must_exist(args.data, "data.yaml")

    reference = run_val(
        weights=reference_weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    candidate = run_val(
        weights=candidate_weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    winner = "candidate" if candidate["map50_95"] >= reference["map50_95"] else "reference"
    delta_map50_95 = candidate["map50_95"] - reference["map50_95"]

    summary = {
        "data_yaml": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "reference": reference,
        "candidate": candidate,
        "winner_by_map50_95": winner,
        "delta_map50_95_candidate_minus_reference": delta_map50_95,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_table(reference=reference, candidate=candidate)
    print(f"\nWinner by mAP50-95: {winner}")
    print(f"Delta (candidate - reference): {delta_map50_95:+.4f}")
    print(f"Saved report: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
