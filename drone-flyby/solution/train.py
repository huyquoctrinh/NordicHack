#!/usr/bin/env python3
"""Train a YOLO model on the tiled aerial object detection dataset.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --model yolov8l-worldv2.pt \
        --data full-data-tile-2/data.yaml \
        --epochs 200 --batch 64 --device 0 --imgsz 640
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolov8l-worldv2.pt",
                        help="Pretrained model weights (default: yolov8l-worldv2.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml for the tiled dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: auto-generated from model name)")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lrf", type=float, default=1e-3)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.3)
    parser.add_argument("--copy-paste", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=30.0)
    parser.add_argument("--translate", type=float, default=0.2)
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    model_stem = args.model.replace(".pt", "").replace("/", "_")
    run_name = args.name or f"{model_stem}_{args.epochs}ep"

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        lrf=args.lrf,
        mosaic=args.mosaic,
        name=run_name,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        scale=args.scale,
        fliplr=0.5,
        flipud=0.5,
        degrees=args.degrees,
        shear=0.0,
        translate=args.translate,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.5,
        close_mosaic=args.close_mosaic,
    )

    metrics = results.results_dict
    print(f"\nResults for {run_name}:")
    print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")


if __name__ == "__main__":
    main()
