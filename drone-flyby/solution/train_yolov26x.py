#!/usr/bin/env python3
"""Train YOLOv26x on the tiled aerial object detection dataset.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_yolov26x.py \
        --data full-data-tile-2/data.yaml \
        --epochs 200 --batch 32 --device 0

    # With custom augmentation
    CUDA_VISIBLE_DEVICES=0 python train_yolov26x.py \
        --data full-data-tile-2/data.yaml \
        --epochs 200 --batch 16 --device 0 \
        --optimizer AdamW --lr0 5e-4 \
        --mosaic 1.0 --mixup 0.3 --copy-paste 0.2
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml for the tiled dataset")
    parser.add_argument("--weights", type=str, default="yolov26x.pt",
                        help="Pretrained weights (default: yolov26x.pt)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32, fits on H100 80GB)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: yolov26x_<epochs>ep)")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=1e-3,
                        help="Final learning rate ratio")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.3)
    parser.add_argument("--copy-paste", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=30.0)
    parser.add_argument("--translate", type=float, default=0.2)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--box", type=float, default=7.5,
                        help="Box loss weight")
    parser.add_argument("--cls", type=float, default=0.5,
                        help="Classification loss weight")
    parser.add_argument("--dfl", type=float, default=1.5,
                        help="DFL loss weight")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.name or f"yolov26x_{args.epochs}ep"

    print(f"Training YOLOv26x")
    print(f"  Data:      {args.data}")
    print(f"  Weights:   {args.weights}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  Image sz:  {args.imgsz}")
    print(f"  Device:    {args.device}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  LR:        {args.lr0}")
    print()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        mosaic=args.mosaic,
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
        name=run_name,
    )

    metrics = results.results_dict
    print(f"\n{'='*50}")
    print(f"YOLOv26x Results ({run_name}):")
    print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
