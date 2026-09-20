#!/usr/bin/env python3
"""Optuna hyperparameter tuning for YOLO models on the tiled dataset.

Tunes hyperparameters for multiple YOLO models sequentially. Each model gets
N Optuna trials; the best trial's hyperparams are used for a final long
training run.

Usage:
    # Full pipeline: tune then train
    CUDA_VISIBLE_DEVICES=0 python optuna_tune.py \
        --data full-data-tile-2/data.yaml \
        --device 0 --n-trials 20 --tune-epochs 30 --final-epochs 200

    # Run specific models only
    CUDA_VISIBLE_DEVICES=0 python optuna_tune.py \
        --models yolov8x-worldv2 yolov26x --device 0

    # Skip tuning, just train with best saved params
    CUDA_VISIBLE_DEVICES=0 python optuna_tune.py --final-only --device 0
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml
from ultralytics import YOLO

MODELS = {
    "yolov8x-worldv2": "yolov8x-worldv2.pt",
    "yolov26x": "yolov26x.pt",
    "yolov8l-worldv2": "yolov8l-worldv2.pt",
    "yolov26l": "yolov26l.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to tune (default: all four)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml for the tiled dataset")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Optuna trials per model for tuning phase")
    parser.add_argument("--tune-epochs", type=int, default=30,
                        help="Epochs per trial during tuning (default: 30)")
    parser.add_argument("--final-epochs", type=int, default=200,
                        help="Epochs for final training with best params (default: 200)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--final-only", action="store_true",
                        help="Skip tuning, run final training with saved best params")
    parser.add_argument("--study-dir", type=str, default="optuna_studies",
                        help="Directory to save study results")
    return parser.parse_args()


def create_objective(model_name: str, weights: str, data: str, device: str,
                     tune_epochs: int, imgsz: int):

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
            "lrf": trial.suggest_float("lrf", 1e-4, 0.1, log=True),
            "momentum": trial.suggest_float("momentum", 0.85, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 1.0, 5.0),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.5, 0.95),
            "box": trial.suggest_float("box", 5.0, 12.0),
            "cls": trial.suggest_float("cls", 0.3, 2.0),
            "dfl": trial.suggest_float("dfl", 0.5, 2.5),
            "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
            "mixup": trial.suggest_float("mixup", 0.0, 0.5),
            "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.5),
            "scale": trial.suggest_float("scale", 0.2, 0.8),
            "degrees": trial.suggest_float("degrees", 0.0, 90.0),
            "translate": trial.suggest_float("translate", 0.05, 0.3),
            "flipud": trial.suggest_float("flipud", 0.0, 0.5),
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.03),
            "hsv_s": trial.suggest_float("hsv_s", 0.2, 0.8),
            "hsv_v": trial.suggest_float("hsv_v", 0.2, 0.8),
            "close_mosaic": trial.suggest_int("close_mosaic", 5, 15),
        }

        if "x" in model_name:
            batch = trial.suggest_categorical("batch", [8, 16, 32])
        else:
            batch = trial.suggest_categorical("batch", [16, 32, 64])

        name = f"optuna_{model_name}_trial{trial.number}"

        try:
            model = YOLO(weights)
            results = model.train(
                data=data,
                epochs=tune_epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=12,
                optimizer="AdamW",
                lr0=params["lr0"],
                lrf=params["lrf"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
                warmup_epochs=params["warmup_epochs"],
                warmup_momentum=params["warmup_momentum"],
                box=params["box"],
                cls=params["cls"],
                dfl=params["dfl"],
                mosaic=params["mosaic"],
                mixup=params["mixup"],
                copy_paste=params["copy_paste"],
                scale=params["scale"],
                degrees=params["degrees"],
                translate=params["translate"],
                fliplr=0.5,
                flipud=params["flipud"],
                hsv_h=params["hsv_h"],
                hsv_s=params["hsv_s"],
                hsv_v=params["hsv_v"],
                close_mosaic=params["close_mosaic"],
                name=name,
                exist_ok=True,
                verbose=False,
            )

            metrics = results.results_dict
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map5095 = metrics.get("metrics/mAP50-95(B)", 0.0)
            score = 0.6 * map50 + 0.4 * map5095

        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {e}", flush=True)
            score = 0.0
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        print(f"  Trial {trial.number}: score={score:.4f} "
              f"(mAP50={map50:.4f}, mAP50-95={map5095:.4f})", flush=True)
        return score

    return objective


def run_final_training(model_name: str, weights: str, best_params: dict,
                       data: str, device: str, final_epochs: int, imgsz: int):
    print(f"\n{'='*60}")
    print(f"FINAL TRAINING: {model_name} for {final_epochs} epochs")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*60}\n", flush=True)

    batch = best_params.pop("batch", 16)
    close_mosaic = int(best_params.pop("close_mosaic", 10))

    model = YOLO(weights)
    results = model.train(
        data=data,
        epochs=final_epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=16,
        optimizer="AdamW",
        lr0=best_params.get("lr0", 5e-4),
        lrf=best_params.get("lrf", 0.01),
        momentum=best_params.get("momentum", 0.937),
        weight_decay=best_params.get("weight_decay", 5e-4),
        warmup_epochs=best_params.get("warmup_epochs", 3.0),
        warmup_momentum=best_params.get("warmup_momentum", 0.8),
        box=best_params.get("box", 7.5),
        cls=best_params.get("cls", 0.5),
        dfl=best_params.get("dfl", 1.5),
        mosaic=best_params.get("mosaic", 1.0),
        mixup=best_params.get("mixup", 0.2),
        copy_paste=best_params.get("copy_paste", 0.2),
        scale=best_params.get("scale", 0.5),
        degrees=best_params.get("degrees", 30.0),
        translate=best_params.get("translate", 0.2),
        fliplr=0.5,
        flipud=best_params.get("flipud", 0.5),
        hsv_h=best_params.get("hsv_h", 0.015),
        hsv_s=best_params.get("hsv_s", 0.5),
        hsv_v=best_params.get("hsv_v", 0.5),
        close_mosaic=close_mosaic,
        name=f"optuna_best_{model_name}_{final_epochs}ep",
        exist_ok=True,
    )

    metrics = results.results_dict
    print(f"\nFINAL RESULTS for {model_name}:")
    print(f"  mAP50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


def main():
    args = parse_args()
    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in args.models:
        weights = MODELS[model_name]
        study_file = study_dir / f"{model_name}_best_params.json"

        if args.final_only:
            if not study_file.exists():
                print(f"SKIP {model_name}: no saved params at {study_file}")
                continue
            best_params = json.loads(study_file.read_text())
            print(f"Loaded saved params for {model_name}", flush=True)
        else:
            print(f"\n{'='*60}")
            print(f"TUNING: {model_name} ({args.n_trials} trials x "
                  f"{args.tune_epochs} epochs)")
            print(f"{'='*60}\n", flush=True)

            study = optuna.create_study(
                direction="maximize",
                study_name=f"tune_{model_name}",
                storage=f"sqlite:///{study_dir / model_name}.db",
                load_if_exists=True,
            )

            objective = create_objective(
                model_name, weights, args.data, args.device,
                args.tune_epochs, args.imgsz,
            )

            study.optimize(objective, n_trials=args.n_trials)

            best_params = study.best_params
            best_value = study.best_value
            print(f"\nBest trial for {model_name}: score={best_value:.4f}")
            print(f"Best params: {json.dumps(best_params, indent=2)}")

            study_file.write_text(json.dumps(best_params, indent=2))
            print(f"Saved to {study_file}", flush=True)

        metrics = run_final_training(
            model_name, weights, best_params.copy(),
            args.data, args.device, args.final_epochs, args.imgsz,
        )
        all_results[model_name] = metrics

    print(f"\n{'='*60}")
    print("OPTUNA TUNING SUMMARY")
    print(f"{'='*60}")
    for model_name, metrics in all_results.items():
        map50 = metrics.get("metrics/mAP50(B)", 0)
        map5095 = metrics.get("metrics/mAP50-95(B)", 0)
        prec = metrics.get("metrics/precision(B)", 0)
        rec = metrics.get("metrics/recall(B)", 0)
        print(f"  {model_name:25s}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
              f"P={prec:.4f}  R={rec:.4f}")

    summary_file = study_dir / "final_results.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()
