# Nordic AI Cup 2026 — Aerial Object Detection

Drone-based object detection pipeline for the Nordic AI Cup 2026 competition. Includes data preparation (merge + tile), hyperparameter tuning (Optuna), and YOLO training.

## Object Classes (16)

hangar, helicopter, jet_plane, large_launcher, large_tower, medium_launcher, medium_plane, mine_roller, small_launcher, small_plane, small_tower, ta-ta, tank, condor, jammer, spacecraft

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA-capable GPU(s).

## Pipeline

### 0. Generate Map-Drone Data (crawl_map_drone_data.py)

Generates the synthetic map-drone training dataset end-to-end. Reproduces the pipeline that creates ~6600 training images by:
1. Extracting object sprites from Helsinki reference frames (GrabCut + colour clustering)
2. Fetching aerial basemap tiles from Esri World Imagery for 22 Nordic sites
3. Compositing objects onto new terrain as simulated 600m drone surveys
4. Exporting 960x540 YOLO-format view crops
5. Then map the object sprite to the random position, and then scaling via image ratio and object mask (by using contour detection) to get the bounding boxes coordinate.

```bash
# Set path to competition repo's Helsinki scene
export NAIC_HELSINKI=/path/to/Nordic-AI-Cup-2026/drone-flyby/src/helsinki

# Run full pipeline
python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --output map-drone-data

# Run individual steps
python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step sprites
python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step basemaps
python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step scenes
python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step export
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--naic-helsinki` | (required) | Path to Helsinki reference scene dir |
| `--output` | map-drone-data/ | Output root directory |
| `--step` | all | Run one step or all: sprites/basemaps/scenes/export |
| `--basemap-source` | esri | Basemap tile provider |
| `--tile-cache` | /tmp/tilecache | Cache dir for downloaded tiles |
| `--per-class` | 2 | Object instances per class per scene |
| `--workers` | 8 | Parallel workers for scene generation |
| `--yolo-levels` | 0 1 2 | View resolution levels for YOLO export |

### 1. Prepare Data (Merge + Tile)

Merges multiple YOLO-format datasets and the Helsinki JSON-annotated dataset into a single tiled dataset. Images are split into a grid to normalize resolution:

- **960×540 images** → 2×1 grid → two 480×540 tiles
- **3840×2160 Helsinki images** → 8×4 grid → 32 tiles of 480×540 each

```bash
python prepare_merged_data.py \
    --sources /path/to/map-drone-data/yolo /path/to/other_yolo_source \
    --helsinki /path/to/helsinki \
    --output full-data-tile-2 \
    --cols 2 --rows 1 \
    --helsinki-cols 8 --helsinki-rows 4 \
    --helsinki-train 15 \
    --val-ratio 0.15 \
    --overwrite
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--sources` | (required) | YOLO dataset dirs (images/ + labels/ + data.yaml) |
| `--helsinki` | None | Helsinki dir (images/ + annotations/ with JSON) |
| `--output` | (required) | Output directory for tiled dataset |
| `--cols` | 2 | Grid columns for YOLO sources |
| `--rows` | 1 | Grid rows for YOLO sources |
| `--helsinki-cols` | 8 | Grid columns for Helsinki images |
| `--helsinki-rows` | 4 | Grid rows for Helsinki images |
| `--helsinki-train` | 15 | Number of Helsinki images for train (rest → val) |
| `--val-ratio` | 0.15 | Val split ratio for sources without train/val dirs |
| `--min-visible-fraction` | 0.25 | Min fraction of bbox visible after clipping |
| `--overwrite` | False | Replace existing output directory |

### 2. Train a Single Model

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model yolov8l-worldv2.pt \
    --data full-data-tile-2/data.yaml \
    --epochs 200 --batch 64 --device 0 --imgsz 640
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | yolov8l-worldv2.pt | Pretrained YOLO weights |
| `--data` | (required) | Path to data.yaml |
| `--epochs` | 200 | Training epochs |
| `--imgsz` | 640 | Input image size |
| `--batch` | 64 | Batch size |
| `--device` | 0 | GPU device |
| `--mosaic` | 1.0 | Mosaic augmentation probability |
| `--mixup` | 0.3 | MixUp augmentation probability |
| `--copy-paste` | 0.2 | Copy-paste augmentation probability |
| `--degrees` | 30.0 | Random rotation degrees |

### 3. Train YOLOv26x (train_yolov26x.py)

Dedicated training script for YOLOv26x with tuned defaults for aerial detection.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolov26x.py \
    --data full-data-tile-2/data.yaml \
    --epochs 200 --batch 32 --device 0

# With custom hyperparameters
CUDA_VISIBLE_DEVICES=0 python train_yolov26x.py \
    --data full-data-tile-2/data.yaml \
    --epochs 200 --batch 16 --device 0 \
    --optimizer AdamW --lr0 5e-4 \
    --mosaic 1.0 --mixup 0.3 --copy-paste 0.2
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data` | (required) | Path to data.yaml |
| `--weights` | yolov26x.pt | Pretrained weights |
| `--epochs` | 200 | Training epochs |
| `--batch` | 32 | Batch size (fits H100 80GB) |
| `--optimizer` | AdamW | Optimizer |
| `--lr0` | 5e-4 | Initial learning rate |
| `--box` | 7.5 | Box loss weight |
| `--cls` | 0.5 | Classification loss weight |
| `--dfl` | 1.5 | DFL loss weight |

### 4. Optuna Hyperparameter Tuning

Tunes 20 hyperparameters per model using Optuna (TPE sampler), then runs a final training with the best configuration.

```bash
CUDA_VISIBLE_DEVICES=0 python optuna_tune.py \
    --data full-data-tile-2/data.yaml \
    --device 0 \
    --n-trials 20 \
    --tune-epochs 50 \
    --final-epochs 200 \
    --models yolov8x-worldv2 yolov26x yolov8l-worldv2 yolov26l
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data` | (required) | Path to data.yaml |
| `--models` | all four | Models to tune |
| `--n-trials` | 20 | Optuna trials per model |
| `--tune-epochs` | 30 | Epochs per trial |
| `--final-epochs` | 200 | Epochs for final training |
| `--imgsz` | 640 | Input image size |
| `--study-dir` | optuna_studies/ | Directory for Optuna SQLite DBs + best params |
| `--final-only` | False | Skip tuning, use saved best params |

**Tuned hyperparameters:** lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, box/cls/dfl loss weights, mosaic, mixup, copy_paste, scale, degrees, translate, flipud, hsv_h/s/v, close_mosaic, batch size.

**Scoring:** `0.6 × mAP50 + 0.4 × mAP50-95`

**Supported models:**
- `yolov8x-worldv2` — YOLOv8 X-Large World
- `yolov8l-worldv2` — YOLOv8 Large World
- `yolov26x` — YOLOv26 X-Large
- `yolov26l` — YOLOv26 Large

## Data Sources

| Source | Resolution | # Images | Format |
|---|---|---|---|
| map-drone-data | 960×540 | ~6600 (train+val) | YOLO |
| Helsinki drone-flyby | 3840×2160 | 25 | JSON (bbox: [x1,y1,x2,y2]) |

## Results

Best results on the tiled dataset (480×540 tiles):

| Model | mAP50 | Epochs |
|---|---|---|
| yolov26s | 0.834 | 32 |
| yolov26x-world | 0.787 | 170 |
| yolov8x-world | 0.771 | 200 |
