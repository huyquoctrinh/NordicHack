"""Full-frame detection API: same /predict protocol as api.py, no tile extraction.

api.py splits each incoming view into a patch grid and merges the results. This
serves the transmitted view in a single forward pass instead, which is the right
shape for a checkpoint trained on whole views rather than tiles -- for example
ckpts/best_drone_v26x.pt, whose train_args record imgsz=960 on full images.

Defaults to port 5060 so it can run alongside the tiled API on port 80.

Run (from the drone-flyby directory):

    $env:DRONE_MODEL_PATH = "...\\sol\\ckpts\\best_drone_v26x.pt"
    .\\sol\\.venv_train\\Scripts\\python.exe -B api_full.py

Environment:
    DRONE_MODEL_PATH  checkpoint to serve (required in practice)
    DRONE_PORT        default 5060
    DRONE_HOST        default 0.0.0.0
    DRONE_IMGSZ       default 960 -- match the checkpoint's training imgsz
    DRONE_CONF        default 0.05
    DRONE_NMS_IOU     default 0.7
    DRONE_MAX_DET     default 300
    DRONE_DEVICE      default "0" when CUDA is available, else "cpu"
    DRONE_SAVE_IMAGES default 1
    DRONE_CLIP        1 enables confidence rescaling (same as --clip)
    DRONE_CLIP_MIN    default 0.5    lower bound of the rescaled band
    DRONE_CLIP_MAX    default 0.999  upper bound, kept strictly below 1

Confidence clipping (--clip):
    Rescales every confidence from [conf_floor, 1.0] onto (CLIP_MIN, CLIP_MAX], so
    nothing is reported below 0.5 and nothing reaches 1.0. The map is linear and
    therefore monotonic: a detection that scored higher before still scores higher
    after. That ordering is what average-precision integrates over, so a flat clamp
    (every low score pinned to one value) would throw away the ranking and can only
    hurt AP. This keeps it.
"""
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
from threading import Lock
import time

from fastapi import FastAPI, HTTPException
import uvicorn

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'sol'))

from dtos import (OBJECT_CLASSES, SOURCE_REGION_SIZES, IMAGE_WIDTH, IMAGE_HEIGHT,
                  DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto,
                  DroneFlybyPredictionDto, RequestedViewDto)
from utils import decode_view, view_bbox_to_global, clip_bbox_to_frame, validate_response

logger = logging.getLogger(__name__)


class CameraPlanner:
    """Raster-sweep the source frame at a fixed resolution level.

    At L0 the whole frame arrives downscaled 4x, so a challenge object is about
    13 px across -- indistinguishable from a car. L1 halves that downscale and L2
    removes it entirely, at the cost of seeing only part of the frame per request.
    The sweep walks a boustrophedon grid of centres that tiles the frame, so every
    region is revisited on a fixed period.

    Every command is clamped to the constraints that arrived with the request:
    the per-level centre bounds and maximum_center_delta. A command the evaluator
    rejects comes back as camera_command_feedback, and we fall back to the current
    position rather than repeating it.
    """

    def __init__(self, level):
        self.level = level
        self.index = 0
        self.rejects = 0
        # Limits learned from rejections, per level the camera was sitting at.
        # The evaluator enforces a stricter per-level cap than the
        # maximum_center_delta that arrives in the request, and it rejects a move
        # equal to the cap, so we learn the real ceiling instead of trusting one.
        self.limits = {}
        self.last_sent_from_level = None
        self.last_distance = 0.0

    @staticmethod
    def _targets(level, bounds):
        """Centres that tile the frame at this level, in boustrophedon order."""
        region_w, region_h = SOURCE_REGION_SIZES[level]
        lo_x, hi_x = bounds.minimum_center_x, bounds.maximum_center_x
        lo_y, hi_y = bounds.minimum_center_y, bounds.maximum_center_y
        cols = max(1, math.ceil(IMAGE_WIDTH / region_w))
        rows = max(1, math.ceil(IMAGE_HEIGHT / region_h))
        xs = [lo_x] if cols == 1 else [round(lo_x + (hi_x - lo_x) * i / (cols - 1)) for i in range(cols)]
        ys = [lo_y] if rows == 1 else [round(lo_y + (hi_y - lo_y) * i / (rows - 1)) for i in range(rows)]
        out = []
        for r, y in enumerate(ys):
            row = xs if r % 2 == 0 else list(reversed(xs))   # serpentine: no long jump back
            out.extend((int(x), int(y)) for x in row)
        return out

    def _usable_limit(self, constraints, current_level):
        declared = float(getattr(constraints, 'maximum_center_delta', 0) or 0)
        # Never ask to move further than one view height. That is a sane sweep step
        # regardless of the rules (a longer jump would skip ground anyway), and it
        # sits under the evaluator's real per-level cap, so the first command is
        # legal instead of being spent discovering the ceiling.
        region_cap = float(min(SOURCE_REGION_SIZES[current_level]))
        limit = min(declared if declared > 0 else float('inf'),
                    region_cap,
                    self.limits.get(current_level, float('inf')))
        if not math.isfinite(limit):
            return float('inf')
        # The cap is exclusive, so stay strictly under it with a pixel of slack.
        return max(0.0, min(limit - 1.0, limit * 0.98))

    def _step_towards(self, current, target, usable):
        """Largest integer step from current towards target with length < usable."""
        cx, cy = current
        dx, dy = target[0] - cx, target[1] - cy
        distance = math.hypot(dx, dy)
        if distance <= usable:
            return int(target[0]), int(target[1]), distance, True
        scale = usable / distance
        for _ in range(64):
            nx, ny = int(round(cx + dx * scale)), int(round(cy + dy * scale))
            moved = math.hypot(nx - cx, ny - cy)
            if moved <= usable:
                return nx, ny, moved, False
            scale *= 0.98
        return int(cx), int(cy), 0.0, False

    def next_view(self, request):
        constraints = request.camera_constraints
        allowed = list(getattr(constraints, 'allowed_resolution_levels', []) or [])
        level = self.level if (not allowed or self.level in allowed) else max(allowed)
        if not hasattr(constraints, 'bounds_for_level'):
            return None
        bounds = constraints.bounds_for_level(level)
        if bounds is None:
            return None

        current_level = int(getattr(request.view, 'resolution_level', level))
        current_x, current_y = int(request.view.center_x), int(request.view.center_y)

        # A rejection means our ceiling for that level was too high. Shrink it to
        # below what was refused and retry the same target, rather than skipping on.
        if getattr(request, 'camera_command_feedback', None) is not None:
            self.rejects += 1
            if self.last_sent_from_level is not None and self.last_distance > 0:
                previous = self.limits.get(self.last_sent_from_level, float('inf'))
                self.limits[self.last_sent_from_level] = min(previous, self.last_distance * 0.9)

        usable = self._usable_limit(constraints, current_level)

        # Change level without moving. A centre valid at one level is valid at the
        # levels below it, so this is always a legal zero-distance command -- and it
        # keeps a level change from being bundled with a long traverse.
        if current_level != level:
            target_x = min(max(current_x, bounds.minimum_center_x), bounds.maximum_center_x)
            target_y = min(max(current_y, bounds.minimum_center_y), bounds.maximum_center_y)
            target_x, target_y, moved, _ = self._step_towards(
                (current_x, current_y), (target_x, target_y), usable)
            self.last_sent_from_level, self.last_distance = current_level, moved
            return RequestedViewDto(resolution_level=int(level),
                                    center_x=int(target_x), center_y=int(target_y))

        targets = self._targets(level, bounds)
        if not targets:
            return None
        target = targets[self.index % len(targets)]
        # Already parked on the target: take the next one.
        if math.hypot(target[0] - current_x, target[1] - current_y) < 1.0:
            self.index += 1
            target = targets[self.index % len(targets)]

        target_x, target_y, moved, arrived = self._step_towards(
            (current_x, current_y), target, usable)
        if arrived:
            self.index += 1
        target_x = int(min(max(target_x, bounds.minimum_center_x), bounds.maximum_center_x))
        target_y = int(min(max(target_y, bounds.minimum_center_y), bounds.maximum_center_y))
        if target_x == current_x and target_y == current_y:
            return None                      # nothing legal to ask for this frame
        self.last_sent_from_level, self.last_distance = current_level, moved
        return RequestedViewDto(resolution_level=int(level),
                                center_x=target_x, center_y=target_y)


class DetectionMemory:
    """Carry detections across frames so a partial view still answers for the frame.

    The protocol scores the whole source frame, and explicitly allows reporting an
    object the camera is no longer pointed at. Inside the region currently visible
    the fresh detections are authoritative -- anything stale there is dropped, so a
    false positive does not persist forever. Outside it, earlier detections age.
    """

    def __init__(self, ttl_frames=60, max_items=500, iou=0.5):
        self.ttl = ttl_frames
        self.max_items = max_items
        self.iou = iou
        self.store = {}

    @staticmethod
    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(1e-9, area_a + area_b - inter)

    def update(self, sequence_id, frame_index, region_norm, fresh):
        kept = []
        for item in self.store.get(sequence_id, []):
            if frame_index - item['seen'] > self.ttl:
                continue
            box = item['bbox']
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            inside = (region_norm[0] <= centre[0] <= region_norm[2]
                      and region_norm[1] <= centre[1] <= region_norm[3])
            if inside:
                continue          # this region was just re-observed; fresh wins
            kept.append(item)
        for bbox, confidence, object_id in fresh:
            kept.append({'bbox': bbox, 'confidence': confidence,
                         'object_id': object_id, 'seen': frame_index})

        kept.sort(key=lambda i: -i['confidence'])
        merged = []
        for item in kept:
            if any(other['object_id'] == item['object_id']
                   and self._iou(other['bbox'], item['bbox']) >= self.iou
                   for other in merged):
                continue
            merged.append(item)
            if len(merged) >= self.max_items:
                break
        self.store[sequence_id] = merged
        return merged


class FullFrameModelService:
    """One forward pass over the whole transmitted view. No patching, no merging."""

    def __init__(self):
        import torch
        from ultralytics import YOLO

        default = ROOT / 'sol/ckpts/best_drone_v26x.pt'
        self.path = Path(os.environ.get('DRONE_MODEL_PATH', str(default))).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {self.path}')
        self.device = os.environ.get('DRONE_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
        self.tiled = False
        self.options = dict(
            device=self.device,
            # 960 matches the transmitted view, so a view-trained checkpoint sees
            # its native scale with no resampling.
            imgsz=int(os.environ.get('DRONE_IMGSZ', '960')),
            conf=float(os.environ.get('DRONE_CONF', '0.05')),
            iou=float(os.environ.get('DRONE_NMS_IOU', '0.7')),
            max_det=int(os.environ.get('DRONE_MAX_DET', '300')),
            verbose=False,
        )
        if self.options['imgsz'] % 32:
            raise ValueError('DRONE_IMGSZ must be divisible by 32')
        if not 0 <= self.options['conf'] <= 1 or not 0 < self.options['iou'] <= 1:
            raise ValueError('Invalid DRONE_CONF or DRONE_NMS_IOU')
        if self.options['max_det'] < 1:
            raise ValueError('DRONE_MAX_DET must be positive')
        self.clip = os.environ.get('DRONE_CLIP', '0') == '1'
        self.clip_min = float(os.environ.get('DRONE_CLIP_MIN', '0.5'))
        self.clip_max = float(os.environ.get('DRONE_CLIP_MAX', '0.999'))
        if not 0 < self.clip_min < self.clip_max < 1:
            raise ValueError('Require 0 < DRONE_CLIP_MIN < DRONE_CLIP_MAX < 1')
        # Camera sweep. 0 disables it and the camera is left where it is (the
        # previous behaviour), which keeps every object at the L0 4x downscale.
        # The evaluator runs no NMS: two boxes on one object means one true
        # positive and one false positive, so duplicates must go before we answer.
        self.dedupe_iou = float(os.environ.get('DRONE_DEDUPE_IOU', '0.5'))
        if not 0 < self.dedupe_iou <= 1:
            raise ValueError('DRONE_DEDUPE_IOU must be in (0, 1]')
        self.sweep_level = int(os.environ.get('DRONE_SWEEP_LEVEL', '0'))
        if self.sweep_level not in (0, 1, 2):
            raise ValueError('DRONE_SWEEP_LEVEL must be 0, 1 or 2')
        self.memory_ttl = int(os.environ.get('DRONE_MEMORY_TTL', '60'))
        if self.memory_ttl < 0:
            raise ValueError('DRONE_MEMORY_TTL must be non-negative')
        self.planner = CameraPlanner(self.sweep_level) if self.sweep_level else None
        self.memory = DetectionMemory(ttl_frames=self.memory_ttl) if self.sweep_level else None
        self.model = YOLO(str(self.path))
        if self.model.task != 'detect' or set(self.model.names.values()) != set(OBJECT_CLASSES):
            raise ValueError('Checkpoint must be a detector with the 16 challenge class names')
        self.lock = Lock()
        self.memory_lock = Lock()
        # Warm up CUDA kernels and the Torchvision NMS binary before serving.
        import numpy as np
        self.detect(np.zeros((540, 960, 3), dtype=np.uint8))

    def suppress_duplicates(self, detections):
        """Class-wise greedy NMS over (bbox, confidence, object_id), highest first."""
        kept = []
        for bbox, confidence, object_id in sorted(detections, key=lambda d: -d[1]):
            duplicate = False
            for other_bbox, _, other_id in kept:
                if other_id != object_id:
                    continue
                ix = max(0.0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0]))
                iy = max(0.0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
                inter = ix * iy
                if inter <= 0:
                    continue
                union = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                         + (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
                         - inter)
                if inter / max(1e-9, union) >= self.dedupe_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((bbox, confidence, object_id))
        return kept

    def rescale_confidence(self, confidence):
        """Map [conf_floor, 1] -> (clip_min, clip_max], linearly and monotonically."""
        if not self.clip:
            return confidence
        floor = self.options['conf']
        span = max(1e-9, 1.0 - floor)
        t = min(1.0, max(0.0, (confidence - floor) / span))
        # Nudge off the bottom so a detection sitting exactly on the confidence
        # floor still reports strictly above clip_min rather than equal to it.
        epsilon = 1e-6
        t = epsilon + (1.0 - epsilon) * t
        return self.clip_min + (self.clip_max - self.clip_min) * t

    def detect(self, image):
        """Infer and suppress duplicates on the GPU, returning one numpy array.

        The boxes are already on the device after inference, so running the
        class-wise NMS there costs almost nothing and avoids shipping every
        candidate box to Python only to drop a third of them. The view-to-source
        transform is a uniform scale (both are 16:9), and IoU is invariant under
        that, so suppressing here is equivalent to suppressing afterwards.
        """
        with self.lock:
            data = self.model.predict(image, **self.options)[0].boxes.data
            if data.numel() and self.dedupe_iou < 1.0:
                from torchvision.ops import batched_nms
                keep = batched_nms(data[:, :4], data[:, 4],
                                   data[:, 5].long(), self.dedupe_iou)
                data = data[keep]
            return data.cpu().numpy()

    def predict(self, request):
        image = decode_view(request.view)
        height, width = image.shape[:2]
        fresh = []
        for x1, y1, x2, y2, confidence, class_id in self.detect(image).tolist():
            bbox = clip_bbox_to_frame(view_bbox_to_global(
                (x1 / width, y1 / height, x2 / width, y2 / height),
                request.view.source_region_xyxy,
                request.original_width, request.original_height))
            if bbox is not None:
                fresh.append((tuple(bbox), float(self.rescale_confidence(confidence)),
                              self.model.names[int(class_id)]))

        requested_view = None
        if self.planner is not None:
            # The view covers only part of the frame now, so answer for the whole
            # frame from memory: fresh detections here, carried-over ones elsewhere.
            left, top, right, bottom = request.view.source_region_xyxy
            region = (left / request.original_width, top / request.original_height,
                      right / request.original_width, bottom / request.original_height)
            with self.memory_lock:
                items = self.memory.update(request.sequence_id, request.frame_index,
                                           region, fresh)
            fresh = [(i['bbox'], i['confidence'], i['object_id']) for i in items]
            try:
                requested_view = self.planner.next_view(request)
            except Exception:
                logger.exception('camera planning failed; leaving the view unchanged')
                requested_view = None

        if self.planner is not None:
            # Memory reintroduces boxes from earlier frames, which the GPU pass
            # never saw; those still need suppressing against the fresh ones.
            fresh = self.suppress_duplicates(fresh)
        fresh.sort(key=lambda d: -d[1])
        annotations = [DroneFlybyPredictionDto(object_id=object_id, bbox=list(bbox),
                                               confidence=confidence)
                       for bbox, confidence, object_id in fresh[:500]]
        response = DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=requested_view)
        validate_response(response)
        return response

    def health(self):
        return {
            'service': 'drone-flyby-yolo-fullframe',
            'ready': True,
            'architecture': self.model.model.yaml.get('yaml_file'),
            'model': self.path.name,
            'model_run': self.path.parent.parent.name,
            'device': self.device,
            'classes': len(self.model.names),
            'imgsz': self.options['imgsz'],
            'confidence_threshold': self.options['conf'],
            'nms_iou': self.options['iou'],
            'max_det': self.options['max_det'],
            'tiled': False,
            'patch_grid': None,
            'clip': self.clip,
            'clip_range': [self.clip_min, self.clip_max] if self.clip else None,
            'dedupe_iou': self.dedupe_iou,
            'sweep_level': self.sweep_level,
            'sweep_enabled': self.planner is not None,
            'memory_ttl_frames': self.memory_ttl if self.planner is not None else None,
            'tracked_sequences': len(self.memory.store) if self.memory is not None else 0,
        }


@asynccontextmanager
async def lifespan(app):
    logger.info('Loading full-frame detection backend...')
    app.state.detector = FullFrameModelService()
    app.state.started = time.monotonic()
    logger.info('Ready: %s on device %s at imgsz %s', app.state.detector.path,
                app.state.detector.device, app.state.detector.options['imgsz'])
    yield
    del app.state.detector


app = FastAPI(title='Drone Flyby Detection API (full frame)', lifespan=lifespan)


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    try:
        response = app.state.detector.predict(request)
    except Exception as exc:
        logger.exception('Prediction failed on frame %s', request.frame)
        raise HTTPException(status_code=500, detail='Model inference failed; see server log') from exc
    logger.info('frame %s L%s: %s detections', request.frame,
                request.view.resolution_level, len(response.annotations))
    return response


@app.get('/')
@app.get('/api')
def health():
    result = app.state.detector.health()
    result['uptime_seconds'] = round(time.monotonic() - app.state.started, 2)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Full-frame detection API')
    parser.add_argument('--model', help='checkpoint path (overrides DRONE_MODEL_PATH)')
    parser.add_argument('--port', type=int, help='listen port (default 5060)')
    parser.add_argument('--host', help='bind address (default 0.0.0.0)')
    parser.add_argument('--imgsz', type=int, help='inference size (default 960)')
    parser.add_argument('--conf', type=float, help='confidence floor (default 0.05)')
    parser.add_argument('--clip', action='store_true',
                        help='rescale confidences into (clip-min, clip-max] so nothing '
                             'is reported below 0.5; monotonic, so ranking is preserved')
    parser.add_argument('--clip-min', type=float, help='default 0.5')
    parser.add_argument('--clip-max', type=float, help='default 0.999')
    parser.add_argument('--dedupe-iou', type=float,
                        help='class-wise NMS on the answer (default 0.5); the evaluator '
                             'runs none, so overlapping duplicates score as false positives')
    parser.add_argument('--sweep-level', type=int, choices=(0, 1, 2),
                        help='0 (default) leaves the camera alone; 1 or 2 raster-sweeps '
                             'the frame at that resolution level, making objects 2x or '
                             '4x larger in the view')
    parser.add_argument('--memory-ttl', type=int,
                        help='frames a detection is carried for while unobserved (default 60)')
    parser.add_argument('--no-save-images', action='store_true',
                        help='disable request image logging')
    args = parser.parse_args()

    # CLI wins over the environment; anything omitted keeps its env/default value.
    for flag, key in (('model', 'DRONE_MODEL_PATH'), ('port', 'DRONE_PORT'),
                      ('host', 'DRONE_HOST'), ('imgsz', 'DRONE_IMGSZ'),
                      ('conf', 'DRONE_CONF'), ('clip_min', 'DRONE_CLIP_MIN'),
                      ('clip_max', 'DRONE_CLIP_MAX'), ('sweep_level', 'DRONE_SWEEP_LEVEL'), ('dedupe_iou', 'DRONE_DEDUPE_IOU'),
                      ('memory_ttl', 'DRONE_MEMORY_TTL')):
        value = getattr(args, flag)
        if value is not None:
            os.environ[key] = str(value)
    if args.clip:
        os.environ['DRONE_CLIP'] = '1'
    if args.no_save_images:
        os.environ['DRONE_SAVE_IMAGES'] = '0'

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get('DRONE_HOST', '0.0.0.0'),
                port=int(os.environ.get('DRONE_PORT', '5060')))
