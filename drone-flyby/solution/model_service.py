"""YOLO inference and protocol conversion shared by the production API."""
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT))

from ultralytics import YOLO
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictResponseDto
from utils import decode_view, view_bbox_to_global, clip_bbox_to_frame, validate_response
from sol.tiled_inference import patch_regions, merge_patch_detections


class ModelService:
    def __init__(self):
        default = ROOT / 'ckpts/yolo11n_tile48_original_plus100_best.pt'
        self.path = Path(os.environ.get('DRONE_MODEL_PATH', str(default))).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {self.path}')
        self.device = os.environ.get('DRONE_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
        self.tiled = os.environ.get('DRONE_TILED', '1') == '1'
        self.patch_overlap = float(os.environ.get('DRONE_PATCH_OVERLAP', '0.1'))
        self.patch_batch = int(os.environ.get('DRONE_PATCH_BATCH', '8'))
        # The next checkpoint is trained on 32 patches. An 8x4 grid keeps
        # patches close to square for the protocol's 960x540 transmitted view.
        self.patch_columns = int(os.environ.get('DRONE_PATCH_COLUMNS', '8'))
        self.patch_rows = int(os.environ.get('DRONE_PATCH_ROWS', '4'))
        self.merge_iou = float(os.environ.get('DRONE_MERGE_IOU', '0.5'))
        self.merge_class_agnostic = os.environ.get('DRONE_MERGE_CLASS_AGNOSTIC', '1') == '1'
        self.cross_patch_ios = float(os.environ.get('DRONE_CROSS_PATCH_IOS', '0.8'))
        self.internal_edge_margin = float(os.environ.get('DRONE_INTERNAL_EDGE_MARGIN', '1.0'))
        self.max_det = int(os.environ.get('DRONE_MAX_DET', '300'))
        if min(self.patch_batch, self.patch_columns, self.patch_rows) < 1 or not 0 <= self.patch_overlap < .5 or not 0 < self.merge_iou <= 1:
            raise ValueError('Invalid patch batch/overlap/merge IoU settings')
        if not 0 < self.cross_patch_ios <= 1 or self.internal_edge_margin < 0 or self.max_det < 1:
            raise ValueError('Invalid patch merge settings')
        self.options = dict(device=self.device, imgsz=int(os.environ.get('DRONE_IMGSZ', '640' if self.tiled else '960')),
                            conf=float(os.environ.get('DRONE_CONF', '0.001')),
                            iou=float(os.environ.get('DRONE_NMS_IOU', '0.7')),
                            max_det=self.max_det, verbose=False)
        self.model = YOLO(str(self.path))
        if self.model.task != 'detect' or set(self.model.names.values()) != set(OBJECT_CLASSES):
            raise ValueError('Checkpoint must be a detector with the 16 challenge class names')
        self.lock = Lock()
        # Warmup also exercises the Torchvision NMS binary before accepting requests.
        self.detect(np.zeros((540, 960, 3), dtype=np.uint8))

    def detect(self, image):
        height, width = image.shape[:2]
        with self.lock:
            if not self.tiled:
                return self.model.predict(image, **self.options)[0].boxes.data.cpu().tolist()
            regions = patch_regions(width, height, columns=self.patch_columns,
                                    rows=self.patch_rows, overlap=self.patch_overlap)
            predictions = []
            for start in range(0, len(regions), self.patch_batch):
                crops = [np.ascontiguousarray(image[y1:y2, x1:x2])
                         for x1, y1, x2, y2 in regions[start:start+self.patch_batch]]
                results = self.model.predict(crops, **self.options)
                predictions.extend(result.boxes.data.cpu().tolist() for result in results)
            return merge_patch_detections(predictions, regions, width, height,
                                          iou=self.merge_iou, max_det=self.max_det,
                                          class_agnostic=self.merge_class_agnostic,
                                          cross_patch_ios=self.cross_patch_ios,
                                          internal_edge_margin=self.internal_edge_margin)

    def predict(self, request):
        image = decode_view(request.view)
        height, width = image.shape[:2]
        rows = self.detect(image)
        annotations = []
        for x1, y1, x2, y2, confidence, class_id in rows:
            bbox = clip_bbox_to_frame(view_bbox_to_global(
                (x1 / width, y1 / height, x2 / width, y2 / height),
                request.view.source_region_xyxy, request.original_width, request.original_height))
            if bbox is not None:
                annotations.append(DroneFlybyPredictionDto(
                    object_id=self.model.names[int(class_id)], bbox=list(bbox), confidence=float(confidence)))
        response = DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame, annotations=annotations,
            requested_view=None)
        validate_response(response)
        return response
