"""Source for ../api.py: serve Grounding DINO or YOLO on the same /predict protocol."""
from contextlib import asynccontextmanager
import logging
import os
import time

from fastapi import FastAPI, HTTPException
import uvicorn

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from sol.model_service import ModelService as YoloModelService

def create_detector():
    backend = os.environ.get('DRONE_BACKEND', 'yolo')

    return YoloModelService()
    # raise ValueError(f'Unknown DRONE_BACKEND: {backend}')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info('Loading and warming up detection backend...')
    app.state.detector = create_detector()
    app.state.started = time.monotonic()
    logger.info('Ready: %s on device %s', app.state.detector.path, app.state.detector.device)
    yield
    del app.state.detector


app = FastAPI(title='Drone Flyby Detection API', lifespan=lifespan)


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    try:
        response = app.state.detector.predict(request)
    except Exception as exc:
        logger.exception('Prediction failed on frame %s', request.frame)
        raise HTTPException(status_code=500, detail='Model inference failed; see server log') from exc
    logger.info('frame %s L%s: %s detections', request.frame, request.view.resolution_level, len(response.annotations))
    return response


@app.get('/')
@app.get('/api')
def health():
    detector = app.state.detector
    if hasattr(detector, 'health'):
        result = detector.health()
        result['uptime_seconds'] = round(time.monotonic() - app.state.started, 2)
        return result
    return {'service': 'drone-flyby-yolo', 'ready': True,
            'architecture': detector.model.model.yaml.get('yaml_file'), 'model': detector.path.name, 'model_run': detector.path.parent.parent.name, 'device': detector.device,
            'classes': len(detector.model.names), 'imgsz': detector.options['imgsz'],
            'confidence_threshold': detector.options['conf'], 'tiled': detector.tiled,
            'patch_grid': [detector.patch_columns, detector.patch_rows] if detector.tiled else None,
            'patch_overlap': detector.patch_overlap, 'patch_batch': detector.patch_batch,
            'merge_iou': detector.merge_iou,
            'merge_class_agnostic': detector.merge_class_agnostic,
            'cross_patch_ios': detector.cross_patch_ios,
            'internal_edge_margin': detector.internal_edge_margin,
            'uptime_seconds': round(time.monotonic() - app.state.started, 2)}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get('DRONE_HOST', '0.0.0.0'),
                port=int(os.environ.get('DRONE_PORT', '9053')))



