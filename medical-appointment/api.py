import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from example import predict
from utils import validate_response

HOST = "0.0.0.0"
PORT = 9054

logging.basicConfig(level=logging.INFO)
app = FastAPI()
start_time = time.time()


@app.post("/predict", response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    response = predict(request)
    validate_response(response, expected_count=len(request.questions))
    return response


@app.get("/api")
def health() -> dict:
    return {
        "service": "medical-appointment-second-place",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index() -> str:
    return "Your endpoint is running!"


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
