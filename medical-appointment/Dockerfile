FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12

ENV LLM_NAME=google/gemma-4-E4B-it \
    WHISPER_MODEL_NAME=base \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY dtos.py utils.py llm_qa.py example.py api.py ./
EXPOSE 9054
CMD ["python", "api.py"]
