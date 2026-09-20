# Medical appointment — second place

**Validation:** `0.7884` · **Final evaluation:** second place

## Approach

- **ASR:** `faster-whisper` with the `base` checkpoint, English VAD, and word-level timestamps (`float16` on CUDA, `int8` on CPU).
- **Question answering:** `google/gemma-4-E4B-it` answers all ten questions jointly in one deterministic generation. The prompt rejects near-misses such as wrong doses, durations, drugs, or body locations.
- **Evidence:** for each true answer, Gemma copies the complete supporting sentence. Normalized fuzzy matching aligns that quote to Whisper words, and the first/last matched word provide the audio timestamps.
- **Runtime:** both models load and warm at startup. Transcription or generation failures return a shape-valid fallback response instead of failing the whole conversation.

The deployed configuration uses one Modal `A100-40GB` container with both model weights baked into the image.

## Local

Python 3.11 and an NVIDIA GPU are recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api.py
```

The endpoint is `http://localhost:9054/predict`. The first start downloads and warms both models.

Docker is also supported:

```bash
docker build -t medical-appointment .
docker run --gpus all -p 9054:9054 medical-appointment
```

## Modal

One-time setup: `pip install modal && modal setup`

```bash
modal deploy modal_app.py
```

Submit the printed URL with `/predict`.
