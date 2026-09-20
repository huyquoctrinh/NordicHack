# Medical appointment — second place


Faster Whisper `base` produces word-level timestamps. Gemma 4 E4B-it answers all ten questions in one pass and returns verbatim evidence; fuzzy alignment maps each quote back to its audio span.

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
