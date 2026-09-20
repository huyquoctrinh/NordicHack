import modal

app = modal.App("medical-appointment-second-place")

NVIDIA_LIBRARY_PATH = (
    "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:"
    "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
    .env(
        {
            "LD_LIBRARY_PATH": NVIDIA_LIBRARY_PATH,
            "LLM_NAME": "google/gemma-4-E4B-it",
            "WHISPER_MODEL_NAME": "base",
        }
    )
    .run_commands(
        "python -c \"from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')\"",
        "python -c \"from transformers import AutoModelForCausalLM, AutoProcessor; AutoProcessor.from_pretrained('google/gemma-4-E4B-it'); AutoModelForCausalLM.from_pretrained('google/gemma-4-E4B-it')\"",
    )
    .add_local_python_source("dtos", "utils", "llm_qa", "example", "api")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    min_containers=0,
    max_containers=1,
    scaledown_window=1200,
    timeout=120,
)
@modal.asgi_app()
def fastapi_app():
    from api import app as web_app

    return web_app
