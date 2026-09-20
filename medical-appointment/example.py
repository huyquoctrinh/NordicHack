import logging
import os
import tempfile
import wave
from typing import List

import ctranslate2
from faster_whisper import WhisperModel

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from llm_qa import Word, answer_questions
from utils import decode_audio

logger = logging.getLogger(__name__)
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "base")
DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
ASR_MODEL = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def _warm_up_asr() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
        with wave.open(audio_file.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
        list(ASR_MODEL.transcribe(audio_file.name, language="en", word_timestamps=True)[0])


def _transcribe(audio_bytes: bytes) -> List[Word]:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
        audio_file.write(audio_bytes)
        audio_file.flush()
        segments, _ = ASR_MODEL.transcribe(
            audio_file.name,
            language="en",
            vad_filter=True,
            word_timestamps=True,
        )
        segments = list(segments)

    return [
        {"start": word.start, "end": word.end, "word": word.word.strip()}
        for segment in segments
        for word in (segment.words or [])
    ]


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    try:
        words = _transcribe(decode_audio(request.audio_base64))
    except Exception:
        logger.exception("Transcription failed for %s", request.audio_filename)
        words = []

    transcript = " ".join(word["word"] for word in words)
    results = answer_questions(transcript, words, request.questions)

    return ASRQuestionResponseDto(
        answers=[answer for answer, _ in results],
        evidence_start=[span[0] if span else None for _, span in results],
        evidence_end=[span[1] if span else None for _, span in results],
    )


_warm_up_asr()
