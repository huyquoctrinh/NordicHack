import base64
import math
from typing import Tuple

from dtos import ASRQuestionResponseDto

Span = Tuple[float, float]


def decode_audio(audio_base64: str) -> bytes:
    return base64.b64decode(audio_base64)


def validate_response(response: ASRQuestionResponseDto, expected_count: int) -> None:
    if not isinstance(response, ASRQuestionResponseDto):
        raise ValueError("predict() must return ASRQuestionResponseDto")

    fields = {
        "answers": response.answers,
        "evidence_start": response.evidence_start,
        "evidence_end": response.evidence_end,
    }
    for name, values in fields.items():
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"{name} must contain one value per question")

    for index, answer in enumerate(response.answers):
        if not isinstance(answer, bool):
            raise ValueError(f"answers[{index}] must be a bool")

        start = response.evidence_start[index]
        end = response.evidence_end[index]
        if (start is None) != (end is None):
            raise ValueError(f"evidence interval {index} is incomplete")
        if start is None:
            continue
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise ValueError(f"evidence interval {index} is invalid")
