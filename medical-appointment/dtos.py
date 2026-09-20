from typing import List, Optional

from pydantic import BaseModel, model_validator


class ASRQuestionRequestDto(BaseModel):
    audio_base64: str
    audio_filename: str
    questions: List[str]


class ASRQuestionResponseDto(BaseModel):
    answers: List[bool]
    evidence_start: List[Optional[float]]
    evidence_end: List[Optional[float]]

    @model_validator(mode="after")
    def evidence_matches_answers(self) -> "ASRQuestionResponseDto":
        if len(self.evidence_start) != len(self.answers) or len(self.evidence_end) != len(self.answers):
            raise ValueError("answers and evidence lists must have equal lengths")
        return self
