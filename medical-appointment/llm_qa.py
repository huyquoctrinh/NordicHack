import difflib
import json
import logging
import os
import re
from typing import List, Optional, Tuple, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from utils import Span

logger = logging.getLogger(__name__)


class Word(TypedDict):
    start: float
    end: float
    word: str


LLM_NAME = os.environ.get("LLM_NAME", "google/gemma-4-E4B-it")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
logger.info("Loading %s on %s", LLM_NAME, DEVICE)

try:
    PROCESSOR = AutoTokenizer.from_pretrained(LLM_NAME)
except ValueError:
    PROCESSOR = AutoProcessor.from_pretrained(LLM_NAME)

try:
    MODEL = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=DTYPE)
except ValueError:
    MODEL = AutoModelForImageTextToText.from_pretrained(LLM_NAME, torch_dtype=DTYPE)

MODEL.to(DEVICE)
MODEL.eval()
TEXT_TOKENIZER = getattr(PROCESSOR, "tokenizer", PROCESSOR)
MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "800"))


def _generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(DEVICE)

    with torch.no_grad():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=TEXT_TOKENIZER.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return TEXT_TOKENIZER.decode(generated, skip_special_tokens=True)


def _extract_json_array(text: str) -> Optional[list]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _build_prompt(full_text: str, questions: List[str]) -> str:
    question_block = "\n".join(
        f"{index + 1}. {question}" for index, question in enumerate(questions)
    )
    return f'''You are reading the full transcript of a doctor-patient conversation, as plain continuous text. Both speakers were recorded on a single channel, so speaker labels are not marked — infer from context (who is asking versus answering, examining versus describing symptoms) which parts are the doctor's and which are the patient's where it matters for a question.

TRANSCRIPT:
{full_text}

Answer each numbered question below using only what the transcript actually states. Many questions are near-misses on a real statement in the transcript — the same drug at a different dose, the same finding in a different place, a plausible detail that was never said. Treat those as "no". Only answer "yes" when the transcript states the exact thing asked, not just the same general topic.

QUESTIONS:
{question_block}

For each question, respond with one JSON object: {{"answer": true or false, "quote": "<text>"}}. "quote" must be copied exactly, word-for-word, from the transcript above — the complete sentence (or two, if the answer only makes sense with its lead-in clause) that states the answer, not a trimmed fragment of it. Omit it (set to null) when the answer is false.

Respond with a JSON array of exactly {len(questions)} objects, one per question, in order. Output nothing but the JSON array.'''


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _locate_quote(quote: Optional[str], words: List[Word]) -> Optional[Span]:
    if not quote or not words:
        return None

    normalized_words = [_normalize(word["word"]) for word in words]
    transcript = " ".join(normalized_words)
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return None

    word_ranges = []
    position = 0
    for index, word in enumerate(normalized_words):
        word_ranges.append((position, position + len(word), index))
        position += len(word) + 1

    match = difflib.SequenceMatcher(
        None, transcript, normalized_quote, autojunk=False
    ).find_longest_match(0, len(transcript), 0, len(normalized_quote))
    if match.size < min(len(normalized_quote), 6):
        return None

    match_start = match.a
    match_end = match.a + match.size
    covering = [
        index
        for start, end, index in word_ranges
        if end > match_start and start < match_end
    ]
    if not covering:
        return None
    return words[covering[0]]["start"], words[covering[-1]]["end"]


def answer_questions(
    full_text: str,
    words: List[Word],
    questions: List[str],
) -> List[Tuple[bool, Optional[Span]]]:
    fallback = [(True, None)] * len(questions)
    if not words:
        return fallback

    try:
        raw = _generate(_build_prompt(full_text, questions))
    except Exception:
        logger.exception("LLM generation failed")
        return fallback

    parsed = _extract_json_array(raw)
    if parsed is None:
        logger.warning("Could not parse LLM output as JSON")
        return fallback

    results: List[Tuple[bool, Optional[Span]]] = []
    for index in range(len(questions)):
        if index >= len(parsed) or not isinstance(parsed[index], dict):
            results.append((True, None))
            continue

        item = parsed[index]
        answer = bool(item.get("answer"))
        if not answer:
            results.append((False, None))
            continue

        quote = item.get("quote")
        span = _locate_quote(quote, words) if isinstance(quote, str) else None
        results.append((True, span))

    return results


_generate("Reply with the single word: ready.", max_new_tokens=8)
