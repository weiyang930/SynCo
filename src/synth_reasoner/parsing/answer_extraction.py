"""High-level Reasoner answer extraction.

Order of preference (matches spec §7.4):
1. Last ``\\boxed{...}`` content.
2. ``Final answer: ...`` line (case-insensitive).
3. ``Answer: ...`` line (case-insensitive).
4. Last non-empty trimmed line as a fallback.

Each layer reports ``has_boxed`` so the reward function can apply the
format bonus only when the boxed convention is followed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .boxed_answer import extract_last_boxed


_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)


@dataclass
class ExtractedAnswer:
    answer: str | None
    method: str  # "boxed", "final_answer", "answer", "last_line", "none"
    has_boxed: bool


def extract_reasoner_answer(text: str) -> ExtractedAnswer:
    """Run the layered extraction and report which method succeeded."""
    if not text:
        return ExtractedAnswer(answer=None, method="none", has_boxed=False)

    boxed = extract_last_boxed(text)
    if boxed is not None and boxed.strip():
        return ExtractedAnswer(answer=boxed.strip(), method="boxed", has_boxed=True)

    # Search line-by-line, scanning from the bottom so we capture the *final* claim.
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        m = _FINAL_ANSWER_RE.search(line)
        if m:
            return ExtractedAnswer(
                answer=m.group(1).strip(), method="final_answer", has_boxed=False
            )
    for line in reversed(lines):
        m = _ANSWER_RE.search(line)
        if m:
            return ExtractedAnswer(
                answer=m.group(1).strip(), method="answer", has_boxed=False
            )
    if lines:
        return ExtractedAnswer(answer=lines[-1].strip(), method="last_line", has_boxed=False)
    return ExtractedAnswer(answer=None, method="none", has_boxed=False)
