"""Load math benchmarks from local JSONL/JSON files.

The evaluation module is intentionally permissive: each benchmark folder
is searched for ``test.jsonl``, ``test.json``, or any ``*.jsonl`` /
``*.json`` file. We accept any of the common keys for problem and
answer; missing files are skipped (returning ``[]``) so a partial setup
does not break a run.

Spec §13: GSM8K / MATH500 / AMC / AIME with graceful skip when local
data is missing. We deliberately do **not** fetch from the network.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

# Common keys we will probe for problem/answer.
_PROBLEM_KEYS = ("problem", "question", "Problem", "Question", "input", "prompt")
_ANSWER_KEYS = (
    "final_answer",
    "answer",
    "Answer",
    "solution_answer",
    "expected_answer",
    "target",
    "label",
)


@dataclass
class EvalExample:
    benchmark: str
    problem: str
    reference_answer: str
    extra: dict = field(default_factory=dict)


def _candidate_files(path: str) -> list[str]:
    """Resolve a benchmark spec ``path`` to one or more JSONL/JSON files.

    Accepts either a file path (used as-is) or a folder path (must contain
    one of the preferred split files). We deliberately do **not** fall
    back to globbing every ``*.jsonl`` in a folder, because that risks
    silently mixing train/test splits or human-annotated subsets.
    """
    if not path:
        return []
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    preferred = ("test.jsonl", "test.json", "validation.jsonl", "valid.jsonl")
    for name in preferred:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return [candidate]
    return []


def _iter_records(path: str) -> Iterable[dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return
    with open(path, "r", encoding="utf-8") as fh:
        try:
            blob = json.load(fh)
        except json.JSONDecodeError:
            return
    if isinstance(blob, list):
        for obj in blob:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(blob, dict):
        for key in ("data", "examples", "items", "rows"):
            if isinstance(blob.get(key), list):
                for obj in blob[key]:
                    if isinstance(obj, dict):
                        yield obj
                return


def _coerce_answer(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            text = _coerce_answer(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in _ANSWER_KEYS:
            if key in value:
                text = _coerce_answer(value[key])
                if text:
                    return text
        return ""
    return str(value).strip()


def _coerce_problem(obj: dict) -> str:
    for key in _PROBLEM_KEYS:
        if key in obj:
            value = obj[key]
            if isinstance(value, str) and value.strip():
                return value.strip()
    # Last-ditch effort: a list of messages.
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                text = m.get("content", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return ""


def load_benchmark(name: str, path: str, max_examples: int = 0) -> list[EvalExample]:
    """Load examples from one benchmark folder.

    ``max_examples <= 0`` disables truncation. Missing folders return [].
    """
    files = _candidate_files(path)
    out: list[EvalExample] = []
    for file_path in files:
        for obj in _iter_records(file_path):
            problem = _coerce_problem(obj)
            if not problem:
                continue
            ans_value = None
            for key in _ANSWER_KEYS:
                if key in obj:
                    ans_value = obj[key]
                    break
            if ans_value is None:
                # Fall through: spec datasets sometimes nest the answer.
                rm = obj.get("reward_model") or obj.get("extra_info") or {}
                if isinstance(rm, dict):
                    for key in _ANSWER_KEYS:
                        if key in rm:
                            ans_value = rm[key]
                            break
            answer = _coerce_answer(ans_value)
            if not answer:
                continue
            extra = {k: v for k, v in obj.items() if k not in _PROBLEM_KEYS}
            out.append(
                EvalExample(
                    benchmark=name,
                    problem=problem,
                    reference_answer=answer,
                    extra=extra,
                )
            )
            if max_examples > 0 and len(out) >= max_examples:
                return out
    return out


__all__ = ["EvalExample", "load_benchmark"]
