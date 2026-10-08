"""Robust JSON extraction for Synthesizer output.

The Synthesizer is instructed to emit a single JSON object matching the
9-field task-card schema. Real LLM output is messy: extra prose, code
fences, trailing commas, single quotes. We try, in order:

1. ``json.loads`` of the raw text (happy path).
2. Strip surrounding markdown code fences and retry.
3. Walk the text and extract the first balanced ``{...}`` block, retry.
4. As a last resort, attempt simple repairs (single→double quotes, trailing
   commas) on the extracted block, retry.

Failures return ``ParsedJSON(ok=False, ...)`` instead of raising. The
caller is expected to mark the task invalid and emit an invalid-format
reward; we never silently drop the failure (spec §6.6).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)

# Qwen3 thinking-mode emits ``<think>...</think>`` before the answer. The
# CoT often contains stray ``{`` tokens (set notation, sequence indices),
# so we must strip the thinking block before hunting for JSON. We tolerate
# both the standard and the rare unclosed-think case (truncation): if the
# closing tag is missing, fall back to the original text.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


@dataclass
class ParsedJSON:
    ok: bool
    data: Any | None
    method: str  # "raw", "fence_stripped", "first_object", "repaired", "failed"
    error: str | None


def _try_loads(text: str, method: str) -> ParsedJSON:
    try:
        return ParsedJSON(ok=True, data=json.loads(text), method=method, error=None)
    except json.JSONDecodeError as e:
        return ParsedJSON(ok=False, data=None, method=method, error=str(e))


def _strip_fences(text: str) -> str:
    out = text.strip()
    out = _FENCE_RE.sub("", out)
    return out.strip()


def _strip_thinking(text: str) -> str:
    """Drop any ``<think>...</think>`` prelude. No-op if no closing tag."""
    m = None
    for m in _THINK_CLOSE_RE.finditer(text):
        pass  # keep last occurrence
    if m is None:
        return text
    return text[m.end() :]


def _first_balanced_object(text: str) -> str | None:
    """Return the substring of the first balanced ``{...}`` JSON object."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair(text: str) -> str:
    # Remove trailing commas inside objects/arrays: ",}" -> "}"
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    return repaired


def parse_synthesizer_json(raw: str | None) -> ParsedJSON:
    """Best-effort parse of a Synthesizer completion as JSON.

    Returns a `ParsedJSON` that records both success/failure and which
    layer succeeded, so we can surface diagnostics in the pool.
    """
    if not raw:
        return ParsedJSON(ok=False, data=None, method="failed", error="empty input")

    # Layer 0: drop Qwen3 ``<think>...</think>`` prelude. The CoT contains
    # set-notation/index braces that defeat the balanced-object scan.
    post_think = _strip_thinking(raw)

    # Layer 1: as-is (post-think).
    direct = _try_loads(post_think, "raw")
    if direct.ok:
        return direct

    # Layer 2: fences.
    stripped = _strip_fences(post_think)
    fenced_error: str | None = direct.error
    if stripped and stripped != post_think:
        fenced = _try_loads(stripped, "fence_stripped")
        if fenced.ok:
            return fenced
        fenced_error = fenced.error

    # Layer 3: first balanced object.
    obj = _first_balanced_object(stripped or post_think)
    if obj is not None:
        first = _try_loads(obj, "first_object")
        if first.ok:
            return first
        # Layer 4: light repair.
        repaired = _repair(obj)
        if repaired != obj:
            r = _try_loads(repaired, "repaired")
            if r.ok:
                return r

    return ParsedJSON(
        ok=False,
        data=None,
        method="failed",
        error=fenced_error or "no JSON object found",
    )
