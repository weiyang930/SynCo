"""Extract \\boxed{...} answers from Reasoner output.

Handles nested braces inside the boxed expression (common in LaTeX-heavy
solutions like ``\\boxed{\\frac{1}{2}}``). When multiple ``\\boxed{}``
expressions are present, the *last* one is returned: this matches the
convention that the Reasoner ends with the final answer.
"""
from __future__ import annotations


_BOXED_PREFIX = "\\boxed{"


def extract_all_boxed(text: str) -> list[str]:
    """Return every balanced ``\\boxed{...}`` content found in ``text``.

    Empty list if none found. Brace balancing is performed manually so
    nested ``{...}`` work, unlike a naive regex.
    """
    if not text or _BOXED_PREFIX not in text:
        return []

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find(_BOXED_PREFIX, i)
        if idx < 0:
            break
        # Walk forward from the opening brace, balancing braces.
        start = idx + len(_BOXED_PREFIX)
        depth = 1
        j = start
        while j < n and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            out.append(text[start:j])
            i = j + 1
        else:
            # unbalanced; bail to avoid infinite loop
            break
    return out


def extract_last_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` content, or ``None`` if absent."""
    boxed = extract_all_boxed(text)
    return boxed[-1] if boxed else None
