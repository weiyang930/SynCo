"""Primitive math equivalence helpers.

Layered checks used by ``answer_checker.py``:
- string normalization (whitespace, latex wrappers, $...$, \\boxed{})
- numeric equivalence (decimal, fraction)
- sympy equivalence for parseable algebraic expressions
- set / tuple normalization

Inspired by ``demo/agentskill/src/eval.py`` and ``src/evaluators/math_symbolic.py``,
but implemented from scratch and slimmed down.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_LATEX_BOXED_PREFIX_RE = re.compile(r"\\boxed\s*\{")
_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^}]*)\}")
_LATEX_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)")
_DOLLARS_RE = re.compile(r"\$+")
_PERCENT_SUFFIX_RE = re.compile(r"(?<=[\d\)])\s*\\?%\s*$")
_WHITESPACE_RE = re.compile(r"\s+")

# \dfrac / \tfrac collapse to \frac for equivalence purposes.
_DFRAC_TFRAC_RE = re.compile(r"\\(?:dfrac|tfrac)\b")
# \sqrt2 (one-char argument) → \sqrt{2}, similarly for variables.
_SQRT_BARE_ARG_RE = re.compile(r"\\sqrt\s*([0-9A-Za-z])")
# \frac43 (two-char braceless arguments) → \frac{4}{3}.
_FRAC_BRACELESS_RE = re.compile(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])")
# Degree marker: "90^\circ", "90^{\circ}", "90 °".
_DEGREE_RE = re.compile(r"\s*\^\s*\{?\s*\\?circ\s*\}?", re.IGNORECASE)
_DEGREE_UNICODE_RE = re.compile(r"\s*°\s*")
# Leading "x=", "y=", "z=" in a final answer ("x=5" → "5").
_LEADING_ASSIGN_RE = re.compile(r"^\s*[a-zA-Z]\s*=\s*")
# Common LaTeX spacing macros to strip.
_LATEX_SPACING_RES = (
    re.compile(r"\\quad\b"),
    re.compile(r"\\qquad\b"),
    re.compile(r"\\:"),
    re.compile(r"\\;"),
    re.compile(r"\\,"),
    re.compile(r"\\!"),
)


def _strip_outer_boxed(s: str) -> str:
    """Recursively peel ``\\boxed{...}`` wrappers (handles nested cases).

    Uses a balanced-brace scan so that nested LaTeX like
    ``\\boxed{\\frac{\\sqrt{65}}{2}}`` peels to
    ``\\frac{\\sqrt{65}}{2}`` instead of stopping at the first ``}``.
    """
    out = s
    for _ in range(4):  # bounded loop to avoid pathological inputs
        m = _LATEX_BOXED_PREFIX_RE.search(out)
        if not m:
            break
        start = m.end()  # one past the opening '{'
        depth = 1
        i = start
        while i < len(out) and depth > 0:
            ch = out[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            break  # unbalanced — leave the original alone
        out = out[start:i].strip()
    return out


def normalize_string(s: str | None) -> str:
    """Aggressive whitespace / latex wrapper normalization.

    Returns the *first* unwrapped form when ``\\boxed{}`` is present.
    Collapses LaTeX synonyms (``\\dfrac``/``\\tfrac`` → ``\\frac``,
    ``\\sqrt2`` → ``\\sqrt{2}``) and strips degree markers / leading
    variable assignments so ``"90^\\circ"`` matches ``"90"`` and
    ``"x = 5"`` matches ``"5"``.
    """
    if not s:
        return ""
    out = str(s).strip()
    out = _strip_outer_boxed(out)
    out = _LATEX_TEXT_RE.sub(r"\1", out)
    out = _LATEX_LEFT_RIGHT_RE.sub("", out)
    out = _DOLLARS_RE.sub("", out)
    out = _DFRAC_TFRAC_RE.sub(r"\\frac", out)
    out = _FRAC_BRACELESS_RE.sub(r"\\frac{\1}{\2}", out)
    out = _SQRT_BARE_ARG_RE.sub(r"\\sqrt{\1}", out)
    out = _DEGREE_RE.sub("", out)
    out = _DEGREE_UNICODE_RE.sub("", out)
    for pattern in _LATEX_SPACING_RES:
        out = pattern.sub("", out)
    out = out.replace("\\ ", " ")
    out = _LEADING_ASSIGN_RE.sub("", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    if out.endswith("."):
        out = out[:-1].rstrip()
    return out


def _strip_commas(s: str) -> str:
    return s.replace(",", "")


def try_parse_decimal(s: str | None) -> Decimal | None:
    """Parse a string as a Decimal. Handles fractions like ``3/4``."""
    if not s:
        return None
    candidate = _strip_commas(str(s).strip())
    if not candidate:
        return None

    # Trailing percent: scale by 1/100.
    percent = False
    if candidate.endswith("%"):
        candidate = candidate[:-1].strip()
        percent = True
    candidate = _PERCENT_SUFFIX_RE.sub("", candidate).strip()

    # Pure decimal.
    try:
        d = Decimal(candidate)
        return d / Decimal(100) if percent else d
    except (InvalidOperation, ValueError):
        pass

    # Fraction: a/b (with optional sign).
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", candidate):
        try:
            f = Fraction(candidate.replace(" ", ""))
            d = Decimal(f.numerator) / Decimal(f.denominator)
            return d / Decimal(100) if percent else d
        except (ZeroDivisionError, ValueError):
            return None

    return None


def numeric_equivalent(a: str, b: str, tolerance: float = 1e-6) -> bool:
    """Return True iff both strings parse to the same number within tolerance."""
    da = try_parse_decimal(a)
    db = try_parse_decimal(b)
    if da is None or db is None:
        return False
    if da == db:
        return True
    try:
        return abs(float(da) - float(db)) <= tolerance
    except (OverflowError, ValueError):
        return False


def sympy_equivalent(a: str, b: str) -> bool:
    """Return True iff sympy thinks ``a`` and ``b`` are symbolically equal.

    Best-effort: parse failures or sympy exceptions are treated as ``False``.
    Imports sympy lazily so the rest of the pipeline runs without it.
    """
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except ImportError:
        return False

    transformations = standard_transformations + (implicit_multiplication_application,)

    def _to_sympy_input(s: str) -> str:
        # Minimal LaTeX→sympy bridge: only \frac{a}{b}, since sympy cannot
        # parse it natively. Other constructs (^, \cdot, \pi, \sqrt) are
        # left to sympy's own parser; if it fails we fall through to a
        # False match rather than aggressively rewriting.
        return re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)

    try:
        ea = parse_expr(_to_sympy_input(a), transformations=transformations, evaluate=True)
        eb = parse_expr(_to_sympy_input(b), transformations=transformations, evaluate=True)
    except (SyntaxError, TypeError, ValueError, Exception):  # noqa: BLE001
        return False

    try:
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:  # noqa: BLE001
        return False


_SET_RE = re.compile(r"^\{(.*)\}$", re.DOTALL)
_TUPLE_RE = re.compile(r"^\((.*)\)$", re.DOTALL)


def _split_top_level(body: str, sep: str = ",") -> list[str]:
    """Split ``body`` on ``sep`` only at brace/paren depth 0."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def set_or_tuple_equivalent(a: str, b: str, tolerance: float = 1e-6) -> bool | None:
    """Return True/False if both strings parse as sets/tuples, else None."""
    a_norm = normalize_string(a)
    b_norm = normalize_string(b)
    set_a = _SET_RE.match(a_norm)
    set_b = _SET_RE.match(b_norm)
    tup_a = _TUPLE_RE.match(a_norm)
    tup_b = _TUPLE_RE.match(b_norm)

    if set_a and set_b:
        items_a = _split_top_level(set_a.group(1))
        items_b = _split_top_level(set_b.group(1))
        if len(items_a) != len(items_b):
            return False
        # Order-insensitive: greedy match.
        used = [False] * len(items_b)
        for x in items_a:
            matched = False
            for j, y in enumerate(items_b):
                if used[j]:
                    continue
                if (
                    normalize_string(x) == normalize_string(y)
                    or numeric_equivalent(x, y, tolerance)
                    or sympy_equivalent(x, y)
                ):
                    used[j] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    if tup_a and tup_b:
        items_a = _split_top_level(tup_a.group(1))
        items_b = _split_top_level(tup_b.group(1))
        if len(items_a) != len(items_b):
            return False
        for x, y in zip(items_a, items_b):
            if (
                normalize_string(x) == normalize_string(y)
                or numeric_equivalent(x, y, tolerance)
                or sympy_equivalent(x, y)
            ):
                continue
            return False
        return True

    return None
