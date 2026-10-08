from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        for key in ("ground_truth", "answer", "target", "solution"):
            if key in value:
                return _to_text(value[key])
    return str(value)


def _strip_boxed(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    idx = raw.rfind("\\boxed")
    if idx >= 0:
        if raw.startswith("\\boxed ", idx):
            return raw[idx + len("\\boxed ") :].split("$")[0].strip()
        brace_start = raw.find("{", idx)
        if brace_start >= 0:
            depth = 0
            for pos in range(brace_start, len(raw)):
                if raw[pos] == "{":
                    depth += 1
                elif raw[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return raw[brace_start + 1 : pos].strip()
    match = re.search(r"\\boxed\s*\{(.+)\}\s*$", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\s+([^\s$]+)", raw)
    if match:
        return match.group(1).strip()
    return raw


def extract_math_answer(value: Any) -> str:
    """Extract a compact final answer from model output or dataset fields."""

    text = _to_text(value).strip()
    if not text:
        return ""

    for tag in ("FINAL_ANSWER", "ANSWER"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            text = str(matches[-1]).strip()
            break

    boxed = _strip_boxed(text)
    if boxed != text:
        return boxed.strip()

    answer_match = re.findall(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)", text)
    if answer_match:
        text = answer_match[-1].strip()

    text = text.strip().strip("$").strip()
    text = re.sub(r"^(?:\\displaystyle\s*)", "", text).strip()
    text = text.rstrip(".。").strip()
    return _strip_boxed(text).strip()


def normalize_math_answer(value: Any) -> str:
    answer = extract_math_answer(value)
    if not answer:
        return ""
    return canonical_math_answer(answer)


def _replace_text_commands(text: str) -> str:
    pattern = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)
    return text


def _replace_latex_fracs(text: str) -> str:
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    while prev != text:
        prev = text

        def repl(match: re.Match) -> str:
            numerator = canonical_math_answer(match.group(1))
            denominator = canonical_math_answer(match.group(2))
            return f"{numerator}/{denominator}"

        text = pattern.sub(repl, text)

    # Common MATH shorthand, e.g. \frac43 or \frac\pi2.
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"\1/\2", text)
    text = re.sub(r"\\frac\s*\\([a-zA-Z]+)\s*([+-]?\d+)", r"\\\1/\2", text)
    return text


def _replace_latex_sqrt(text: str) -> str:
    text = re.sub(r"\\sqrt\{([^{}]+)\}", lambda m: f"sqrt({canonical_math_answer(m.group(1))})", text)
    text = re.sub(r"\\sqrt\s*([0-9a-zA-Z]+)", r"sqrt(\1)", text)
    return text


def canonical_math_answer(value: Any) -> str:
    """Canonical string used for non-symbolic benchmark answer comparison."""

    text = _to_text(value).strip()
    if not text:
        return ""

    text = _strip_boxed(text)
    text = text.strip().strip("$").strip()
    text = text.rstrip(".。")
    text = text.replace("\\displaystyle", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\leq", "\\le").replace("\\geq", "\\ge")
    text = text.replace("\\le", "<=").replace("\\ge", ">=")
    text = text.replace("\\neq", "!=").replace("\\ne", "!=")
    text = text.replace("\\pm", "pm")
    text = text.replace("\\pi", "pi")
    text = re.sub(r"\^\{?\\circ\}?", "", text)
    text = text.replace("\\degree", "")
    text = _replace_text_commands(text)
    text = _replace_latex_sqrt(text)
    text = _replace_latex_fracs(text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\\\", ",")
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("，", ",")
    text = re.sub(r"\s+", "", text)
    text = text.casefold()
    return text


def _fraction_maybe(value: str) -> Fraction | None:
    text = normalize_math_answer(value)
    if not text:
        return None
    text = text.replace(",", "")
    frac_match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
    if frac_match:
        denominator = int(frac_match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(frac_match.group(1)), denominator)
    simple_frac = re.fullmatch(r"([-+]?\d+)/([-+]?\d+)", text)
    if simple_frac:
        denominator = int(simple_frac.group(2))
        if denominator == 0:
            return None
        return Fraction(int(simple_frac.group(1)), denominator)
    number_match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if number_match:
        try:
            return Fraction(text)
        except Exception:
            return None
    return None


def math_answer_equal(prediction: Any, ground_truth: Any) -> bool:
    pred = extract_math_answer(prediction)
    gt = extract_math_answer(ground_truth)
    if not pred or not gt:
        return False

    pred_num = _fraction_maybe(pred)
    gt_num = _fraction_maybe(gt)
    if pred_num is not None and gt_num is not None:
        return pred_num == gt_num

    pred_norm = normalize_math_answer(pred)
    gt_norm = normalize_math_answer(gt)
    if pred_norm and pred_norm == gt_norm:
        return True

    return False


def grade_math_answer(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    pred = extract_math_answer(prediction)
    gt = extract_math_answer(ground_truth)
    acc = math_answer_equal(pred, gt)
    return {
        "acc": bool(acc),
        "score": 1.0 if acc else 0.0,
        "pred": pred,
        "ground_truth": gt,
        "pred_normalized": normalize_math_answer(pred),
        "ground_truth_normalized": normalize_math_answer(gt),
    }
