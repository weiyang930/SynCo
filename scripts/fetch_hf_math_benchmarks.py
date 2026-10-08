#!/usr/bin/env python3
"""Fetch the Hugging Face math benchmarks used in the SynCo paper and normalize them to the local
eval format: data/eval/<name>/test.jsonl with keys {benchmark,id,problem,answer,metadata}.

Only SHORT-ANSWER benchmarks (numeric / expression / set) are included so the
existing AnswerChecker (exact/numeric/sympy/set_tuple) can grade them reliably.
Run: python scripts/fetch_hf_math_benchmarks.py
"""
from __future__ import annotations
import json, os, re
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO_ROOT, "data", "eval")


def _write(name: str, rows: list[dict]) -> None:
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "test.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {name:14s} n={len(rows):5d} -> {path}")


def _boxed(ans: str) -> str:
    """Strip \\boxed{...} / surrounding $ if present, keep the inner answer."""
    ans = ans.strip()
    m = re.search(r"\\boxed\{(.+?)\}\s*\$?\s*$", ans)
    if m:
        return m.group(1).strip()
    return ans.strip("$ ").strip()


def build_minerva() -> None:
    ds = load_dataset("math-ai/minervamath", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({"benchmark": "minerva", "id": str(i),
                     "problem": r["question"].strip(),
                     "answer": _boxed(str(r["answer"])),
                     "metadata": {}})
    _write("minerva", rows)


def build_gsm_hard() -> None:
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    rows = []
    for i, r in enumerate(ds):
        tgt = r["target"]
        # targets are floats; render ints without trailing .0
        if isinstance(tgt, float) and tgt.is_integer():
            tgt = int(tgt)
        rows.append({"benchmark": "gsm_hard", "id": str(i),
                     "problem": str(r["input"]).strip(),
                     "answer": str(tgt),
                     "metadata": {}})
    _write("gsm_hard", rows)


def build_svamp() -> None:
    ds = load_dataset("ChilleD/SVAMP", split="test")
    rows = []
    for i, r in enumerate(ds):
        body = str(r["Body"]).strip()
        q = str(r["Question"]).strip()
        prob = (body + " " + q).strip() if body and not body.endswith(("?", ".")) else (body + " " + q).strip()
        ans = r["Answer"]
        if isinstance(ans, float) and ans.is_integer():
            ans = int(ans)
        rows.append({"benchmark": "svamp", "id": str(r.get("ID", i)),
                     "problem": prob,
                     "answer": str(ans),
                     "metadata": {"type": r.get("Type", "")}})
    _write("svamp", rows)


def build_asdiv() -> None:
    ds = load_dataset("MU-NLPC/Calc-asdiv_a", split="test")
    rows = []
    for i, r in enumerate(ds):
        ans = r.get("result_float", None)
        if ans is None or (isinstance(ans, float) and ans != ans):  # NaN
            ans = r.get("result", "")
        if isinstance(ans, float) and ans.is_integer():
            ans = int(ans)
        rows.append({"benchmark": "asdiv", "id": str(r.get("id", i)),
                     "problem": str(r["question"]).strip(),
                     "answer": str(ans),
                     "metadata": {"grade": r.get("grade", "")}})
    _write("asdiv", rows)


if __name__ == "__main__":
    for fn in (build_minerva, build_gsm_hard, build_svamp, build_asdiv):
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
