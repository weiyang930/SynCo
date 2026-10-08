"""Materialize math eval benchmarks under ``data/eval/<bench>/test.jsonl``.

We pull every benchmark from HuggingFace so the on-disk corpus is
self-contained and reproducible.
Each adapter normalizes its raw rows into our unified schema:

    {
      "benchmark": str,            # canonical short name
      "id":        str,            # stable per-row identifier
      "problem":   str,            # the question text
      "answer":    str,            # canonical reference answer
      "metadata":  {...}           # optional source-specific extras
    }

Run with:
    python scripts/download_eval_data.py            # gsm8k math500 aime24 aime25
    python scripts/download_eval_data.py gsm8k math500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "eval")


@dataclass
class BenchmarkSpec:
    name: str
    repo: str
    config: str | None
    split: str
    adapter: Callable[[dict, int], dict | None]
    expected_min: int  # sanity floor on row count


def _gsm8k_adapter(row: dict, idx: int) -> dict | None:
    q = (row.get("question") or "").strip()
    a_raw = row.get("answer") or ""
    # GSM8K answers are formatted as "<chain-of-thought>\n#### <number>".
    # We keep only the post-#### numeric token as the canonical answer.
    m = re.search(r"####\s*(.+?)\s*$", a_raw, re.MULTILINE)
    if not (q and m):
        return None
    answer = m.group(1).strip().replace(",", "")
    return {
        "benchmark": "gsm8k",
        "id": f"gsm8k_{idx}",
        "problem": q,
        "answer": answer,
        "metadata": {"raw_answer": a_raw},
    }


def _math500_adapter(row: dict, idx: int) -> dict | None:
    q = (row.get("problem") or "").strip()
    a = (row.get("answer") or "").strip()
    if not (q and a):
        return None
    return {
        "benchmark": "math500",
        "id": str(row.get("unique_id") or f"math500_{idx}"),
        "problem": q,
        "answer": a,
        "metadata": {
            "subject": row.get("subject"),
            "level": row.get("level"),
        },
    }


def _aime_adapter(name: str) -> Callable[[dict, int], dict | None]:
    def _adapt(row: dict, idx: int) -> dict | None:
        q = (row.get("problem") or "").strip()
        a = row.get("answer")
        if a is None or not q:
            return None
        a_str = str(a).strip()
        return {
            "benchmark": name,
            "id": str(row.get("id") if row.get("id") is not None else f"{name}_{idx}"),
            "problem": q,
            "answer": a_str,
            "metadata": {"year": row.get("year"), "url": row.get("url")},
        }
    return _adapt


SPECS: dict[str, BenchmarkSpec] = {
    "gsm8k":    BenchmarkSpec("gsm8k",    "openai/gsm8k",                    "main",                   "test",  _gsm8k_adapter,           1300),
    "math500":  BenchmarkSpec("math500",  "HuggingFaceH4/MATH-500",          None,                     "test",  _math500_adapter,         500),
    "aime24":   BenchmarkSpec("aime24",   "HuggingFaceH4/aime_2024",         None,                     "train", _aime_adapter("aime24"),  30),
    "aime25":   BenchmarkSpec("aime25",   "yentinglin/aime_2025",            None,                     "train", _aime_adapter("aime25"),  30),
}

# Benchmarks handled by this script; the remaining paper benchmarks are
# downloaded by ``fetch_hf_math_benchmarks.py``.
DEFAULT_BENCHMARKS = ("gsm8k", "math500", "aime24", "aime25")


def _materialize(spec: BenchmarkSpec) -> tuple[int, str]:
    from datasets import load_dataset

    if spec.config:
        ds = load_dataset(spec.repo, spec.config, split=spec.split)
    else:
        ds = load_dataset(spec.repo, split=spec.split)

    out_dir = os.path.join(DATA_ROOT, spec.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, row in enumerate(ds):
            record = spec.adapter(dict(row), idx)
            if record is None:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if written < spec.expected_min:
        raise RuntimeError(
            f"{spec.name}: only {written} rows written, "
            f"expected at least {spec.expected_min}"
        )
    return written, out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        help=(
            f"Subset to materialize (default: {list(DEFAULT_BENCHMARKS)}). "
            f"Choices: {sorted(SPECS)}"
        ),
    )
    args = parser.parse_args(argv)

    targets = args.benchmarks or list(DEFAULT_BENCHMARKS)
    unknown = [t for t in targets if t not in SPECS]
    if unknown:
        parser.error(f"unknown benchmark(s): {unknown}")

    print(f"[download_eval_data] writing to {DATA_ROOT}")
    for name in targets:
        spec = SPECS[name]
        print(f"  - {name}  <-  {spec.repo}"
              + (f" ({spec.config})" if spec.config else "")
              + f"  split={spec.split}")
        try:
            n, path = _materialize(spec)
        except Exception as e:
            print(f"    [FAIL] {type(e).__name__}: {e}")
            return 1
        print(f"    [ok] {n} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
