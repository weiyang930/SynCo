"""Evaluate a (merged, Hugging Face format) checkpoint on the math benchmarks.

Protocol (the one used for all numbers in the paper):
  * prompt  = Reasoner system prompt + Reasoner user template, rendered with the model's chat
              template (Qwen3 default thinking mode);
  * decoding: temperature 0.6, top_p 0.95, one sample per problem, per-benchmark max_tokens
              (MAX_TOKENS below);
  * answer  = last \\boxed{} (with fallbacks, ``extract_reasoner_answer``), graded by
              ``AnswerChecker`` (exact -> numeric -> sympy -> set/tuple).

Two sub-commands:
  run        evaluate one shard of every benchmark on the visible GPU(s); example j of a
             benchmark goes to shard j % num_shards, so N processes (one per GPU) split the work.
  aggregate  merge all shards, check completeness, write accuracy.json.

  python scripts/evaluate.py run --model-path M --out-dir O [--shard-id 0 --num-shards 8]
  python scripts/evaluate.py aggregate --out-dir O
(scripts/eval_checkpoint.sh does both on one 8-GPU node.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from synth_reasoner.agents.prompts import REASONER_SYSTEM_PROMPT, REASONER_USER_TEMPLATE  # noqa: E402
from synth_reasoner.evaluation.benchmark_loader import load_benchmark  # noqa: E402
from synth_reasoner.parsing.answer_extraction import extract_reasoner_answer  # noqa: E402
from synth_reasoner.verification.answer_checker import AnswerChecker  # noqa: E402

# Main-table benchmarks.
DEFAULT_BENCHMARKS = ["gsm8k", "svamp", "asdiv", "gsm_hard", "math500", "aime24", "aime25", "minerva"]
MAX_TOKENS = {
    "gsm8k": 4096,
    "svamp": 2048,
    "asdiv": 2048,
    "gsm_hard": 4096,
    "math500": 8192,
    "aime24": 32768,
    "aime25": 32768,
    "minerva": 16384,
    "amc23": 16384,
    "olympiad": 32768,
    "gaokao2023en": 16384,
}
TEMPERATURE, TOP_P = 0.6, 0.95


def bench_path(name: str) -> str:
    return os.path.join(ROOT, "data", "eval", name, "test.jsonl")


def build_prompt(problem: str, tokenizer) -> str:
    msgs = [
        {"role": "system", "content": REASONER_SYSTEM_PROMPT},
        {"role": "user", "content": REASONER_USER_TEMPLATE.format(problem=problem)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def cmd_run(args) -> int:
    work = []  # (bench, example_index, example)
    for b in args.benchmarks:
        for j, ex in enumerate(load_benchmark(b, bench_path(b), args.max_examples)):
            if args.max_examples > 0 or j % args.num_shards == args.shard_id:
                work.append((b, j, ex))
    print(f"[eval] shard {args.shard_id}/{args.num_shards}: {len(work)} problems", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16", trust_remote_code=True)
    prompts = [build_prompt(ex.problem, tokenizer) for _, _, ex in work]
    params = {b: SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS[b]) for b in args.benchmarks}
    t0 = time.time()
    outputs = llm.generate(prompts, [params[b] for b, _, _ in work])
    print(f"[eval] generation {time.time() - t0:.0f}s", flush=True)

    checker = AnswerChecker()
    files = {}
    for (b, j, ex), out in zip(work, outputs):
        if b not in files:
            d = os.path.join(args.out_dir, "raw", b)
            os.makedirs(d, exist_ok=True)
            files[b] = open(os.path.join(d, f"shard_{args.shard_id}.jsonl"), "w", encoding="utf-8")
        o = out.outputs[0]
        ext = extract_reasoner_answer(o.text)
        try:
            chk = checker.check(ext.answer, ex.reference_answer)
            correct, method, err = bool(chk.is_correct), chk.method, None
        except Exception as e:  # recorded, never silently dropped
            correct, method, err = False, "error", repr(e)
        files[b].write(json.dumps({
            "benchmark": b, "example_index": j, "id": ex.extra.get("id") or "", "problem": ex.problem,
            "completion": o.text, "predicted_answer": ext.answer, "extraction_method": ext.method,
            "has_boxed": ext.has_boxed, "reference_answer": ex.reference_answer, "check_method": method,
            "verifier_error": err, "is_correct": correct, "finish_reason": o.finish_reason,
            "num_output_tokens": len(o.token_ids),
        }, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    return 0


def cmd_aggregate(args) -> int:
    acc, checks = {}, {}
    for b in args.benchmarks:
        gold = load_benchmark(b, bench_path(b), 0)
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(args.out_dir, "raw", b, "*.jsonl"))) for l in open(f)]
        rows.sort(key=lambda r: r["example_index"])
        complete = [r["example_index"] for r in rows] == list(range(len(gold)))
        a = 100.0 * sum(r["is_correct"] for r in rows) / max(1, len(rows))
        acc[b] = round(a, 2) if complete else None
        checks[b] = {"n": len(rows), "expected": len(gold), "complete": complete, "accuracy": a,
                     "no_boxed": sum(not r["has_boxed"] for r in rows),
                     "length_truncated": sum(r["finish_reason"] == "length" for r in rows)}
    vals = [v for v in acc.values() if v is not None]
    acc["average"] = round(sum(vals) / len(vals), 2) if len(vals) == len(args.benchmarks) else None
    json.dump(acc, open(os.path.join(args.out_dir, "accuracy.json"), "w"), indent=2)
    json.dump(checks, open(os.path.join(args.out_dir, "eval_checks.json"), "w"), indent=2)
    print(json.dumps(acc))
    return 0 if acc["average"] is not None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-path", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, choices=sorted(MAX_TOKENS))
    r.add_argument("--shard-id", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--tensor-parallel-size", type=int, default=1)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--max-examples", type=int, default=0, help="smoke test: first N problems per benchmark")
    g = sub.add_parser("aggregate")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, choices=sorted(MAX_TOKENS))
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
