"""Build the JSONL of Synthesizer queries consumed by ``star_ppo`` training.

The verl ``RLHFDataset`` reads each row verbatim into ``non_tensor_batch``
and only requires a ``prompt`` field (chat messages list) for chat-template
tokenization. The Synthesizer-Reasoner workflow adapter pulls the rest of
the iteration state (pool/diagnostic/rejection reports, weak/over-generated
terms, recent problems) off the same row via
``TraceWorkflowRunner._extract_from_batch``.

This tool snapshots the SyntheticPool at the chosen run, builds one Report
via the existing ReportBuilder, and emits N identical-state rows (one per
task slot in the iteration) so the trainer can fan out N synthesizer queries
in parallel. The Synthesizer's stochasticity (high temperature) gives each
slot a distinct generation.

The output filename ends in ``.json`` because ``RLHFDataset`` only accepts
``.json`` / ``.parquet``; ``datasets.load_dataset("json")`` handles JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ..agents.prompts import (
    DEFAULT_EMPTY_DIAGNOSTIC_REPORT,
    DEFAULT_EMPTY_POOL_REPORT,
    DEFAULT_EMPTY_REJECTION_SUMMARY,
    SYNTHESIZER_OUTPUT_SCHEMA_TEXT,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_TEMPLATE,
)
from ..pool.storage import RunPaths
from ..pool.synthetic_pool import SyntheticPool
from ..reports.report_builder import LearningProgressSummary, ReportBuilder


def _build_state(
    *,
    pool_root: str | None,
    iteration: int,
    recent_window: int,
    max_report_chars: int,
) -> dict[str, Any]:
    """Snapshot the pool + render reports. Falls back to empty defaults if no pool."""
    if pool_root and os.path.isdir(pool_root):
        paths = RunPaths(root=pool_root)
        pool = SyntheticPool(paths=paths, recent_window=recent_window)
        pool.warm_load()
        records = pool.recent_records()
    else:
        records = []

    if records:
        report = ReportBuilder(max_chars=max_report_chars).build(
            records=records,
            iteration=iteration,
            learning=LearningProgressSummary(),
            recent_window=recent_window,
        )
        return {
            "pool_report": report.rendered or DEFAULT_EMPTY_POOL_REPORT,
            "diagnostic_report": (
                report.diagnostic_rendered or DEFAULT_EMPTY_DIAGNOSTIC_REPORT
            ),
            "rejection_summary": (
                report.rejection_rendered or DEFAULT_EMPTY_REJECTION_SUMMARY
            ),
            "weak_terms": list(report.weak_terms),
            "over_generated_terms": list(report.over_generated_terms),
            "recent_problems": list(report.recent_problems),
        }

    return {
        "pool_report": DEFAULT_EMPTY_POOL_REPORT,
        "diagnostic_report": DEFAULT_EMPTY_DIAGNOSTIC_REPORT,
        "rejection_summary": DEFAULT_EMPTY_REJECTION_SUMMARY,
        "weak_terms": [],
        "over_generated_terms": [],
        "recent_problems": [],
    }


def _render_user_prompt(state: dict[str, Any]) -> str:
    return SYNTHESIZER_USER_TEMPLATE.format(
        pool_report=state["pool_report"],
        reasoner_diagnostic_report=state["diagnostic_report"],
        recent_rejection_summary=state["rejection_summary"],
        output_schema=SYNTHESIZER_OUTPUT_SCHEMA_TEXT,
    )


def _build_row(
    *,
    state: dict[str, Any],
    iteration: int,
    task_index: int,
    user_prompt: str,
) -> dict[str, Any]:
    query_id = f"iter{iteration:04d}-task{task_index:04d}"
    return {
        "query_id": query_id,
        "iteration": int(iteration),
        "task_index": int(task_index),
        "pool_report": state["pool_report"],
        "diagnostic_report": state["diagnostic_report"],
        "rejection_summary": state["rejection_summary"],
        "weak_terms": list(state["weak_terms"]),
        "over_generated_terms": list(state["over_generated_terms"]),
        "recent_problems": list(state["recent_problems"]),
        # ``RLHFDataset`` requires ``prompt`` (chat messages list) for the
        # tokenizer's apply_chat_template; the Synthesizer adapter ignores
        # this in favor of ``synth_inp`` rebuilt from the state fields, but
        # verl uses it when sampling per-engine prompts.
        "prompt": [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # extra_info is required by RLHFDataset.__getitem__ defaults; keep
        # it harmlessly populated so the dataset passes filter checks.
        "extra_info": {
            "index": int(task_index),
            "split": "train",
            "data_source": "synth_reasoner_query",
        },
        "data_source": "synth_reasoner_query",
        # Empty ground-truth list — Reasoner answers are checked against
        # the Synthesizer-emitted gold inside the workflow, not this row.
        "answer": "",
        "reward_model": {"style": "synth_reasoner", "ground_truth": ""},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Snapshot the synth-reasoner pool and emit a training-query JSONL"
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write JSONL (must end in .json so RLHFDataset accepts it)",
    )
    p.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Number of query rows to emit (= tasks per iteration)",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration index recorded on each row (default: 0)",
    )
    p.add_argument(
        "--pool-run-dir",
        default=None,
        help=(
            "Optional path to an existing run directory (e.g. "
            "runs/<run_id>/) whose pool/ subdir holds prior generated tasks. "
            "When omitted, an empty pool is used and the report falls back to "
            "the cold-start defaults."
        ),
    )
    p.add_argument(
        "--recent-window",
        type=int,
        default=500,
        help="Pool warm-load window (matches pool.recent_window in the YAML).",
    )
    p.add_argument(
        "--max-report-chars",
        type=int,
        default=4000,
        help="Truncate the natural-language report at this many chars.",
    )
    args = p.parse_args(argv)

    if not args.output.endswith(".json"):
        raise SystemExit(
            "--output must end in .json (verl's RLHFDataset rejects other "
            "extensions); JSONL contents are still fine."
        )

    state = _build_state(
        pool_root=args.pool_run_dir,
        iteration=args.iteration,
        recent_window=args.recent_window,
        max_report_chars=args.max_report_chars,
    )
    user_prompt = _render_user_prompt(state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for task_index in range(int(args.n_tasks)):
            row = _build_row(
                state=state,
                iteration=int(args.iteration),
                task_index=task_index,
                user_prompt=user_prompt,
            )
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    print(
        f"[prepare_train_jsonl] wrote {args.n_tasks} rows -> {args.output} "
        f"(iteration={args.iteration}, pool_run_dir={args.pool_run_dir})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
