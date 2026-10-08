"""Persistent synthetic pool of generated tasks.

Backed by JSONL files under ``runs/{run_id}/pool/``. The pool keeps an
in-memory ring buffer of recent tasks for cheap dedup / report queries
without re-reading files on every iteration.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Iterable, Iterator

from .schemas import TaskRecord
from .storage import RunPaths


class SyntheticPool:
    def __init__(self, paths: RunPaths, recent_window: int = 500) -> None:
        self.paths = paths
        self.recent_window = recent_window
        # In-memory windows of recent task records and recent problem texts.
        self._recent: deque[TaskRecord] = deque(maxlen=recent_window)
        self._recent_problems: deque[str] = deque(maxlen=max(recent_window * 2, 1000))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record: TaskRecord) -> None:
        """Persist a record and update in-memory windows."""
        self._append_jsonl(self.paths.generated_tasks_path, record.to_dict())
        if record.task_verification.is_valid:
            self._append_jsonl(self.paths.valid_tasks_path, record.to_dict())
        else:
            self._append_jsonl(self.paths.invalid_tasks_path, record.to_dict())

        self._recent.append(record)
        if record.task_card.problem:
            self._recent_problems.append(record.task_card.problem)

    def add_rollout_log(self, rollout: dict) -> None:
        self._append_jsonl(self.paths.reasoner_rollouts_path, rollout)

    def add_reward_log(self, reward: dict) -> None:
        self._append_jsonl(self.paths.rewards_path, reward)

    def add_delayed_reward_log(self, payload: dict) -> None:
        self._append_jsonl(self.paths.delayed_rewards_path, payload)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def recent_records(self, n: int | None = None) -> list[TaskRecord]:
        if n is None or n >= len(self._recent):
            return list(self._recent)
        return list(self._recent)[-n:]

    def recent_problems(self, n: int | None = None) -> list[str]:
        if n is None or n >= len(self._recent_problems):
            return list(self._recent_problems)
        return list(self._recent_problems)[-n:]

    def __len__(self) -> int:  # number of in-memory recent records
        return len(self._recent)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: str, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")

    @staticmethod
    def iter_jsonl(path: str) -> Iterator[dict]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def warm_load(self, paths: Iterable[str] | None = None) -> int:
        """Load existing JSONL records into memory (e.g., on resume).

        Returns the number of records loaded.
        """
        sources = list(paths) if paths is not None else [self.paths.generated_tasks_path]
        loaded = 0
        for path in sources:
            try:
                for obj in self.iter_jsonl(path):
                    rec = _record_from_dict(obj)
                    if rec is not None:
                        self._recent.append(rec)
                        if rec.task_card.problem:
                            self._recent_problems.append(rec.task_card.problem)
                        loaded += 1
            except FileNotFoundError:
                continue
        return loaded


def _record_from_dict(obj: dict) -> TaskRecord | None:
    """Best-effort reconstruction of a TaskRecord from a dict on disk.

    Returns ``None`` if the row is too malformed; this lets warm_load
    skip junk without crashing.
    """
    from .schemas import (
        RolloutRecord,
        SynthesizerRewardRecord,
        TaskCard,
        TaskVerificationRecord,
    )

    try:
        card = TaskCard.from_dict(obj.get("task_card", {}))
        ver = obj.get("task_verification", {}) or {}
        verification = TaskVerificationRecord(
            is_valid=bool(ver.get("is_valid", False)),
            scores=dict(ver.get("scores", {}) or {}),
            reasons=list(ver.get("reasons", []) or []),
        )
        rollouts = [
            RolloutRecord(
                rollout_id=r.get("rollout_id", ""),
                raw_output=r.get("raw_output", ""),
                extracted_answer=r.get("extracted_answer"),
                extraction_method=r.get("extraction_method", "none"),
                is_correct=bool(r.get("is_correct", False)),
                answer_check_method=r.get("answer_check_method", "none"),
                confidence=float(r.get("confidence", 0.0)),
                reward=float(r.get("reward", 0.0)),
            )
            for r in (obj.get("reasoner_rollouts") or [])
        ]
        rew = obj.get("synthesizer_reward", {}) or {}
        synth_reward = SynthesizerRewardRecord(
            quality=float(rew.get("quality", 0.0)),
            reliability=float(rew.get("reliability", 0.0)),
            teachability=float(rew.get("teachability", 0.0)),
            novelty=float(rew.get("novelty", 0.0)),
            report_alignment=float(rew.get("report_alignment", 0.0)),
            immediate_total=float(rew.get("immediate_total", 0.0)),
            delayed_bonus=rew.get("delayed_bonus"),
            total=float(rew.get("total", 0.0)),
        )
        return TaskRecord(
            task_id=str(obj.get("task_id", "")),
            iteration=int(obj.get("iteration", 0)),
            timestamp=str(obj.get("timestamp", "")),
            synthesizer_model=str(obj.get("synthesizer_model", "")),
            synthesizer_prompt_hash=str(obj.get("synthesizer_prompt_hash", "")),
            raw_synthesizer_output=str(obj.get("raw_synthesizer_output", "")),
            task_card=card,
            task_verification=verification,
            reasoner_rollouts=rollouts,
            success_rate=float(obj.get("success_rate", 0.0)),
            synthesizer_reward=synth_reward,
            metadata=dict(obj.get("metadata", {}) or {}),
        )
    except (TypeError, ValueError):
        return None
