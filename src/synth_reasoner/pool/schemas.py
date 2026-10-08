"""Dataclasses describing the synthetic pool record schema (spec §10).

These are intentionally pure-Python dataclasses (no torch / pydantic) so
serialization to JSONL is trivial and the module has no heavy dependencies.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskCard:
    teaching_intent: str = ""
    target_reasoning_style: str = ""
    expected_reasoner_weakness: str = ""
    difficulty_estimate: str = ""
    problem: str = ""
    solution: str = ""
    final_answer: str = ""
    verification_notes: str = ""
    novelty_notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskCard":
        kwargs = {f.name: d.get(f.name, "") for f in dataclasses.fields(cls)}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TrainingUnit:
    """Richer pedagogical output (JOINT_NEXT_PHASE_PLAN.md, E3).

    Superset of TaskCard. The Reasoner still receives only `problem`; the extra
    fields are consumed by the verifier / reward / diagnoser. `to_task_card()`
    projects onto the legacy TaskCard so existing verify/reward paths keep working.
    """
    teaching_intent: str = ""
    target_reasoning_style: str = ""
    target_failure_mode: str = ""
    difficulty_estimate: str = ""
    problem: str = ""
    reference_solution: str = ""
    final_answer: str = ""
    key_reasoning_steps: list = field(default_factory=list)
    common_wrong_paths: list = field(default_factory=list)
    verification_checks: list = field(default_factory=list)
    near_variants: list = field(default_factory=list)
    novelty_notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingUnit":
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            default = [] if f.name in (
                "key_reasoning_steps", "common_wrong_paths",
                "verification_checks", "near_variants") else ""
            val = d.get(f.name, default)
            # tolerate a string where a list is expected (wrap it)
            if f.name in ("key_reasoning_steps", "common_wrong_paths",
                          "verification_checks") and isinstance(val, str):
                val = [val] if val else []
            kwargs[f.name] = val
        return cls(**{k: v for k, v in kwargs.items() if k in names})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_task_card(self) -> "TaskCard":
        """Project onto the legacy TaskCard so existing verify/reward paths work."""
        return TaskCard(
            teaching_intent=self.teaching_intent,
            target_reasoning_style=self.target_reasoning_style,
            expected_reasoner_weakness=self.target_failure_mode,
            difficulty_estimate=self.difficulty_estimate,
            problem=self.problem,
            solution=self.reference_solution,
            final_answer=self.final_answer,
            verification_notes="; ".join(str(c) for c in self.verification_checks),
            novelty_notes=self.novelty_notes,
        )


@dataclass
class TaskVerificationRecord:
    is_valid: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class RolloutRecord:
    rollout_id: str
    raw_output: str
    extracted_answer: str | None
    extraction_method: str
    is_correct: bool
    answer_check_method: str
    confidence: float
    reward: float


@dataclass
class SynthesizerRewardRecord:
    quality: float = 0.0
    reliability: float = 0.0
    teachability: float = 0.0
    novelty: float = 0.0
    report_alignment: float = 0.0
    immediate_total: float = 0.0
    delayed_bonus: float | None = None
    total: float = 0.0


@dataclass
class TaskRecord:
    """One full row in the synthetic pool."""

    task_id: str
    iteration: int
    timestamp: str
    synthesizer_model: str
    synthesizer_prompt_hash: str
    raw_synthesizer_output: str
    task_card: TaskCard
    task_verification: TaskVerificationRecord
    reasoner_rollouts: list[RolloutRecord] = field(default_factory=list)
    success_rate: float = 0.0
    synthesizer_reward: SynthesizerRewardRecord = field(
        default_factory=SynthesizerRewardRecord
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "synthesizer_model": self.synthesizer_model,
            "synthesizer_prompt_hash": self.synthesizer_prompt_hash,
            "raw_synthesizer_output": self.raw_synthesizer_output,
            "task_card": self.task_card.to_dict(),
            "task_verification": dataclasses.asdict(self.task_verification),
            "reasoner_rollouts": [dataclasses.asdict(r) for r in self.reasoner_rollouts],
            "success_rate": self.success_rate,
            "synthesizer_reward": dataclasses.asdict(self.synthesizer_reward),
            "metadata": dict(self.metadata),
        }
