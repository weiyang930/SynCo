"""Reasoner reward (spec §9.1).

Default behavior:
- correct + boxed -> 1.0
- correct without boxed -> 0.9
- incorrect -> 0.0 (or a small format-only reward if configured)

Uses ``ExtractedAnswer`` and ``AnswerCheckResult`` to compute per-rollout
rewards. Fully deterministic / stateless.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..parsing.answer_extraction import ExtractedAnswer
from ..verification.answer_checker import AnswerCheckResult


@dataclass
class ReasonerRewardConfig:
    use_format_bonus: bool = True
    format_bonus: float = 0.1
    normalize_to_unit: bool = True
    incorrect_format_only_reward: float = 0.0  # reward when boxed present but answer wrong


@dataclass
class ReasonerRewardBreakdown:
    answer_correctness: float
    format_bonus: float
    total: float


def compute_reasoner_reward(
    extracted: ExtractedAnswer,
    check: AnswerCheckResult,
    config: ReasonerRewardConfig,
) -> ReasonerRewardBreakdown:
    correctness = 1.0 if check.is_correct else 0.0
    bonus = config.format_bonus if (config.use_format_bonus and extracted.has_boxed) else 0.0

    if config.normalize_to_unit:
        if check.is_correct and extracted.has_boxed:
            total = 1.0
        elif check.is_correct:
            total = 0.9
        else:
            total = config.incorrect_format_only_reward if extracted.has_boxed else 0.0
    else:
        total = correctness + bonus

    return ReasonerRewardBreakdown(
        answer_correctness=correctness,
        format_bonus=bonus,
        total=total,
    )
