"""Helper analytics over recent pool records (used by ReportBuilder)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ..pool.schemas import TaskRecord


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


@dataclass
class GenerationStats:
    n_total: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    valid_rate: float = 0.0
    invalid_rate: float = 0.0
    duplicate_rate: float = 0.0
    avg_synth_reward: float = 0.0


@dataclass
class ReasonerStats:
    avg_success_rate: float = 0.0
    too_easy_count: int = 0      # success_rate >= 0.95
    learnable_count: int = 0     # 0.2 <= success_rate <= 0.8
    too_hard_count: int = 0      # success_rate < 0.05
    midband_count: int = 0       # everything else
    invalid_count: int = 0       # tasks not even verified


@dataclass
class DiversitySummary:
    frequent_styles: list[tuple[str, int]] = field(default_factory=list)
    over_generated: list[str] = field(default_factory=list)
    under_explored_hint: list[str] = field(default_factory=list)


@dataclass
class RejectionSummary:
    common_reasons: list[tuple[str, int]] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def compute_generation_stats(records: Iterable[TaskRecord]) -> GenerationStats:
    records = list(records)
    n = len(records)
    if n == 0:
        return GenerationStats()
    n_valid = sum(1 for r in records if r.task_verification.is_valid)
    n_invalid = n - n_valid
    dup_rate = sum(
        1
        for r in records
        if (r.task_verification.scores.get("dedup_score", 1.0)) < 0.3
    ) / n
    avg_reward = sum(r.synthesizer_reward.total for r in records) / n
    return GenerationStats(
        n_total=n,
        n_valid=n_valid,
        n_invalid=n_invalid,
        valid_rate=n_valid / n,
        invalid_rate=n_invalid / n,
        duplicate_rate=dup_rate,
        avg_synth_reward=avg_reward,
    )


def compute_reasoner_stats(records: Iterable[TaskRecord]) -> ReasonerStats:
    records = list(records)
    if not records:
        return ReasonerStats()
    success_rates: list[float] = []
    too_easy = learnable = too_hard = mid = invalid = 0
    for r in records:
        if not r.task_verification.is_valid:
            invalid += 1
            continue
        sr = r.success_rate
        success_rates.append(sr)
        if sr >= 0.95:
            too_easy += 1
        elif sr < 0.05:
            too_hard += 1
        elif 0.2 <= sr <= 0.8:
            learnable += 1
        else:
            mid += 1
    avg = sum(success_rates) / len(success_rates) if success_rates else 0.0
    return ReasonerStats(
        avg_success_rate=avg,
        too_easy_count=too_easy,
        learnable_count=learnable,
        too_hard_count=too_hard,
        midband_count=mid,
        invalid_count=invalid,
    )


def compute_diversity(records: Iterable[TaskRecord], top_k: int = 8) -> DiversitySummary:
    style_counter: Counter[str] = Counter()
    for r in records:
        for tok in _tokens(r.task_card.target_reasoning_style):
            if len(tok) >= 4:
                style_counter[tok] += 1
    frequent = style_counter.most_common(top_k)

    # Heuristic over-generated: anything with > 30% share of tokens.
    total = sum(c for _, c in frequent)
    over = [w for w, c in frequent if total > 0 and c / total > 0.3]

    # A small canned list of buckets we want to encourage when missing.
    canon = [
        "modular",
        "combinator",
        "geometry",
        "probability",
        "polynomial",
        "diophantine",
        "constraint",
        "sequence",
    ]
    seen = {w for w, _ in frequent}
    under = [w for w in canon if w not in seen]
    return DiversitySummary(
        frequent_styles=frequent,
        over_generated=over,
        under_explored_hint=under,
    )


def compute_rejection_summary(records: Iterable[TaskRecord], top_k: int = 5) -> RejectionSummary:
    counter: Counter[str] = Counter()
    for r in records:
        if r.task_verification.is_valid:
            continue
        for reason in r.task_verification.reasons:
            counter[reason] += 1
    return RejectionSummary(common_reasons=counter.most_common(top_k))
