"""TaskVerifier: validate a Synthesizer-produced task card.

Spec §8.1 calls for five categories of checks (schema / problem / answer /
format / dedup) and a `TaskVerificationResult` dataclass with per-category
scores plus an overall_quality_score in [0, 1]. We follow that contract.

The dedup score is computed against an externally-supplied list of recent
task texts so this module stays pure and stateless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..agents.prompts import SYNTHESIZER_REQUIRED_FIELDS
from ..pool.dedup import ngram_max_overlap


_DIAGRAM_PATTERNS = (
    "as shown in the diagram",
    "see the figure",
    "in the figure below",
    "as illustrated",
    "shown in the picture",
    "the diagram shows",
)
_BENCHMARK_NAMES = ("gsm8k", "amc", "aime", "math500", "math 500")


@dataclass
class TaskVerificationScores:
    schema_score: float = 0.0
    problem_score: float = 0.0
    answer_score: float = 0.0
    format_score: float = 0.0
    dedup_score: float = 0.0
    overall_quality_score: float = 0.0


@dataclass
class TaskVerificationResult:
    is_valid: bool
    scores: TaskVerificationScores
    reasons: list[str] = field(default_factory=list)


@dataclass
class TaskVerifierConfig:
    min_problem_chars: int = 20
    max_problem_chars: int = 3000
    max_solution_chars: int = 6000
    dedup_overlap_threshold: float = 0.85  # >= threshold → near-duplicate
    require_final_answer_in_solution_substring: bool = False


class TaskVerifier:
    def __init__(self, config: TaskVerifierConfig | None = None) -> None:
        self.config = config or TaskVerifierConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        task_card: dict | None,
        *,
        json_parse_ok: bool = True,
        recent_problems: Iterable[str] = (),
    ) -> TaskVerificationResult:
        scores = TaskVerificationScores()
        reasons: list[str] = []

        scores.format_score = 1.0 if json_parse_ok else 0.0
        if not json_parse_ok:
            reasons.append("synthesizer output failed JSON parse")

        if task_card is None or not isinstance(task_card, dict):
            scores.overall_quality_score = 0.0
            return TaskVerificationResult(False, scores, reasons or ["no task card"])

        scores.schema_score = self._score_schema(task_card, reasons)
        scores.problem_score = self._score_problem(task_card, reasons)
        scores.answer_score = self._score_answer(task_card, reasons)
        scores.dedup_score = self._score_dedup(task_card, recent_problems, reasons)

        scores.overall_quality_score = self._aggregate(scores)
        is_valid = (
            json_parse_ok
            and scores.schema_score >= 1.0
            and scores.problem_score >= 0.5
            and scores.answer_score >= 0.5
            and scores.dedup_score >= 0.3
        )
        return TaskVerificationResult(is_valid=is_valid, scores=scores, reasons=reasons)

    # ------------------------------------------------------------------
    # Per-category checks
    # ------------------------------------------------------------------

    def _score_schema(self, card: dict, reasons: list[str]) -> float:
        missing = [f for f in SYNTHESIZER_REQUIRED_FIELDS if not card.get(f)]
        if missing:
            reasons.append(f"missing/empty fields: {missing}")
            return 0.0
        return 1.0

    def _score_problem(self, card: dict, reasons: list[str]) -> float:
        problem = str(card.get("problem", "")).strip()
        score = 1.0
        if not problem:
            reasons.append("problem empty")
            return 0.0
        if len(problem) < self.config.min_problem_chars:
            reasons.append("problem too short")
            score *= 0.3
        if len(problem) > self.config.max_problem_chars:
            reasons.append("problem too long")
            score *= 0.3
        lowered = problem.lower()
        if any(p in lowered for p in _DIAGRAM_PATTERNS):
            reasons.append("problem references a diagram")
            score *= 0.3
        if any(b in lowered for b in _BENCHMARK_NAMES):
            reasons.append("problem mentions a benchmark name")
            score *= 0.5
        if "solution:" in lowered or "answer:" in lowered:
            reasons.append("problem appears to leak solution/answer text")
            score *= 0.5
        return max(0.0, min(score, 1.0))

    def _score_answer(self, card: dict, reasons: list[str]) -> float:
        final_answer = str(card.get("final_answer", "")).strip()
        solution = str(card.get("solution", "")).strip()
        if not final_answer:
            reasons.append("final_answer empty")
            return 0.0
        score = 1.0
        if not solution:
            reasons.append("solution empty")
            score *= 0.4
        elif len(solution) > self.config.max_solution_chars:
            reasons.append("solution too long")
            score *= 0.5
        # Soft signal: does the final answer string appear inside the
        # last 25% of the solution? Many solutions end with the answer.
        if solution and final_answer.lower() not in solution.lower():
            tail = solution[max(0, int(len(solution) * 0.6)) :].lower()
            # Try a relaxed alphanumeric-only match.
            relaxed_tail = re.sub(r"\W+", "", tail)
            relaxed_ans = re.sub(r"\W+", "", final_answer.lower())
            if relaxed_ans and relaxed_ans in relaxed_tail:
                pass  # OK
            else:
                reasons.append("final_answer not found near end of solution")
                score *= 0.7
                if self.config.require_final_answer_in_solution_substring:
                    score *= 0.3
        return max(0.0, min(score, 1.0))

    def _score_dedup(
        self, card: dict, recent_problems: Iterable[str], reasons: list[str]
    ) -> float:
        problem = str(card.get("problem", "")).strip()
        if not problem:
            return 0.0
        recent = [p for p in recent_problems if p]
        if not recent:
            return 1.0  # nothing to compare against → maximally novel
        max_overlap = ngram_max_overlap(problem, recent, n=4)
        if max_overlap >= self.config.dedup_overlap_threshold:
            reasons.append(f"near-duplicate of recent task (overlap={max_overlap:.2f})")
            return max(0.0, 1.0 - max_overlap)
        return float(max(0.0, 1.0 - max_overlap))

    @staticmethod
    def _aggregate(scores: TaskVerificationScores) -> float:
        # Weight schema/problem/answer most; format is a hard gate; dedup is a tiebreaker.
        return (
            0.30 * scores.schema_score
            + 0.30 * scores.problem_score
            + 0.25 * scores.answer_score
            + 0.05 * scores.format_score
            + 0.10 * scores.dedup_score
        )
