"""Build the natural-language report consumed by the Synthesizer prompt.

Spec §11 specifies the six sections. The Report carries both:
- ``rendered``: the natural-language string injected into the prompt.
- structured fields used by the reward computation
  (``weak_terms``, ``over_generated_terms``, ``recent_problems``, etc.).

This module contains no I/O; the workflow is responsible for persisting
the JSON form under ``runs/{run_id}/reports/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..pool.schemas import TaskRecord
from .diagnostics import (
    DiversitySummary,
    GenerationStats,
    ReasonerStats,
    RejectionSummary,
    compute_diversity,
    compute_generation_stats,
    compute_reasoner_stats,
    compute_rejection_summary,
)


@dataclass
class LearningProgressSummary:
    latest_eval_score: float | None = None
    previous_eval_score: float | None = None
    global_gain: float | None = None
    cluster_gains: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    iteration: int
    rendered: str  # natural-language form for Synthesizer prompt
    diagnostic_rendered: str  # the "Reasoner diagnostic report" sub-block
    rejection_rendered: str  # the "Recent rejection summary" sub-block
    generation: GenerationStats
    reasoner: ReasonerStats
    diversity: DiversitySummary
    rejection: RejectionSummary
    learning_progress: LearningProgressSummary
    weak_terms: list[str] = field(default_factory=list)
    over_generated_terms: list[str] = field(default_factory=list)
    recent_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "rendered": self.rendered,
            "diagnostic_rendered": self.diagnostic_rendered,
            "rejection_rendered": self.rejection_rendered,
            "generation": self.generation.__dict__,
            "reasoner": self.reasoner.__dict__,
            "diversity": {
                "frequent_styles": self.diversity.frequent_styles,
                "over_generated": self.diversity.over_generated,
                "under_explored_hint": self.diversity.under_explored_hint,
            },
            "rejection": {"common_reasons": self.rejection.common_reasons},
            "learning_progress": self.learning_progress.__dict__,
            "weak_terms": self.weak_terms,
            "over_generated_terms": self.over_generated_terms,
        }


class ReportBuilder:
    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def build(
        self,
        *,
        records: Iterable[TaskRecord],
        iteration: int,
        learning: LearningProgressSummary | None = None,
        recent_window: int | None = None,
    ) -> Report:
        records = list(records)
        if recent_window is not None and len(records) > recent_window:
            records = records[-recent_window:]

        gen = compute_generation_stats(records)
        reas = compute_reasoner_stats(records)
        div = compute_diversity(records)
        rej = compute_rejection_summary(records)
        learning = learning or LearningProgressSummary()

        recent_problems = [r.task_card.problem for r in records if r.task_card.problem]

        weak_terms, over_terms = self._extract_terms(reas, div)

        rendered = self._render_main(gen, reas, div, rej, learning)
        diag = self._render_diagnostic(reas, weak_terms)
        rejection_block = self._render_rejection(rej)

        if len(rendered) > self.max_chars:
            rendered = rendered[: self.max_chars - 20] + "\n... (truncated)"

        return Report(
            iteration=iteration,
            rendered=rendered,
            diagnostic_rendered=diag,
            rejection_rendered=rejection_block,
            generation=gen,
            reasoner=reas,
            diversity=div,
            rejection=rej,
            learning_progress=learning,
            weak_terms=weak_terms,
            over_generated_terms=over_terms,
            recent_problems=recent_problems,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_terms(
        reas: ReasonerStats,
        div: DiversitySummary,
    ) -> tuple[list[str], list[str]]:
        weak_terms = list(div.under_explored_hint)
        over_terms = list(div.over_generated)
        if reas.too_easy_count > reas.learnable_count:
            over_terms.append("too easy")
        if reas.too_hard_count > reas.learnable_count:
            weak_terms.append("hard but well-posed")
        return weak_terms, over_terms

    @staticmethod
    def _render_main(
        gen: GenerationStats,
        reas: ReasonerStats,
        div: DiversitySummary,
        rej: RejectionSummary,
        learning: LearningProgressSummary,
    ) -> str:
        if gen.n_total == 0:
            return (
                "No historical synthetic pool exists yet.\n"
                "Generate a diverse, valid, pedagogically useful mathematical reasoning task."
            )
        lines = [
            "Recent synthetic pool summary:",
            f"- {gen.n_total} tasks generated, {gen.n_valid} valid, {gen.n_invalid} rejected.",
            f"- Valid rate: {gen.valid_rate:.2f}; duplicate-flag rate: {gen.duplicate_rate:.2f}.",
            f"- Average Synthesizer reward: {gen.avg_synth_reward:.3f}.",
            "",
            "Reasoner performance on valid tasks:",
            f"- Average success rate: {reas.avg_success_rate:.2f}.",
            f"- Distribution: too_easy={reas.too_easy_count}, learnable={reas.learnable_count}, "
            f"too_hard={reas.too_hard_count}, mid_band={reas.midband_count}.",
        ]

        if div.frequent_styles:
            top = ", ".join(f"{w}({c})" for w, c in div.frequent_styles[:5])
            lines += ["", "Frequent reasoning-style terms:", f"- {top}."]
        if div.over_generated:
            lines += [f"- Over-generated patterns: {', '.join(div.over_generated)}."]
        if div.under_explored_hint:
            lines += [f"- Under-explored hint terms: {', '.join(div.under_explored_hint)}."]

        if rej.common_reasons:
            lines += ["", "Common rejection reasons:"]
            lines += [f"- {reason} ({count})." for reason, count in rej.common_reasons]

        if learning.global_gain is not None:
            lines += [
                "",
                "Learning progress:",
                f"- Latest eval score: {learning.latest_eval_score}.",
                f"- Previous eval score: {learning.previous_eval_score}.",
                f"- Global gain: {learning.global_gain:+.3f}.",
            ]

        lines += [
            "",
            "Recommendation for next batch:",
            "- Generate more problems requiring multi-step exact reasoning, hidden constraints, "
            "and verifiable final answers.",
            "- Avoid problems that depend on a diagram, are ambiguous, or use known benchmark phrasing.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_diagnostic(reas: ReasonerStats, weak_terms: list[str]) -> str:
        if reas.avg_success_rate == 0 and reas.invalid_count == 0:
            return "No Reasoner diagnostic information available yet."
        lines = [
            f"Reasoner average success rate on valid tasks: {reas.avg_success_rate:.2f}.",
        ]
        if reas.too_easy_count > 0 or reas.too_hard_count > 0:
            lines.append(
                f"Recent extremes: {reas.too_easy_count} too-easy, "
                f"{reas.too_hard_count} too-hard tasks."
            )
        if weak_terms:
            lines.append(
                "Reasoner appears weakest on tasks involving: " + ", ".join(weak_terms) + "."
            )
        return "\n".join(lines)

    @staticmethod
    def _render_rejection(rej: RejectionSummary) -> str:
        if not rej.common_reasons:
            return "No recent rejections."
        return "; ".join(f"{r} (x{c})" for r, c in rej.common_reasons)
