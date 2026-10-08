"""AnswerChecker: compare a Reasoner-extracted answer against a reference.

Uses the layered strategy from spec §8.2:
1. Exact normalized string match.
2. Numeric equivalence (decimal, fraction, percentage).
3. Sympy equivalence for parseable algebraic expressions.
4. Set/tuple normalization.
5. Optional LLM judge (gated by config; never required).

A ``method`` string and a ``confidence`` are returned alongside the
boolean so the pool can record *why* something matched.
"""
from __future__ import annotations

from dataclasses import dataclass

from .math_equivalence import (
    normalize_string,
    numeric_equivalent,
    set_or_tuple_equivalent,
    sympy_equivalent,
)


@dataclass
class AnswerCheckResult:
    is_correct: bool
    confidence: float
    method: str  # "exact", "numeric", "sympy", "set_tuple", "llm_judge", "none"
    normalized_prediction: str
    normalized_reference: str
    reason: str


class AnswerChecker:
    """Stateless checker. Configure via constructor."""

    def __init__(
        self,
        use_sympy: bool = True,
        numeric_tolerance: float = 1e-6,
        llm_judge=None,
    ) -> None:
        self.use_sympy = use_sympy
        self.numeric_tolerance = numeric_tolerance
        # Optional callable: ``(pred, ref) -> bool``. Disabled by default.
        self.llm_judge = llm_judge

    def check(self, prediction: str | None, reference: str | None) -> AnswerCheckResult:
        norm_pred = normalize_string(prediction or "")
        norm_ref = normalize_string(reference or "")

        if not norm_pred or not norm_ref:
            return AnswerCheckResult(
                is_correct=False,
                confidence=0.0,
                method="none",
                normalized_prediction=norm_pred,
                normalized_reference=norm_ref,
                reason="empty prediction or reference",
            )

        # 1. Exact normalized.
        if norm_pred == norm_ref:
            return AnswerCheckResult(
                is_correct=True,
                confidence=1.0,
                method="exact",
                normalized_prediction=norm_pred,
                normalized_reference=norm_ref,
                reason="exact normalized string match",
            )

        # 2. Numeric.
        if numeric_equivalent(norm_pred, norm_ref, self.numeric_tolerance):
            return AnswerCheckResult(
                is_correct=True,
                confidence=0.95,
                method="numeric",
                normalized_prediction=norm_pred,
                normalized_reference=norm_ref,
                reason="numeric equivalence",
            )

        # 3. Sympy.
        if self.use_sympy and sympy_equivalent(norm_pred, norm_ref):
            return AnswerCheckResult(
                is_correct=True,
                confidence=0.9,
                method="sympy",
                normalized_prediction=norm_pred,
                normalized_reference=norm_ref,
                reason="sympy symbolic equivalence",
            )

        # 4. Set / tuple.
        st = set_or_tuple_equivalent(norm_pred, norm_ref, self.numeric_tolerance)
        if st is True:
            return AnswerCheckResult(
                is_correct=True,
                confidence=0.85,
                method="set_tuple",
                normalized_prediction=norm_pred,
                normalized_reference=norm_ref,
                reason="set/tuple element-wise equivalence",
            )

        # 5. LLM judge (optional).
        if self.llm_judge is not None:
            try:
                if bool(self.llm_judge(norm_pred, norm_ref)):
                    return AnswerCheckResult(
                        is_correct=True,
                        confidence=0.7,
                        method="llm_judge",
                        normalized_prediction=norm_pred,
                        normalized_reference=norm_ref,
                        reason="LLM judge accepted",
                    )
            except Exception as e:  # noqa: BLE001
                return AnswerCheckResult(
                    is_correct=False,
                    confidence=0.0,
                    method="none",
                    normalized_prediction=norm_pred,
                    normalized_reference=norm_ref,
                    reason=f"LLM judge raised: {e!r}",
                )

        return AnswerCheckResult(
            is_correct=False,
            confidence=0.0,
            method="none",
            normalized_prediction=norm_pred,
            normalized_reference=norm_ref,
            reason="no equivalence layer matched",
        )
