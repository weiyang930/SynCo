"""Synthesizer reward.

Two modes, selected by SynthesizerRewardConfig.reward_mode:

gated (default; the mode used in the paper)::

    R = gate(quality, t_q) * gate(reliability, t_r) * teachability_peak(sr)

    gate(x, t)            = clip((x - t) / (1 - t), 0, 1)               (soft gate, t = 0.5)
    teachability_peak(sr) = 2*sr                    if sr <= 0.5
                          = max(0, 1 - s*2*(sr-0.5)) if sr > 0.5         (s = easy_slope = 1.0)

  quality / reliability only act as pass/fail ramps; the payoff is the triangular
  difficulty peak, maximal when the Reasoner solves the task in half of its K rollouts.

additive (optional switch; the original weighted-sum design)::

    R = w_q*quality + w_r*reliability + w_t*teachability_band(sr) + w_n*novelty + w_a*report_alignment

  with teachability_band: sr < 0.05 or sr > 0.95 -> 0.2, 0.2 <= sr <= 0.8 -> 1.0, otherwise 0.6.

- quality     : TaskVerifier overall_quality_score.
- reliability : TaskVerifier answer_score (solution / final_answer consistency).
- sr          : fraction of the K Reasoner rollouts judged correct.
- novelty     : 1 - max 4-gram overlap with recent problems (logged in both modes).
- report_alignment : heuristic match against the report's weak / over-generated terms
                     (logged in both modes).

Invalid task -> invalid_reward (the trainer's allocator uses invalid_synth_reward, -0.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..pool.dedup import ngram_max_overlap


@dataclass
class TeachabilityConfig:
    very_easy_threshold: float = 0.95
    very_hard_threshold: float = 0.05
    learnable_low: float = 0.2
    learnable_high: float = 0.8
    very_easy_score: float = 0.2
    very_hard_score: float = 0.2
    learnable_score: float = 1.0
    midband_score: float = 0.6
    use_triangular_blend: bool = False


@dataclass
class SynthesizerRewardConfig:
    invalid_reward: float = -0.5
    quality_weight: float = 0.30
    reliability_weight: float = 0.20
    teachability_weight: float = 0.20
    novelty_weight: float = 0.15
    report_alignment_weight: float = 0.15
    # "gated" (default): soft gates on quality / reliability x continuous teachability peak.
    # "additive": the weighted sum of the five components above.
    reward_mode: str = "gated"
    quality_gate_threshold: float = 0.5
    reliability_gate_threshold: float = 0.5
    # Steepness of the too-easy (sr > 0.5) side of the gated teachability peak; 1.0 = symmetric.
    teachability_easy_slope: float = 1.0
    teachability: TeachabilityConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.teachability is None:
            self.teachability = TeachabilityConfig()


@dataclass
class SynthesizerRewardBreakdown:
    quality: float
    reliability: float
    teachability: float
    novelty: float
    report_alignment: float
    immediate_total: float
    invalid: bool


def teachability_score(success_rate: float, cfg: TeachabilityConfig) -> float:
    """Banded teachability over the Reasoner K-rollout success rate."""
    sr = max(0.0, min(1.0, success_rate))
    if sr < cfg.very_hard_threshold:
        band = cfg.very_hard_score
    elif sr > cfg.very_easy_threshold:
        band = cfg.very_easy_score
    elif cfg.learnable_low <= sr <= cfg.learnable_high:
        band = cfg.learnable_score
    else:
        band = cfg.midband_score
    if cfg.use_triangular_blend:
        triangular = 1.0 - 2.0 * abs(sr - 0.5)
        triangular = max(0.0, triangular)
        return 0.5 * band + 0.5 * triangular
    return band


def teachability_peak(success_rate: float, easy_slope: float = 1.0) -> float:
    """Triangular teachability peak for the gated mode: 1.0 at sr=0.5, 0 at sr=0 and (for
    easy_slope=1) at sr=1. easy_slope > 1 makes the too-easy side decay faster."""
    sr = max(0.0, min(1.0, success_rate))
    if sr <= 0.5:
        return 2.0 * sr
    return max(0.0, 1.0 - easy_slope * 2.0 * (sr - 0.5))


def _soft_gate(value: float, threshold: float) -> float:
    """Linear-ramp soft gate in [0, 1]: 0 below threshold, ramping to 1 at 1.0.

    Keeps a gradient above the threshold (unlike a hard 0/1 gate) while driving
    the reward toward 0 for low-quality / unreliable tasks."""
    if threshold >= 1.0:
        return 1.0 if value >= 1.0 else 0.0
    return max(0.0, min(1.0, (value - threshold) / (1.0 - threshold)))


def novelty_score(problem: str, recent_problems: Iterable[str], n: int = 4) -> float:
    """1 - max n-gram overlap with recent problems. Empty pool → 1.0."""
    others = [p for p in recent_problems if p]
    if not others or not problem:
        return 1.0
    return float(max(0.0, 1.0 - ngram_max_overlap(problem, others, n)))


def report_alignment_score(
    task_card: dict, weak_terms: Sequence[str], over_generated_terms: Sequence[str]
) -> float:
    """Heuristic alignment score in [0, 1].

    - Empty report → neutral 0.5.
    - +0.5 for each match against weak_terms (capped at +0.5 total).
    - -0.5 for each match against over_generated_terms (capped at -0.4 total).
    - Final clamped to [0, 1].
    """
    if not weak_terms and not over_generated_terms:
        return 0.5
    haystack = " ".join(
        str(task_card.get(k, "")).lower()
        for k in (
            "teaching_intent",
            "target_reasoning_style",
            "expected_reasoner_weakness",
            "novelty_notes",
            "verification_notes",
        )
    )
    score = 0.5
    pos_hits = sum(1 for t in weak_terms if t and t.lower() in haystack)
    neg_hits = sum(1 for t in over_generated_terms if t and t.lower() in haystack)
    score += min(0.5, 0.25 * pos_hits)
    score -= min(0.4, 0.2 * neg_hits)
    return float(max(0.0, min(1.0, score)))


def compute_synthesizer_immediate_reward(
    *,
    is_valid: bool,
    quality: float,
    reliability: float,
    success_rate: float,
    problem: str,
    recent_problems: Iterable[str],
    task_card: dict,
    weak_terms: Sequence[str] = (),
    over_generated_terms: Sequence[str] = (),
    config: SynthesizerRewardConfig | None = None,
) -> SynthesizerRewardBreakdown:
    cfg = config or SynthesizerRewardConfig()

    if not is_valid:
        return SynthesizerRewardBreakdown(
            quality=quality,
            reliability=reliability,
            teachability=0.0,
            novelty=0.0,
            report_alignment=0.0,
            immediate_total=cfg.invalid_reward,
            invalid=True,
        )

    nov = novelty_score(problem, recent_problems)
    align = report_alignment_score(task_card, weak_terms, over_generated_terms)

    if cfg.reward_mode == "gated":
        teach = teachability_peak(success_rate, easy_slope=cfg.teachability_easy_slope)
        gate = _soft_gate(quality, cfg.quality_gate_threshold) * _soft_gate(
            reliability, cfg.reliability_gate_threshold
        )
        total = gate * teach
    elif cfg.reward_mode == "additive":
        teach = teachability_score(success_rate, cfg.teachability)
        total = (
            cfg.quality_weight * quality
            + cfg.reliability_weight * reliability
            + cfg.teachability_weight * teach
            + cfg.novelty_weight * nov
            + cfg.report_alignment_weight * align
        )
    else:
        raise ValueError(f"unknown synthesizer reward_mode {cfg.reward_mode!r} (expected 'gated' or 'additive')")
    return SynthesizerRewardBreakdown(
        quality=quality,
        reliability=reliability,
        teachability=teach,
        novelty=nov,
        report_alignment=align,
        immediate_total=float(total),
        invalid=False,
    )
