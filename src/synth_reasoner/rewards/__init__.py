"""Reward functions for the Synthesizer and the Reasoner."""
from .reasoner_reward import (
    ReasonerRewardBreakdown,
    ReasonerRewardConfig,
    compute_reasoner_reward,
)
from .synthesizer_reward import (
    SynthesizerRewardBreakdown,
    SynthesizerRewardConfig,
    TeachabilityConfig,
    compute_synthesizer_immediate_reward,
    novelty_score,
    report_alignment_score,
    teachability_peak,
    teachability_score,
)

__all__ = [
    "ReasonerRewardBreakdown",
    "ReasonerRewardConfig",
    "SynthesizerRewardBreakdown",
    "SynthesizerRewardConfig",
    "TeachabilityConfig",
    "compute_reasoner_reward",
    "compute_synthesizer_immediate_reward",
    "novelty_score",
    "report_alignment_score",
    "teachability_peak",
    "teachability_score",
]
