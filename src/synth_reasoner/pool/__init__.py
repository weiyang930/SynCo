"""Synthetic pool: persistent storage and dedup for generated tasks."""
from .dedup import ngram_jaccard, ngram_max_overlap, normalized_hash
from .schemas import (
    RolloutRecord,
    SynthesizerRewardRecord,
    TaskCard,
    TaskRecord,
    TaskVerificationRecord,
)
from .storage import RunPaths, make_run_paths
from .synthetic_pool import SyntheticPool

__all__ = [
    "RolloutRecord",
    "RunPaths",
    "SyntheticPool",
    "SynthesizerRewardRecord",
    "TaskCard",
    "TaskRecord",
    "TaskVerificationRecord",
    "make_run_paths",
    "ngram_jaccard",
    "ngram_max_overlap",
    "normalized_hash",
]
