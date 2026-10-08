from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from collections import Counter

from verl.experimental.star_ppo.workflows.schema import RewardAssignment, WorkflowTrace


class RewardAllocator(ABC):
    """Convert a workflow trace into trainable scalar rewards."""

    def __init__(self, trainer, config, runner=None):
        self.trainer = trainer
        self.config = config
        self.runner = runner

    @abstractmethod
    def allocate(self, trace: WorkflowTrace) -> tuple[list[RewardAssignment], dict[str, float]]:
        raise NotImplementedError

    @staticmethod
    def _normalize_text(text: str) -> str:
        s = str(text or "").lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @classmethod
    def f1_score(cls, pred: str, gt: str) -> float:
        pred_tokens = cls._normalize_text(pred).split()
        gt_tokens = cls._normalize_text(gt).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        overlap = sum(common.values())
        if overlap <= 0:
            return 0.0
        precision = overlap / float(len(pred_tokens))
        recall = overlap / float(len(gt_tokens))
        return 2.0 * precision * recall / (precision + recall)

    @classmethod
    def max_f1_score(cls, pred: str, ground_truth: list[str]) -> float:
        if not ground_truth:
            return 0.0
        return float(max(cls.f1_score(pred, gt) for gt in ground_truth))

    @staticmethod
    def compose_reward(task_reward: float, format_reward: float = 0.0, format_weight: float = 0.0) -> float:
        return float(task_reward) + float(format_weight) * float(format_reward)

