from __future__ import annotations

from abc import ABC, abstractmethod

from verl import DataProto


class WorkflowRunner(ABC):
    """Workflow runner plugin interface.

    A runner encapsulates query-level workflow orchestration and returns reward
    trajectories ready for commit into per-model rollout buffers.
    """

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config

    @abstractmethod
    async def run_batch(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        raise NotImplementedError

