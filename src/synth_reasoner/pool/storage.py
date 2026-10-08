"""Path layout helpers for run artifacts under ``runs/{run_id}/``.

Spec §10 specifies these subpaths; centralizing them here keeps the rest
of the pipeline insulated from filesystem layout changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RunPaths:
    root: str

    @property
    def pool_dir(self) -> str:
        return os.path.join(self.root, "pool")

    @property
    def rollouts_dir(self) -> str:
        return os.path.join(self.root, "rollouts")

    @property
    def rewards_dir(self) -> str:
        return os.path.join(self.root, "rewards")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.root, "reports")

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.root, "eval")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.root, "logs")

    @property
    def generated_tasks_path(self) -> str:
        return os.path.join(self.pool_dir, "generated_tasks.jsonl")

    @property
    def valid_tasks_path(self) -> str:
        return os.path.join(self.pool_dir, "valid_tasks.jsonl")

    @property
    def invalid_tasks_path(self) -> str:
        return os.path.join(self.pool_dir, "invalid_tasks.jsonl")

    @property
    def reasoner_rollouts_path(self) -> str:
        return os.path.join(self.rollouts_dir, "reasoner_rollouts.jsonl")

    @property
    def rewards_path(self) -> str:
        return os.path.join(self.rewards_dir, "rewards.jsonl")

    @property
    def delayed_rewards_path(self) -> str:
        return os.path.join(self.rewards_dir, "delayed.jsonl")

    def report_path(self, iteration: int) -> str:
        return os.path.join(self.reports_dir, f"report_iter_{iteration}.json")

    def eval_path(self, iteration: int) -> str:
        return os.path.join(self.eval_dir, f"eval_iter_{iteration}.json")

    @property
    def eval_summary_csv(self) -> str:
        return os.path.join(self.eval_dir, "summary.csv")

    def ensure_dirs(self) -> None:
        for d in (
            self.root,
            self.pool_dir,
            self.rollouts_dir,
            self.rewards_dir,
            self.reports_dir,
            self.eval_dir,
            self.logs_dir,
        ):
            os.makedirs(d, exist_ok=True)


def make_run_paths(output_dir: str, run_id: str) -> RunPaths:
    paths = RunPaths(root=os.path.join(output_dir, run_id))
    paths.ensure_dirs()
    return paths
