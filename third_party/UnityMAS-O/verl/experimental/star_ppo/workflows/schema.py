from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from verl import DataProto


@dataclass
class WorkflowExecutionRecord:
    """One executed workflow node, optionally backed by a trainable rollout trajectory."""

    query_id: str
    node_id: str
    agent_id: str
    model_id: str
    turn_id: int
    step_id: int
    node_type: str
    raw_output: str = ""
    parsed_output: Any = None
    thin: Optional[DataProto] = None
    trainable: bool = True
    state_before: Any = None
    state_after: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardAssignment:
    """A scalar reward assigned to one execution record."""

    record: WorkflowExecutionRecord
    reward: float
    reward_type: str = "custom"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTrace:
    """Full execution trace for one query."""

    query_id: str
    question: str
    ground_truth: list[str]
    records: list[WorkflowExecutionRecord] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    dropped: bool = False
    drop_reason: str = ""
    drop_error: str = ""
    debug_dump: Optional[str] = None

