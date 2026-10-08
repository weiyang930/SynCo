from __future__ import annotations

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.reward_allocators.math_final_answer import MathFinalAnswerRewardAllocator
from verl.experimental.star_ppo.tools.math_answer import extract_math_answer, grade_math_answer, math_answer_equal
from verl.experimental.star_ppo.workflows.math_multi_agent_workflow import MathMultiAgentWorkflowRunner
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace


def _thin(traj_id: str) -> DataProto:
    return DataProto.from_dict(
        tensors={},
        non_tensors={
            "traj_id": np.array([traj_id], dtype=object),
            "model_id": np.array(["m"], dtype=object),
            "query_id": np.array(["q"], dtype=object),
            "agent_id": np.array(["a"], dtype=object),
        },
    )


def _llm_record(node_id: str, agent_id: str, format_reward: float = 0.0) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="q",
        node_id=node_id,
        agent_id=agent_id,
        model_id="m",
        turn_id=0,
        step_id=0,
        node_type="llm",
        thin=_thin(f"{node_id}-traj"),
        meta={"format_reward": format_reward, "format_weight": 1.0},
    )


def test_math_answer_extraction_and_equivalence():
    assert extract_math_answer(r"\boxed{294}") == "294"
    assert extract_math_answer("<FINAL_ANSWER>\n96\n</FINAL_ANSWER>") == "96"
    assert math_answer_equal("96", 96.0)
    assert math_answer_equal(r"\frac{1}{2}", "1/2")
    assert math_answer_equal(r"\left( 3, \frac{\pi}{2} \right)", "(3,pi/2)")
    assert math_answer_equal(r"\text{Evelyn}", "Evelyn")
    assert math_answer_equal(r"90^\circ", "90")
    assert math_answer_equal(r"3\sqrt{13}", "3sqrt(13)")
    assert math_answer_equal(r"\frac43", "4/3")
    assert math_answer_equal("p - q", "p-q")
    assert grade_math_answer("<FINAL_ANSWER>468</FINAL_ANSWER>", 468)["acc"] is True


def test_math_workflow_strict_tag_parser():
    parsed, legal = MathMultiAgentWorkflowRunner._parse_required_tags(
        "<SOLUTION>x</SOLUTION>\n<ANSWER>3</ANSWER>",
        ("SOLUTION", "ANSWER"),
    )
    assert legal is True
    assert parsed["solution"] == "x"
    assert parsed["answer"] == "3"

    _, legal = MathMultiAgentWorkflowRunner._parse_required_tags(
        "extra\n<SOLUTION>x</SOLUTION>\n<ANSWER>3</ANSWER>",
        ("SOLUTION", "ANSWER"),
    )
    assert legal is False

    _, legal = MathMultiAgentWorkflowRunner._parse_required_tags(
        "<STATUS>maybe</STATUS><ERROR_TYPE>none</ERROR_TYPE><FEEDBACK>x</FEEDBACK>",
        ("STATUS", "ERROR_TYPE", "FEEDBACK"),
    )
    assert legal is False


def test_math_reward_allocator_assigns_final_acc_and_format_penalty_to_all_agents():
    allocator = MathFinalAnswerRewardAllocator(trainer=None, config=None)
    trace = WorkflowTrace(
        query_id="q",
        question="p",
        ground_truth=["3"],
        records=[
            _llm_record("solver", "solver_agent"),
            _llm_record("verifier", "verifier_agent", format_reward=-1.0),
            _llm_record("refiner", "refiner_agent"),
            _llm_record("finalizer", "finalizer_agent"),
        ],
        metrics={"workflow/math/final_acc": 1.0},
    )

    assignments, metrics = allocator.allocate(trace)
    by_node = {assignment.record.node_id: assignment.reward for assignment in assignments}

    assert by_node["solver"] == 1.0
    assert by_node["verifier"] == 0.0
    assert by_node["refiner"] == 1.0
    assert by_node["finalizer"] == 1.0
    assert metrics["workflow/math/final_acc"] == 1.0
    assert metrics["workflow/math/reward_assignments"] == 4.0
