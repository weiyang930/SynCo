from __future__ import annotations

import numpy as np

from verl import DataProto
from verl.experimental.star_ppo.reward_allocators.code_turn_level import CodeTurnLevelRewardAllocator
from verl.experimental.star_ppo.tools.code_verifier import CodeVerifierTool
from verl.experimental.star_ppo.workflows.code_iterative_workflow import CodeIterativeWorkflowRunner
from verl.experimental.star_ppo.workflows.schema import WorkflowExecutionRecord, WorkflowTrace


def test_code_verifier_standard_input_passes():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\nimport sys\nprint(sys.stdin.read().strip())\n</code>",
            "tests": '[{"input": "hello\\n", "output": "hello"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["runner_count"] == 1


def test_code_verifier_expands_batched_stdio_cases():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "problem": "The first line contains a single integer t — the number of test cases.",
            "code": (
                "<code>\n"
                "t = int(input())\n"
                "for _ in range(t):\n"
                "    x, y = map(int, input().split())\n"
                "    print(min(x, y), max(x, y))\n"
                "</code>"
            ),
            "tests": '[{"input": "3\\n1 9\\n8 4\\n2 0\\n", "output": "1 9\\n4 8\\n0 2\\n"}]',
        }
    )
    assert result["total_raw"] == 1
    assert result["total"] == 3
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_verifier_does_not_expand_single_case_n_line_input():
    tests = '[{"input": "3\\na\\nb\\nc\\n", "output": "a\\nb\\nc\\n"}]'
    expanded = CodeVerifierTool.normalize_and_expand_tests(
        tests,
        problem="Given n strings. Print the strings in order.",
    )
    assert len(expanded) == 1


def test_code_verifier_supports_call_based_tests():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\ndef add(a, b):\n    return a + b\n</code>",
            "tests": {"inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"], "fn_name": "add"},
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["runner_count"] == 1


def test_code_verifier_supports_single_line_list_as_multiple_call_args():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\ndef max_multiple(divisor, bound):\n    return bound - bound % divisor\n</code>",
            "tests": {"inputs": ["[2, 7]", "10\n50"], "outputs": ["6", "50"], "fn_name": "max_multiple"},
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_verifier_preserves_outer_fn_name_for_nested_tests():
    for tests in (
        {"fn_name": "add", "tests": [{"input": "1\n2", "output": "3"}]},
        {"fn_name": "add", "public_tests": [{"input": "1\n2", "output": "3"}]},
        {"fn_name": "add", "public_tests": {"inputs": ["1\n2"], "outputs": ["3"]}},
    ):
        cases = CodeVerifierTool.normalize_tests(tests)
        assert len(cases) == 1
        assert cases[0]["fn_name"] == "add"


def test_code_verifier_applies_runtime_test_budget():
    verifier = CodeVerifierTool(timeout_seconds=2, max_tests_per_example=2)
    result = verifier(
        {
            "code": "<code>\nprint(input())\n</code>",
            "tests": [
                {"input": "a\n", "output": "a\n"},
                {"input": "b\n", "output": "b\n"},
                {"input": "c\n", "output": "c\n"},
            ],
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["total_available"] == 3
    assert result["total"] == 2
    assert result["skipped_by_limit"] == 1


def test_code_verifier_parses_problem_time_limit():
    verifier = CodeVerifierTool(timeout_seconds=5, respect_problem_time_limit=True)
    result = verifier(
        {
            "problem": "Time Limit: 1000 ms\nMemory Limit: 256 MB",
            "code": "<code>\nprint('ok')\n</code>",
            "tests": [{"input": "", "output": "ok\n"}],
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["timeout_seconds"] == 1.0


def test_code_workflow_formats_visible_tests_summary():
    label = '{"inputs": ["2\\n7", "10\\n50"], "outputs": ["6", "50"], "fn_name": "max_multiple"}'
    summary = CodeIterativeWorkflowRunner._format_visible_tests(label, max_examples=1, max_chars=1000)
    assert "fn_name=max_multiple" in summary
    assert "2\n7" in summary
    assert "6" in summary
    assert "1 more visible tests omitted" in summary


def test_code_verifier_supports_common_typing_imports_for_call_based_tests():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\ndef first(xs: List[int]):\n    return xs[0]\n</code>",
            "tests": {"inputs": ["[3, 4]"], "outputs": ["3"], "fn_name": "first"},
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_verifier_preserves_zero_exit_for_stdio_tests():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\nprint('ok')\nexit()\n</code>",
            "tests": '[{"input": "", "output": "ok"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_verifier_accepts_yes_no_case_inside_witness_output():
    verifier = CodeVerifierTool(timeout_seconds=2)
    result = verifier(
        {
            "code": "<code>\nprint('Yes')\nprint('1 2')\n</code>",
            "tests": '[{"input": "", "output": "YES\\n1 2"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_verifier_rearrange_string_special_judge_accepts_alternate_answer():
    verifier = CodeVerifierTool(timeout_seconds=2)
    problem = "Rearrange the characters of s to form a new string r that is not equal to s, or report impossible."
    result = verifier(
        {
            "problem": problem,
            "code": "<code>\nprint('YES')\nprint('oc')\n</code>",
            "tests": '[{"input": "1\\nco", "output": "YES\\noc"}]',
        }
    )
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1
    assert result["checker_type"] == "cf_rearrange_string"


def test_code_verifier_expands_batched_special_judge_cases():
    verifier = CodeVerifierTool(timeout_seconds=2)
    problem = (
        "The first line contains a single integer t — the number of test cases. "
        "Rearrange the characters of s to form a new string r that is not equal to s, or report impossible."
    )
    result = verifier(
        {
            "problem": problem,
            "code": (
                "<code>\n"
                "t = int(input())\n"
                "for _ in range(t):\n"
                "    s = input().strip()\n"
                "    if len(set(s)) == 1:\n"
                "        print('NO')\n"
                "    else:\n"
                "        print('YES')\n"
                "        print(s[::-1])\n"
                "</code>"
            ),
            "tests": '[{"input": "2\\nco\\naaaaa\\n", "output": "YES\\noc\\nNO\\n"}]',
        }
    )
    assert result["total_raw"] == 1
    assert result["total"] == 2
    assert result["pass_rate"] == 1.0
    assert result["all_passed"] == 1


def test_code_parser_requires_single_outer_tag():
    value, legal = CodeIterativeWorkflowRunner._parse_tagged_text("<pseudocode>read input</pseudocode>", "pseudocode")
    assert value == "read input"
    assert legal is True

    value, legal = CodeIterativeWorkflowRunner._parse_tagged_text("Plan:\n<pseudocode>x</pseudocode>", "pseudocode")
    assert value == "x"
    assert legal is False


def test_code_workflow_extracts_function_mode_from_marti_label():
    label = '{"inputs": ["2\\n7"], "outputs": ["6"], "fn_name": "max_multiple"}'
    assert CodeIterativeWorkflowRunner._find_fn_name_in_value(label) == "max_multiple"

    instruction = CodeIterativeWorkflowRunner._execution_instruction("max_multiple")
    assert "call-based function task" in instruction
    assert "Do not read from stdin" in instruction

    example = CodeIterativeWorkflowRunner._format_example_code("max_multiple")
    assert "def max_multiple" in example


def test_code_workflow_coder_instruction_switches_by_fn_name():
    call_instruction = CodeIterativeWorkflowRunner._coder_task_instruction("max_multiple")
    assert 'define "max_multiple" exactly' in call_instruction
    assert "Do not parse stdin" in call_instruction
    assert "Return the result" in call_instruction

    stdio_instruction = CodeIterativeWorkflowRunner._coder_task_instruction("")
    assert "stdin/stdout program" in stdio_instruction
    assert "print the required output" in stdio_instruction


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


def _llm_record(node_id: str, turn: int, agent_id: str, format_reward: float = 0.0) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="q",
        node_id=node_id,
        agent_id=agent_id,
        model_id="m",
        turn_id=turn,
        step_id=turn,
        node_type="llm",
        thin=_thin(f"{node_id}-traj"),
        meta={"format_reward": format_reward, "format_weight": 1.0},
    )


def _verifier_record(turn: int, pass_rate: float) -> WorkflowExecutionRecord:
    return WorkflowExecutionRecord(
        query_id="q",
        node_id=f"verifier_{turn}",
        agent_id="code_verifier",
        model_id="tool",
        turn_id=turn,
        step_id=turn,
        node_type="tool",
        trainable=False,
        parsed_output={"pass_rate": pass_rate, "all_passed": int(pass_rate == 1.0)},
    )


def test_code_reward_allocator_turn_deltas_and_format_penalty():
    allocator = CodeTurnLevelRewardAllocator(trainer=None, config=None)
    trace = WorkflowTrace(
        query_id="q",
        question="p",
        ground_truth=[],
        records=[
            _llm_record("planner_0", 0, "planner_agent"),
            _llm_record("coder_0", 0, "coder_agent", format_reward=-1.0),
            _verifier_record(0, 0.25),
            _llm_record("reflection_0", 0, "reflection_agent"),
            _llm_record("planner_1", 1, "planner_agent"),
            _llm_record("coder_1", 1, "coder_agent"),
            _verifier_record(1, 0.75),
        ],
    )

    assignments, metrics = allocator.allocate(trace)
    by_node = {assignment.record.node_id: assignment.reward for assignment in assignments}

    assert by_node["planner_0"] == 0.25
    assert by_node["coder_0"] == -0.75
    assert by_node["reflection_0"] == 0.5
    assert by_node["planner_1"] == 0.5
    assert by_node["coder_1"] == 0.5
    assert metrics["workflow/code/final_pass_rate"] == 0.75
