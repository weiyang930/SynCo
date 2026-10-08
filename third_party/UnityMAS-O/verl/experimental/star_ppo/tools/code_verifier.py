from __future__ import annotations

import ast
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from typing import Any


class CodeVerifierTool:
    """Run generated Python code against code-task tests.

    This verifier keeps the current STAR workflow schema:
    input payload: {"problem": str, "code": str, "tests": Any, ...}
    output: pass_rate/all_passed/passed/total/error/failed_test_index.

    It also accepts MARTI/LiveCodeBench-style tests:
    {"inputs": [...], "outputs": [...], "fn_name": "..."}.
    If fn_name is present, the generated code is evaluated as a callable
    solution. Otherwise it is executed as a standard-input program.
    """

    _CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
    _PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        timeout_seconds: float = 1.0,
        max_error_chars: int = 2000,
        python_executable: str | None = None,
        allow_custom_checker: bool = False,
        default_checker_type: str = "auto",
        respect_problem_time_limit: bool = True,
        problem_time_limit_multiplier: float = 1.0,
        min_timeout_seconds: float = 0.0,
        max_tests_per_example: int = 0,
        max_case_input_chars: int = 0,
        max_case_output_chars: int = 0,
        max_total_test_chars: int = 0,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.max_error_chars = int(max_error_chars)
        self.python_executable = python_executable or sys.executable
        self.allow_custom_checker = bool(allow_custom_checker)
        self.default_checker_type = str(default_checker_type or "auto")
        self.respect_problem_time_limit = bool(respect_problem_time_limit)
        self.problem_time_limit_multiplier = max(0.01, float(problem_time_limit_multiplier or 1.0))
        self.min_timeout_seconds = max(0.0, float(min_timeout_seconds or 0.0))
        self.max_tests_per_example = max(0, int(max_tests_per_example or 0))
        self.max_case_input_chars = max(0, int(max_case_input_chars or 0))
        self.max_case_output_chars = max(0, int(max_case_output_chars or 0))
        self.max_total_test_chars = max(0, int(max_total_test_chars or 0))

    @classmethod
    def extract_code(cls, value: Any, min_fenced_length: int = 1, strict_syntax: bool = True) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""

        tag_match = cls._CODE_TAG_RE.search(raw)
        if tag_match:
            return tag_match.group(1).strip()

        valid_blocks: list[str] = []
        for block in cls._PY_FENCE_RE.findall(raw):
            code = str(block or "").strip()
            if len(code) < min_fenced_length:
                continue
            if strict_syntax:
                try:
                    ast.parse(code, mode="exec")
                except (SyntaxError, IndentationError):
                    continue
            valid_blocks.append(code)
        if valid_blocks:
            return valid_blocks[-1]

        return raw

    @staticmethod
    def _loads_maybe(value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return value

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        value = CodeVerifierTool._loads_maybe(value)
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def normalize_tests(cls, tests: Any) -> list[dict[str, Any]]:
        """Backward-compatible list-of-cases view."""

        return cls.normalize_test_spec(tests)["cases"]

    @classmethod
    def normalize_and_expand_tests(
        cls,
        tests: Any,
        problem: str = "",
        default_checker_type: str = "auto",
    ) -> list[dict[str, Any]]:
        cases = cls.normalize_tests(tests)
        return cls.expand_batched_stdio_tests(cases, problem=problem, default_checker_type=default_checker_type)

    @classmethod
    def expand_batched_stdio_tests(
        cls,
        cases: list[dict[str, Any]],
        problem: str = "",
        default_checker_type: str = "auto",
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for case in cases:
            expanded.extend(cls._split_batched_stdio_case(case, problem, default_checker_type))
        return expanded

    @staticmethod
    def _nonempty_lines(value: Any) -> list[str]:
        return [line.strip() for line in str(value or "").splitlines() if line.strip()]

    @staticmethod
    def _problem_has_batched_stdio_cases(problem: str) -> bool:
        p = str(problem or "").lower()
        return any(
            marker in p
            for marker in (
                "number of test cases",
                "multiple test cases",
                "each test consists of multiple test cases",
                "the first line contains a single integer t",
                "the first line contains an integer t",
                "the first line of the input contains t",
            )
        )

    @classmethod
    def _split_batched_stdio_case(
        cls,
        case: dict[str, Any],
        problem: str,
        default_checker_type: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(case, dict):
            return []
        if str(case.get("fn_name") or "").strip():
            return [case]
        if not cls._problem_has_batched_stdio_cases(problem):
            return [case]

        input_lines = cls._nonempty_lines(case.get("input", ""))
        if len(input_lines) <= 2:
            return [case]
        try:
            t = int(input_lines[0])
        except Exception:
            return [case]
        if t <= 1 or len(input_lines[1:]) != t:
            return [case]

        checker_type = cls._infer_checker_type(problem, case, default_checker_type)
        output_lines = cls._nonempty_lines(case.get("output", ""))
        if checker_type in {"standard", "unordered_tokens"}:
            if len(output_lines) != t:
                return [case]
            outputs = output_lines
        elif checker_type == "cf_rearrange_string":
            outputs = [""] * t
        else:
            return [case]

        split_cases: list[dict[str, Any]] = []
        for idx, (input_line, output) in enumerate(zip(input_lines[1:], outputs)):
            split_case = dict(case)
            split_case["input"] = f"1\n{input_line}\n"
            split_case["output"] = f"{output}\n" if output else ""
            split_case["checker_type"] = checker_type
            split_case["batch_parent_index"] = idx
            split_case["batch_parent_size"] = t
            split_cases.append(split_case)
        return split_cases

    @classmethod
    def normalize_test_spec(cls, tests: Any) -> dict[str, Any]:
        tests = cls._loads_maybe(tests)
        outer_common: dict[str, Any] = {}
        if isinstance(tests, dict):
            outer_common = {
                "fn_name": tests.get("fn_name") or tests.get("function_name") or "",
                "checker_type": tests.get("checker_type") or tests.get("special_judge") or "",
                "checker_code": tests.get("checker_code") or tests.get("checker") or "",
            }
        if isinstance(tests, dict) and "tests" in tests:
            nested = cls._loads_maybe(tests["tests"])
            if isinstance(nested, dict):
                for key, value in outer_common.items():
                    if value and not (nested.get(key) or nested.get("function_name" if key == "fn_name" else key)):
                        nested[key] = value
            tests = nested
        if isinstance(tests, dict):
            for key, value in outer_common.items():
                if value and not tests.get(key):
                    tests[key] = value

        # MARTI answer schema: use public tests by default for training-time rewards.
        if isinstance(tests, dict) and ("public_tests" in tests or "private_tests" in tests):
            nested = cls._loads_maybe(tests.get("public_tests") or tests.get("private_tests") or {})
            if isinstance(nested, dict):
                inherited_common = {
                    "fn_name": tests.get("fn_name") or tests.get("function_name") or outer_common.get("fn_name") or "",
                    "checker_type": (
                        tests.get("checker_type")
                        or tests.get("special_judge")
                        or outer_common.get("checker_type")
                        or ""
                    ),
                    "checker_code": tests.get("checker_code") or tests.get("checker") or outer_common.get("checker_code") or "",
                }
                for key, value in inherited_common.items():
                    if value and not nested.get(key):
                        nested[key] = value
            tests = nested

        common: dict[str, Any] = {}
        cases: list[dict[str, Any]] = []
        if isinstance(tests, dict):
            common = {
                "fn_name": tests.get("fn_name") or tests.get("function_name") or "",
                "checker_type": tests.get("checker_type") or tests.get("special_judge") or "",
                "checker_code": tests.get("checker_code") or tests.get("checker") or "",
            }
            if "inputs" in tests and "outputs" in tests:
                inputs = cls._as_list(tests.get("inputs"))
                outputs = cls._as_list(tests.get("outputs"))
                cases = [{"input": inp, "output": out, **common} for inp, out in zip(inputs, outputs)]
            elif "input" in tests and "output" in tests:
                cases = [{"input": tests.get("input", ""), "output": tests.get("output", ""), **common}]

        elif isinstance(tests, list):
            inferred_fn_name = outer_common.get("fn_name", "")
            inferred_checker_type = outer_common.get("checker_type", "")
            inferred_checker_code = outer_common.get("checker_code", "")
            for item in tests:
                item = cls._loads_maybe(item)
                if not isinstance(item, dict):
                    continue
                inferred_fn_name = inferred_fn_name or item.get("fn_name") or item.get("function_name") or ""
                inferred_checker_type = (
                    inferred_checker_type or item.get("checker_type") or item.get("special_judge") or ""
                )
                inferred_checker_code = inferred_checker_code or item.get("checker_code") or item.get("checker") or ""
                if "input" in item or "output" in item:
                    inp = item.get("input", item.get("stdin", ""))
                    out = item.get("output", item.get("stdout", ""))
                    cases.append(
                        {
                            "input": inp,
                            "output": out,
                            "fn_name": item.get("fn_name") or item.get("function_name") or outer_common.get("fn_name") or "",
                            "checker_type": (
                                item.get("checker_type")
                                or item.get("special_judge")
                                or outer_common.get("checker_type")
                                or ""
                            ),
                            "checker_code": item.get("checker_code") or item.get("checker") or outer_common.get("checker_code") or "",
                        }
                    )
                elif "inputs" in item and "outputs" in item:
                    item_common = {
                        "fn_name": item.get("fn_name") or item.get("function_name") or outer_common.get("fn_name") or "",
                        "checker_type": (
                            item.get("checker_type")
                            or item.get("special_judge")
                            or outer_common.get("checker_type")
                            or ""
                        ),
                        "checker_code": item.get("checker_code") or item.get("checker") or outer_common.get("checker_code") or "",
                    }
                    for inp, out in zip(cls._as_list(item["inputs"]), cls._as_list(item["outputs"])):
                        cases.append({"input": inp, "output": out, **item_common})
            common = {
                "fn_name": inferred_fn_name,
                "checker_type": inferred_checker_type,
                "checker_code": inferred_checker_code,
            }

        for case in cases:
            for key in ("fn_name", "checker_type", "checker_code"):
                if not case.get(key) and common.get(key):
                    case[key] = common[key]
            if isinstance(case.get("input"), list):
                case["input"] = "\n".join(str(x) for x in case["input"])
            if isinstance(case.get("output"), list) and not case.get("fn_name"):
                case["output"] = "\n".join(str(x) for x in case["output"])

        return {"cases": cases, **common}

    @staticmethod
    def _guard_prelude() -> str:
        return textwrap.dedent(
            """
            import builtins as __verl_builtins
            def __verl_exit(*args):
                raise SystemExit(args[0] if args else 0)
            __verl_builtins.exit = __verl_exit
            __verl_builtins.quit = __verl_exit
            try:
                import os as __verl_os
                __verl_os.environ["OMP_NUM_THREADS"] = "1"
                for __verl_name in (
                    "kill", "system", "putenv", "remove", "removedirs", "rmdir",
                    "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
                    "renames", "truncate", "replace", "unlink", "fchmod",
                    "fchown", "chmod", "chown", "chroot", "lchflags",
                    "lchmod", "lchown", "chdir",
                ):
                    if hasattr(__verl_os, __verl_name):
                        setattr(__verl_os, __verl_name, None)
            except Exception:
                pass
            try:
                import shutil as __verl_shutil
                for __verl_name in ("rmtree", "move", "chown"):
                    if hasattr(__verl_shutil, __verl_name):
                        setattr(__verl_shutil, __verl_name, None)
            except Exception:
                pass
            try:
                import subprocess as __verl_subprocess
                __verl_subprocess.Popen = None
            except Exception:
                pass
            """
        ).strip()

    @staticmethod
    def _common_code_prelude() -> str:
        return textwrap.dedent(
            """
            from string import *
            from re import *
            from datetime import *
            from collections import *
            from heapq import *
            from bisect import *
            from copy import *
            from math import *
            from random import *
            from statistics import *
            from itertools import *
            from functools import *
            from operator import *
            from io import *
            from sys import *
            from json import *
            from typing import *
            import string
            import re
            import datetime
            import collections
            import heapq
            import bisect
            import copy
            import math
            import random
            import statistics
            import itertools
            import functools
            import operator
            import io
            import sys
            import json
            try:
                sys.setrecursionlimit(6 * 10**5)
            except Exception:
                pass
            """
        ).strip()

    @staticmethod
    def _preexec_memory_limit() -> Any:
        if platform.system() == "Darwin":
            return None

        def _limit() -> None:
            try:
                import resource

                max_bytes = 512 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
            except Exception:
                pass

        return _limit

    @staticmethod
    def _json_safe_expr() -> str:
        return textwrap.dedent(
            """
            def __verl_json_safe(value):
                if isinstance(value, tuple):
                    return [__verl_json_safe(x) for x in value]
                if isinstance(value, list):
                    return [__verl_json_safe(x) for x in value]
                if isinstance(value, dict):
                    return {str(k): __verl_json_safe(v) for k, v in value.items()}
                return value
            """
        ).strip()

    def _write_solution(self, tmpdir: str, code: str) -> str:
        solution_path = os.path.join(tmpdir, "solution.py")
        source = self._guard_prelude() + "\n\n" + self._common_code_prelude() + "\n\n" + code + "\n"
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(source)
        return solution_path

    @staticmethod
    def _strip_main_guard(code: str) -> str:
        try:
            tree = ast.parse(str(code or ""))
            body = []
            changed = False
            for node in tree.body:
                if isinstance(node, ast.If):
                    try:
                        condition = ast.unparse(node.test).replace('"', "'").strip()
                    except Exception:
                        condition = ""
                    if condition == "__name__ == '__main__'":
                        changed = True
                        continue
                body.append(node)
            if not changed:
                return code
            tree.body = body
            return ast.unparse(tree)
        except Exception:
            return code

    def _write_call_runner(self, tmpdir: str, code: str, fn_name: str) -> str:
        runner_path = os.path.join(tmpdir, "call_runner.py")
        code = self._strip_main_guard(code)
        runner = (
            self._guard_prelude()
            + "\n\n"
            + self._common_code_prelude()
            + "\n\n"
            + code
            + "\n\n"
            + self._json_safe_expr()
            + "\n"
            + textwrap.dedent(
                f"""
                import json as __verl_json
                import sys as __verl_sys
                __verl_payload = __verl_sys.stdin.read()
                __verl_lines = [line for line in __verl_payload.splitlines() if line.strip()]
                __verl_args = [__verl_json.loads(line) for line in __verl_lines]
                if not __verl_args and __verl_payload.strip():
                    __verl_args = [__verl_json.loads(__verl_payload)]
                __verl_target = globals().get({fn_name!r})
                if __verl_target is None and "Solution" in globals():
                    __verl_obj = globals()["Solution"]()
                    __verl_target = getattr(__verl_obj, {fn_name!r})
                if __verl_target is None:
                    raise AttributeError("function not found: " + {fn_name!r})
                __verl_output = __verl_target(*__verl_args)
                __verl_sys.stdout.write(__verl_json.dumps(__verl_json_safe(__verl_output), ensure_ascii=False))
                """
            ).strip()
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner)
        return runner_path

    def _write_stdio_batch_runner(self, tmpdir: str, code: str, case_timeout_seconds: float | None = None) -> str:
        runner_path = os.path.join(tmpdir, "stdio_batch_runner.py")
        case_timeout_seconds = self.timeout_seconds if case_timeout_seconds is None else float(case_timeout_seconds)
        runner = (
            self._guard_prelude()
            + "\n\n"
            + textwrap.dedent(
                f"""
                import contextlib as __verl_contextlib
                import io as __verl_io
                import json as __verl_json
                import signal as __verl_signal
                import sys as __verl_sys
                import traceback as __verl_traceback

                __verl_user_code = {str(code or "")!r}
                __verl_case_timeout = {float(case_timeout_seconds)!r}
                __verl_compiled = compile({self._common_code_prelude()!r} + "\\n\\n" + __verl_user_code, "solution.py", "exec")
                __verl_payload = __verl_json.loads(__verl_sys.stdin.read() or "[]")
                __verl_results = []

                class __VerlCaseTimeout(Exception):
                    pass

                def __verl_timeout_handler(__verl_signum, __verl_frame):
                    raise __VerlCaseTimeout(
                        f"Time limit exceeded after {{__verl_case_timeout:.2f}}s."
                    )

                for __verl_case in __verl_payload:
                    __verl_stdin = str(__verl_case.get("input", ""))
                    __verl_stdout = __verl_io.StringIO()
                    __verl_stderr = __verl_io.StringIO()
                    __verl_old_stdin, __verl_old_stdout, __verl_old_stderr = (
                        __verl_sys.stdin,
                        __verl_sys.stdout,
                        __verl_sys.stderr,
                    )
                    __verl_returncode = 0
                    __verl_error = ""
                    try:
                        __verl_signal.signal(__verl_signal.SIGALRM, __verl_timeout_handler)
                        __verl_signal.setitimer(__verl_signal.ITIMER_REAL, __verl_case_timeout)
                        __verl_sys.stdin = __verl_io.StringIO(__verl_stdin)
                        __verl_sys.stdout = __verl_stdout
                        __verl_sys.stderr = __verl_stderr
                        __verl_ns = {{"__name__": "__main__", "__file__": "solution.py"}}
                        exec(__verl_compiled, __verl_ns, __verl_ns)
                    except SystemExit as __verl_exc:
                        __verl_code = __verl_exc.code
                        if __verl_code not in (None, 0):
                            __verl_returncode = int(__verl_code) if isinstance(__verl_code, int) else 1
                            __verl_error = "SystemExit: " + str(__verl_code)
                    except __VerlCaseTimeout as __verl_exc:
                        __verl_returncode = -1
                        __verl_error = str(__verl_exc)
                    except BaseException:
                        __verl_returncode = 1
                        __verl_error = __verl_traceback.format_exc(limit=8)
                    finally:
                        __verl_signal.setitimer(__verl_signal.ITIMER_REAL, 0.0)
                        __verl_sys.stdin = __verl_old_stdin
                        __verl_sys.stdout = __verl_old_stdout
                        __verl_sys.stderr = __verl_old_stderr

                    __verl_results.append({{
                        "returncode": __verl_returncode,
                        "stdout": __verl_stdout.getvalue(),
                        "stderr": __verl_stderr.getvalue(),
                        "error": __verl_error,
                    }})

                __verl_sys.stdout.write(__verl_json.dumps(__verl_results, ensure_ascii=False))
                """
            ).strip()
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner)
        return runner_path

    def _write_call_batch_runner(self, tmpdir: str, code: str, case_timeout_seconds: float | None = None) -> str:
        runner_path = os.path.join(tmpdir, "call_batch_runner.py")
        code = self._strip_main_guard(code)
        case_timeout_seconds = self.timeout_seconds if case_timeout_seconds is None else float(case_timeout_seconds)
        runner = (
            self._guard_prelude()
            + "\n\n"
            + self._common_code_prelude()
            + "\n\n"
            + code
            + "\n\n"
            + self._json_safe_expr()
            + "\n"
            + textwrap.dedent(
                """
                import json as __verl_json
                import signal as __verl_signal
                import sys as __verl_sys
                import traceback as __verl_traceback

                __verl_case_timeout = __VERL_CASE_TIMEOUT__

                def __verl_parse_args(__verl_payload, __verl_target):
                    __verl_payload = str(__verl_payload or "")
                    __verl_lines = [line for line in __verl_payload.splitlines() if line.strip()]
                    if len(__verl_lines) > 1:
                        return [__verl_json.loads(line) for line in __verl_lines]
                    if len(__verl_lines) == 1:
                        __verl_value = __verl_json.loads(__verl_lines[0])
                        if not isinstance(__verl_value, list):
                            return [__verl_value]
                        try:
                            import inspect as __verl_inspect

                            __verl_sig = __verl_inspect.signature(__verl_target)
                            __verl_positional = [
                                p for p in __verl_sig.parameters.values()
                                if p.kind in (
                                    p.POSITIONAL_ONLY,
                                    p.POSITIONAL_OR_KEYWORD,
                                    p.VAR_POSITIONAL,
                                )
                            ]
                            __verl_has_varargs = any(p.kind == p.VAR_POSITIONAL for p in __verl_positional)
                            if __verl_has_varargs or len(__verl_positional) != 1:
                                return list(__verl_value)
                        except Exception:
                            pass
                        return [__verl_value]
                    if __verl_payload.strip():
                        __verl_value = __verl_json.loads(__verl_payload)
                        return list(__verl_value) if isinstance(__verl_value, list) else [__verl_value]
                    return []

                class __VerlCaseTimeout(Exception):
                    pass

                def __verl_timeout_handler(__verl_signum, __verl_frame):
                    raise __VerlCaseTimeout(
                        f"Time limit exceeded after {__verl_case_timeout:.2f}s."
                    )

                __verl_cases = __verl_json.loads(__verl_sys.stdin.read() or "[]")
                __verl_results = []
                for __verl_case in __verl_cases:
                    __verl_fn_name = str(__verl_case.get("fn_name") or "")
                    __verl_returncode = 0
                    __verl_stdout = ""
                    __verl_stderr = ""
                    __verl_error = ""
                    try:
                        __verl_signal.signal(__verl_signal.SIGALRM, __verl_timeout_handler)
                        __verl_signal.setitimer(__verl_signal.ITIMER_REAL, __verl_case_timeout)
                        __verl_target = globals().get(__verl_fn_name)
                        if __verl_target is None and "Solution" in globals():
                            __verl_obj = globals()["Solution"]()
                            __verl_target = getattr(__verl_obj, __verl_fn_name)
                        if __verl_target is None:
                            raise AttributeError("function not found: " + __verl_fn_name)
                        __verl_args = __verl_parse_args(__verl_case.get("input", ""), __verl_target)
                        __verl_output = __verl_target(*__verl_args)
                        __verl_stdout = __verl_json.dumps(__verl_json_safe(__verl_output), ensure_ascii=False)
                    except __VerlCaseTimeout as __verl_exc:
                        __verl_returncode = -1
                        __verl_error = str(__verl_exc)
                    except BaseException:
                        __verl_returncode = 1
                        __verl_error = __verl_traceback.format_exc(limit=8)
                    finally:
                        __verl_signal.setitimer(__verl_signal.ITIMER_REAL, 0.0)
                    __verl_results.append({
                        "returncode": __verl_returncode,
                        "stdout": __verl_stdout,
                        "stderr": __verl_stderr,
                        "error": __verl_error,
                    })

                __verl_sys.stdout.write(__verl_json.dumps(__verl_results, ensure_ascii=False))
                """
            ).strip()
        )
        runner = runner.replace("__VERL_CASE_TIMEOUT__", repr(float(case_timeout_seconds)))
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner)
        return runner_path

    def _run_python(
        self,
        script_path: str,
        stdin: str,
        cwd: str,
        timeout_seconds: float | None = None,
    ) -> tuple[int, str, str, str]:
        timeout_seconds = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        try:
            proc = subprocess.run(
                [self.python_executable, script_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=cwd,
                preexec_fn=self._preexec_memory_limit(),
            )
            return proc.returncode, proc.stdout, proc.stderr, ""
        except subprocess.TimeoutExpired:
            return -1, "", "", f"Time limit exceeded after {timeout_seconds:.2f}s."
        except Exception as exc:
            return -1, "", "", f"{type(exc).__name__}: {exc}"

    def _run_batch_runner(
        self,
        script_path: str,
        cases: list[dict[str, Any]],
        cwd: str,
        timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        runner_input = json.dumps(cases, ensure_ascii=False)
        returncode, stdout, stderr, run_error = self._run_python(
            script_path,
            runner_input,
            cwd,
            timeout_seconds=timeout_seconds,
        )
        if run_error:
            return [
                {"returncode": returncode, "stdout": "", "stderr": stderr, "error": run_error}
                for _ in cases
            ]
        if returncode != 0:
            err = stderr or f"Batch runner exited with non-zero code: {returncode}"
            return [
                {"returncode": returncode, "stdout": "", "stderr": stderr, "error": err}
                for _ in cases
            ]
        try:
            parsed = json.loads(stdout or "[]")
        except Exception as exc:
            err = f"Batch runner produced invalid JSON: {type(exc).__name__}: {exc}"
            return [{"returncode": -1, "stdout": stdout, "stderr": stderr, "error": err} for _ in cases]
        if not isinstance(parsed, list):
            err = "Batch runner output is not a list."
            return [{"returncode": -1, "stdout": stdout, "stderr": stderr, "error": err} for _ in cases]
        results: list[dict[str, Any]] = []
        for idx in range(len(cases)):
            item = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
            results.append(
                {
                    "returncode": int(item.get("returncode", 0) or 0),
                    "stdout": str(item.get("stdout", "") or ""),
                    "stderr": str(item.get("stderr", "") or ""),
                    "error": str(item.get("error", "") or ""),
                }
            )
        return results

    @staticmethod
    def _parse_json_maybe(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value

    @classmethod
    def _objects_equal(cls, actual: Any, expected: Any) -> bool:
        actual = cls._parse_json_maybe(actual)
        expected = cls._parse_json_maybe(expected)
        if isinstance(actual, tuple):
            actual = list(actual)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual == expected:
            return True
        if isinstance(expected, list) and len(expected) == 1 and actual == expected[0]:
            return True
        if isinstance(actual, list) and isinstance(expected, list):
            return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(
                expected, sort_keys=True, ensure_ascii=False
            )
        return False

    @staticmethod
    def _float_tokens_close(actual_tokens: list[str], expected_tokens: list[str]) -> bool:
        if len(actual_tokens) != len(expected_tokens):
            return False
        try:
            actual_f = [float(x) for x in actual_tokens]
            expected_f = [float(x) for x in expected_tokens]
        except Exception:
            return False
        return all(abs(a - e) <= 1e-6 * max(1.0, abs(e)) for a, e in zip(actual_f, expected_f))

    @staticmethod
    def _yes_no_case_insensitive_match(actual_lines: list[str], expected_lines: list[str]) -> bool:
        if len(actual_lines) != len(expected_lines):
            return False
        yn = {"yes", "no"}
        for a, e in zip(actual_lines, expected_lines):
            if e.lower() not in yn:
                return False
            if a.lower() != e.lower():
                return False
        return True

    @staticmethod
    def _tokens_equal_with_yes_no_casefold(actual_tokens: list[str], expected_tokens: list[str]) -> bool:
        if len(actual_tokens) != len(expected_tokens):
            return False
        yn = {"yes", "no"}
        for actual, expected in zip(actual_tokens, expected_tokens):
            if expected.lower() in yn:
                if actual.lower() != expected.lower():
                    return False
            elif actual != expected:
                return False
        return True

    @classmethod
    def _standard_output_match(cls, actual: str, expected: str) -> bool:
        actual_s = str(actual or "").strip()
        expected_s = str(expected or "").strip()
        if actual_s == expected_s:
            return True

        actual_lines = [line.strip() for line in actual_s.splitlines() if line.strip()]
        expected_lines = [line.strip() for line in expected_s.splitlines() if line.strip()]
        if actual_lines == expected_lines:
            return True
        if cls._yes_no_case_insensitive_match(actual_lines, expected_lines):
            return True

        actual_tokens = actual_s.split()
        expected_tokens = expected_s.split()
        if actual_tokens == expected_tokens:
            return True
        if cls._tokens_equal_with_yes_no_casefold(actual_tokens, expected_tokens):
            return True
        if cls._float_tokens_close(actual_tokens, expected_tokens):
            return True
        return False

    @staticmethod
    def _infer_checker_type(problem: str, case: dict[str, Any], default_checker_type: str) -> str:
        explicit = str(case.get("checker_type") or "").strip()
        if explicit:
            return explicit
        if default_checker_type and default_checker_type != "auto":
            return default_checker_type
        p = str(problem or "").lower()
        if "rearrange the characters" in p and "not equal to" in p and "impossible" in p:
            return "cf_rearrange_string"
        return "standard"

    @staticmethod
    def _check_cf_rearrange_string(input_data: str, actual_output: str) -> tuple[bool, str]:
        tokens = str(input_data or "").split()
        if not tokens:
            return False, "empty input"
        try:
            t = int(tokens[0])
        except Exception:
            return False, "cannot parse number of test cases"
        strings = tokens[1 : 1 + t]
        if len(strings) != t:
            return False, f"expected {t} strings, got {len(strings)}"

        out_lines = [line.strip() for line in str(actual_output or "").splitlines() if line.strip()]
        ptr = 0
        for case_idx, s in enumerate(strings):
            impossible = len(set(s)) <= 1
            if ptr >= len(out_lines):
                return False, f"missing verdict for case {case_idx}"
            verdict = out_lines[ptr].lower()
            ptr += 1
            if impossible:
                if verdict != "no":
                    return False, f"case {case_idx}: expected NO"
                continue
            if verdict != "yes":
                return False, f"case {case_idx}: expected YES"
            if ptr >= len(out_lines):
                return False, f"case {case_idx}: missing rearranged string"
            r = out_lines[ptr]
            ptr += 1
            if len(r) != len(s) or Counter(r) != Counter(s):
                return False, f"case {case_idx}: output is not a permutation"
            if r == s:
                return False, f"case {case_idx}: output equals original string"
        if ptr != len(out_lines):
            return False, "extra output lines"
        return True, ""

    def _run_custom_checker(
        self,
        checker_code: str,
        input_data: str,
        expected_output: str,
        actual_output: str,
        problem: str,
        metadata: Any,
    ) -> tuple[bool, str]:
        if not self.allow_custom_checker:
            return False, "custom checker present but allow_custom_checker=false"
        ns: dict[str, Any] = {}
        try:
            exec(str(checker_code), ns)
            check_fn = ns.get("check")
            if not callable(check_fn):
                return False, "checker_code must define check(input_data, expected_output, actual_output, problem, metadata)"
            ok = check_fn(input_data, expected_output, actual_output, problem, metadata)
            return bool(ok), "" if ok else "custom checker returned false"
        except Exception as exc:
            return False, f"custom checker error: {type(exc).__name__}: {exc}"

    def _outputs_match(
        self,
        *,
        actual: str,
        expected: Any,
        input_data: Any,
        problem: str,
        case: dict[str, Any],
        metadata: Any,
        call_based: bool,
    ) -> tuple[bool, str, str]:
        if call_based:
            ok = self._objects_equal(actual, expected)
            return ok, "call_based", "" if ok else "call-based output mismatch"

        checker_type = self._infer_checker_type(problem, case, self.default_checker_type)
        checker_code = str(case.get("checker_code") or "")
        if checker_code:
            ok, reason = self._run_custom_checker(
                checker_code,
                str(input_data or ""),
                str(expected or ""),
                str(actual or ""),
                problem,
                metadata,
            )
            return ok, "custom_checker", reason

        if checker_type == "cf_rearrange_string":
            ok, reason = self._check_cf_rearrange_string(str(input_data or ""), str(actual or ""))
            return ok, checker_type, reason
        if checker_type == "unordered_tokens":
            ok = Counter(str(actual or "").split()) == Counter(str(expected or "").split())
            return ok, checker_type, "" if ok else "unordered token multiset mismatch"

        ok = self._standard_output_match(str(actual or ""), str(expected or ""))
        return ok, checker_type, "" if ok else "standard output mismatch"

    def _truncate(self, value: Any) -> str:
        text = str(value or "")
        if len(text) <= self.max_error_chars:
            return text
        keep = max(0, self.max_error_chars // 2)
        return text[:keep] + "\n...(truncated)...\n" + text[-keep:]

    def _select_runtime_tests(self, tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        selected: list[dict[str, Any]] = []
        skipped = 0
        total_chars = 0
        for test in tests:
            input_chars = len(str(test.get("input", "") or ""))
            output_chars = len(str(test.get("output", "") or ""))
            if self.max_case_input_chars > 0 and input_chars > self.max_case_input_chars:
                skipped += 1
                continue
            if self.max_case_output_chars > 0 and output_chars > self.max_case_output_chars:
                skipped += 1
                continue
            case_chars = input_chars + output_chars
            if self.max_total_test_chars > 0 and selected and total_chars + case_chars > self.max_total_test_chars:
                skipped += 1
                continue
            selected.append(test)
            total_chars += case_chars
            if self.max_tests_per_example > 0 and len(selected) >= self.max_tests_per_example:
                skipped += max(0, len(tests) - len(selected) - skipped)
                break
        return selected, skipped

    @staticmethod
    def _parse_problem_time_limit_seconds(problem: str) -> float | None:
        text = str(problem or "")
        patterns = (
            r"time\s*limit(?:\s*per\s*test)?\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?)\b",
            r"time\s*limit(?:\s*per\s*test)?\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|secs|second|seconds)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value / 1000.0 if unit.startswith("m") else value
        return None

    def _effective_timeout_seconds(self, problem: str) -> float:
        timeout = float(self.timeout_seconds)
        if self.respect_problem_time_limit:
            parsed = self._parse_problem_time_limit_seconds(problem)
            if parsed is not None:
                timeout = float(parsed) * self.problem_time_limit_multiplier
        if self.min_timeout_seconds > 0:
            timeout = max(timeout, self.min_timeout_seconds)
        return max(0.05, timeout)

    def __call__(self, payload: Any) -> dict[str, Any]:
        start = time.perf_counter()
        payload = payload if isinstance(payload, dict) else {}
        problem = str(payload.get("problem", "") or "")
        metadata = payload.get("metadata", payload.get("extra_info", {}))
        code = self.extract_code(payload.get("code", ""))
        raw_tests = self.normalize_tests(payload.get("tests", []))
        if isinstance(metadata, str):
            metadata = self._loads_maybe(metadata)
        if isinstance(metadata, dict):
            metadata_fn_name = str(metadata.get("func_name") or metadata.get("function_name") or "").strip()
            if metadata_fn_name:
                for test in raw_tests:
                    if not test.get("fn_name"):
                        test["fn_name"] = metadata_fn_name
        tests = self.expand_batched_stdio_tests(
            raw_tests,
            problem=problem,
            default_checker_type=self.default_checker_type,
        )
        total_available = len(tests)
        tests, skipped_by_limit = self._select_runtime_tests(tests)

        result: dict[str, Any] = {
            "pass_rate": 0.0,
            "all_passed": 0,
            "passed": 0,
            "total": len(tests),
            "total_raw": len(raw_tests),
            "total_available": total_available,
            "skipped_by_limit": skipped_by_limit,
            "error": "",
            "error_code": 0,
            "error_message": "",
            "failed_test_index": -1,
            "test_results": [],
            "checker_type": "",
            "runner_count": 0,
            "batch_mode": "",
            "timeout_seconds": 0.0,
            "duration_s": 0.0,
        }
        if not code.strip():
            result["error"] = "No code found."
            result["error_code"] = -1
            result["error_message"] = "No code found"
            result["duration_s"] = float(time.perf_counter() - start)
            return result
        if not tests:
            result["error"] = "No runnable tests found after applying verifier test limits."
            result["error_code"] = -1
            result["error_message"] = "No runnable tests found"
            result["duration_s"] = float(time.perf_counter() - start)
            return result

        case_timeout = self._effective_timeout_seconds(problem)
        result["timeout_seconds"] = float(case_timeout)
        batch_timeout = case_timeout * max(1, len(tests)) + 5.0
        passed = 0
        first_error = ""
        first_error_code = 0
        first_error_message = ""
        first_failed_idx = -1
        checker_types: list[str] = []

        with tempfile.TemporaryDirectory(prefix="verl_code_verify_") as tmpdir:
            batch_results: list[dict[str, Any] | None] = [None] * len(tests)
            stdio_indices = [idx for idx, test in enumerate(tests) if not str(test.get("fn_name") or "").strip()]
            call_indices = [idx for idx, test in enumerate(tests) if str(test.get("fn_name") or "").strip()]
            runner_count = 0
            batch_modes: list[str] = []

            if stdio_indices:
                runner_count += 1
                batch_modes.append("stdio")
                stdio_runner = self._write_stdio_batch_runner(tmpdir, code, case_timeout_seconds=case_timeout)
                stdio_cases = [{"input": str(tests[idx].get("input", ""))} for idx in stdio_indices]
                stdio_results = self._run_batch_runner(stdio_runner, stdio_cases, tmpdir, batch_timeout)
                for idx, item in zip(stdio_indices, stdio_results):
                    batch_results[idx] = item

            if call_indices:
                runner_count += 1
                batch_modes.append("call")
                call_runner = self._write_call_batch_runner(tmpdir, code, case_timeout_seconds=case_timeout)
                call_cases = [
                    {
                        "input": str(tests[idx].get("input", "")),
                        "fn_name": str(tests[idx].get("fn_name") or "").strip(),
                    }
                    for idx in call_indices
                ]
                call_results = self._run_batch_runner(call_runner, call_cases, tmpdir, batch_timeout)
                for idx, item in zip(call_indices, call_results):
                    batch_results[idx] = item

            result["runner_count"] = int(runner_count)
            result["batch_mode"] = "+".join(batch_modes)

            for idx, test in enumerate(tests):
                failure_kind = ""
                fn_name = str(test.get("fn_name") or "").strip()
                runner_result = batch_results[idx] if idx < len(batch_results) else None
                if not isinstance(runner_result, dict):
                    returncode = -1
                    stdout = ""
                    stderr = ""
                    run_error = "Batch verifier did not produce a result for this test."
                    ok = False
                    failure_kind = "runtime"
                else:
                    returncode = int(runner_result.get("returncode", 0) or 0)
                    stdout = str(runner_result.get("stdout", "") or "")
                    stderr = str(runner_result.get("stderr", "") or "")
                    run_error = str(runner_result.get("error", "") or "")

                    if run_error:
                        ok = False
                        failure_kind = "timeout" if "time" in run_error.lower() else "runtime"
                    elif returncode != 0:
                        ok = False
                        failure_kind = "runtime"
                    else:
                        ok, checker_type, reason = self._outputs_match(
                            actual=stdout,
                            expected=test.get("output", ""),
                            input_data=test.get("input", ""),
                            problem=problem,
                            case=test,
                            metadata=metadata,
                            call_based=bool(fn_name),
                        )
                        checker_types.append(checker_type)
                        if not ok:
                            run_error = reason
                            failure_kind = "wrong_answer"

                result["test_results"].append(bool(ok))
                if ok:
                    passed += 1
                    continue

                if first_failed_idx < 0:
                    first_failed_idx = idx
                    if failure_kind in {"timeout", "runtime"} and run_error:
                        first_error_code = -3 if failure_kind == "timeout" else -4
                        first_error_message = "Time Limit Exceeded" if failure_kind == "timeout" else "Runtime Error"
                        first_error = run_error
                    elif returncode != 0:
                        first_error_code = -4
                        first_error_message = "Runtime Error"
                        first_error = stderr or f"Non-zero exit code: {returncode}"
                    else:
                        first_error_code = -2
                        first_error_message = "Wrong Answer"
                        first_error = (
                            f"{run_error or 'Wrong Answer'}\n"
                            f"Input:\n{test.get('input', '')}\n\n"
                            f"Expected:\n{test.get('output', '')}\n\n"
                            f"Actual:\n{stdout}"
                        )

        total = max(1, len(tests))
        result["passed"] = int(passed)
        result["pass_rate"] = float(passed / total)
        result["all_passed"] = int(passed == len(tests))
        result["failed_test_index"] = int(first_failed_idx)
        result["error"] = self._truncate(first_error)
        result["error_code"] = int(first_error_code)
        result["error_message"] = first_error_message
        result["checker_type"] = ",".join(sorted(set(checker_types))) if checker_types else ""
        result["duration_s"] = float(time.perf_counter() - start)
        return result

    async def acall(self, payload: Any) -> dict[str, Any]:
        """Async tool entrypoint used by STAR workflow runners."""

        return await asyncio.to_thread(self.__call__, payload)
