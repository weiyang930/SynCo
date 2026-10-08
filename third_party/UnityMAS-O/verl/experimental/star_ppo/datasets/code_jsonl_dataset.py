from __future__ import annotations

import copy
import json
import os

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


class CodeJsonlDataset(RLHFDataset):
    """Dataset adapter for code-task json/jsonl rows.

    Expected raw fields include `problem` and `tests`. The adapter creates the
    chat-style `prompt` field needed by RLHFDataset while preserving all original
    fields for the STAR workflow verifier.
    """

    @staticmethod
    def _json_dumps_maybe(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def _json_loads_maybe(value):
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return value

    @classmethod
    def _read_json_rows(cls, path: str) -> list[dict]:
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        stripped = content.strip()
        if not stripped:
            return rows
        parsed = cls._json_loads_maybe(stripped)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("data"), list):
                return [row for row in parsed["data"] if isinstance(row, dict)]
            return [parsed]

        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            row = cls._json_loads_maybe(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @classmethod
    def _extract_problem_text(cls, row: dict, prompt_key: str) -> str:
        for key in ("problem", "question", "query"):
            value = row.get(key)
            if value:
                return cls._clean_problem_text(str(value))
        prompt_value = row.get(prompt_key) or row.get("prompt")
        if isinstance(prompt_value, str):
            return cls._clean_problem_text(prompt_value)
        if isinstance(prompt_value, list):
            texts = []
            for message in prompt_value:
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                texts.append(str(part.get("text", "")))
                            elif isinstance(part, str):
                                texts.append(part)
                else:
                    texts.append(str(message))
            return cls._clean_problem_text("\n".join(text for text in texts if text).strip())
        return cls._clean_problem_text(str(prompt_value or ""))

    @staticmethod
    def _clean_problem_text(text: str) -> str:
        """Remove dataset-level answer-format wrappers that conflict with STAR prompts."""

        text = str(text or "").strip()
        if "### Question:" in text:
            text = text.split("### Question:", 1)[1].strip()

        cut_markers = (
            "\nNow solve the problem and return the code.",
            "\n### Format:",
            "\n### Answer:",
        )
        cut_positions = [idx for marker in cut_markers if (idx := text.find(marker)) >= 0]
        if cut_positions:
            text = text[: min(cut_positions)].strip()
        return text

    @classmethod
    def _extract_tests_value(cls, row: dict):
        for key in ("tests", "test_cases", "answer", "label"):
            value = row.get(key)
            if value not in (None, ""):
                return value, key
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict):
            for key in ("ground_truth", "answer", "target"):
                value = reward_model.get(key)
                if value not in (None, ""):
                    return value, f"reward_model.{key}"
        elif reward_model not in (None, ""):
            return reward_model, "reward_model"
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            for key in ("tests", "public_test_cases", "private_test_cases"):
                value = extra_info.get(key)
                if value not in (None, ""):
                    return value, f"extra_info.{key}"
        return "", "none"

    @classmethod
    def _normalize_metadata(cls, row: dict, extra_info: dict) -> dict:
        metadata = cls._json_loads_maybe(row.get("metadata", extra_info.get("metadata", {})))
        if not isinstance(metadata, dict):
            metadata = {}

        for source in (row.get("label"), extra_info):
            source = cls._json_loads_maybe(source)
            if not isinstance(source, dict):
                continue
            fn_name = source.get("fn_name") or source.get("function_name") or source.get("func_name")
            if fn_name and not (metadata.get("func_name") or metadata.get("function_name")):
                metadata["func_name"] = str(fn_name)
                break
        return metadata

    @classmethod
    def _count_test_cases(cls, value) -> int:
        value = cls._json_loads_maybe(value)
        if isinstance(value, dict) and "tests" in value:
            value = cls._json_loads_maybe(value.get("tests"))
        if isinstance(value, dict) and ("public_tests" in value or "private_tests" in value):
            value = cls._json_loads_maybe(value.get("public_tests") or value.get("private_tests") or {})
        if isinstance(value, dict):
            if "inputs" in value and "outputs" in value:
                inputs = cls._json_loads_maybe(value.get("inputs"))
                outputs = cls._json_loads_maybe(value.get("outputs"))
                return min(len(inputs or []), len(outputs or [])) if isinstance(inputs, list) and isinstance(outputs, list) else 0
            if "input" in value or "output" in value:
                return 1
            return 0
        if isinstance(value, list):
            total = 0
            for item in value:
                item = cls._json_loads_maybe(item)
                if isinstance(item, dict) and "inputs" in item and "outputs" in item:
                    inputs = cls._json_loads_maybe(item.get("inputs"))
                    outputs = cls._json_loads_maybe(item.get("outputs"))
                    if isinstance(inputs, list) and isinstance(outputs, list):
                        total += min(len(inputs), len(outputs))
                elif isinstance(item, dict) and ("input" in item or "output" in item):
                    total += 1
            return total
        return 0

    @classmethod
    def _normalize_raw_row(cls, row: dict, prompt_key: str, row_index: int) -> dict:
        extra_info_raw = row.get("extra_info")
        if isinstance(extra_info_raw, str):
            extra_info_raw = cls._json_loads_maybe(extra_info_raw)
        if not isinstance(extra_info_raw, dict):
            extra_info_raw = {}

        metadata = cls._normalize_metadata(row, extra_info_raw)
        starter_code = row.get("starter_code", extra_info_raw.get("starter_code", "")) or ""
        problem = cls._extract_problem_text(row, prompt_key)
        source = str(row.get("source") or row.get("data_source") or row.get("datasource") or extra_info_raw.get("source") or "code")
        split = str(row.get("split") or extra_info_raw.get("split") or "unknown")
        uid = str(row.get("uid") or row.get("query_id") or row.get("id") or "")
        if not uid:
            uid = f"{source}/{split}/{row.get('row_id', extra_info_raw.get('row_id', row_index))}"

        metadata_str = cls._json_dumps_maybe(metadata)
        tests_value, tests_source = cls._extract_tests_value(row)
        tests_count_raw = cls._count_test_cases(tests_value)
        try:
            from verl.experimental.star_ppo.tools.code_verifier import CodeVerifierTool

            tests_count = len(CodeVerifierTool.normalize_and_expand_tests(tests_value, problem=problem))
        except Exception:
            tests_count = tests_count_raw
        tests_str = cls._json_dumps_maybe(tests_value)
        normalized_extra_info = {
            "index": int(row_index),
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "need_tools_kwargs": False,
            "metadata": metadata_str,
            "starter_code": str(starter_code),
            "source": source,
            "split": split,
            "tests_source": str(tests_source),
            "tests_count_raw": int(tests_count_raw),
            "tests_count": int(tests_count),
        }

        return {
            prompt_key: [{"role": "user", "content": problem}],
            "prompt": [{"role": "user", "content": problem}],
            "problem": problem,
            "query_id": uid,
            "uid": uid,
            "data_source": source,
            "source": source,
            "split": split,
            "starter_code": str(starter_code),
            "tests": tests_str,
            "tests_source": str(tests_source),
            "tests_count_raw": int(tests_count_raw),
            "tests_count": int(tests_count),
            "metadata": metadata_str,
            "extra_info": normalized_extra_info,
        }

    def _read_files_and_tokenize(self):
        rows = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
                rows.extend(dict(row) for row in dataframe)
            elif data_file.endswith((".json", ".jsonl")):
                rows.extend(self._read_json_rows(data_file))
            else:
                raise ValueError(f"Unsupported file format for CodeJsonlDataset: {data_file}")

        total = len(rows)
        print(f"dataset len: {total}")
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            rows = [rows[int(i)] for i in indices]
            print(f"selected {self.max_samples} random samples out of {total}")

        prompt_key = self.prompt_key
        normalized_rows = [self._normalize_raw_row(row, prompt_key, idx) for idx, row in enumerate(rows)]
        self.dataframe = datasets.Dataset.from_list(normalized_rows)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, data_file in enumerate(data_files):
            expanded = os.path.expanduser(str(data_file))
            self.data_files[i] = copy_to_local(src=expanded, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        if isinstance(data_files, (list, tuple)) or data_files.__class__.__name__ == "ListConfig":
            self.original_data_files = copy.deepcopy(list(data_files))
        else:
            self.original_data_files = copy.deepcopy([data_files])
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
