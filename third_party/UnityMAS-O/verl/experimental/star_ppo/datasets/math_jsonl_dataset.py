from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from verl.experimental.star_ppo.tools.math_answer import extract_math_answer
from verl.utils.dataset.rl_dataset import RLHFDataset


class MathJsonlDataset(RLHFDataset):
    """Dataset adapter for STAR math workflows.

    It accepts JSON, JSONL, and Parquet rows from DAPO-Math, MATH-500, AIME,
    AMC, and similar math benchmark formats. The adapter preserves only the
    fields needed by the workflow: problem text, answer, data_source, and ids.
    """

    @staticmethod
    def _json_loads_maybe(value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return value
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
            row = cls._json_loads_maybe(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _extract_from_messages(messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                return "".join(parts).strip()
        return ""

    @classmethod
    def _extract_question(cls, row: dict, prompt_key: str) -> str:
        for key in ("question", "problem", "query"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        for key in (prompt_key, "prompt", "raw_prompt"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            parsed = cls._extract_from_messages(value)
            if parsed:
                return parsed
        return ""

    @classmethod
    def _extract_answer(cls, row: dict) -> str:
        reward_model = cls._json_loads_maybe(row.get("reward_model"))
        if isinstance(reward_model, dict):
            for key in ("ground_truth", "answer", "target"):
                value = reward_model.get(key)
                if value not in (None, ""):
                    return extract_math_answer(value)
        for key in ("answer", "ground_truth", "target"):
            value = row.get(key)
            if value not in (None, ""):
                return extract_math_answer(value)
        solution = row.get("solution")
        if solution not in (None, ""):
            return extract_math_answer(solution)
        return ""

    @staticmethod
    def _extract_extra_info(row: dict) -> dict:
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except Exception:
                extra_info = {}
        return dict(extra_info) if isinstance(extra_info, dict) else {}

    @staticmethod
    def _infer_data_source(row: dict, data_file: str) -> str:
        url = str(row.get("url") or "").lower()
        unique_id = str(row.get("unique_id") or "").lower()
        stem = Path(data_file).stem.lower()
        parent = Path(data_file).parent.name.lower()
        path_text = str(data_file).lower()
        joined = " ".join([url, unique_id, stem, parent, path_text])

        if "math-500" in joined or (unique_id.startswith("test/") and row.get("subject") is not None):
            return "math-500"
        if "2023_amc" in joined or "amc23" in joined or "amc_2023" in joined:
            return "amc23"
        if "2024_aime" in joined or "aime24" in joined or "aime2024" in joined:
            return "aime24"
        if "2025_aime" in joined or "aime25" in joined or "aime2025" in joined:
            return "aime25"
        if "2026_aime" in joined or "aime26" in joined or "aime2026" in joined:
            return "aime26"
        if "gsm8k" in joined:
            return "gsm8k"
        if "deepscaler" in joined or "deepscaler-preview-dataset" in joined:
            return "deepscaler"

        source_default = Path(data_file).stem
        if source_default.lower() in {"train", "test", "val", "validation", "dev"} and Path(data_file).parent.name:
            source_default = Path(data_file).parent.name
        return source_default or "math"

    @classmethod
    def _normalize_raw_row(cls, row: dict, prompt_key: str, row_index: int, data_file: str) -> dict:
        extra_info = cls._extract_extra_info(row)
        question = cls._extract_question(row, prompt_key)
        answer = cls._extract_answer(row)
        data_source = str(
            row.get("data_source")
            or row.get("source")
            or row.get("dataset")
            or extra_info.get("data_source")
            or extra_info.get("source")
            or cls._infer_data_source(row, data_file)
        )
        split = str(row.get("split") or extra_info.get("split") or "unknown")
        query_id = str(
            row.get("query_id")
            or row.get("uid")
            or row.get("unique_id")
            or row.get("id")
            or extra_info.get("index")
            or f"{data_source}/{split}/{row_index}"
        )

        normalized_extra = copy.deepcopy(extra_info)
        normalized_extra.update(
            {
                "index": normalized_extra.get("index", row_index),
                "split": split,
                "answer": answer,
                "data_source": data_source,
                "source_file": data_file,
                "tools_kwargs": normalized_extra.get("tools_kwargs", {}),
                "interaction_kwargs": normalized_extra.get("interaction_kwargs", {}),
                "need_tools_kwargs": bool(normalized_extra.get("need_tools_kwargs", False)),
            }
        )

        return {
            prompt_key: [{"role": "user", "content": question}],
            "prompt": [{"role": "user", "content": question}],
            "question": question,
            "problem": question,
            "answer": answer,
            "query_id": query_id,
            "uid": query_id,
            "data_source": data_source,
            "source": data_source,
            "split": split,
            "ability": str(row.get("ability") or "MATH"),
            "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": answer},
            "extra_info": normalized_extra,
        }

    def _read_files_and_tokenize(self):
        rows: list[tuple[dict, str]] = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
                rows.extend((dict(row), data_file) for row in dataframe)
            elif data_file.endswith((".json", ".jsonl")):
                rows.extend((row, data_file) for row in self._read_json_rows(data_file))
            else:
                raise ValueError(f"Unsupported file format for MathJsonlDataset: {data_file}")

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
        normalized_rows = [
            self._normalize_raw_row(row, prompt_key, idx, data_file)
            for idx, (row, data_file) in enumerate(rows)
            if isinstance(row, dict)
        ]
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
