"""``TraceWorkflowRunner`` subclass for the Synthesizer-Reasoner pipeline.

This module is imported by UnityMAS-O at training time via the Hydra path
``pkg://synth_reasoner.unitymas_adapter.workflow_adapter``. ``verl`` is
imported eagerly at module load so the class can subclass
``TraceWorkflowRunner``.

Per-query flow (one query = one Synthesizer task to be generated):

1. The query batch is treated as a thin slot — only the iteration index,
   the rendered report, and recent-problem dedup hints are extracted from
   ``non_tensor_batch``.
2. We call the Synthesizer model (``synthesizer_qwen3_8b``) once and
   produce a ``WorkflowExecutionRecord`` with ``trainable=cfg.train_synthesizer``.
3. Run the (CPU-side, deterministic) ``TaskVerifier``. Append a
   non-trainable record so the trace is complete.
4. If the task is valid, fan out K Reasoner rollouts on the
   ``reasoner_qwen3_8b`` model. Each rollout is a record with
   ``trainable=cfg.train_reasoner``. Then run ``AnswerChecker`` per rollout
   and append a non-trainable record.
5. Pack the immediate Synthesizer reward components and reasoner success
   rate into ``trace.state`` so the reward allocator can pick them up.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from verl.experimental.star_ppo.workflows.schema import (  # type: ignore[import]
    WorkflowExecutionRecord,
    WorkflowTrace,
)
from verl.experimental.star_ppo.workflows.trace_workflow import (  # type: ignore[import]
    TraceWorkflowRunner,
)

from ..agents.prompts import (
    SynthesizerInput,
    render_reasoner_prompt,
    render_synthesizer_prompt,
)
from ..parsing.answer_extraction import extract_reasoner_answer
from ..parsing.json_parser import parse_synthesizer_json
from ..pool.schemas import (
    RolloutRecord,
    SynthesizerRewardRecord,
    TaskCard,
    TaskRecord,
    TaskVerificationRecord,
)
from ..pool.storage import RunPaths
from ..pool.synthetic_pool import SyntheticPool
from ..reports.report_builder import LearningProgressSummary, ReportBuilder
from ..rewards import (
    ReasonerRewardConfig,
    SynthesizerRewardConfig,
    compute_reasoner_reward,
    compute_synthesizer_immediate_reward,
)
from ..verification.answer_checker import AnswerChecker
from ..verification.task_verifier import TaskVerifier
from . import roles

logger = logging.getLogger(__name__)


@dataclass
class _SynthesizerCfg:
    model_id: str
    agent_id: str
    max_tokens: int
    temperature: float
    top_p: float
    train: bool
    format_weight: float


@dataclass
class _ReasonerCfg:
    model_id: str
    agent_id: str
    rollouts_per_task: int
    max_tokens: int
    temperature: float
    top_p: float
    train: bool
    format_weight: float


class SynthReasonerWorkflowRunner(TraceWorkflowRunner):
    """One query → one synthesized task + K reasoner rollouts."""

    def __init__(self, trainer, config):
        super().__init__(trainer=trainer, config=config)
        synth_cfg = dict(self.workflow_cfg.get("synth_reasoner", {}))
        synth_node = dict(synth_cfg.get("synthesizer", {}))
        reasoner_node = dict(synth_cfg.get("reasoner", {}))
        self._synth = _SynthesizerCfg(
            model_id=str(synth_node.get("model_id", roles.SYNTHESIZER_MODEL_ID_FULL_SEPARATION)),
            agent_id=str(synth_node.get("agent_id", roles.SYNTHESIZER_AGENT_ID)),
            max_tokens=int(synth_node.get("max_tokens", 2048)),
            temperature=float(synth_node.get("temperature", 1.0)),
            top_p=float(synth_node.get("top_p", 0.95)),
            train=bool(synth_node.get("trainable", True)),
            format_weight=float(dict(synth_node.get("reward", {})).get("format_weight", 0.0)),
        )
        self._reasoner = _ReasonerCfg(
            model_id=str(reasoner_node.get("model_id", roles.REASONER_MODEL_ID_FULL_SEPARATION)),
            agent_id=str(reasoner_node.get("agent_id", roles.REASONER_AGENT_ID)),
            rollouts_per_task=int(reasoner_node.get("rollouts_per_task", 8)),
            max_tokens=int(reasoner_node.get("max_tokens", 2048)),
            temperature=float(reasoner_node.get("temperature", 1.0)),
            top_p=float(reasoner_node.get("top_p", 0.95)),
            train=bool(reasoner_node.get("trainable", True)),
            format_weight=float(dict(reasoner_node.get("reward", {})).get("format_weight", 0.0)),
        )
        self._synth_reward_cfg = SynthesizerRewardConfig(
            **dict(synth_cfg.get("synth_reward_config", {}))
        )
        self._reasoner_reward_cfg = ReasonerRewardConfig(
            **dict(synth_cfg.get("reasoner_reward_config", {}))
        )
        self._task_verifier = TaskVerifier()
        self._answer_checker = AnswerChecker()

        # Stage 0: persist every generated task back to a per-run pool so the
        # report can be rebuilt live during training (closed-loop). Cold-start:
        # the pool begins empty and fills with this run's own generations.
        self._pool = self._init_pool()
        # Serialize JSONL appends since run_batch fans out queries concurrently.
        self._pool_write_lock = asyncio.Lock()

        # Stage 1: rebuild the Synthesizer's report from the live pool every
        # ``report_refresh_every`` training steps (0 = never; static behavior).
        self._report_refresh_every = int(
            self.workflow_cfg.get("report_refresh_every", 0)
        )
        self._report_max_chars = int(self.workflow_cfg.get("report_max_chars", 4000))
        self._report_builder = ReportBuilder(max_chars=self._report_max_chars)
        self._train_batch_counter = 0

    def _init_pool(self) -> SyntheticPool | None:
        """Build a SyntheticPool rooted at runs/<run_id>/. Returns None if we
        cannot resolve a run directory (then persistence is silently skipped)."""
        run_id = str(self.config.get("run_id", "") or "")
        if not run_id:
            logger.warning("[synth-reasoner] no run_id in config; pool persistence disabled")
            return None
        # Mirror the launcher's layout: runs/<run_id>/ lives under the repo root.
        # default_local_dir points at ckpts; the run dir is a sibling "runs" tree.
        repo_root = os.environ.get(
            "SYNTH_REASONER_REPO_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        )
        runs_root = os.environ.get(
            "SYNTH_REASONER_RUNS_ROOT", os.path.join(repo_root, "runs")
        )
        paths = RunPaths(root=os.path.join(runs_root, run_id))
        paths.ensure_dirs()
        recent_window = int(self.workflow_cfg.get("pool_recent_window", 500))
        pool = SyntheticPool(paths=paths, recent_window=recent_window)
        # On resume, warm-load whatever this run already wrote so the report
        # reflects the full history, not just post-resume generations.
        try:
            loaded = pool.warm_load()
            logger.info("[synth-reasoner] pool at %s (warm-loaded %d records)", paths.pool_dir, loaded)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[synth-reasoner] pool warm_load failed: %s", exc)
        return pool

    # ------------------------------------------------------------------
    # Helpers to extract the iteration-level inputs from the query batch.
    # ------------------------------------------------------------------

    def _extract_state_strings(self, query_batch) -> dict[str, Any]:
        """Pull report/diagnostic/rejection text + recent problems off the batch."""
        return {
            "iteration": int(self._extract_from_batch(query_batch, "iteration") or 0),
            "task_index": int(self._extract_from_batch(query_batch, "task_index") or 0),
            "pool_report": str(self._extract_from_batch(query_batch, "pool_report") or ""),
            "diagnostic_report": str(
                self._extract_from_batch(query_batch, "diagnostic_report") or ""
            ),
            "rejection_summary": str(
                self._extract_from_batch(query_batch, "rejection_summary") or ""
            ),
            "weak_terms": list(self._extract_from_batch(query_batch, "weak_terms") or []),
            "over_generated_terms": list(
                self._extract_from_batch(query_batch, "over_generated_terms") or []
            ),
            "recent_problems": list(
                self._extract_from_batch(query_batch, "recent_problems") or []
            ),
        }

    # ------------------------------------------------------------------
    # Stage 1: refresh the Synthesizer report from the live pool, then run.
    # ------------------------------------------------------------------

    async def run_batch(self, batch, epoch, stage: str = "train"):
        """Override: before fanning out queries, optionally rebuild the report
        from the live pool and inject it into the batch's non_tensor fields so
        the Synthesizer sees up-to-date pool/Reasoner-performance context."""
        if str(stage or "train") == "train":
            self._train_batch_counter += 1
            if (
                self._report_refresh_every > 0
                and self._pool is not None
                and self._train_batch_counter % self._report_refresh_every == 0
            ):
                self._refresh_report_into_batch(batch)
        return await super().run_batch(batch, epoch, stage=stage)

    def _refresh_report_into_batch(self, batch) -> None:
        """Rebuild the report from the current pool and overwrite the per-row
        report fields in batch.non_tensor_batch (all rows get the same state)."""
        if self._pool is None:
            return
        try:
            records = self._pool.recent_records()
            if not records:
                return  # cold-start: nothing generated yet, keep seed report
            report = self._report_builder.build(
                records=records,
                iteration=self._train_batch_counter,
                learning=LearningProgressSummary(),
                recent_window=self._pool.recent_window,
            )
            bsz = len(batch)
            fields = {
                "pool_report": report.rendered,
                "diagnostic_report": report.diagnostic_rendered,
                "rejection_summary": report.rejection_rendered,
                "weak_terms": report.weak_terms,
                "over_generated_terms": report.over_generated_terms,
                "recent_problems": report.recent_problems,
            }
            for key, value in fields.items():
                if key in ("pool_report", "diagnostic_report", "rejection_summary"):
                    batch.non_tensor_batch[key] = np.array([value] * bsz, dtype=object)
                else:
                    # list-valued fields: must be a 1-D object array whose elements
                    # are the lists themselves. np.array([list]*bsz) would build a
                    # 2-D array, so vec[0] returns an ndarray and downstream
                    # `list(x or [])` raises "truth value of an array is ambiguous"
                    # — dropping every query on each refresh step. Match the seed
                    # format (1-D object array of lists) via explicit assignment.
                    col = np.empty(bsz, dtype=object)
                    col[:] = [list(value) for _ in range(bsz)]
                    batch.non_tensor_batch[key] = col
            # Use print(flush) not logger.info: this adapter's logger has no
            # handler and verl sets root to WARN, so logger.info is swallowed.
            # star_ppo itself logs progress via print(); match that so the
            # refresh is actually observable in the training log.
            print(
                f"[synth-reasoner] refreshed report at train batch "
                f"{self._train_batch_counter} (pool={len(records)} records, "
                f"avg_success={report.reasoner.avg_success_rate:.3f})",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - must never crash training
            print(f"[synth-reasoner] report refresh failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Main per-query coroutine.
    # ------------------------------------------------------------------

    async def run_query(
        self,
        query_batch,
        query_local_idx: int,
        debug: bool,
        debug_max_chars: int | None = None,
    ) -> WorkflowTrace:
        state = self._extract_state_strings(query_batch)
        query_id = str(self._extract_from_batch(query_batch, "query_id") or "")
        records: list[WorkflowExecutionRecord] = []

        # 1. Synthesizer.
        synth_inp = SynthesizerInput(
            pool_report=state["pool_report"],
            reasoner_diagnostic_report=state["diagnostic_report"],
            recent_rejection_summary=state["rejection_summary"],
        )
        synth_prompt = render_synthesizer_prompt(synth_inp)
        synth_record = await self._execute_llm_step(
            query_batch=query_batch,
            node_id=roles.SYNTHESIZER_NODE_ID,
            turn_id=0,
            step_id=0,
            node_cfg={
                "model_id": self._synth.model_id,
                "agent_id": self._synth.agent_id,
                "prompt_template": synth_prompt["user"],
                "parser": {"type": "raw"},
                "reward": {"format_weight": self._synth.format_weight},
            },
            prompt_context={},
            state_before=copy.deepcopy(state),
        )
        synth_record.trainable = bool(self._synth.train)
        synth_record.query_id = query_id
        parsed = parse_synthesizer_json(synth_record.raw_output)
        card = (
            TaskCard.from_dict(parsed.data)
            if parsed.ok and isinstance(parsed.data, dict)
            else None
        )
        synth_record.parsed_output = card.to_dict() if card else None
        synth_record.meta["json_parse_method"] = parsed.method
        synth_record.meta["json_parse_ok"] = parsed.ok
        records.append(synth_record)

        # 2. Task verifier.
        ver = self._task_verifier.verify(
            card.to_dict() if card else None,
            json_parse_ok=parsed.ok,
            recent_problems=state["recent_problems"],
        )
        records.append(
            WorkflowExecutionRecord(
                query_id=query_id,
                node_id=roles.TASK_VERIFIER_NODE_ID,
                agent_id="tool",
                model_id="",
                turn_id=0,
                step_id=1,
                node_type="tool",
                raw_output="",
                parsed_output={
                    "is_valid": ver.is_valid,
                    "scores": ver.scores.__dict__,
                    "reasons": ver.reasons,
                },
                trainable=False,
                state_before=copy.deepcopy(state),
                meta={"duration_s": 0.0},
            )
        )

        rollouts_meta: list[dict[str, Any]] = []
        n_correct = 0
        K = max(1, int(self._reasoner.rollouts_per_task))

        # 3. Reasoner rollouts (only on valid tasks).
        if ver.is_valid and card is not None and card.problem and card.final_answer:
            reasoner_prompt = render_reasoner_prompt(card.problem)
            for k in range(K):
                step_id = 2 + 2 * k
                reasoner_record = await self._execute_llm_step(
                    query_batch=query_batch,
                    node_id=f"{roles.REASONER_NODE_ID}_{k}",
                    turn_id=0,
                    step_id=step_id,
                    node_cfg={
                        "model_id": self._reasoner.model_id,
                        "agent_id": self._reasoner.agent_id,
                        "prompt_template": reasoner_prompt["user"],
                        "parser": {"type": "raw"},
                        "reward": {"format_weight": self._reasoner.format_weight},
                    },
                    prompt_context={},
                    state_before=copy.deepcopy(state),
                    extra_meta={"rollout_index": k},
                )
                reasoner_record.trainable = bool(self._reasoner.train)
                reasoner_record.query_id = query_id
                extracted = extract_reasoner_answer(reasoner_record.raw_output)
                reasoner_record.parsed_output = {
                    "answer": extracted.answer,
                    "method": extracted.method,
                    "has_boxed": extracted.has_boxed,
                }
                reasoner_record.meta["extraction_method"] = extracted.method
                reasoner_record.meta["has_boxed"] = bool(extracted.has_boxed)
                records.append(reasoner_record)

                # Per-rollout answer check (CPU tool node).
                check_t0 = time.perf_counter()
                check = self._answer_checker.check(extracted.answer, card.final_answer)
                check_dt = time.perf_counter() - check_t0
                reward = compute_reasoner_reward(extracted, check, self._reasoner_reward_cfg)
                if check.is_correct:
                    n_correct += 1
                # Annotate the reasoner record so the allocator can split the
                # per-rollout reward into per-trajectory pieces.
                reasoner_record.meta["is_correct"] = bool(check.is_correct)
                reasoner_record.meta["answer_check_method"] = check.method
                reasoner_record.meta["reasoner_reward"] = float(reward.total)
                reasoner_record.meta["reasoner_reward_breakdown"] = {
                    "answer_correctness": float(reward.answer_correctness),
                    "format_bonus": float(reward.format_bonus),
                    "total": float(reward.total),
                }
                rollouts_meta.append(
                    {
                        "rollout_index": k,
                        "is_correct": bool(check.is_correct),
                        "reward": float(reward.total),
                        "extraction_method": extracted.method,
                    }
                )
                records.append(
                    WorkflowExecutionRecord(
                        query_id=query_id,
                        node_id=f"{roles.ANSWER_CHECKER_NODE_ID}_{k}",
                        agent_id="tool",
                        model_id="",
                        turn_id=0,
                        step_id=step_id + 1,
                        node_type="tool",
                        raw_output="",
                        parsed_output={
                            "is_correct": check.is_correct,
                            "method": check.method,
                            "confidence": check.confidence,
                            "normalized_prediction": check.normalized_prediction,
                            "normalized_reference": check.normalized_reference,
                            "reason": check.reason,
                        },
                        trainable=False,
                        state_before=copy.deepcopy(state),
                        meta={"duration_s": float(check_dt)},
                    )
                )

        success_rate = float(n_correct) / float(K) if K > 0 else 0.0

        # 4. Synthesizer reward (the allocator reads it from ``trace.state``).
        synth_breakdown = compute_synthesizer_immediate_reward(
            is_valid=ver.is_valid,
            quality=ver.scores.overall_quality_score,
            reliability=ver.scores.answer_score,
            success_rate=success_rate,
            problem=card.problem if card else "",
            recent_problems=state["recent_problems"],
            task_card=card.to_dict() if card else {},
            weak_terms=state["weak_terms"],
            over_generated_terms=state["over_generated_terms"],
            config=self._synth_reward_cfg,
        )

        # Per-task success-rate buckets are aggregated into the fraction of
        # tasks in each difficulty band and reported as training metrics.
        is_too_easy = ver.is_valid and success_rate >= 0.95
        is_too_hard = ver.is_valid and success_rate <= 0.05
        is_learnable = ver.is_valid and 0.3 <= success_rate <= 0.7
        is_midband = (
            ver.is_valid
            and (0.05 < success_rate < 0.3 or 0.7 < success_rate < 0.95)
        )
        # A task is non-degenerate iff its K rollouts are mixed (0 < sr < 1), i.e. it yields a
        # non-zero Reasoner GRPO advantage. Batch mean = fraction of gradient-bearing tasks.
        is_nondegenerate = ver.is_valid and 0.0 < success_rate < 1.0

        metrics: dict[str, float] = {
            "workflow/synth_reasoner/is_valid": float(1.0 if ver.is_valid else 0.0),
            "workflow/synth_reasoner/quality": float(ver.scores.overall_quality_score),
            "workflow/synth_reasoner/reliability": float(ver.scores.answer_score),
            "workflow/synth_reasoner/success_rate": float(success_rate),
            "workflow/synth_reasoner/synth_immediate_total": float(
                synth_breakdown.immediate_total
            ),
            "workflow/synth_reasoner/synth_quality": float(synth_breakdown.quality),
            "workflow/synth_reasoner/synth_reliability": float(synth_breakdown.reliability),
            "workflow/synth_reasoner/synth_teachability": float(synth_breakdown.teachability),
            "workflow/synth_reasoner/synth_novelty": float(synth_breakdown.novelty),
            "workflow/synth_reasoner/synth_report_alignment": float(
                synth_breakdown.report_alignment
            ),
            "workflow/synth_reasoner/json_parse_ok": float(1.0 if parsed.ok else 0.0),
            "workflow/synth_reasoner/n_reasoner_rollouts": float(len(rollouts_meta)),
            "workflow/synth_reasoner/frac_too_easy": float(1.0 if is_too_easy else 0.0),
            "workflow/synth_reasoner/frac_too_hard": float(1.0 if is_too_hard else 0.0),
            "workflow/synth_reasoner/frac_learnable": float(1.0 if is_learnable else 0.0),
            "workflow/synth_reasoner/frac_midband": float(1.0 if is_midband else 0.0),
            "workflow/synth_reasoner/frac_nondegenerate": float(
                1.0 if is_nondegenerate else 0.0
            ),
        }

        trace = WorkflowTrace(
            query_id=query_id,
            question=card.problem if card else "",
            ground_truth=[card.final_answer] if card else [],
            records=records,
            state={
                "iteration": state["iteration"],
                "task_index": state["task_index"],
                "is_valid": bool(ver.is_valid),
                "task_card": card.to_dict() if card else None,
                "task_verification": {
                    "is_valid": ver.is_valid,
                    "scores": ver.scores.__dict__,
                    "reasons": list(ver.reasons),
                },
                "success_rate": float(success_rate),
                "rollouts": rollouts_meta,
                "synth_immediate_breakdown": {
                    "quality": float(synth_breakdown.quality),
                    "reliability": float(synth_breakdown.reliability),
                    "teachability": float(synth_breakdown.teachability),
                    "novelty": float(synth_breakdown.novelty),
                    "report_alignment": float(synth_breakdown.report_alignment),
                    "immediate_total": float(synth_breakdown.immediate_total),
                },
                "json_parse_ok": bool(parsed.ok),
            },
            metrics=metrics,
        )

        if debug:
            max_chars = (
                debug_max_chars if debug_max_chars is not None else self.debug_max_chars
            )

            def _trim(s: str) -> str:
                if max_chars and max_chars > 0 and len(s) > max_chars:
                    return s[:max_chars] + "...(truncated)..."
                return s

            lines = [
                "[synth-reasoner] ===== example begin =====",
                f"[synth-reasoner] iteration={state['iteration']} task_index={state['task_index']}",
                f"[synth-reasoner] is_valid={ver.is_valid} success_rate={success_rate:.2f}",
                f"[synth-reasoner] synth_total={synth_breakdown.immediate_total:.3f}",
                f"[synth-reasoner] synth_raw:\n{_trim(synth_record.raw_output)}",
                f"[synth-reasoner] task_card:\n{_trim(str(card.to_dict()) if card else 'None')}",
            ]
            if rollouts_meta:
                lines.append(f"[synth-reasoner] rollouts={rollouts_meta[:3]} ...")
            lines.append("[synth-reasoner] ===== example end =====")
            trace.debug_dump = "\n".join(lines)

        # Stage 0: persist this task (card + verification + rollouts + reward)
        # to the per-run pool so the report can be rebuilt live next iteration.
        await self._persist_task_record(
            query_id=query_id,
            iteration=state["iteration"],
            synth_record=synth_record,
            card=card,
            ver=ver,
            rollouts_records=[r for r in records if r.node_id.startswith(roles.REASONER_NODE_ID)],
            success_rate=success_rate,
            synth_breakdown=synth_breakdown,
        )

        return trace

    async def _persist_task_record(
        self,
        *,
        query_id: str,
        iteration: int,
        synth_record,
        card,
        ver,
        rollouts_records: list,
        success_rate: float,
        synth_breakdown,
    ) -> None:
        """Build a TaskRecord and append it to the pool (concurrency-safe)."""
        if self._pool is None:
            return
        try:
            rollout_recs = []
            for k, rec in enumerate(rollouts_records):
                parsed = rec.parsed_output or {}
                rollout_recs.append(
                    RolloutRecord(
                        rollout_id=f"{query_id}-r{k}",
                        raw_output=rec.raw_output or "",
                        extracted_answer=parsed.get("answer"),
                        extraction_method=str(parsed.get("method", "none")),
                        is_correct=bool(rec.meta.get("is_correct", False)),
                        answer_check_method=str(rec.meta.get("answer_check_method", "none")),
                        confidence=float(rec.meta.get("confidence", 0.0) or 0.0),
                        reward=float(rec.meta.get("reasoner_reward", 0.0) or 0.0),
                    )
                )
            record = TaskRecord(
                task_id=query_id,
                iteration=int(iteration),
                timestamp=datetime.now(timezone.utc).isoformat(),
                synthesizer_model=self._synth.model_id,
                synthesizer_prompt_hash=hashlib.sha1(
                    (synth_record.raw_output or "").encode("utf-8")
                ).hexdigest()[:16],
                raw_synthesizer_output=synth_record.raw_output or "",
                task_card=card if card is not None else TaskCard(),
                task_verification=TaskVerificationRecord(
                    is_valid=bool(ver.is_valid),
                    scores=dict(ver.scores.__dict__),
                    reasons=list(ver.reasons),
                ),
                reasoner_rollouts=rollout_recs,
                success_rate=float(success_rate),
                synthesizer_reward=SynthesizerRewardRecord(
                    quality=float(synth_breakdown.quality),
                    reliability=float(synth_breakdown.reliability),
                    teachability=float(synth_breakdown.teachability),
                    novelty=float(synth_breakdown.novelty),
                    report_alignment=float(synth_breakdown.report_alignment),
                    immediate_total=float(synth_breakdown.immediate_total),
                    total=float(synth_breakdown.immediate_total),
                ),
                metadata={"invalid": bool(synth_breakdown.invalid)},
            )
            async with self._pool_write_lock:
                self._pool.add(record)
        except Exception as exc:  # pragma: no cover - persistence must never crash training
            logger.warning("[synth-reasoner] pool persist failed for %s: %s", query_id, exc)


__all__ = ["SynthReasonerWorkflowRunner"]
