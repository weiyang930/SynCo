# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import math
import os
import time

import numpy as np
import ray
from ray.exceptions import GetTimeoutError

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if len(self.agent_loop_workers) == 0:
            raise RuntimeError("No agent loop workers available for async generation.")

        manager_start = time.perf_counter()

        # Validation/workflow may issue tiny batches (e.g. size=1).
        # DataProto.chunk requires equal split when padding is disabled, so fallback to
        # split() for non-divisible cases to avoid assertion failures.
        target_workers = min(len(self.agent_loop_workers), max(1, len(prompts)))
        if len(prompts) % target_workers == 0:
            chunks = prompts.chunk(target_workers)
        else:
            split_size = math.ceil(len(prompts) / target_workers)
            chunks = prompts.split(split_size)

        # Use round-robin worker selection so tiny batches (especially len=1 in STAR
        # per-node workflow execution) do not always hit worker-0.
        if not hasattr(self, "_rr_worker_cursor"):
            self._rr_worker_cursor = 0
        start = int(self._rr_worker_cursor % len(self.agent_loop_workers))
        workers = [self.agent_loop_workers[(start + i) % len(self.agent_loop_workers)] for i in range(len(chunks))]
        self._rr_worker_cursor += len(chunks)

        manager_prep_s = time.perf_counter() - manager_start

        def _merge_worker_timings(outputs: list[DataProto]) -> dict[str, float]:
            merged: dict[str, float] = {}
            weighted_values: dict[str, list[tuple[float, int]]] = {}
            max_values: dict[str, list[float]] = {}
            scalar_values: dict[str, list[tuple[float, int]]] = {}

            for output in outputs:
                meta_info = output.meta_info or {}
                chunk_timing = meta_info.get("timing", None)
                if not isinstance(chunk_timing, dict):
                    continue
                weight = max(len(output), 1)
                for key, value in chunk_timing.items():
                    if not isinstance(value, int | float | np.integer | np.floating):
                        continue
                    key_str = str(key)
                    if not (
                        key_str.startswith("agent_loop/worker/")
                        or key_str.startswith("agent_loop/server_")
                        or key_str.startswith("agent_loop/server/")
                    ):
                        continue
                    value_f = float(value)
                    if key_str.endswith("/mean"):
                        weighted_values.setdefault(key_str, []).append((value_f, weight))
                    elif key_str.endswith("/max"):
                        max_values.setdefault(key_str, []).append(value_f)
                    else:
                        scalar_values.setdefault(key_str, []).append((value_f, weight))

            for key, pairs in weighted_values.items():
                total_weight = float(sum(weight for _, weight in pairs))
                if total_weight > 0:
                    merged[key] = float(sum(value * weight for value, weight in pairs) / total_weight)
            for key, values in max_values.items():
                if values:
                    merged[key] = float(max(values))
            for key, pairs in scalar_values.items():
                total_weight = float(sum(weight for _, weight in pairs))
                if total_weight <= 0:
                    continue
                mean_key = f"{key}/mean"
                max_key = f"{key}/max"
                merged[mean_key] = float(sum(value * weight for value, weight in pairs) / total_weight)
                merged[max_key] = float(max(value for value, _ in pairs))

            return merged

        rpc_timeout_s = float(
            os.environ.get(
                "STAR_WORKER_CALL_TIMEOUT_SECONDS",
                os.environ.get("STAR_RAY_GET_TIMEOUT_SECONDS", "0"),
            )
        )

        async def _run_chunk(worker, chunk):
            if chunk.meta_info is None:
                chunk.meta_info = {}
            chunk.meta_info["__agent_loop_dispatch_ts__"] = time.perf_counter()
            rpc_start = time.perf_counter()
            obj_ref = worker.generate_sequences.remote(chunk)
            try:
                if rpc_timeout_s > 0:
                    output = await asyncio.to_thread(ray.get, obj_ref, timeout=rpc_timeout_s)
                else:
                    output = await asyncio.to_thread(ray.get, obj_ref)
            except GetTimeoutError as exc:
                # Best-effort cancellation to avoid zombie RPC/task accumulation.
                try:
                    ray.cancel(obj_ref, force=False, recursive=False)
                except Exception:
                    pass
                raise TimeoutError(
                    f"AgentLoop worker RPC timed out: method=generate_sequences timeout_s={rpc_timeout_s}"
                ) from exc
            except asyncio.CancelledError:
                # Upstream timeout/cancel should also try to release backend pressure.
                try:
                    ray.cancel(obj_ref, force=False, recursive=False)
                except Exception:
                    pass
                raise
            return output, time.perf_counter() - rpc_start

        worker_rpc_start = time.perf_counter()
        chunk_results = await asyncio.gather(
            *[_run_chunk(worker, chunk) for worker, chunk in zip(workers, chunks, strict=True)]
        )
        worker_rpc_elapsed_s = time.perf_counter() - worker_rpc_start
        outputs = [output for output, _ in chunk_results]
        worker_rpc_times = [elapsed_s for _, elapsed_s in chunk_results]

        concat_start = time.perf_counter()
        concat_outputs = []
        for item in outputs:
            meta_info = dict(item.meta_info or {})
            # Worker timing differs per chunk and is merged below; do not let it
            # participate in DataProto.concat's strict meta_info consistency check.
            meta_info.pop("timing", None)
            meta_info.pop("__agent_loop_dispatch_ts__", None)
            concat_outputs.append(
                DataProto(batch=item.batch, non_tensor_batch=item.non_tensor_batch, meta_info=meta_info)
            )
        output = DataProto.concat(concat_outputs)
        manager_concat_s = time.perf_counter() - concat_start

        # calculate performance metrics
        metrics_start = time.perf_counter()
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        manager_metrics_reduce_s = time.perf_counter() - metrics_start
        timing.update(_merge_worker_timings(outputs))

        worker_rpc_times_np = np.array(worker_rpc_times, dtype=np.float64)
        manager_total_s = time.perf_counter() - manager_start
        timing["agent_loop/manager/prep"] = manager_prep_s
        timing["agent_loop/manager/worker_rpc_wait"] = worker_rpc_elapsed_s
        timing["agent_loop/manager/worker_rpc_mean"] = worker_rpc_times_np.mean()
        timing["agent_loop/manager/worker_rpc_max"] = worker_rpc_times_np.max()
        timing["agent_loop/manager/concat"] = manager_concat_s
        timing["agent_loop/manager/metrics_reduce"] = manager_metrics_reduce_s
        timing["agent_loop/manager/total"] = manager_total_s
        timing["agent_loop/manager/overhead"] = max(
            manager_total_s - float(worker_rpc_times_np.max()),
            0.0,
        )

        output_meta = dict(outputs[0].meta_info or {})
        output_meta.pop("metrics", None)
        output_meta.pop("timing", None)
        output_meta.pop("__agent_loop_dispatch_ts__", None)
        output.meta_info = {**output_meta, "timing": timing}
        return output

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
