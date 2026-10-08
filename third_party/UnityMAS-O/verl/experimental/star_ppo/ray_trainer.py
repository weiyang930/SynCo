# Modified by the SynCo authors (2026): see SYNCO_MODIFICATIONS.md at the root of this directory.
import asyncio
import json
import os
import time
import traceback
import uuid
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from ray.exceptions import GetTimeoutError
from ray.util.collective import collective
from torch.utils.data import DataLoader, Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

from verl import DataProto
from verl.experimental.star_ppo.trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
from verl.experimental.star_ppo.types import EngineSpec
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.device import get_nccl_backend
from verl.utils.import_utils import load_extern_object
from verl.utils import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


@dataclass
class ModelWorkerContext:
    model_id: str
    resource_pool: RayResourcePool
    actor_wg: RayWorkerGroup
    rollout_wg: RayWorkerGroup
    rollout_manager: Optional[Any] = None
    critic_wg: Optional[RayWorkerGroup] = None
    ref_policy_wg: Optional[RayWorkerGroup] = None
    rm_wg: Optional[RayWorkerGroup] = None


@dataclass
class RolloutMicrobatchRequest:
    model_id: str
    batch: DataProto
    timing_state: Optional[dict[str, Any]]
    future: asyncio.Future
    enqueue_ts: float


class StarRayTrainer:
    """Star-topology PPO skeleton trainer with multi-engine routing."""

    def __init__(
        self,
        config,
        tokenizer,
        engine_specs: list[EngineSpec],
        role_worker_mapping,
        ray_worker_group_cls=RayWorkerGroup,
        processor=None,
        reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.engine_specs = engine_specs
        self.role_worker_mapping = role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = self.config.trainer.device

        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = Role.RewardModel in self.role_worker_mapping

        self.model_ids = [spec.model_id for spec in self.engine_specs]
        self.engine_cfg_by_model_id = {
            str(engine.model_id): engine for engine in self.config.trainer.get("llm_engines", [])
        }
        self.model_contexts: dict[str, ModelWorkerContext] = {}
        self.kl_ctrl_by_model = {}
        self.query_reward_ledger: dict[str, float] = defaultdict(float)
        self.global_steps = 0
        self._train_loader_state_loaded = False
        self._max_parallel_rollouts_per_model = int(self.config.star.workflow.get("max_parallel_rollouts_per_model", 32))
        self._rollout_semaphore_by_model: dict[str, asyncio.Semaphore] = {}
        microbatch_flag = str(os.environ.get("STAR_LLM_MICROBATCH_ENABLE", "true")).strip().lower()
        self._llm_microbatch_enabled = microbatch_flag in {"1", "true", "yes", "on"}
        self._llm_microbatch_max_size = max(1, int(os.environ.get("STAR_LLM_MICROBATCH_MAX_SIZE", "8")))
        self._llm_microbatch_max_wait_s = max(
            0.0,
            float(os.environ.get("STAR_LLM_MICROBATCH_MAX_WAIT_MS", "100")) / 1000.0,
        )
        self._llm_microbatch_queues_by_model: dict[str, list[RolloutMicrobatchRequest]] = defaultdict(list)
        self._llm_microbatch_locks_by_model: dict[str, asyncio.Lock] = {}
        self._llm_microbatch_flush_tasks_by_model: dict[str, asyncio.Task] = {}
        self._llm_microbatch_flush_task_kind_by_model: dict[str, str] = {}
        # For tiny batches (especially bsz=1), ND dispatch padding can bias samples
        # to shard-0 if we always slice the first item. Use round-robin shard pick to
        # spread committed trajectories across rollout shards.
        self._thin_pick_cursor_by_model: dict[str, int] = defaultdict(int)
        local_build_thin_flag = str(os.environ.get("STAR_LOCAL_BUILD_THIN", "true")).strip().lower()
        self._local_build_thin_enabled = local_build_thin_flag in {"1", "true", "yes", "on"}
        self._local_build_thin_max_bsz = max(
            1,
            int(os.environ.get("STAR_LOCAL_BUILD_THIN_MAX_BSZ", str(self._llm_microbatch_max_size))),
        )
        buffer_cfg = self.config.star.get("buffer", {})
        shuffle_ready = buffer_cfg.get("shuffle_ready", True)
        self._shuffle_ready_buffer = (
            shuffle_ready.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(shuffle_ready, str)
            else bool(shuffle_ready)
        )
        self._local_traj_buffers_by_model: dict[str, TrajectoryBuffer] = {
            model_id: TrajectoryBuffer(
                max_items=int(buffer_cfg.get("max_items", 100000)),
                ttl_seconds=int(buffer_cfg.get("ttl_seconds", 7200)),
                dropped_query_ttl_seconds=int(buffer_cfg.get("dropped_query_ttl_seconds", 120)),
            )
            for model_id in self.model_ids
        }
        timing_print_flag = str(os.environ.get("STAR_TIMING_PRINT", "true")).strip().lower()
        self._timing_print_enabled = timing_print_flag in {"1", "true", "yes", "on"}
        self._timing_print_every_n_batches = max(1, int(os.environ.get("STAR_TIMING_PRINT_EVERY_N_BATCHES", "1")))
        self._timing_group_topk = max(1, int(os.environ.get("STAR_TIMING_GROUP_TOPK", "8")))
        wandb_timing_filter_flag = str(os.environ.get("STAR_WANDB_FILTER_FINE_TIMING", "true")).strip().lower()
        self._wandb_filter_fine_timing = wandb_timing_filter_flag in {"1", "true", "yes", "on"}
        self._ray_get_timeout_seconds = float(os.environ.get("STAR_RAY_GET_TIMEOUT_SECONDS", "0"))
        self._worker_call_timeout_seconds = float(
            os.environ.get("STAR_WORKER_CALL_TIMEOUT_SECONDS", self._ray_get_timeout_seconds)
        )
        self._weight_sync_timeout_seconds = float(
            os.environ.get("STAR_WEIGHT_SYNC_TIMEOUT_SECONDS", self._ray_get_timeout_seconds)
        )
        self._workflow_batch_timeout_seconds = float(os.environ.get("STAR_WORKFLOW_BATCH_TIMEOUT_SECONDS", "0"))
        self._stall_detect_seconds = float(os.environ.get("STAR_STALL_DETECT_SECONDS", "0"))
        self._stall_heartbeat_seconds = float(os.environ.get("STAR_STALL_HEARTBEAT_SECONDS", "30"))
        self._last_progress_ts = time.time()
        self._last_progress_stage = "init"
        self._last_progress_step = 0
        self.workflow_runner = self._create_workflow_runner()
        if self.config.algorithm.use_kl_in_reward:
            for model_id in self.model_ids:
                self.kl_ctrl_by_model[model_id] = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, train_dataset)

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.data.val_batch_size or len(val_dataset),
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=False,
            collate_fn=collate_fn,
            shuffle=self.config.data.get("validation_shuffle", True),
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception:
            pass

    def _mark_progress(self, stage: str, step: Optional[int] = None) -> None:
        self._last_progress_ts = time.time()
        self._last_progress_stage = str(stage)
        if step is not None:
            self._last_progress_step = int(step)

    async def _stall_watchdog(self, stop_event: asyncio.Event) -> None:
        if self._stall_detect_seconds <= 0:
            return
        interval = max(5.0, min(self._stall_heartbeat_seconds, self._stall_detect_seconds))
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass
            stalled_for = time.time() - self._last_progress_ts
            if stalled_for < self._stall_detect_seconds:
                continue
            print(
                f"[star-watchdog] no progress for {stalled_for:.1f}s "
                f"(stage={self._last_progress_stage}, step={self._last_progress_step}, "
                f"inflight={int(self.config.star.workflow.get('max_inflight_queries', 32))})"
            )

    @staticmethod
    def _flatten_object_refs(value) -> list[ray.ObjectRef]:
        if isinstance(value, ray.ObjectRef):
            return [value]
        if isinstance(value, list | tuple):
            out: list[ray.ObjectRef] = []
            for item in value:
                out.extend(StarRayTrainer._flatten_object_refs(item))
            return out
        return []

    def _ray_get_with_timeout(self, refs, timeout_s: float, op_name: str):
        obj_refs = self._flatten_object_refs(refs)
        if not obj_refs:
            return refs
        try:
            if timeout_s > 0:
                return ray.get(obj_refs, timeout=timeout_s)
            return ray.get(obj_refs)
        except GetTimeoutError as exc:
            raise TimeoutError(
                f"Ray get timeout: op={op_name} timeout_s={timeout_s} refs={len(obj_refs)}"
            ) from exc

    def _clone_actor_rollout_cfg_for_model(self, model_id: str):
        cfg = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
        cfg.model_id = model_id
        cfg.star_buffer = OmegaConf.to_container(self.config.star.buffer, resolve=True)
        with open_dict(cfg):
            if OmegaConf.select(cfg, "rollout.custom") is None:
                cfg.rollout.custom = {}
            # Multi-model async server actors must use distinct names in the same Ray namespace.
            cfg.rollout.custom["server_name_prefix"] = f"star_{model_id}_"
        engine_cfg = self.engine_cfg_by_model_id.get(model_id, None)
        if engine_cfg is not None:
            # Optional per-engine model path override for true multi-LLM training.
            engine_model_path = engine_cfg.get("model_path", None)
            if engine_model_path is not None:
                cfg.model.path = str(engine_model_path)
        return cfg

    def _build_manager_cfg_for_model(self, actor_rollout_cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        with open_dict(cfg):
            cfg.actor_rollout_ref = OmegaConf.create(OmegaConf.to_container(actor_rollout_cfg, resolve=True))
        return cfg

    def _clone_critic_cfg_for_model(self, model_id: str, actor_model_path: Optional[str] = None):
        cfg = OmegaConf.create(OmegaConf.to_container(self.config.critic, resolve=True))
        engine_cfg = self.engine_cfg_by_model_id.get(model_id, None)
        model_path = None
        if engine_cfg is not None:
            engine_model_path = engine_cfg.get("model_path", None)
            if engine_model_path:
                model_path = str(engine_model_path)
        if not model_path and actor_model_path:
            model_path = str(actor_model_path)

        if model_path:
            with open_dict(cfg):
                if OmegaConf.select(cfg, "model") is None:
                    cfg.model = {}
                cfg.model.path = model_path
                # Keep critic tokenizer aligned with actor model to guarantee same-model pairing.
                cfg.model.tokenizer_path = model_path
        return cfg

    def init_workers(self):
        actor_rollout_cfg_by_model_id = {}
        init_targets_by_role: dict[str, list[tuple[str, RayWorkerGroup]]] = {
            "actor": [],
            "rollout": [],
            "critic": [],
            "ref": [],
            "rm": [],
        }

        def _enqueue_role_init(model_id: str, role_name: str, wg: RayWorkerGroup):
            worker0 = wg.workers[0]
            method_candidates = [f"{role_name}_init_model", "init_model"]
            method_name = None
            for candidate in method_candidates:
                if hasattr(worker0, candidate):
                    method_name = candidate
                    break
            if method_name is None:
                raise AttributeError(
                    f"No init method found for role={role_name}. "
                    f"Tried candidates={method_candidates}"
                )
            refs = wg.execute_all_async(method_name)
            print(
                f"[star] enqueue init_model model={model_id} role={role_name} "
                f"remote_method={method_name} workers={len(wg.workers)} calls={len(refs)}"
            )
            return refs

        for spec in self.engine_specs:
            resource_pool = RayResourcePool(
                process_on_nodes=[spec.n_gpus_per_node] * spec.nnodes,
                use_gpu=True,
                max_colocate_count=3,
                name_prefix=f"star_{spec.model_id}",
                accelerator_type=spec.accelerator_type,
            )

            actor_rollout_cfg = self._clone_actor_rollout_cfg_for_model(spec.model_id)
            class_dict = {
                "actor": RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Actor], config=actor_rollout_cfg, role=str(Role.Actor)
                ),
                "rollout": RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Rollout], config=actor_rollout_cfg, role=str(Role.Rollout)
                ),
            }

            if self.use_critic:
                actor_model_path = OmegaConf.select(actor_rollout_cfg, "model.path")
                critic_cfg = omega_conf_to_dataclass(
                    self._clone_critic_cfg_for_model(spec.model_id, actor_model_path=actor_model_path)
                )
                class_dict["critic"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Critic], config=critic_cfg
                )
                print(
                    f"[star] model={spec.model_id} actor_model={actor_model_path} "
                    f"critic_model={critic_cfg.model.path}"
                )

            if self.use_reference_policy:
                class_dict["ref"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.RefPolicy], config=actor_rollout_cfg, role=str(Role.RefPolicy)
                )

            if self.use_rm:
                class_dict["rm"] = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.RewardModel],
                    config=omega_conf_to_dataclass(self.config.reward_model),
                )

            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # Star stateless weight-sync may need actor/rollout RPCs to run concurrently
            # on the same colocated WorkerDict process.
            worker_max_concurrency = int(os.environ.get("STAR_WORKER_MAX_CONCURRENCY", "4"))
            worker_dict_cls.update_options({"max_concurrency": worker_max_concurrency})
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
            )
            spawned = wg_dict.spawn(prefix_set=class_dict.keys())

            actor_wg = spawned["actor"]
            rollout_wg = spawned["rollout"]

            critic_wg = spawned.get("critic")
            ref_wg = spawned.get("ref")
            rm_wg = spawned.get("rm")
            self.model_contexts[spec.model_id] = ModelWorkerContext(
                model_id=spec.model_id,
                resource_pool=resource_pool,
                actor_wg=actor_wg,
                rollout_wg=rollout_wg,
                rollout_manager=None,
                critic_wg=critic_wg,
                ref_policy_wg=ref_wg,
                rm_wg=rm_wg,
            )
            actor_rollout_cfg_by_model_id[spec.model_id] = actor_rollout_cfg

            init_targets_by_role["actor"].append((spec.model_id, actor_wg))
            init_targets_by_role["rollout"].append((spec.model_id, rollout_wg))
            if critic_wg is not None:
                init_targets_by_role["critic"].append((spec.model_id, critic_wg))
            if ref_wg is not None:
                init_targets_by_role["ref"].append((spec.model_id, ref_wg))
            if rm_wg is not None:
                init_targets_by_role["rm"].append((spec.model_id, rm_wg))

        # Init order is role-serial (safe on colocated WorkerDict), model-parallel per role.
        for role_name in ("actor", "rollout", "critic", "ref", "rm"):
            role_refs = []
            for model_id, wg in init_targets_by_role[role_name]:
                role_refs.extend(_enqueue_role_init(model_id, role_name, wg))
            if role_refs:
                print(
                    f"[star] parallel init_model role={role_name} "
                    f"models={len(init_targets_by_role[role_name])} total_remote_calls={len(role_refs)}"
                )
                ray.get(role_refs)
                print(f"[star] parallel init_model role={role_name} done")

        def _post_init_model(spec: EngineSpec):
            ctx = self.model_contexts[spec.model_id]
            local_buffer = self._local_traj_buffers_by_model.get(spec.model_id, None)
            if local_buffer is not None:
                # Worker-side buffers are per rollout-DP shard. The local buffer is
                # per model, so preserve equivalent aggregate residency capacity.
                local_buffer.max_items *= max(1, self._get_dp_size(ctx.rollout_wg, "rollout"))
            actor_rollout_cfg = actor_rollout_cfg_by_model_id[spec.model_id]
            rollout_mode = str(OmegaConf.select(actor_rollout_cfg, "rollout.mode") or "async")
            if rollout_mode == "async":
                from verl.experimental.one_step_off_policy.agent_loop.agent_loop import OneStepOffAgentLoopManager

                manager_cfg = self._build_manager_cfg_for_model(actor_rollout_cfg)
                ctx.rollout_manager = OneStepOffAgentLoopManager(config=manager_cfg, worker_group=ctx.rollout_wg)
                print(f"[star] async rollout manager ready model={spec.model_id}")

            self._init_weight_sync_group(spec.model_id, ctx)
            self._sync_rollout_weights(spec.model_id, ctx)
            return spec.model_id

        parallel_post_init = str(os.environ.get("STAR_PARALLEL_POST_INIT", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        post_init_parallelism = max(
            1,
            int(os.environ.get("STAR_POST_INIT_PARALLELISM", str(len(self.engine_specs)))),
        )
        if parallel_post_init and len(self.engine_specs) > 1:
            print(
                f"[star] parallel post-init enabled models={len(self.engine_specs)} "
                f"parallelism={post_init_parallelism}"
            )
            with ThreadPoolExecutor(max_workers=post_init_parallelism) as executor:
                futures = {executor.submit(_post_init_model, spec): spec.model_id for spec in self.engine_specs}
                for future in as_completed(futures):
                    model_id = futures[future]
                    try:
                        future.result()
                        print(f"[star] post-init done model={model_id}")
                    except Exception as exc:
                        print(f"[star] post-init failed model={model_id} err={type(exc).__name__}: {exc}")
                        raise
        else:
            for spec in self.engine_specs:
                _post_init_model(spec)

    def _init_weight_sync_group(self, model_id: str, ctx: ModelWorkerContext):
        weights_info = ctx.actor_wg.get_actor_weights_info()[0]
        ctx.rollout_wg.set_actor_weights_info(weights_info)

        group_name = f"actor_rollout_{model_id}"
        ctx.actor_wg.set_weight_sync_group_name(group_name)
        ctx.rollout_wg.set_weight_sync_group_name(group_name)

        actor_rollout_workers = ctx.actor_wg.workers + ctx.rollout_wg.workers
        n_workers = len(actor_rollout_workers)

        def _to_ref_list(x):
            if x is None:
                return []
            if isinstance(x, list | tuple):
                return list(x)
            return [x]

        # Weight-sync mode:
        # - auto (default): try Ray collective first, fallback to local_pair on CUDA, stateless on NPU.
        # - collective: use Ray collective only.
        # - stateless: force stateless process group (NPU) or local_pair on CUDA.
        # - local_pair: colocated actor/rollout in-process sync without cross-role NCCL.
        mode = str(os.environ.get("STAR_WEIGHT_SYNC_MODE", "auto")).strip().lower()
        if mode not in {"auto", "collective", "stateless", "local_pair"}:
            print(f"[star] invalid STAR_WEIGHT_SYNC_MODE={mode}, fallback to auto")
            mode = "auto"

        master_address = ray.get(ctx.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
        fixed_port = int(os.environ.get("STAR_WEIGHT_SYNC_MASTER_PORT", "0"))
        model_idx = self.model_ids.index(model_id) if model_id in self.model_ids else 0
        max_retries = int(os.environ.get("STAR_WEIGHT_SYNC_RETRIES", "3"))
        retry_stride = int(os.environ.get("STAR_WEIGHT_SYNC_PORT_RETRY_STRIDE", "10"))
        collective_ready = False
        if self.device_name != "npu" and mode in {"auto", "collective"}:
            try:
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                collective_ready = True
                print(f"[star] Ray collective group ready model={model_id} group={group_name}")
            except Exception as e:
                print(f"[star] Ray collective unavailable model={model_id}: {e}")
                if mode == "collective":
                    raise

        # CUDA + colocated WorkerDict cannot safely build a 2*N stateless NCCL communicator,
        # because actor/rollout are on the same local GPU and trigger duplicate GPU ranks.
        use_local_pair = self.device_name != "npu" and (
            mode == "local_pair" or ((mode in {"auto", "stateless"}) and not collective_ready)
        )
        if use_local_pair:
            ctx.actor_wg.set_weight_sync_mode("local_pair")
            ctx.rollout_wg.set_weight_sync_mode("local_pair")
            print(f"[star] local_pair weight sync ready model={model_id} group={group_name}")
            return

        ctx.actor_wg.set_weight_sync_mode("collective")
        ctx.rollout_wg.set_weight_sync_mode("collective")

        need_stateless = self.device_name == "npu" and (mode in {"auto", "stateless"} or not collective_ready)
        if need_stateless:
            last_err = None
            for attempt in range(max_retries):
                if fixed_port > 0:
                    # Per-model stable port + retry stride to avoid collisions with stale groups.
                    master_port = fixed_port + model_idx + attempt * retry_stride
                else:
                    master_port = ray.get(ctx.actor_wg.workers[0]._get_free_port.remote())
                print(
                    f"[star] init stateless weight sync model={model_id} attempt={attempt + 1}/{max_retries} "
                    f"addr={master_address}:{master_port} workers={n_workers}"
                )
                actor_refs = ctx.actor_wg.create_weight_sync_group(master_address, master_port, 0, n_workers)
                rollout_refs = ctx.rollout_wg.create_weight_sync_group(
                    master_address, master_port, len(ctx.actor_wg.workers), n_workers
                )
                try:
                    self._ray_get_with_timeout(
                        _to_ref_list(actor_refs) + _to_ref_list(rollout_refs),
                        timeout_s=self._weight_sync_timeout_seconds,
                        op_name=f"create_weight_sync_group(model={model_id},attempt={attempt + 1})",
                    )
                    last_err = None
                    print(f"[star] stateless weight sync ready model={model_id}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[star] stateless weight sync init failed model={model_id} attempt={attempt + 1}: {e}")
                    time.sleep(2)
            if last_err is not None:
                raise last_err

    def _sync_rollout_weights(self, model_id: str, ctx: ModelWorkerContext):
        actor_refs = ctx.actor_wg.sync_rollout_weights()
        rollout_refs = ctx.rollout_wg.sync_rollout_weights()
        self._ray_get_with_timeout(
            [actor_refs, rollout_refs],
            timeout_s=self._weight_sync_timeout_seconds,
            op_name=f"sync_rollout_weights(model={model_id})",
        )

    def _ensure_routing_fields(self, batch: DataProto):
        bsz = len(batch)
        if "query_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["query_id"] = np.array([uuid.uuid4().hex for _ in range(bsz)], dtype=object)
        if "agent_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_id"] = np.array(["agent_0"] * bsz, dtype=object)

    def _route_batch(self, batch: DataProto, epoch: int) -> dict[str, list[int]]:
        routed: dict[str, list[int]] = {model_id: [] for model_id in self.model_ids}
        query_ids = batch.non_tensor_batch["query_id"]
        policy = self.config.trainer.routing.policy

        for idx, query_id in enumerate(query_ids):
            base = str(query_id)
            if policy == "epoch_dynamic":
                base = f"{epoch}:{base}"
            slot = zlib.crc32(base.encode("utf-8")) % len(self.model_ids)
            routed[self.model_ids[slot]].append(idx)

        return {k: v for k, v in routed.items() if v}

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Skeleton version: keep all fields for easy routing/reward alignment.
        return batch

    @staticmethod
    def _extract_worker_rollout_timing(batch: DataProto) -> dict[str, float]:
        timing: dict[str, float] = {}
        for key, value in batch.non_tensor_batch.items():
            if not str(key).startswith("__star_timing_"):
                continue
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            flat = value.reshape(-1)
            if np.issubdtype(flat.dtype, np.number):
                timing[str(key).replace("__star_timing_", "", 1)] = float(np.mean(flat.astype(np.float64)))
        return timing

    def _decode_action_text(self, response_tokens: torch.Tensor | None) -> str:
        if response_tokens is None:
            return ""
        tokens = response_tokens.detach().cpu().tolist()
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            return ""

    @staticmethod
    def _attach_rollout_timing(batch: DataProto, timing: dict[str, Any]) -> None:
        merged: dict[str, float] = {}
        for key, value in batch.non_tensor_batch.items():
            if not str(key).startswith("__star_timing_"):
                continue
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            flat = value.reshape(-1)
            if np.issubdtype(flat.dtype, np.number):
                merged[str(key).replace("__star_timing_", "", 1)] = float(np.mean(flat.astype(np.float64)))
        for key, value in timing.items():
            if isinstance(value, int | float | np.integer | np.floating):
                merged[str(key)] = float(value)
        bsz = len(batch)
        for key, value in merged.items():
            batch.non_tensor_batch[f"__star_timing_{key}"] = np.full((bsz,), float(value), dtype=np.float64)

    @staticmethod
    def _extract_inner_rollout_timing(full_batch: DataProto) -> dict[str, float]:
        timing: dict[str, float] = {}
        meta_info = full_batch.meta_info or {}

        inner_timing = meta_info.get("timing", None)
        if isinstance(inner_timing, dict):
            generate_s = inner_timing.get("generate_sequences", None)
            if isinstance(generate_s, int | float | np.integer | np.floating):
                timing["engine_generate_s"] = float(generate_s)
            timing_aliases = {
                "agent_loop/generate_sequences/mean": "engine_generate_s",
                "agent_loop/generate_sequences/max": "engine_generate_max_s",
                "agent_loop/tool_calls/mean": "agent_loop_tool_calls_s",
                "agent_loop/tool_calls/max": "agent_loop_tool_calls_max_s",
                "agent_loop/server_rpc_roundtrip/mean": "agent_server_rpc_roundtrip_s",
                "agent_loop/server_rpc_roundtrip/max": "agent_server_rpc_roundtrip_max_s",
                "agent_loop/server_total/mean": "agent_server_total_s",
                "agent_loop/server_total/max": "agent_server_total_max_s",
                "agent_loop/server_rpc_overhead/mean": "agent_server_rpc_overhead_s",
                "agent_loop/server_rpc_overhead/max": "agent_server_rpc_overhead_max_s",
                "agent_loop/server_first_token/mean": "agent_server_first_token_s",
                "agent_loop/server_first_token/max": "agent_server_first_token_max_s",
                "agent_loop/server_decode_tail/mean": "agent_server_decode_tail_s",
                "agent_loop/server_decode_tail/max": "agent_server_decode_tail_max_s",
                "agent_loop/worker/start_lag/mean": "agent_worker_start_lag_s",
                "agent_loop/worker/start_lag/max": "agent_worker_start_lag_max_s",
                "agent_loop/worker/prep/mean": "agent_worker_prep_s",
                "agent_loop/worker/prep/max": "agent_worker_prep_max_s",
                "agent_loop/worker/run_loops/mean": "agent_worker_run_loops_s",
                "agent_loop/worker/run_loops/max": "agent_worker_run_loops_max_s",
                "agent_loop/worker/postprocess/mean": "agent_worker_postprocess_s",
                "agent_loop/worker/postprocess/max": "agent_worker_postprocess_max_s",
                "agent_loop/worker/total/mean": "agent_worker_total_s",
                "agent_loop/worker/total/max": "agent_worker_total_max_s",
                "agent_loop/worker/non_loop_overhead/mean": "agent_worker_non_loop_overhead_s",
                "agent_loop/worker/non_loop_overhead/max": "agent_worker_non_loop_overhead_max_s",
                "agent_loop/manager/prep": "agent_loop_manager_prep_s",
                "agent_loop/manager/worker_rpc_wait": "agent_loop_manager_worker_rpc_wait_s",
                "agent_loop/manager/worker_rpc_mean": "agent_loop_manager_worker_rpc_mean_s",
                "agent_loop/manager/worker_rpc_max": "agent_loop_manager_worker_rpc_max_s",
                "agent_loop/manager/concat": "agent_loop_manager_concat_s",
                "agent_loop/manager/metrics_reduce": "agent_loop_manager_metrics_reduce_s",
                "agent_loop/manager/total": "agent_loop_manager_total_s",
                "agent_loop/manager/overhead": "agent_loop_manager_overhead_s",
            }
            for src_key, dst_key in timing_aliases.items():
                value = inner_timing.get(src_key, None)
                if isinstance(value, int | float | np.integer | np.floating):
                    timing[dst_key] = float(value)

        metric_list = meta_info.get("metrics", None)
        if isinstance(metric_list, list):
            metric_acc: dict[str, list[float]] = {}
            for item in metric_list:
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    if not isinstance(value, int | float | np.integer | np.floating):
                        continue
                    metric_acc.setdefault(str(key), []).append(float(value))
            if metric_acc.get("generate_sequences") and "engine_generate_s" not in timing:
                timing["engine_generate_s"] = float(np.mean(metric_acc["generate_sequences"]))
            if metric_acc.get("tool_calls"):
                timing["agent_loop_tool_calls_s"] = float(np.mean(metric_acc["tool_calls"]))

        for key, value in full_batch.non_tensor_batch.items():
            if not str(key).startswith("__star_timing_"):
                continue
            if not isinstance(value, np.ndarray) or value.size == 0:
                continue
            flat = value.reshape(-1)
            if np.issubdtype(flat.dtype, np.number):
                timing[str(key).replace("__star_timing_", "", 1)] = float(np.mean(flat.astype(np.float64)))

        return timing

    @staticmethod
    def _strip_concat_volatile_meta(data: DataProto) -> DataProto:
        meta_info = dict(data.meta_info or {})
        # Per-request/per-worker diagnostics are not part of the sample payload.
        # They can legitimately differ and break DataProto.concat consistency checks.
        meta_info.pop("timing", None)
        meta_info.pop("metrics", None)
        return DataProto(batch=data.batch, non_tensor_batch=data.non_tensor_batch, meta_info=meta_info)

    @classmethod
    def _concat_data_proto_safe(cls, parts: list[DataProto]) -> DataProto:
        cleaned = [cls._strip_concat_volatile_meta(part) for part in parts]
        return DataProto.concat(cleaned) if len(cleaned) > 1 else cleaned[0]

    def _build_thin_from_generated_local(self, model_id: str, full_batch: DataProto) -> DataProto:
        build_start = time.perf_counter()
        bsz = len(full_batch)
        query_ids = full_batch.non_tensor_batch.get("query_id", np.array(["unknown"] * bsz, dtype=object))
        agent_ids = full_batch.non_tensor_batch.get("agent_id", np.array(["agent_0"] * bsz, dtype=object))
        keep_mask = full_batch.non_tensor_batch.get("__star_keep_in_buffer__", None)
        if keep_mask is None:
            keep_mask = np.ones((bsz,), dtype=bool)
        else:
            keep_mask = np.array(keep_mask, dtype=bool).reshape(-1)
            if keep_mask.shape[0] != bsz:
                keep_mask = np.ones((bsz,), dtype=bool)

        traj_ids = np.empty((bsz,), dtype=object)
        model_ids = np.empty((bsz,), dtype=object)
        action_text = np.empty((bsz,), dtype=object)
        created_ts = np.empty((bsz,), dtype=np.float64)

        responses = full_batch.batch.get("responses", None)
        now = time.time()
        decode_action_text_s = 0.0
        buffer_put_s = 0.0
        local_buffer = self._local_traj_buffers_by_model[model_id]
        for i in range(bsz):
            traj_id = uuid.uuid4().hex if bool(keep_mask[i]) else ""
            traj_ids[i] = traj_id
            model_ids[i] = model_id
            created_ts[i] = now

            response_tokens = responses[i] if responses is not None else None
            decode_start = time.perf_counter()
            action_text[i] = self._decode_action_text(response_tokens)
            decode_action_text_s += float(time.perf_counter() - decode_start)

            if bool(keep_mask[i]):
                fat_item = full_batch[i : i + 1]
                put_start = time.perf_counter()
                local_buffer.put(
                    TrajectoryEntry(
                        traj_id=traj_id,
                        model_id=model_id,
                        query_id=str(query_ids[i]),
                        agent_id=str(agent_ids[i]),
                        fat_data=fat_item,
                    )
                )
                buffer_put_s += float(time.perf_counter() - put_start)

        thin = DataProto.from_dict(
            non_tensors={
                "traj_id": traj_ids,
                "query_id": query_ids.astype(object),
                "agent_id": agent_ids.astype(object),
                "model_id": model_ids,
                "action_text": action_text,
                "created_ts": created_ts,
            },
            meta_info={"thin_only": True},
        )
        thin_build_s = float(time.perf_counter() - build_start)
        inherited_timing = self._extract_inner_rollout_timing(full_batch)
        inherited_timing.update(
            {
                "worker_thin_build_s": thin_build_s,
                "worker_decode_action_text_s": float(decode_action_text_s),
                "worker_buffer_put_s": float(buffer_put_s),
                "worker_build_overhead_s": float(max(thin_build_s - decode_action_text_s - buffer_put_s, 0.0)),
                "worker_total_s": thin_build_s,
                "local_build_thin_used": 1.0,
            }
        )
        self._attach_rollout_timing(thin, inherited_timing)
        return thin

    async def _rollout_model_async(
        self,
        model_id: str,
        batch: DataProto,
        timing_state: Optional[dict[str, Any]] = None,
    ):
        ctx = self.model_contexts[model_id]
        if model_id not in self._rollout_semaphore_by_model:
            self._rollout_semaphore_by_model[model_id] = asyncio.Semaphore(max(1, self._max_parallel_rollouts_per_model))
        gen_batch = self._get_gen_batch(batch)
        request_start = time.perf_counter()
        request_bsz = int(len(gen_batch))
        rollout_dp_size = 0
        dp_padding_applied = 0.0
        dp_padding_added = 0.0
        dp_padding_factor = 1.0
        local_build_thin_used = 0.0
        manager_generate_s = 0.0
        build_thin_s = 0.0
        data_proto_pad_select_s = 0.0
        if timing_state is not None:
            timing_state["queue_wait_s"] = 0.0
            timing_state["rollout_exec_s"] = 0.0
            timing_state["rollout_total_s"] = 0.0
            timing_state["rpc_roundtrip_s"] = 0.0
            timing_state["queue_acquired"] = False
        worker_timing: dict[str, float] = {}
        async with self._rollout_semaphore_by_model[model_id]:
            exec_start = time.perf_counter()
            queue_wait_s = float(exec_start - request_start)
            if timing_state is not None:
                timing_state["queue_wait_s"] = queue_wait_s
                timing_state["queue_acquired"] = True
            rollout_dp_size = self._get_dp_size(ctx.rollout_wg, "rollout")
            try:
                if ctx.rollout_manager is None:
                    bsz = len(gen_batch)
                    if bsz > 0 and rollout_dp_size > 1 and bsz % rollout_dp_size != 0:
                        pad = rollout_dp_size - (bsz % rollout_dp_size)
                        dp_padding_applied = 1.0
                        dp_padding_added = float(pad)
                        dp_padding_factor = float((bsz + pad) / max(1, bsz))
                        pad_select_start = time.perf_counter()
                        padded_indices = list(range(bsz)) + [bsz - 1] * pad
                        padded_batch = gen_batch.select_idxs(padded_indices)
                        keep_mask = np.zeros((len(padded_indices),), dtype=bool)
                        if bsz == 1 and len(padded_indices) >= rollout_dp_size:
                            pick = int(self._thin_pick_cursor_by_model[model_id] % rollout_dp_size)
                            self._thin_pick_cursor_by_model[model_id] += 1
                            keep_mask[pick] = True
                        else:
                            keep_mask[:bsz] = True
                        padded_batch.non_tensor_batch["__star_keep_in_buffer__"] = keep_mask
                        data_proto_pad_select_s += float(time.perf_counter() - pad_select_start)
                        build_thin_start = time.perf_counter()
                        thin_padded = await asyncio.to_thread(ctx.rollout_wg.generate_sequences_thin, padded_batch)
                        build_thin_s += float(time.perf_counter() - build_thin_start)
                        pad_select_start = time.perf_counter()
                        if bsz == 1 and len(padded_indices) >= rollout_dp_size:
                            pick_idx = int(np.argmax(keep_mask))
                            thin = thin_padded.select_idxs([pick_idx])
                        else:
                            thin = thin_padded.select_idxs(list(range(bsz)))
                        data_proto_pad_select_s += float(time.perf_counter() - pad_select_start)
                    else:
                        build_thin_start = time.perf_counter()
                        thin = await asyncio.to_thread(ctx.rollout_wg.generate_sequences_thin, gen_batch)
                        build_thin_s += float(time.perf_counter() - build_thin_start)
                    worker_timing = self._extract_worker_rollout_timing(thin)
                else:
                    manager_start = time.perf_counter()
                    fat = await ctx.rollout_manager.generate_sequences_async(gen_batch)
                    manager_generate_s = float(time.perf_counter() - manager_start)
                    # Avoid DataProto.union() conflicts on object-typed non-tensor fields
                    # (e.g. raw_prompt) that may be semantically equivalent but not deeply equal.
                    # The async agent-loop output already carries the required rollout tensors.
                    full_batch = fat
                    pad_select_start = time.perf_counter()
                    for key in ("query_id", "agent_id"):
                        if key not in full_batch.non_tensor_batch and key in gen_batch.non_tensor_batch:
                            full_batch.non_tensor_batch[key] = gen_batch.non_tensor_batch[key]
                    data_proto_pad_select_s += float(time.perf_counter() - pad_select_start)
                    bsz = len(full_batch)
                    use_local_build_thin = (
                        self._local_build_thin_enabled
                        and bsz > 0
                        and bsz <= self._local_build_thin_max_bsz
                    )
                    if use_local_build_thin:
                        local_build_thin_used = 1.0
                        build_thin_start = time.perf_counter()
                        thin = self._build_thin_from_generated_local(model_id, full_batch)
                        build_thin_s += float(time.perf_counter() - build_thin_start)
                    elif bsz > 0 and rollout_dp_size > 1 and bsz % rollout_dp_size != 0:
                        # ND dispatch requires equal chunks. For tiny validation/workflow batches,
                        # pad by repeating the last sample, then trim back after conversion.
                        pad = rollout_dp_size - (bsz % rollout_dp_size)
                        dp_padding_applied = 1.0
                        dp_padding_added = float(pad)
                        dp_padding_factor = float((bsz + pad) / max(1, bsz))
                        pad_select_start = time.perf_counter()
                        padded_indices = list(range(bsz)) + [bsz - 1] * pad
                        padded_batch = full_batch.select_idxs(padded_indices)
                        keep_mask = np.zeros((len(padded_indices),), dtype=bool)
                        if bsz == 1 and len(padded_indices) >= rollout_dp_size:
                            pick = int(self._thin_pick_cursor_by_model[model_id] % rollout_dp_size)
                            self._thin_pick_cursor_by_model[model_id] += 1
                            keep_mask[pick] = True
                        else:
                            keep_mask[:bsz] = True
                        padded_batch.non_tensor_batch["__star_keep_in_buffer__"] = keep_mask
                        data_proto_pad_select_s += float(time.perf_counter() - pad_select_start)
                        build_thin_start = time.perf_counter()
                        thin_padded = await asyncio.to_thread(ctx.rollout_wg.build_thin_from_generated, padded_batch)
                        build_thin_s += float(time.perf_counter() - build_thin_start)
                        pad_select_start = time.perf_counter()
                        if bsz == 1 and len(padded_indices) >= rollout_dp_size:
                            pick_idx = int(np.argmax(keep_mask))
                            thin = thin_padded.select_idxs([pick_idx])
                        else:
                            thin = thin_padded.select_idxs(list(range(bsz)))
                        data_proto_pad_select_s += float(time.perf_counter() - pad_select_start)
                    else:
                        build_thin_start = time.perf_counter()
                        thin = await asyncio.to_thread(ctx.rollout_wg.build_thin_from_generated, full_batch)
                        build_thin_s += float(time.perf_counter() - build_thin_start)
                    worker_timing = self._extract_worker_rollout_timing(thin)
            finally:
                exec_elapsed_s = float(time.perf_counter() - exec_start)
                total_elapsed_s = float(time.perf_counter() - request_start)
                if timing_state is not None:
                    timing_state["rollout_exec_s"] = exec_elapsed_s
                    timing_state["rollout_total_s"] = total_elapsed_s
                    timing_state["rpc_roundtrip_s"] = exec_elapsed_s
                    for key, value in worker_timing.items():
                        timing_state[key] = float(value)
                    worker_total_s = worker_timing.get("worker_total_s", None)
                    if isinstance(worker_total_s, int | float | np.integer | np.floating):
                        timing_state["rpc_overhead_s"] = float(max(exec_elapsed_s - float(worker_total_s), 0.0))
                    timing_state["request_bsz"] = float(request_bsz)
                    timing_state["rollout_dp_size"] = float(rollout_dp_size)
                    timing_state["dp_padding_applied"] = float(dp_padding_applied)
                    timing_state["dp_padding_added"] = float(dp_padding_added)
                    timing_state["dp_padding_factor"] = float(dp_padding_factor)
                    timing_state["local_build_thin_used"] = float(local_build_thin_used)
                    timing_state["manager_generate_s"] = float(manager_generate_s)
                    timing_state["build_thin_s"] = float(build_thin_s)
                    timing_state["data_proto_pad_select_s"] = float(data_proto_pad_select_s)
        timing_info = {
            "queue_wait_s": float(timing_state["queue_wait_s"]) if timing_state is not None else queue_wait_s,
            "rollout_exec_s": float(timing_state["rollout_exec_s"]) if timing_state is not None else exec_elapsed_s,
            "rollout_total_s": float(timing_state["rollout_total_s"]) if timing_state is not None else total_elapsed_s,
            "rpc_roundtrip_s": float(timing_state["rpc_roundtrip_s"]) if timing_state is not None else exec_elapsed_s,
            "request_bsz": float(request_bsz),
            "rollout_dp_size": float(rollout_dp_size),
            "dp_padding_applied": float(dp_padding_applied),
            "dp_padding_added": float(dp_padding_added),
            "dp_padding_factor": float(dp_padding_factor),
            "local_build_thin_used": float(local_build_thin_used),
            "manager_generate_s": float(manager_generate_s),
            "build_thin_s": float(build_thin_s),
            "data_proto_pad_select_s": float(data_proto_pad_select_s),
        }
        if timing_state is not None:
            for key, value in timing_state.items():
                if key in timing_info or key == "queue_acquired":
                    continue
                if isinstance(value, int | float | np.integer | np.floating):
                    timing_info[key] = float(value)
        return model_id, thin, batch, timing_info

    def _get_microbatch_lock(self, model_id: str) -> asyncio.Lock:
        lock = self._llm_microbatch_locks_by_model.get(model_id, None)
        if lock is None:
            lock = asyncio.Lock()
            self._llm_microbatch_locks_by_model[model_id] = lock
        return lock

    @staticmethod
    def _microbatch_pending_size(queue: list[RolloutMicrobatchRequest]) -> int:
        return int(sum(len(req.batch) for req in queue if not req.future.cancelled()))

    async def _rollout_model_async_batched(
        self,
        model_id: str,
        batch: DataProto,
        timing_state: Optional[dict[str, Any]] = None,
    ):
        if (
            not self._llm_microbatch_enabled
            or len(batch) == 0
            or len(batch) >= self._llm_microbatch_max_size
        ):
            if timing_state is not None:
                timing_state["microbatch_enabled"] = 0.0
                timing_state["microbatch_size"] = float(len(batch))
                timing_state["microbatch_request_count"] = 1.0
                timing_state["microbatch_wait_s"] = 0.0
                timing_state["microbatch_wait_max_s"] = 0.0
                timing_state["microbatch_batch_exec_s"] = 0.0
                timing_state["microbatch_flush_timeout"] = 0.0
                timing_state["microbatch_flush_size"] = 0.0
                timing_state["microbatch_saved_calls"] = 0.0
                timing_state["microbatch_saved_call_ratio"] = 0.0
            return await self._rollout_model_async(model_id, batch, timing_state=timing_state)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = RolloutMicrobatchRequest(
            model_id=model_id,
            batch=batch,
            timing_state=timing_state,
            future=future,
            enqueue_ts=time.perf_counter(),
        )
        if timing_state is not None:
            timing_state["microbatch_enabled"] = 1.0
            timing_state["microbatch_size"] = 1.0
            timing_state["microbatch_wait_s"] = 0.0

        await self._enqueue_rollout_microbatch_request(request)
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            await self._remove_rollout_microbatch_request(request)
            raise

    async def _enqueue_rollout_microbatch_request(self, request: RolloutMicrobatchRequest) -> None:
        model_id = request.model_id
        lock = self._get_microbatch_lock(model_id)
        async with lock:
            queue = self._llm_microbatch_queues_by_model[model_id]
            queue.append(request)
            pending_size = self._microbatch_pending_size(queue)
            should_flush_now = pending_size >= self._llm_microbatch_max_size or self._llm_microbatch_max_wait_s <= 0
            if should_flush_now:
                reason = "size" if pending_size >= self._llm_microbatch_max_size else "timeout"
                task = self._llm_microbatch_flush_tasks_by_model.get(model_id, None)
                task_kind = self._llm_microbatch_flush_task_kind_by_model.get(model_id, "")
                if task is not None and not task.done():
                    if task_kind == "timer":
                        task.cancel()
                    else:
                        return
                self._llm_microbatch_flush_tasks_by_model[model_id] = asyncio.create_task(
                    self._flush_rollout_microbatch(model_id, reason=reason)
                )
                self._llm_microbatch_flush_task_kind_by_model[model_id] = "flush"
                return

            task = self._llm_microbatch_flush_tasks_by_model.get(model_id, None)
            if task is None or task.done():
                self._llm_microbatch_flush_tasks_by_model[model_id] = asyncio.create_task(
                    self._flush_rollout_microbatch_after_wait(model_id)
                )
                self._llm_microbatch_flush_task_kind_by_model[model_id] = "timer"

    async def _remove_rollout_microbatch_request(self, request: RolloutMicrobatchRequest) -> None:
        lock = self._get_microbatch_lock(request.model_id)
        async with lock:
            queue = self._llm_microbatch_queues_by_model.get(request.model_id, [])
            self._llm_microbatch_queues_by_model[request.model_id] = [req for req in queue if req is not request]
            if len(self._llm_microbatch_queues_by_model[request.model_id]) == 0:
                self._llm_microbatch_flush_tasks_by_model.pop(request.model_id, None)
                self._llm_microbatch_flush_task_kind_by_model.pop(request.model_id, None)

    async def _flush_rollout_microbatch_after_wait(self, model_id: str) -> None:
        try:
            await asyncio.sleep(self._llm_microbatch_max_wait_s)
            if self._llm_microbatch_flush_tasks_by_model.get(model_id) is not asyncio.current_task():
                return
            self._llm_microbatch_flush_tasks_by_model[model_id] = asyncio.create_task(
                self._flush_rollout_microbatch(model_id, reason="timeout")
            )
            self._llm_microbatch_flush_task_kind_by_model[model_id] = "flush"
        except asyncio.CancelledError:
            return

    async def _flush_rollout_microbatch(self, model_id: str, reason: str) -> None:
        selected: list[RolloutMicrobatchRequest] = []
        lock = self._get_microbatch_lock(model_id)
        async with lock:
            queue = [
                req
                for req in self._llm_microbatch_queues_by_model.get(model_id, [])
                if not req.future.cancelled()
            ]
            selected_size = 0
            remaining: list[RolloutMicrobatchRequest] = []
            for req in queue:
                req_size = len(req.batch)
                if selected and selected_size + req_size > self._llm_microbatch_max_size:
                    remaining.append(req)
                    continue
                selected.append(req)
                selected_size += req_size

            self._llm_microbatch_queues_by_model[model_id] = remaining
            current_task = asyncio.current_task()
            if self._llm_microbatch_flush_tasks_by_model.get(model_id) is current_task:
                self._llm_microbatch_flush_tasks_by_model.pop(model_id, None)
                self._llm_microbatch_flush_task_kind_by_model.pop(model_id, None)

            if remaining:
                pending_size = self._microbatch_pending_size(remaining)
                if pending_size >= self._llm_microbatch_max_size:
                    next_task = asyncio.create_task(self._flush_rollout_microbatch(model_id, reason="size"))
                    next_kind = "flush"
                else:
                    next_task = asyncio.create_task(self._flush_rollout_microbatch_after_wait(model_id))
                    next_kind = "timer"
                self._llm_microbatch_flush_tasks_by_model[model_id] = next_task
                self._llm_microbatch_flush_task_kind_by_model[model_id] = next_kind

        if not selected:
            return
        await self._run_rollout_microbatch(model_id, selected, reason=reason)

    async def _run_rollout_microbatch(
        self,
        model_id: str,
        requests: list[RolloutMicrobatchRequest],
        reason: str,
    ) -> None:
        active_requests = [req for req in requests if not req.future.cancelled()]
        if not active_requests:
            return

        flush_ts = time.perf_counter()
        wait_times = [max(0.0, flush_ts - req.enqueue_ts) for req in active_requests]
        microbatch_request_count = len(active_requests)
        microbatch_size = int(sum(len(req.batch) for req in active_requests))

        batch_timing_state: dict[str, Any] = {}
        batch_start = time.perf_counter()
        try:
            concat_batches = [self._strip_concat_volatile_meta(req.batch) for req in active_requests]
            batched_batch = (
                concat_batches[0]
                if len(concat_batches) == 1
                else DataProto.concat(concat_batches)
            )
            _, thin, _, batch_timing_info = await self._rollout_model_async(
                model_id,
                batched_batch,
                timing_state=batch_timing_state,
            )
            batch_exec_s = float(time.perf_counter() - batch_start)
        except Exception as exc:
            for req in active_requests:
                if not req.future.cancelled():
                    req.future.set_exception(exc)
            return

        offset = 0
        max_wait_s = float(max(wait_times)) if wait_times else 0.0
        for req, wait_s in zip(active_requests, wait_times, strict=True):
            req_len = len(req.batch)
            indices = list(range(offset, offset + req_len))
            offset += req_len
            try:
                req_thin = thin.select_idxs(indices)
            except Exception as exc:
                if not req.future.cancelled():
                    req.future.set_exception(exc)
                continue

            req_timing = dict(batch_timing_info or {})
            req_timing["rollout_total_s"] = float(req_timing.get("rollout_total_s", 0.0) + wait_s)
            req_timing["microbatch_enabled"] = 1.0
            req_timing["microbatch_size"] = float(microbatch_size)
            req_timing["microbatch_request_count"] = float(microbatch_request_count)
            req_timing["microbatch_wait_s"] = float(wait_s)
            req_timing["microbatch_wait_max_s"] = max_wait_s
            req_timing["microbatch_batch_exec_s"] = batch_exec_s
            req_timing["microbatch_flush_timeout"] = 1.0 if reason == "timeout" else 0.0
            req_timing["microbatch_flush_size"] = 1.0 if reason == "size" else 0.0
            req_timing["microbatch_saved_calls"] = float(max(0, microbatch_request_count - 1))
            req_timing["microbatch_saved_call_ratio"] = float(
                max(0, microbatch_request_count - 1) / max(1, microbatch_request_count)
            )

            if req.timing_state is not None:
                req.timing_state.update(req_timing)
            if not req.future.cancelled():
                req.future.set_result((model_id, req_thin, req.batch, req_timing))

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # fallback: find the first balanced json object
        start = raw.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(raw)):
                ch = raw[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        segment = raw[start : i + 1]
                        try:
                            obj = json.loads(segment)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            start = raw.find("{", start + 1)
        return None

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            if "ground_truth" in value:
                return StarRayTrainer._to_str_list(value["ground_truth"])
            if "target" in value:
                return StarRayTrainer._to_str_list(value["target"])
            return [json.dumps(value, ensure_ascii=False)]
        if isinstance(value, list | tuple):
            return [str(x) for x in value if str(x).strip()]
        if isinstance(value, np.ndarray):
            return [str(x) for x in value.tolist() if str(x).strip()]
        return [str(value)]

    def _extract_string_vector(self, source_batch: DataProto, keys: list[str], default: str = "") -> np.ndarray:
        bsz = len(source_batch)
        for key in keys:
            if key in source_batch.non_tensor_batch:
                values = source_batch.non_tensor_batch[key]
                out = [str(v) if v is not None else default for v in values]
                return np.array(out, dtype=object)
        return np.array([default] * bsz, dtype=object)

    def _extract_ground_truth_lists(self, source_batch: DataProto, keys: list[str]) -> list[list[str]]:
        bsz = len(source_batch)
        out: list[list[str]] = [[] for _ in range(bsz)]
        for i in range(bsz):
            picked = None
            for key in keys:
                if key not in source_batch.non_tensor_batch:
                    continue
                value = source_batch.non_tensor_batch[key][i]
                if value is None:
                    continue
                picked = value
                break
            out[i] = self._to_str_list(picked)
        return out

    def _build_workflow_prompt_batch(
        self,
        source_batch: DataProto,
        raw_prompts: list[list[dict]],
        agent_id: str,
    ) -> DataProto:
        bsz = len(source_batch)
        if source_batch.batch is not None and "dummy_tensor" in source_batch.batch.keys():
            tensors = {"dummy_tensor": source_batch.batch["dummy_tensor"]}
        else:
            tensors = {"dummy_tensor": torch.zeros((bsz, 1), dtype=torch.uint8)}

        non_tensors = {k: v.copy() for k, v in source_batch.non_tensor_batch.items()}
        non_tensors["raw_prompt"] = np.array(raw_prompts, dtype=object)
        non_tensors["agent_id"] = np.array([agent_id] * bsz, dtype=object)
        meta_info = dict(source_batch.meta_info or {})
        # Workflow LLM requests only need prompt/sample payload. Per-query timing
        # and metrics are volatile and can break DataProto.concat consistency checks.
        meta_info.pop("timing", None)
        meta_info.pop("metrics", None)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _build_commit_rewards_from_thin(self, thin_batch: DataProto, reward: np.ndarray) -> DataProto:
        done = np.ones((len(thin_batch),), dtype=bool)
        return DataProto.from_dict(
            tensors={
                "reward": torch.tensor(reward, dtype=torch.float32),
                "done": torch.tensor(done, dtype=torch.bool),
            },
            non_tensors={
                "traj_id": thin_batch.non_tensor_batch["traj_id"],
                "model_id": thin_batch.non_tensor_batch["model_id"],
                "query_id": thin_batch.non_tensor_batch["query_id"],
                "agent_id": thin_batch.non_tensor_batch["agent_id"],
            },
        )

    @staticmethod
    def _empty_rewards() -> DataProto:
        return DataProto.from_dict(
            tensors={"reward": torch.zeros((0,), dtype=torch.float32), "done": torch.zeros((0,), dtype=torch.bool)},
            non_tensors={
                "traj_id": np.array([], dtype=object),
                "model_id": np.array([], dtype=object),
                "query_id": np.array([], dtype=object),
                "agent_id": np.array([], dtype=object),
            },
        )

    async def _run_single_agent_workflow(self, batch: DataProto, epoch: int) -> tuple[DataProto, dict[str, float]]:
        routed = self._route_batch(batch, epoch)
        tasks = [self._rollout_model_async(model_id, batch.select_idxs(idxs)) for model_id, idxs in routed.items()]
        rollout_results = await asyncio.gather(*tasks)

        reward_parts = []
        for _, thin_output, source_sub_batch, _ in rollout_results:
            if len(thin_output) == 0:
                continue
            reward_parts.append(self._assemble_rewards(thin_output, source_batch=source_sub_batch))

        if len(reward_parts) == 0:
            return self._empty_rewards(), {}
        rewards = self._concat_data_proto_safe(reward_parts)
        return rewards, {}

    def _create_workflow_runner(self):
        workflow_cfg = self.config.star.get("workflow", {})
        runner_cfg = workflow_cfg.get("runner", {})
        if "path" in runner_cfg and "name" in runner_cfg:
            runner_cls = load_extern_object(str(runner_cfg.get("path")), str(runner_cfg.get("name")))
            return runner_cls(trainer=self, config=self.config)

        from verl.experimental.star_ppo.workflows.builtin import BuiltinWorkflowRunner

        return BuiltinWorkflowRunner(trainer=self, config=self.config)

    @staticmethod
    def _extract_optional_vector(source_batch: Optional[DataProto], keys: list[str], default: float, size: int) -> np.ndarray:
        if source_batch is None:
            return np.full((size,), default, dtype=np.float32)
        for key in keys:
            if source_batch.batch is not None and key in source_batch.batch.keys():
                return source_batch.batch[key].detach().cpu().float().reshape(-1).numpy()
            if key in source_batch.non_tensor_batch:
                return np.array(source_batch.non_tensor_batch[key], dtype=np.float32).reshape(-1)
        return np.full((size,), default, dtype=np.float32)

    @staticmethod
    def _extract_bool_vector(source_batch: Optional[DataProto], keys: list[str], default: bool, size: int) -> np.ndarray:
        if source_batch is None:
            return np.full((size,), default, dtype=bool)
        for key in keys:
            if source_batch.batch is not None and key in source_batch.batch.keys():
                return source_batch.batch[key].detach().cpu().bool().reshape(-1).numpy()
            if key in source_batch.non_tensor_batch:
                return np.array(source_batch.non_tensor_batch[key], dtype=bool).reshape(-1)
        return np.full((size,), default, dtype=bool)

    def _assemble_rewards(self, thin_batch: DataProto, source_batch: Optional[DataProto] = None) -> DataProto:
        action_text = thin_batch.non_tensor_batch.get("action_text", np.array([], dtype=object))
        bsz = len(action_text)
        if bsz == 0:
            return DataProto.from_dict(
                tensors={"reward": torch.zeros((0,), dtype=torch.float32), "done": torch.zeros((0,), dtype=torch.bool)},
                non_tensors={"traj_id": np.array([], dtype=object), "model_id": np.array([], dtype=object)},
            )

        format_reward = self._extract_optional_vector(
            source_batch,
            keys=["format_reward", "format_rewards", "step_reward", "rule_reward"],
            default=np.nan,
            size=bsz,
        )
        fallback_reward = np.array([1.0 if len(str(text).strip()) > 0 else 0.0 for text in action_text], dtype=np.float32)
        format_reward = np.where(np.isnan(format_reward), fallback_reward, format_reward).astype(np.float32)

        outcome_reward = self._extract_optional_vector(
            source_batch,
            keys=["outcome_reward", "final_reward", "task_reward"],
            default=0.0,
            size=bsz,
        ).astype(np.float32)
        workflow_done = self._extract_bool_vector(
            source_batch,
            keys=["workflow_done", "query_done", "is_done", "done"],
            default=True,
            size=bsz,
        )

        reward = np.zeros((bsz,), dtype=np.float32)
        reward_mode = self.config.star.reward.get("mode", "streaming")
        emit_intermediate = bool(self.config.star.reward.get("emit_intermediate_format", True))
        query_ids = thin_batch.non_tensor_batch["query_id"]
        for i in range(bsz):
            query_id = str(query_ids[i])
            self.query_reward_ledger[query_id] += float(format_reward[i])

            if reward_mode == "terminal" and not bool(workflow_done[i]):
                reward[i] = float(format_reward[i]) if emit_intermediate else 0.0
            elif reward_mode == "terminal" and bool(workflow_done[i]):
                reward[i] = float(self.query_reward_ledger[query_id] + outcome_reward[i])
                self.query_reward_ledger.pop(query_id, None)
            else:
                reward[i] = float(format_reward[i] + (outcome_reward[i] if workflow_done[i] else 0.0))
                if workflow_done[i]:
                    self.query_reward_ledger.pop(query_id, None)

        # In V3 we always release trajectories to training queue each step.
        done = np.full((bsz,), True, dtype=bool)
        rewards = DataProto.from_dict(
            tensors={"reward": torch.tensor(reward, dtype=torch.float32), "done": torch.tensor(done, dtype=torch.bool)},
            non_tensors={
                "traj_id": thin_batch.non_tensor_batch["traj_id"],
                "model_id": thin_batch.non_tensor_batch["model_id"],
                "query_id": thin_batch.non_tensor_batch["query_id"],
                "agent_id": thin_batch.non_tensor_batch["agent_id"],
            },
        )
        return rewards

    def _commit_rewards(self, rewards: DataProto) -> dict[str, float]:
        model_ids = rewards.non_tensor_batch["model_id"]
        metrics = {}
        for model_id in self.model_ids:
            indices = [i for i, mid in enumerate(model_ids) if str(mid) == model_id]
            if not indices:
                continue
            sub = rewards.select_idxs(indices)
            model_metrics, remote_indices = self._commit_rewards_to_local_buffer(model_id, sub)
            if remote_indices:
                remote_sub = sub.select_idxs(remote_indices)
                worker_outputs = self.model_contexts[model_id].rollout_wg.commit_rewards(remote_sub)
                reduced = self._reduce_worker_metrics(worker_outputs)
                for k, v in reduced.items():
                    if k in {"star/committed", "buffer/total", "buffer/ready", "buffer/dropped_queries"}:
                        model_metrics[k] = float(model_metrics.get(k, 0.0) + float(v))
                    elif k != "star/reward_in":
                        model_metrics[k] = float(v)
                model_metrics["star/remote_reward_in"] = float(len(remote_indices))
            model_metrics["star/reward_in"] = float(len(sub))
            for k, v in model_metrics.items():
                metrics[f"model/{model_id}/{k}"] = v
        # reward metrics by agent id (for independent curves on tracking backend)
        reward_vec = rewards.batch["reward"].detach().cpu().numpy() if len(rewards) > 0 else np.array([])
        agent_ids = rewards.non_tensor_batch.get("agent_id", np.array([], dtype=object))
        for agent_id in np.unique(agent_ids):
            mask = agent_ids == agent_id
            if mask.sum() > 0:
                metrics[f"agent/{agent_id}/reward_mean"] = float(np.mean(reward_vec[mask]))
                metrics[f"agent/{agent_id}/samples"] = float(np.sum(mask))
        return metrics

    def _commit_rewards_to_local_buffer(self, model_id: str, rewards: DataProto) -> tuple[dict[str, float], list[int]]:
        local_buffer = self._local_traj_buffers_by_model.get(model_id, None)
        if local_buffer is None or len(rewards) == 0:
            return {}, list(range(len(rewards)))

        traj_ids = rewards.non_tensor_batch.get("traj_id", np.array([], dtype=object))
        reward_vec = rewards.batch.get("reward", None) if rewards.batch is not None else None
        done_vec = rewards.batch.get("done", None) if rewards.batch is not None else None
        if reward_vec is None:
            reward_vec = torch.zeros((len(traj_ids),), dtype=torch.float32)
        if done_vec is None:
            done_vec = torch.ones((len(traj_ids),), dtype=torch.bool)

        committed = 0
        remote_indices: list[int] = []
        for i, traj_id in enumerate(traj_ids):
            ok = local_buffer.commit_reward(
                str(traj_id),
                reward=reward_vec[i].reshape(()).to(torch.float32),
                done=bool(done_vec[i].item()),
            )
            if ok:
                committed += 1
            else:
                remote_indices.append(i)

        rollout_dp_size = max(1, self._get_dp_size(self.model_contexts[model_id].rollout_wg, "rollout"))
        stats = local_buffer.stats()
        metrics = {
            "star/local_committed": float(committed),
            "star/local_reward_in": float(len(traj_ids)),
            "star/local_buffer_total": float(stats["buffer/total"]),
            "star/local_buffer_ready": float(stats["buffer/ready"]),
            "star/local_buffer_dropped_queries": float(stats["buffer/dropped_queries"]),
            # Preserve the old worker-reduced scale for unprefixed buffer metrics.
            "star/committed": float(committed / rollout_dp_size),
            "buffer/total": float(stats["buffer/total"] / rollout_dp_size),
            "buffer/ready": float(stats["buffer/ready"] / rollout_dp_size),
            "buffer/dropped_queries": float(stats["buffer/dropped_queries"] / rollout_dp_size),
        }
        return metrics, remote_indices

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        return isinstance(exc, (TimeoutError, asyncio.TimeoutError, GetTimeoutError))

    def _commit_rewards_safe(self, rewards: DataProto, stage: str, step: int) -> tuple[dict[str, float], bool]:
        try:
            return self._commit_rewards(rewards), True
        except Exception as exc:
            timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
            print(
                f"[star] commit_rewards failed: stage={stage} step={step} "
                f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
            )
            return {
                f"{stage}/commit_failed": 1.0,
                f"{stage}/commit_failed_timeout": timeout_flag,
            }, False

    def _drain_rollout_ready_queues_safe(self, stage: str, step: int) -> dict[str, float]:
        try:
            self._drain_rollout_ready_queues()
            return {}
        except Exception as exc:
            timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
            print(
                f"[star] drain_rollout_ready_queues failed: stage={stage} step={step} "
                f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
            )
            return {
                f"{stage}/drain_failed": 1.0,
                f"{stage}/drain_failed_timeout": timeout_flag,
            }

    def _run_model_ppo_update_safe(self, model_id: str, ctx: ModelWorkerContext, batch: DataProto, global_step: int):
        try:
            return self._run_model_ppo_update(model_id, ctx, batch, global_step)
        except Exception as exc:
            timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
            print(
                f"[star] model ppo update failed: model={model_id} step={global_step} "
                f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
            )
            return {
                f"model/{model_id}/star/update_failed": 1.0,
                f"model/{model_id}/star/update_failed_timeout": timeout_flag,
                f"model/{model_id}/star/consumed": 0.0,
            }

    def _collect_workflow_dropped_query_ids(self) -> list[str]:
        getter = getattr(self.workflow_runner, "pop_dropped_query_ids", None)
        if not callable(getter):
            return []
        try:
            query_ids = getter()
        except Exception as exc:
            print(f"[star] failed to collect dropped query ids from workflow runner: {exc}")
            return []
        if not isinstance(query_ids, list | tuple):
            return []
        return [str(q).strip() for q in query_ids if str(q).strip()]

    def _drop_queries_from_rollout_buffers(self, query_ids: list[str]) -> dict[str, float]:
        if not query_ids:
            return {}
        unique_ids = list(dict.fromkeys(query_ids))
        metrics: dict[str, float] = {
            "workflow/query_dropped_cleanup": float(len(unique_ids)),
        }
        for model_id, ctx in self.model_contexts.items():
            local_buffer = self._local_traj_buffers_by_model.get(model_id, None)
            if local_buffer is not None:
                local_purged = local_buffer.mark_queries_dropped(unique_ids)
                local_stats = local_buffer.stats()
                metrics[f"model/{model_id}/star/local_dropped_queries"] = float(len(unique_ids))
                metrics[f"model/{model_id}/star/local_purged_traj"] = float(local_purged)
                metrics[f"model/{model_id}/star/local_buffer_total"] = float(local_stats["buffer/total"])
                metrics[f"model/{model_id}/star/local_buffer_ready"] = float(local_stats["buffer/ready"])
                metrics[f"model/{model_id}/star/local_buffer_dropped_queries"] = float(
                    local_stats["buffer/dropped_queries"]
                )
            try:
                outputs = ctx.rollout_wg.drop_queries(unique_ids)
                reduced = self._reduce_worker_metrics(outputs)
                for key, val in reduced.items():
                    metrics[f"model/{model_id}/{key}"] = float(val)
            except Exception as exc:
                timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
                print(
                    f"[star] drop_queries cleanup failed: model={model_id} "
                    f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
                )
                metrics[f"model/{model_id}/star/drop_cleanup_failed"] = 1.0
                metrics[f"model/{model_id}/star/drop_cleanup_failed_timeout"] = timeout_flag
        return metrics

    async def _run_workflow_batch(self, batch: DataProto, epoch: int, stage: str) -> tuple[DataProto, dict[str, float]]:
        timeout_s = float(self._workflow_batch_timeout_seconds)
        try:
            if timeout_s > 0:
                rewards, workflow_metrics = await asyncio.wait_for(
                    self.workflow_runner.run_batch(batch, epoch, stage=stage),
                    timeout=timeout_s,
                )
            else:
                rewards, workflow_metrics = await self.workflow_runner.run_batch(batch, epoch, stage=stage)
        except asyncio.TimeoutError:
            query_ids = []
            if "query_id" in batch.non_tensor_batch:
                query_ids = [str(q).strip() for q in batch.non_tensor_batch["query_id"] if str(q).strip()]
            print(
                f"[star] workflow batch timeout: stage={stage} timeout_s={timeout_s} "
                f"batch_size={len(batch)} dropped_queries={len(query_ids)}"
            )
            cleanup_metrics = self._drop_queries_from_rollout_buffers(query_ids)
            timeout_metrics: dict[str, float] = {
                "workflow/query_dropped": float(len(query_ids)),
                "workflow/query_drop_ratio": float(len(query_ids) / max(1, len(batch))),
                "workflow/query_drop/workflow_batch_timeout": float(len(query_ids)),
            }
            timeout_metrics.update(cleanup_metrics)
            return self._empty_rewards(), timeout_metrics
        except Exception as exc:
            query_ids = []
            if "query_id" in batch.non_tensor_batch:
                query_ids = [str(q).strip() for q in batch.non_tensor_batch["query_id"] if str(q).strip()]
            timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
            print(
                f"[star] workflow batch failed: stage={stage} timeout={bool(timeout_flag)} "
                f"batch_size={len(batch)} dropped_queries={len(query_ids)} "
                f"err={type(exc).__name__}: {exc}"
            )
            cleanup_metrics = self._drop_queries_from_rollout_buffers(query_ids)
            fail_metrics: dict[str, float] = {
                "workflow/query_dropped": float(len(query_ids)),
                "workflow/query_drop_ratio": float(len(query_ids) / max(1, len(batch))),
                "workflow/batch_failed": 1.0,
                "workflow/batch_failed_timeout": timeout_flag,
            }
            fail_metrics.update(cleanup_metrics)
            return self._empty_rewards(), fail_metrics

        dropped_query_ids = self._collect_workflow_dropped_query_ids()
        drop_cleanup_metrics = self._drop_queries_from_rollout_buffers(dropped_query_ids)
        if drop_cleanup_metrics:
            workflow_metrics.update(drop_cleanup_metrics)
        return rewards, workflow_metrics

    @staticmethod
    def _reduce_worker_metrics(worker_outputs) -> dict[str, float]:
        if isinstance(worker_outputs, dict):
            return {k: float(v) for k, v in worker_outputs.items() if isinstance(v, int | float)}
        if not isinstance(worker_outputs, list) or len(worker_outputs) == 0:
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in worker_outputs:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, int | float):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / max(1, counts[k]) for k in sums}

    def _get_dp_size(self, worker_group, role: str) -> int:
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    @staticmethod
    def _empty_batch() -> DataProto:
        return DataProto.from_dict(non_tensors={"traj_id": np.array([], dtype=object)})

    @staticmethod
    def _pad_fill_value_for_key(key: str, tensor: torch.Tensor):
        key_lower = str(key).lower()
        if tensor.dtype == torch.bool:
            return False
        if "label" in key_lower:
            return -100
        if tensor.is_floating_point():
            return 0.0
        return 0

    @classmethod
    def _pad_tensor_to_shape(cls, key: str, tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        fill_value = cls._pad_fill_value_for_key(key, tensor)
        padded = torch.full(target_shape, fill_value=fill_value, dtype=tensor.dtype, device=tensor.device)
        copy_slices = tuple(slice(0, min(src, dst)) for src, dst in zip(tensor.shape, target_shape))
        padded[copy_slices] = tensor[copy_slices]
        return padded

    def _align_fat_batch_shapes_for_concat(self, fat_list: list[DataProto]) -> list[DataProto]:
        if len(fat_list) <= 1:
            return fat_list

        target_shapes: dict[str, tuple[int, ...]] = {}
        prototypes: dict[str, torch.Tensor] = {}
        for fat in fat_list:
            if fat.batch is None:
                continue
            for key in fat.batch.keys():
                tensor = fat.batch[key]
                if not isinstance(tensor, torch.Tensor):
                    continue
                key = str(key)
                shape = tuple(int(x) for x in tensor.shape)
                if key not in target_shapes:
                    target_shapes[key] = shape
                    prototypes[key] = tensor
                    continue
                prev = target_shapes[key]
                if len(prev) != len(shape):
                    raise RuntimeError(
                        f"Inconsistent tensor rank for key={key}: shape={shape} vs prev_shape={prev}"
                    )
                target_shapes[key] = tuple(max(a, b) for a, b in zip(prev, shape))

        if not target_shapes:
            return fat_list

        aligned: list[DataProto] = []
        for fat in fat_list:
            if fat.batch is None:
                aligned.append(fat)
                continue

            tensors: dict[str, torch.Tensor] = {}
            changed = False
            bsz = int(fat.batch.batch_size[0])

            for key, target_shape in target_shapes.items():
                if key in fat.batch.keys():
                    tensor = fat.batch[key]
                    if not isinstance(tensor, torch.Tensor):
                        continue
                else:
                    proto = prototypes[key]
                    cur_shape = list(target_shape)
                    cur_shape[0] = bsz
                    tensors[key] = torch.full(
                        tuple(cur_shape),
                        fill_value=self._pad_fill_value_for_key(key, proto),
                        dtype=proto.dtype,
                        device=proto.device,
                    )
                    changed = True
                    continue

                cur_target = list(target_shape)
                cur_target[0] = int(tensor.shape[0])
                cur_target_t = tuple(cur_target)
                if tuple(int(x) for x in tensor.shape) != cur_target_t:
                    tensors[key] = self._pad_tensor_to_shape(key, tensor, cur_target_t)
                    changed = True
                else:
                    tensors[key] = tensor

            if changed:
                aligned.append(
                    DataProto.from_dict(
                        tensors=tensors,
                        non_tensors=fat.non_tensor_batch,
                        meta_info=self._strip_concat_volatile_meta(fat).meta_info,
                    )
                )
            else:
                aligned.append(self._strip_concat_volatile_meta(fat))

        return aligned

    @staticmethod
    def _summarize_fat_shapes(fat_list: list[DataProto], max_items: int = 8) -> str:
        items = []
        for i, fat in enumerate(fat_list[:max_items]):
            if fat.batch is None:
                items.append(f"{i}:<none>")
                continue
            key_shapes = []
            for key in sorted(fat.batch.keys()):
                tensor = fat.batch[key]
                if isinstance(tensor, torch.Tensor):
                    key_shapes.append(f"{key}:{tuple(int(x) for x in tensor.shape)}")
            items.append(f"{i}:{{{', '.join(key_shapes[:6])}}}")
        suffix = f", ...+{len(fat_list) - max_items}" if len(fat_list) > max_items else ""
        return "[" + "; ".join(items) + suffix + "]"

    def _build_ready_train_batch_from_local_buffer(self, model_id: str, max_items: int = 0) -> DataProto:
        local_buffer = self._local_traj_buffers_by_model.get(model_id, None)
        if local_buffer is None:
            return self._empty_batch()
        entries = local_buffer.pop_ready(
            max_items=max_items if max_items and max_items > 0 else None,
            shuffle=self._shuffle_ready_buffer,
        )
        if len(entries) == 0:
            return self._empty_batch()

        fat_list = []
        for entry in entries:
            fat = entry.fat_data
            if fat.meta_info is None:
                fat.meta_info = {}
            else:
                fat.meta_info.pop("timing", None)
                fat.meta_info.pop("metrics", None)
            fat_list.append(fat)
        fat_list = self._align_fat_batch_shapes_for_concat(fat_list)
        try:
            batch = self._concat_data_proto_safe(fat_list)
        except RuntimeError as exc:
            shape_summary = self._summarize_fat_shapes(fat_list)
            raise RuntimeError(f"Failed to concat local ready fat batches. shapes={shape_summary}") from exc
        batch.meta_info = {}

        response_mask = batch.batch.get("response_mask", None)
        responses = batch.batch.get("responses", None)
        if responses is None:
            return batch

        bsz, resp_len = responses.shape[0], responses.shape[1]
        token_level_scores = torch.zeros((bsz, resp_len), dtype=torch.float32)

        reward_scalar = torch.tensor([float(entry.reward.item()) if entry.reward is not None else 0.0 for entry in entries])
        if response_mask is None:
            token_level_scores[:, -1] = reward_scalar
        else:
            last_pos = response_mask.to(torch.long).sum(dim=-1) - 1
            last_pos = torch.clamp(last_pos, min=0)
            token_level_scores[torch.arange(bsz), last_pos] = reward_scalar

        extra = DataProto.from_dict(
            tensors={
                "token_level_scores": token_level_scores,
                "token_level_rewards": token_level_scores.clone(),
                "reward": reward_scalar,
                "done": torch.tensor([entry.done for entry in entries], dtype=torch.bool),
            },
            non_tensors={
                "traj_id": np.array([entry.traj_id for entry in entries], dtype=object),
                "query_id": np.array([entry.query_id for entry in entries], dtype=object),
                "agent_id": np.array([entry.agent_id for entry in entries], dtype=object),
                "model_id": np.array([entry.model_id for entry in entries], dtype=object),
            },
        )
        return batch.union(extra)

    def _merge_ready_batches(self, ready_parts: list[DataProto]) -> Optional[DataProto]:
        valid = [x for x in ready_parts if isinstance(x, DataProto) and len(x) > 0]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        aligned = self._align_fat_batch_shapes_for_concat(valid)
        try:
            return self._concat_data_proto_safe(aligned)
        except RuntimeError as exc:
            shape_summary = self._summarize_fat_shapes(aligned)
            raise RuntimeError(f"Failed to concat ready batches. shapes={shape_summary}") from exc

    def _shuffle_ready_batch(self, batch: DataProto) -> DataProto:
        if not self._shuffle_ready_buffer or len(batch) <= 1:
            return batch
        return batch.select_idxs(np.random.permutation(len(batch)).tolist())

    def _maybe_drop_last(self, batch: DataProto, dp_size: int) -> tuple[DataProto, int]:
        enforce_divisible_batch = bool(self.config.star.train.get("enforce_divisible_batch", True))
        if dp_size <= 1:
            return batch, 0
        if not enforce_divisible_batch and not self.config.star.train.drop_last:
            return batch, 0
        bsz = len(batch)
        keep = (bsz // dp_size) * dp_size
        if keep <= 0:
            return batch.select_idxs([]), bsz
        if keep == bsz:
            return batch, 0
        indices = np.random.permutation(bsz)[:keep].tolist()
        return batch.select_idxs(indices), bsz - keep

    def _run_model_ppo_update(self, model_id: str, ctx: ModelWorkerContext, batch: DataProto, global_step: int):
        metrics: dict[str, float] = {}
        if len(batch) == 0:
            return metrics

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        old_log_prob = ctx.actor_wg.compute_log_prob(batch)
        if "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        if self.use_reference_policy and ctx.ref_policy_wg is not None:
            ref_log_prob = ctx.ref_policy_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        if self.use_critic and ctx.critic_wg is not None:
            values = ctx.critic_wg.compute_values(batch)
            batch = batch.union(values)

        if self.config.algorithm.use_kl_in_reward and "ref_log_prob" in batch.batch.keys():
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl_by_model[model_id],
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            for key, val in kl_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)
        else:
            if "token_level_rewards" not in batch.batch and "token_level_scores" in batch.batch:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        # PATCH (SynCo): GRPO advantage requires `uid` to group
        # samples that share a baseline. star_ppo's local-pair path only
        # populates `query_id`; mirror it into `uid` so the upstream
        # compute_grpo_outcome_advantage path works without modifying
        # the upstream verl trainer. See SYNCO_MODIFICATIONS.md.
        #
        # Grouping differs by agent:
        #  - Reasoner: K rollouts share one query_id -> group by query_id
        #    (K samples per group, standard GRPO).
        #  - Synthesizer: ONE record per query, so query_id grouping gives
        #    singleton groups -> GRPO degenerates (mean=0, no baseline). Use a
        #    single batch-level group so the N synthesizer samples in this
        #    update form one GRPO group (batch-mean baseline). Toggle via env
        #    STAR_SYNTH_BATCH_LEVEL_GRPO (default on).
        if "uid" not in batch.non_tensor_batch and "query_id" in batch.non_tensor_batch:
            synth_batch_level = (
                os.environ.get("STAR_SYNTH_BATCH_LEVEL_GRPO", "1") == "1"
            )
            agent_ids = batch.non_tensor_batch.get("agent_id", None)
            is_synth_batch = (
                synth_batch_level
                and agent_ids is not None
                and len(agent_ids) > 0
                and all(str(a) == "synthesizer_agent" for a in agent_ids)
            )
            if is_synth_batch:
                bsz = len(batch.non_tensor_batch["query_id"])
                batch.non_tensor_batch["uid"] = np.array(
                    [f"synth_batch_{global_step}"] * bsz, dtype=object
                )
            else:
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["query_id"]

        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        if "agent_id" in batch.non_tensor_batch and "reward" in batch.batch.keys():
            agent_ids = batch.non_tensor_batch["agent_id"]
            reward_vec = batch.batch["reward"].detach().cpu().numpy().reshape(-1)
            adv_vec = batch.batch["advantages"].detach().cpu().float().mean(dim=-1).numpy()
            for agent_id in np.unique(agent_ids):
                mask = agent_ids == agent_id
                if mask.sum() == 0:
                    continue
                metrics[f"model/{model_id}/agent/{agent_id}/reward_mean"] = float(np.mean(reward_vec[mask]))
                metrics[f"model/{model_id}/agent/{agent_id}/adv_mean"] = float(np.mean(adv_vec[mask]))
                metrics[f"model/{model_id}/agent/{agent_id}/samples"] = float(np.sum(mask))

        if self.use_critic and ctx.critic_wg is not None:
            critic_output = ctx.critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info.get("metrics", {}))
            for key, val in critic_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)

        if self.config.trainer.critic_warmup <= global_step:
            rollout_cfg = self.config.actor_rollout_ref.rollout
            batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
            batch.meta_info["temperature"] = rollout_cfg.temperature
            actor_output = ctx.actor_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_output.meta_info.get("metrics", {}))
            for key, val in actor_metrics.items():
                metrics[f"model/{model_id}/{key}"] = float(val)
            self._sync_rollout_weights(model_id, ctx)

        metrics[f"model/{model_id}/star/consumed"] = float(len(batch))
        return metrics

    async def _global_sync_and_update(self) -> dict[str, float]:
        metrics = {}
        max_ready_items = int(self.config.star.train.get("max_ready_items", 0))
        update_jobs: list[tuple[str, ModelWorkerContext, DataProto]] = []

        for model_id, ctx in self.model_contexts.items():
            try:
                ready_parts = ctx.rollout_wg.build_ready_train_batch(max_items=max_ready_items)
                remote_ready_batch = self._merge_ready_batches(ready_parts if isinstance(ready_parts, list) else [ready_parts])
                remote_ready_count = len(remote_ready_batch) if isinstance(remote_ready_batch, DataProto) else 0
                local_max_items = (
                    max(0, int(max_ready_items) - int(remote_ready_count))
                    if max_ready_items and max_ready_items > 0
                    else 0
                )
                local_ready = self._build_ready_train_batch_from_local_buffer(model_id, max_items=local_max_items)
                ready_batch = self._merge_ready_batches([remote_ready_batch, local_ready])
                metrics[f"model/{model_id}/star/remote_ready_consumed"] = float(remote_ready_count)
                metrics[f"model/{model_id}/star/local_ready_consumed"] = float(len(local_ready))
                if ready_batch is None:
                    metrics[f"model/{model_id}/star/consumed"] = 0.0
                    metrics[f"model/{model_id}/star/dropped"] = 0.0
                    continue

                ready_batch = self._shuffle_ready_batch(ready_batch)
                actor_dp_size = self._get_dp_size(ctx.actor_wg, "actor")
                metrics[f"model/{model_id}/star/drop_divisor"] = float(actor_dp_size)
                metrics[f"model/{model_id}/star/buffer_shuffle_ready"] = float(self._shuffle_ready_buffer)
                ready_batch, dropped = self._maybe_drop_last(ready_batch, actor_dp_size)
                metrics[f"model/{model_id}/star/dropped"] = float(dropped)
            except Exception as exc:
                timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
                print(
                    f"[star] build_ready_train_batch failed: model={model_id} step={self._global_step} "
                    f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
                )
                metrics[f"model/{model_id}/star/build_ready_failed"] = 1.0
                metrics[f"model/{model_id}/star/build_ready_failed_timeout"] = timeout_flag
                metrics[f"model/{model_id}/star/consumed"] = 0.0
                continue

            update_jobs.append((model_id, ctx, ready_batch))

        if update_jobs:
            # Different models use disjoint worker groups/resource pools, so they can
            # update in parallel instead of serial model-by-model execution.
            ppo_results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        self._run_model_ppo_update_safe,
                        model_id,
                        ctx,
                        ready_batch,
                        self._global_step,
                    )
                    for model_id, ctx, ready_batch in update_jobs
                ]
            )
            for ppo_metrics in ppo_results:
                metrics.update(ppo_metrics)

        return metrics

    def _get_checkpoint_root(self) -> str:
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
        return checkpoint_root

    def _save_checkpoint(self, step: int):
        save_start = time.time()
        checkpoint_root = self._get_checkpoint_root()
        global_step_folder = os.path.join(checkpoint_root, f"global_step_{step}")
        os.makedirs(global_step_folder, exist_ok=True)

        # Checkpointing an 8B multi-agent run can legitimately take longer than
        # normal worker RPCs.  If we reuse STAR_WORKER_CALL_TIMEOUT_SECONDS here,
        # the driver may time out, skip the tracker write, and leave an otherwise
        # usable checkpoint invisible to auto-resume.  Keep a dedicated checkpoint
        # timeout; 0 means wait until the worker-side save completes.
        checkpoint_timeout = str(
            os.environ.get("STAR_CHECKPOINT_TIMEOUT_SECONDS", os.environ.get("STAR_CKPT_TIMEOUT_SECONDS", "0"))
        )
        old_worker_call_timeout = os.environ.get("STAR_WORKER_CALL_TIMEOUT_SECONDS")
        os.environ["STAR_WORKER_CALL_TIMEOUT_SECONDS"] = checkpoint_timeout

        try:
            print(
                f"[star] checkpoint save start: step={step} "
                f"path={global_step_folder} worker_call_timeout={checkpoint_timeout}"
            )

            remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
            max_actor_ckpt_to_keep = (
                self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
            )
            max_critic_ckpt_to_keep = (
                self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
            )

            saved_models = []
            for model_id, ctx in self.model_contexts.items():
                model_folder = os.path.join(global_step_folder, model_id)
                actor_local_path = os.path.join(model_folder, "actor")
                actor_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{step}", model_id, "actor")
                )
                print(f"[star] checkpoint save actor: step={step} model={model_id} path={actor_local_path}")
                ctx.actor_wg.save_checkpoint(
                    actor_local_path,
                    actor_remote_path,
                    step,
                    max_ckpt_to_keep=max_actor_ckpt_to_keep,
                )

                model_meta = {"model_id": model_id, "actor_path": actor_local_path}
                if self.use_critic and ctx.critic_wg is not None:
                    critic_local_path = os.path.join(model_folder, str(Role.Critic))
                    critic_remote_path = (
                        None
                        if self.config.trainer.default_hdfs_dir is None
                        else os.path.join(
                            self.config.trainer.default_hdfs_dir, f"global_step_{step}", model_id, str(Role.Critic)
                        )
                    )
                    print(f"[star] checkpoint save critic: step={step} model={model_id} path={critic_local_path}")
                    ctx.critic_wg.save_checkpoint(
                        critic_local_path,
                        critic_remote_path,
                        step,
                        max_ckpt_to_keep=max_critic_ckpt_to_keep,
                    )
                    model_meta["critic_path"] = critic_local_path
                saved_models.append(model_meta)

            # save train dataloader cursor/sampler state for strict resume
            dataloader_local_path = os.path.join(global_step_folder, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

            duration_s = float(time.time() - save_start)
            star_meta = {
                "global_step": step,
                "models": sorted(self.model_contexts.keys()),
                "saved_models": saved_models,
                "completed": True,
                "duration_s": duration_s,
                "worker_call_timeout_seconds": float(checkpoint_timeout),
            }
            with open(os.path.join(global_step_folder, "star_meta.json"), "w", encoding="utf-8") as f:
                json.dump(star_meta, f, ensure_ascii=False, indent=2)
            with open(os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt"), "w", encoding="utf-8") as f:
                f.write(str(step))

            print(
                f"[star] checkpoint save done: step={step} "
                f"path={global_step_folder} duration_s={duration_s:.1f}"
            )
        finally:
            if old_worker_call_timeout is None:
                os.environ.pop("STAR_WORKER_CALL_TIMEOUT_SECONDS", None)
            else:
                os.environ["STAR_WORKER_CALL_TIMEOUT_SECONDS"] = old_worker_call_timeout

    def _load_checkpoint(self) -> int:
        resume_mode = self.config.trainer.resume_mode
        if resume_mode == "disable":
            self.global_steps = 0
            return 0

        if resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            checkpoint_root = self._get_checkpoint_root()
            global_step_folder = find_latest_ckpt_path(checkpoint_root)
            if global_step_folder is None:
                self.global_steps = 0
                return 0

        if global_step_folder is None or not os.path.exists(global_step_folder):
            self.global_steps = 0
            return 0

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        for model_id, ctx in self.model_contexts.items():
            model_folder = os.path.join(global_step_folder, model_id)
            actor_path = os.path.join(model_folder, "actor")
            critic_path = os.path.join(model_folder, str(Role.Critic))
            ctx.actor_wg.load_checkpoint(
                actor_path if os.path.exists(actor_path) else None,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
            if self.use_critic and ctx.critic_wg is not None:
                ctx.critic_wg.load_checkpoint(
                    critic_path if os.path.exists(critic_path) else None,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )

            self._sync_rollout_weights(model_id, ctx)

        # restore train dataloader cursor/sampler state if present
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            self._train_loader_state_loaded = True
        else:
            self._train_loader_state_loaded = False
            print(f"[star] no dataloader state at {dataloader_local_path}, resume from sampler start")
        return self.global_steps

    def _drain_rollout_ready_queues(self):
        # Validation also uses thin->commit flow, so drain ready queue to avoid
        # mixing validation trajectories into subsequent training updates.
        for model_id, ctx in self.model_contexts.items():
            _ = self._build_ready_train_batch_from_local_buffer(model_id, max_items=0)
            _ = ctx.rollout_wg.build_ready_train_batch(max_items=0)

    @staticmethod
    def _pick_metric(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        if isinstance(value, int | float):
            return float(value)
        return float(default)

    def _pick_progress_reward(self, metrics: dict[str, float]) -> float:
        # Graph workflows report outcome reward; trace workflows report the summed
        # assigned reward per trace. Fall back to agent means for older configs.
        for key in (
            "workflow/outcome_reward_mean",
            "workflow/trace_reward_sum",
            "agent/answer_agent/reward_mean",
            "agent/search_agent/reward_mean",
            "agent/summary_agent/reward_mean",
            "agent/update_agent/reward_mean",
        ):
            value = metrics.get(key, None)
            if isinstance(value, int | float):
                return float(value)
        return 0.0

    @staticmethod
    def _is_fine_grained_timing_key(metric_key: str) -> bool:
        key = str(metric_key)
        return ("/timing/node/" in key) or ("/timing/group/" in key)

    def _log_metrics(self, logger: Tracking, data: dict[str, float], step: int) -> None:
        # Keep full metrics for local backends (console/file), but trim very
        # high-cardinality timing keys from WandB payloads.
        if not self._wandb_filter_fine_timing:
            logger.log(data=data, step=step)
            return

        wandb_backends = [b for b in ("wandb", "vemlp_wandb") if b in logger.logger]
        other_backends = [b for b in logger.logger.keys() if b not in {"wandb", "vemlp_wandb"}]

        if other_backends:
            logger.log(data=data, step=step, backend=other_backends)

        if wandb_backends:
            filtered = {k: v for k, v in data.items() if not self._is_fine_grained_timing_key(k)}
            if filtered:
                logger.log(data=filtered, step=step, backend=wandb_backends)

    def _print_batch_timing(
        self,
        stage: str,
        batch_idx: int,
        batch_size: int,
        metrics: dict[str, float],
        extra_timings: dict[str, float] | None = None,
    ):
        if not self._timing_print_enabled:
            return
        if batch_idx % self._timing_print_every_n_batches != 0:
            return

        workflow_total = self._pick_metric(metrics, "workflow/timing/batch_total_s")
        query_mean = self._pick_metric(metrics, "workflow/timing/query_s_mean")
        llm_mean = self._pick_metric(metrics, "workflow/timing/llm_node_s_mean")
        llm_queue_mean = self._pick_metric(metrics, "workflow/timing/llm_queue_wait_s_mean")
        llm_exec_mean = self._pick_metric(metrics, "workflow/timing/llm_rollout_exec_s_mean")
        llm_rpc_overhead_mean = self._pick_metric(metrics, "workflow/timing/llm_rpc_overhead_s_mean")
        llm_engine_mean = self._pick_metric(metrics, "workflow/timing/llm_engine_generate_s_mean")
        llm_server_mean = self._pick_metric(metrics, "workflow/timing/llm_agent_server_total_s_mean")
        llm_worker_start_mean = self._pick_metric(metrics, "workflow/timing/llm_agent_worker_start_lag_s_mean")
        llm_mgr_mean = self._pick_metric(metrics, "workflow/timing/llm_agent_loop_manager_total_s_mean")
        tool_mean = self._pick_metric(metrics, "workflow/timing/tool_node_s_mean")
        node_calls = int(self._pick_metric(metrics, "workflow/timing/node_invocations"))
        group_pairs: list[tuple[str, float]] = []
        group_prefix = "workflow/timing/group/"
        group_suffix = "_s_mean"
        for key, value in metrics.items():
            if not isinstance(value, int | float):
                continue
            if not key.startswith(group_prefix) or not key.endswith(group_suffix):
                continue
            group_name = key[len(group_prefix) : -len(group_suffix)]
            if not group_name:
                continue
            if f"{group_prefix}{group_name}_count" not in metrics:
                continue
            group_pairs.append((group_name, float(value)))
        group_pairs.sort(key=lambda x: x[1], reverse=True)
        group_pairs = group_pairs[: self._timing_group_topk]
        group_text = ""
        if group_pairs:
            group_text = " groups=" + ",".join([f"{name}:{val:.3f}s" for name, val in group_pairs])

        extra_parts = []
        if extra_timings:
            for key in ("workflow_wall_s", "commit_s", "drain_s", "sync_update_s", "batch_elapsed_s"):
                if key in extra_timings:
                    extra_parts.append(f"{key}={float(extra_timings[key]):.3f}s")
        extra_text = (" " + " ".join(extra_parts)) if extra_parts else ""
        print(
            "[star-timing] "
            f"stage={stage} batch={batch_idx} size={batch_size} "
            f"workflow={workflow_total:.3f}s query_mean={query_mean:.3f}s "
            f"llm_mean={llm_mean:.3f}s llm_queue_mean={llm_queue_mean:.3f}s "
            f"llm_exec_mean={llm_exec_mean:.3f}s llm_rpc_ovh_mean={llm_rpc_overhead_mean:.3f}s "
            f"llm_engine_mean={llm_engine_mean:.3f}s llm_server_mean={llm_server_mean:.3f}s "
            f"llm_worker_start_mean={llm_worker_start_mean:.3f}s llm_mgr_mean={llm_mgr_mean:.3f}s "
            f"tool_mean={tool_mean:.3f}s "
            f"node_calls={node_calls}{group_text}{extra_text}"
        )

    async def _run_validation(self, epoch: int, global_step: int) -> dict[str, float]:
        max_batches = int(self.config.trainer.get("val_max_batches", -1))
        val_progress_every = int(os.environ.get("STAR_VAL_PROGRESS_EVERY", "0"))
        tqdm_disable = str(os.environ.get("STAR_TQDM_DISABLE", "false")).strip().lower() in {"1", "true", "yes", "on"}
        batch_count = 0
        reward_sum = 0.0
        reward_count = 0
        workflow_acc: dict[str, list[float]] = defaultdict(list)

        if val_progress_every > 0:
            print(
                f"[star] validation start epoch={epoch} global_step={global_step} "
                f"max_batches={max_batches}"
            )

        total_val_batches = len(self.val_dataloader)
        if max_batches > 0:
            total_val_batches = min(total_val_batches, max_batches)
        val_iter = tqdm(
            enumerate(self.val_dataloader),
            total=total_val_batches,
            desc=f"[star-val] e{epoch} gs{global_step}",
            leave=True,
            dynamic_ncols=True,
            disable=tqdm_disable,
        )
        try:
            for batch_idx, batch_dict in val_iter:
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                self._mark_progress(stage=f"val_batch_start_{batch_idx}", step=global_step)
                batch_start = time.time()
                batch_count += 1
                batch = DataProto.from_single_dict(batch_dict)
                if val_progress_every > 0 and (batch_idx % val_progress_every == 0):
                    print(
                        f"[star] validation batch_start idx={batch_idx} size={len(batch)} "
                        f"inflight={self.config.star.workflow.get('max_inflight_queries', 32)}"
                    )
                self._ensure_routing_fields(batch)
                workflow_t0 = time.time()
                rewards, workflow_metrics = await self._run_workflow_batch(batch, epoch, stage="validation")
                workflow_wall_s = time.time() - workflow_t0
                workflow_metrics["workflow/timing/validation_workflow_wall_s"] = float(workflow_wall_s)
                commit_s = 0.0
                drain_s = 0.0
                if val_progress_every > 0 and (batch_idx % val_progress_every == 0):
                    print(
                        f"[star] validation batch_done idx={batch_idx} "
                        f"elapsed={time.time() - batch_start:.2f}s reward_samples={len(rewards)}"
                    )

                if len(rewards) > 0:
                    reward_vec = rewards.batch["reward"].detach().cpu().float().reshape(-1).numpy()
                    reward_sum += float(np.sum(reward_vec))
                    reward_count += int(reward_vec.shape[0])
                    # Commit to local buffers so worker-side trajectory states are consistent.
                    commit_t0 = time.time()
                    commit_metrics, commit_ok = self._commit_rewards_safe(
                        rewards,
                        stage="validation",
                        step=global_step,
                    )
                    commit_s = time.time() - commit_t0
                    for key, val in commit_metrics.items():
                        workflow_metrics[f"workflow/{key}"] = float(val)
                    if commit_ok:
                        drain_t0 = time.time()
                        drain_metrics = self._drain_rollout_ready_queues_safe(
                            stage="validation",
                            step=global_step,
                        )
                        drain_s = time.time() - drain_t0
                        for key, val in drain_metrics.items():
                            workflow_metrics[f"workflow/{key}"] = float(val)
                workflow_metrics["workflow/timing/validation_commit_s"] = float(commit_s)
                workflow_metrics["workflow/timing/validation_drain_s"] = float(drain_s)

                for key, val in workflow_metrics.items():
                    if isinstance(val, int | float):
                        workflow_acc[key].append(float(val))

                self._print_batch_timing(
                    stage="val",
                    batch_idx=batch_idx,
                    batch_size=len(batch),
                    metrics=workflow_metrics,
                    extra_timings={
                        "workflow_wall_s": workflow_wall_s,
                        "commit_s": commit_s,
                        "drain_s": drain_s,
                        "batch_elapsed_s": float(time.time() - batch_start),
                    },
                )

                val_iter.set_postfix(
                    {
                        "inflight": int(self.config.star.workflow.get("max_inflight_queries", 32)),
                        "samples": int(reward_count),
                        "rmean": float(reward_sum / max(1, reward_count)),
                    },
                    refresh=False,
                )
                self._mark_progress(stage=f"val_batch_done_{batch_idx}", step=global_step)
        finally:
            val_iter.close()

        metrics: dict[str, float] = {
            "validation/global_step": float(global_step),
            "validation/epoch": float(epoch),
            "validation/batches": float(batch_count),
            "validation/samples": float(reward_count),
            "validation/reward_mean": float(reward_sum / max(1, reward_count)),
        }
        for key, values in workflow_acc.items():
            if values:
                metrics[f"validation/{key}"] = float(np.mean(values))
        if val_progress_every > 0:
            print(f"[star] validation end metrics={metrics}")
        return metrics

    async def _run_validation_safe(self, epoch: int, global_step: int) -> dict[str, float]:
        try:
            return await self._run_validation(epoch=epoch, global_step=global_step)
        except Exception as exc:
            timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
            print(
                f"[star] validation failed: epoch={epoch} step={global_step} "
                f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
            )
            return {
                "validation/global_step": float(global_step),
                "validation/epoch": float(epoch),
                "validation/failed": 1.0,
                "validation/failed_timeout": timeout_flag,
            }

    async def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        tqdm_disable = str(os.environ.get("STAR_TQDM_DISABLE", "false")).strip().lower() in {"1", "true", "yes", "on"}
        print(
            f"[star] tracking_backends={list(logger.logger.keys())} "
            f"wandb_fine_timing_filter={self._wandb_filter_fine_timing}"
        )
        watchdog_stop = asyncio.Event()
        watchdog_task = None
        if self._stall_detect_seconds > 0:
            watchdog_task = asyncio.create_task(self._stall_watchdog(watchdog_stop))
        old_worker_call_timeout = os.environ.get("STAR_WORKER_CALL_TIMEOUT_SECONDS")
        if self._worker_call_timeout_seconds > 0:
            os.environ["STAR_WORKER_CALL_TIMEOUT_SECONDS"] = str(self._worker_call_timeout_seconds)
        try:
            global_step = self._load_checkpoint()
            self._global_step = global_step
            self._mark_progress(stage="fit_start", step=global_step)
            self._log_metrics(
                logger,
                {
                    "system/run_initialized": 1.0,
                    "system/global_step_at_start": float(global_step),
                },
                global_step,
            )
            start_epoch = global_step // max(1, len(self.train_dataloader))
            val_before_train = bool(self.config.trainer.get("val_before_train", False))
            test_freq = int(self.config.trainer.get("test_freq", -1))

            if val_before_train:
                print(f"[star] pre-train validation start epoch={start_epoch} global_step={global_step}")
                val_metrics = await self._run_validation_safe(epoch=start_epoch, global_step=global_step)
                self._log_metrics(logger, val_metrics, global_step)
                print(f"[star] pre-train validation={val_metrics}")
                self._mark_progress(stage="pre_train_validation_done", step=global_step)

            if bool(self.config.trainer.get("val_only", False)):
                return

            for epoch in range(start_epoch, self.config.trainer.total_epochs):
                empty_reward_streak = 0
                resume_batch_offset = 0
                if self._train_loader_state_loaded and epoch == start_epoch and len(self.train_dataloader) > 0:
                    resume_batch_offset = int(global_step % len(self.train_dataloader))
                train_iter = tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc=f"[star-train] e{epoch}",
                    initial=resume_batch_offset,
                    leave=True,
                    dynamic_ncols=True,
                    disable=tqdm_disable,
                )
                try:
                    for _, batch_dict in train_iter:
                        global_step += 1
                        self._global_step = global_step
                        self._mark_progress(stage="train_step_start", step=global_step)
                        if global_step > self.total_training_steps:
                            break

                        batch = DataProto.from_single_dict(batch_dict)
                        self._ensure_routing_fields(batch)
                        workflow_t0 = time.time()
                        rewards, workflow_metrics = await self._run_workflow_batch(batch, epoch, stage="train")
                        workflow_wall_s = time.time() - workflow_t0
                        workflow_metrics["workflow/timing/train_workflow_wall_s"] = float(workflow_wall_s)
                        self._mark_progress(stage="train_workflow_done", step=global_step)
                        if len(rewards) == 0:
                            empty_reward_streak += 1
                            empty_metrics = {
                                "training/global_step": float(global_step),
                                "training/epoch": float(epoch),
                                "training/timing/workflow_wall_s": float(workflow_wall_s),
                                **workflow_metrics,
                                "workflow/empty_reward_batch": 1.0,
                                "workflow/empty_reward_streak": float(empty_reward_streak),
                            }
                            self._print_batch_timing(
                                stage="train",
                                batch_idx=global_step,
                                batch_size=len(batch),
                                metrics=empty_metrics,
                                extra_timings={
                                    "workflow_wall_s": workflow_wall_s,
                                },
                            )
                            self._log_metrics(logger, empty_metrics, global_step)
                            if global_step % max(1, self.config.trainer.get("log_freq", 1)) == 0:
                                print(f"[star] step={global_step} empty_reward_batch streak={empty_reward_streak}")
                            train_iter.set_postfix({"gstep": int(global_step), "samples": 0}, refresh=False)
                            self._mark_progress(stage="train_empty_batch_done", step=global_step)
                            continue
                        empty_reward_streak = 0

                        commit_t0 = time.time()
                        commit_metrics, commit_ok = self._commit_rewards_safe(
                            rewards,
                            stage="training",
                            step=global_step,
                        )
                        commit_s = time.time() - commit_t0
                        self._mark_progress(stage="train_commit_done", step=global_step)
                        sync_t0 = time.time()
                        if commit_ok:
                            try:
                                sync_metrics = await self._global_sync_and_update()
                            except Exception as exc:
                                timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
                                print(
                                    f"[star] global_sync_and_update failed: step={global_step} "
                                    f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
                                )
                                sync_metrics = {
                                    "training/sync_failed": 1.0,
                                    "training/sync_failed_timeout": timeout_flag,
                                }
                        else:
                            sync_metrics = {
                                "training/sync_skipped_due_to_commit_failed": 1.0,
                            }
                        sync_update_s = time.time() - sync_t0
                        self._mark_progress(stage="train_sync_update_done", step=global_step)

                        step_metrics = {
                            "training/global_step": float(global_step),
                            "training/epoch": float(epoch),
                            "training/timing/workflow_wall_s": float(workflow_wall_s),
                            "training/timing/commit_s": float(commit_s),
                            "training/timing/sync_update_s": float(sync_update_s),
                            **workflow_metrics,
                            **commit_metrics,
                            **sync_metrics,
                        }
                        self._print_batch_timing(
                            stage="train",
                            batch_idx=global_step,
                            batch_size=len(batch),
                            metrics=step_metrics,
                            extra_timings={
                                "workflow_wall_s": workflow_wall_s,
                                "commit_s": commit_s,
                                "sync_update_s": sync_update_s,
                            },
                        )
                        self._log_metrics(logger, step_metrics, global_step)
                        if global_step % max(1, self.config.trainer.get("log_freq", 1)) == 0:
                            print(f"[star] step={global_step} batch_update={step_metrics}")

                        train_iter.set_postfix(
                            {
                                "gstep": int(global_step),
                                "inflight": int(self.config.star.workflow.get("max_inflight_queries", 32)),
                                "samples": int(workflow_metrics.get("workflow/samples", 0)),
                                "reward": self._pick_progress_reward(step_metrics),
                            },
                            refresh=False,
                        )

                        is_last_step = global_step >= self.total_training_steps
                        if test_freq > 0 and (is_last_step or global_step % test_freq == 0):
                            val_metrics = await self._run_validation_safe(epoch=epoch, global_step=global_step)
                            self._log_metrics(logger, val_metrics, global_step)
                            print(f"[star] step={global_step} validation={val_metrics}")
                            self._mark_progress(stage="train_periodic_validation_done", step=global_step)

                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or global_step % self.config.trainer.save_freq == 0
                        ):
                            checkpoint_t0 = time.time()
                            try:
                                self._save_checkpoint(global_step)
                                self._log_metrics(
                                    logger,
                                    {
                                        "training/checkpoint_saved": 1.0,
                                        "training/checkpoint_save_s": float(time.time() - checkpoint_t0),
                                    },
                                    global_step,
                                )
                                self._mark_progress(stage="train_checkpoint_saved", step=global_step)
                            except Exception as exc:
                                timeout_flag = 1.0 if self._is_timeout_error(exc) else 0.0
                                print(
                                    f"[star] checkpoint save failed: step={global_step} "
                                    f"timeout={bool(timeout_flag)} err={type(exc).__name__}: {exc}"
                                )
                                print(traceback.format_exc())
                                self._log_metrics(
                                    logger,
                                    {
                                        "training/checkpoint_failed": 1.0,
                                        "training/checkpoint_failed_timeout": timeout_flag,
                                        "training/checkpoint_save_s": float(time.time() - checkpoint_t0),
                                    },
                                    global_step,
                                )
                finally:
                    train_iter.close()

                if global_step >= self.total_training_steps:
                    break
        finally:
            if old_worker_call_timeout is None:
                os.environ.pop("STAR_WORKER_CALL_TIMEOUT_SECONDS", None)
            else:
                os.environ["STAR_WORKER_CALL_TIMEOUT_SECONDS"] = old_worker_call_timeout
            if watchdog_task is not None:
                watchdog_stop.set()
                await watchdog_task
