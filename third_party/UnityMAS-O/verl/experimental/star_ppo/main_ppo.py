"""Entry point for star-topology multi-engine PPO skeleton."""

import asyncio
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.one_step_off_policy.utils import need_critic
from verl.experimental.star_ppo.ray_trainer import StarRayTrainer
from verl.experimental.star_ppo.types import EngineSpec
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def create_engine_specs(config) -> list[EngineSpec]:
    specs = []
    for engine in config.trainer.llm_engines:
        specs.append(
            EngineSpec(
                model_id=str(engine.model_id),
                nnodes=int(engine.nnodes),
                n_gpus_per_node=int(engine.n_gpus_per_node),
                accelerator_type=engine.get("accelerator_type", None),
                strategy=str(engine.get("strategy", "fsdp2")),
            )
        )
    return specs


def create_role_worker_mapping(config):
    strategy = config.actor_rollout_ref.actor.strategy
    if strategy not in ["fsdp", "fsdp2"]:
        raise NotImplementedError(f"Star PPO skeleton currently supports fsdp/fsdp2 only, got {strategy}")
    if strategy != config.critic.strategy:
        raise ValueError("actor strategy and critic strategy must be consistent")

    from verl.experimental.star_ppo.star_fsdp_workers import (
        CriticWorker,
        RewardModelWorker,
        StarDetachActorWorker,
        StarDetachAsyncRolloutWorker,
    )

    role_worker_mapping = {
        Role.Actor: ray.remote(StarDetachActorWorker),
        Role.Rollout: ray.remote(StarDetachAsyncRolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(StarDetachActorWorker)

    return role_worker_mapping


@ray.remote(num_cpus=10, max_concurrency=100)
class StarTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"StarTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        role_worker_mapping = create_role_worker_mapping(config)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = StarRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            engine_specs=create_engine_specs(config),
            role_worker_mapping=role_worker_mapping,
            reward_fn=reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="star_ppo_trainer", version_base=None)
def main(config):
    from time import time

    start_time = time()
    auto_set_device(config)
    run_ppo(config, task_runner_class=StarTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
