from dataclasses import dataclass


@dataclass
class EngineSpec:
    """Physical deployment spec for one isolated L2 model engine."""

    model_id: str
    nnodes: int
    n_gpus_per_node: int
    accelerator_type: str | None = None
    strategy: str = "fsdp2"
