"""Thin trainer entrypoint that defers to UnityMAS-O ``main_ppo``.

This module exists so users can run::

    python -m synth_reasoner.unitymas_adapter.trainer_entry --config-name star_synth_reasoner_trainer

UnityMAS-O is responsible for the heavy lifting (Hydra parsing, Ray init,
PPO loop). Our adapter ships only the workflow runner, the reward allocator,
and a Hydra YAML that points to them.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin shim
    # Imported lazily so importing this module does not require ``verl``.
    from verl.experimental.star_ppo.main_ppo import main as star_main  # type: ignore[import]

    if argv is not None:
        sys.argv = ["main_ppo"] + list(argv)
    star_main()


if __name__ == "__main__":  # pragma: no cover
    main()
