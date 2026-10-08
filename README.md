<div align="center">

# SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning

### **NeurIPS 2026 Workshop on Towards Test-Time Continual Learning Agents (TTCL)**

[![Conference](https://img.shields.io/badge/NeurIPS%20TTCL-2026-blueviolet?style=for-the-badge)](https://openreview.net/forum?id=0SbD6GdfQZ)
[![Framework](https://img.shields.io/badge/Framework-SynCo-success?style=for-the-badge)](#-overview)
[![Training](https://img.shields.io/badge/Optimization-Multi--Agent%20GRPO-orange?style=for-the-badge)](#-training)
[![Code](https://img.shields.io/badge/Code-Open%20Source-black?style=for-the-badge)](#-environment-setup)

<p align="center">
  <b>SynCo</b>: A multi-agent reinforcement learning framework that jointly trains a <b>Synthesizer</b> and a <b>Reasoner</b>, enabling the agent and its training data to continually co-evolve through online interaction.
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=0SbD6GdfQZ"><b>Paper</b></a> •
  <a href="#-overview"><b>Overview</b></a> •
  <a href="#-method"><b>Method</b></a> •
  <a href="#-dataset"><b>Dataset</b></a> •
  <a href="#-environment-setup"><b>Setup</b></a> •
  <a href="#-training"><b>Training</b></a> •
  <a href="#-evaluation"><b>Evaluation</b></a> •
  <a href="#-citation"><b>Citation</b></a>
</p>

</div>

---

## 📌 Overview

Self-evolving LLM agents are expected to improve continuously through interaction and accumulated experience. However, most existing training pipelines still rely on **static synthetic datasets** or synthesis models that are optimized separately from the downstream learner. As the learner evolves, previously useful tasks may become trivial, while overly difficult tasks may remain uninformative, creating a growing mismatch between the agent and its training distribution.

We introduce **SynCo**, an agentic data synthesis co-training framework that treats **data synthesis itself as a learnable component of agent training**. SynCo jointly optimizes two independently parameterized agents:

- **Synthesizer** — generates structured and verifiable training tasks conditioned on the Reasoner's evolving capability state.
- **Reasoner** — learns from the synthesized tasks through online reinforcement learning.

Rather than alternating between data generation and learner optimization, SynCo places both agents in a **shared online multi-agent reinforcement learning loop**. Each synthesized task induces multiple Reasoner rollouts. These interactions simultaneously provide correctness-based learning signals for the Reasoner and outcome-grounded learning signals for the Synthesizer.

The Synthesizer is optimized with a gated objective that combines **task quality**, **answer reliability**, and **realized teachability**. Teachability is determined by the Reasoner's actual rollout outcomes, favoring tasks near the learner's current capability boundary. As both policies update, the Reasoner's evolving capability reshapes the next synthesis context, while the updated Synthesizer changes the Reasoner's future training distribution. This creates a closed **agent–data self-evolution loop**.

This repository contains the official training and evaluation code, the bundled UnityMAS-O training framework, and the evaluation benchmarks used by SynCo.

---

## 📝 Abstract

> Self-evolving large language model agents require training data that remains informative as their capabilities change. Existing approaches commonly rely on static synthetic datasets or data generators optimized independently from the learner, causing the training distribution to become increasingly mismatched with the learner's current needs. We introduce **SynCo**, a data synthesis co-training framework that jointly optimizes a Synthesizer and a Reasoner through online multi-agent reinforcement learning. The Synthesizer produces structured, verifiable reasoning tasks conditioned on the Reasoner's evolving capability state, while the Reasoner learns from multiple rollouts on these tasks. The resulting outcomes provide correctness-based rewards to the Reasoner and outcome-grounded rewards to the Synthesizer. A gated objective combines task quality, answer reliability, and learner-relative teachability, encouraging the Synthesizer to generate tasks near the Reasoner's capability boundary. By feeding interaction outcomes back into subsequent synthesis rounds, SynCo creates a closed loop in which the agent and its training data continually co-evolve.

---

## ✨ Key Features

- **Joint Synthesizer–Reasoner training** in a shared online reinforcement learning process.
- **Capability-conditioned synthesis** informed by the Reasoner's recent empirical performance.
- **Structured and verifiable task generation** with schema, problem, answer, format, and deduplication checks.
- **Outcome-grounded teachability rewards** based on the Reasoner's realized rollout success rate.
- **Multi-agent GRPO optimization** for both independently parameterized policies.
- **Closed-loop self-evolution** through a continuously refreshed synthetic-task pool and capability report.
- End-to-end scripts for environment setup, distributed training, checkpoint merging, and benchmark evaluation.

---

## 🔄 Method

SynCo consists of three tightly coupled stages:

1. **Capability-Conditioned Data Synthesis**  
   The Synthesizer observes the recent synthetic-data pool, the Reasoner's empirical capability profile, and feedback from rejected tasks, then generates a structured mathematical reasoning task with a verifiable answer.

2. **Interaction-Grounded Multi-Agent Co-Training**  
   The Reasoner performs multiple rollouts on every admitted task. Rollout correctness trains the Reasoner, while task quality, answer reliability, and learner-relative teachability train the Synthesizer. Both agents are optimized online with GRPO in the same training process.

3. **Closed-Loop Self-Evolution**  
   The resulting tasks, verification outcomes, rollouts, and rewards are stored in the live pool and summarized for the next synthesis round, continually updating both the learner state and the training distribution.

At a high level:

```text
Reasoner Capability State
          ↓
    Synthesizer ──→ Synthetic Task ──→ Task Verifier
          ↑                                  ↓
          │                           Reasoner Rollouts
          │                                  ↓
          └────── Outcome-Grounded Feedback ─┘
```

### Training Workflow

Each training step contains 32 task slots:

1. The **Synthesizer** reads the current report and writes one nine-field JSON task card.
2. The **TaskVerifier** validates its schema, problem, answer, format, and novelty.
3. The **Reasoner** solves every valid task `K = 4` times by default.
4. The **AnswerChecker** grades responses using exact, numeric, symbolic, and set/tuple equivalence.
5. Role-specific rewards are assigned and both policies are updated with GRPO.
6. The live pool is updated, and the Synthesizer report is rebuilt every five steps by default.

---

## 📂 Dataset

### Online Synthetic Training Data

SynCo generates its training data online. Every generated task is stored under the current run directory together with verification results, Reasoner rollouts, and rewards:

```text
runs/<RUN_ID>/pool/
└── valid_tasks.jsonl
```

`valid_tasks.jsonl` contains the admitted synthetic tasks and can be used as the synthesized dataset produced by a run.

SynCo starts from an empty task pool by default. At launch, `scripts/train.sh` automatically creates the framework-required query slots inside the current run directory. No pre-generated training-query file is required.

### Evaluation Benchmarks

Evaluation data is not included in this repository. After downloading it, place the normalized benchmark files under `data/eval/`. The default evaluation suite contains:

- GSM8K
- SVAMP
- ASDiv
- GSM-Hard
- MATH-500
- AIME 2024
- AIME 2025
- Minerva Math

To download or rebuild the benchmark files from Hugging Face:

```bash
python scripts/download_eval_data.py
python scripts/fetch_hf_math_benchmarks.py
```

The expected data layout is:

```text
data/
└── eval/
    ├── asdiv/test.jsonl
    ├── aime24/test.jsonl
    ├── aime25/test.jsonl
    ├── gsm8k/test.jsonl
    ├── gsm_hard/test.jsonl
    ├── math500/test.jsonl
    ├── minerva/test.jsonl
    └── svamp/test.jsonl
```

---

## ⚙️ Environment Setup

### Requirements

- Python 3.10
- CUDA 12.x
- Two nodes with eight GPUs each for the default paper training configuration
- One or more GPUs for evaluation

The paper configuration assigns one eight-GPU node to the Synthesizer and one eight-GPU node to the Reasoner. Both policies use Qwen3-8B with FSDP2 and vLLM tensor parallelism.

### Installation

From the repository root, create a virtual environment and install the pinned dependencies:

```bash
VENV=.venv bash scripts/setup_env.sh
```

The setup script installs:

- the pinned Python packages from `requirements-lock.txt`;
- the bundled and patched UnityMAS-O framework under `third_party/UnityMAS-O/`; and
- the SynCo package in editable mode.

The paper environment uses PyTorch 2.8.0, vLLM 0.10.2, Transformers 4.56.2, and Ray 2.55.1.

Download the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) checkpoint and set `BASE_MODEL_PATH` to its local path before training.

---

## 🚀 Training

### Option 1: Launch with a SLURM Worker Node

Use the current machine as the Ray head node and connect a second node through an existing SLURM allocation:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
WORKER_NODE_IP=<worker-node-ip> \
WORKER_SLURM_JOBID=<slurm-job-id> \
bash scripts/train.sh
```

### Option 2: Launch with an SSH Worker Node

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
WORKER_NODE_IP=<worker-node-ip> \
WORKER_SSH=<user@worker-host> \
bash scripts/train.sh
```

### Option 3: Use an Existing Ray Cluster

The Ray cluster must expose at least 16 GPUs:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
RAY_ADDRESS=<head-ip>:<port> \
bash scripts/train.sh
```

Hydra overrides can be appended directly to the command:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
RAY_ADDRESS=<head-ip>:<port> \
bash scripts/train.sh trainer.save_freq=20
```

### Main Training Options

| Variable | Default | Description |
|---|---:|---|
| `MODE` | `cotrain` | `cotrain` updates both agents; `frozen` keeps the Synthesizer fixed |
| `SYNTH_REWARD_MODE` | `gated` | Synthesizer reward mode: `gated` or `additive` |
| `TOTAL_EPOCHS` | `100` | Number of training steps |
| `GEN_BATCH_SIZE` | `32` | Synthesized task slots per step |
| `REASONER_ROLLOUTS_PER_TASK` | `4` | Reasoner rollouts per admitted task |
| `REPORT_REFRESH_EVERY` | `5` | Steps between live capability-report refreshes |
| `STAR_WORKFLOW_DEBUG` | `false` | Set to `true` to emit sampled workflow traces for debugging |
| `TRAIN_QUERIES` | auto-generated | Optional custom query file; when omitted, SynCo generates query slots from an empty pool |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | `0.3` | vLLM GPU-memory fraction |
| `RESUME_MODE` | `disable` | Set to `auto` to resume from the latest checkpoint |
| `RUN_ID` | auto-generated | Run name and artifact directory |

The full training configuration is defined in `configs/synco_trainer.yaml`.

### Training Outputs

Each run writes its artifacts to `runs/<RUN_ID>/`:

```text
runs/<RUN_ID>/
├── pool/                 # generated tasks, verification, rollouts, and rewards
│   └── valid_tasks.jsonl # admitted synthetic dataset
├── ckpts/                # training checkpoints
├── tensorboard_log/      # TensorBoard logs
└── train_query.json      # generated empty-pool query slots, or a copy of a custom query file
```

> **Memory tip:** On GPUs with less memory than B200, reduce `ROLLOUT_GPU_MEMORY_UTILIZATION` and `ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU`, or allocate additional nodes to each policy with `SYNTHESIZER_NNODES` and `REASONER_NNODES`.

---

## 🔬 Evaluation

### Step 1: Merge a Reasoner Checkpoint

Convert the distributed FSDP checkpoint into a Hugging Face model directory:

```bash
bash scripts/merge_checkpoint.sh \
  runs/<RUN_ID>/ckpts/global_step_100 \
  eval_results/<RUN_ID>/model
```

To merge the Synthesizer instead, pass `synthesizer_qwen3_8b` as the third argument.

### Step 2: Run Benchmark Evaluation

Evaluate the merged Reasoner model on all default benchmarks using every GPU on the current node:

```bash
bash scripts/eval_checkpoint.sh \
  eval_results/<RUN_ID>/model \
  eval_results/<RUN_ID>
```

Evaluate a selected subset of benchmarks:

```bash
bash scripts/eval_checkpoint.sh \
  eval_results/<RUN_ID>/model \
  eval_results/<RUN_ID> \
  gsm8k math500 aime24 aime25
```

The aggregated results are written to:

```bash
cat eval_results/<RUN_ID>/accuracy.json
```

---

## 📁 Repository Structure

```text
SynCo/
├── README.md
├── requirements-lock.txt
├── pyproject.toml
├── configs/
│   └── synco_trainer.yaml
├── data/                 # intentionally empty; download evaluation data separately
├── scripts/
│   ├── setup_env.sh
│   ├── train.sh
│   ├── merge_checkpoint.sh
│   ├── eval_checkpoint.sh
│   ├── evaluate.py
│   ├── download_eval_data.py
│   └── fetch_hf_math_benchmarks.py
├── src/synth_reasoner/
│   ├── agents/
│   ├── evaluation/
│   ├── parsing/
│   ├── pool/
│   ├── reports/
│   ├── rewards/
│   ├── unitymas_adapter/
│   └── verification/
└── third_party/
    └── UnityMAS-O/
```

---

## 🙏 Acknowledgements

SynCo builds on [UnityMAS-O](https://github.com/chenyiqun/UnityMAS-O), which is bundled under `third_party/UnityMAS-O/` at commit `7f1616a`. The SynCo-specific modifications for multi-agent GRPO are documented in `third_party/UnityMAS-O/SYNCO_MODIFICATIONS.md`.

We also thank the open-source communities behind [Qwen3](https://huggingface.co/Qwen/Qwen3-8B), [vLLM](https://github.com/vllm-project/vllm), [Ray](https://github.com/ray-project/ray), and [verl](https://github.com/volcengine/verl).

---

## 📖 Citation

If you find SynCo useful in your research, please cite:

```bibtex
@inproceedings{yang2026synco,
  title     = {SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning},
  author    = {Yang, Wei and Li, Shawn and Qin, Yuehan and Wang, Yawei and Wang, Mingxi and Li, Shixuan and Yang, Tiankai and Li, Jiate and Thomason, Jesse and Ma, Xuezhe and Zhao, Yue},
  booktitle = {NeurIPS 2026 Workshop on Towards Test-Time Continual Learning Agents},
  year      = {2026}
}
```

---

<div align="center">

**Built for the co-evolution of agents and data.**

</div>
