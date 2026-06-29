"""Training and LoRA configuration for TechMentor-LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
STATS_PATH = ROOT / "reports" / "dataset_stats.json"

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
EVAL_PATH = PROCESSED_DIR / "eval.jsonl"
OUTPUTS_DIR = TRAINING_DIR / "outputs"
LOGS_DIR = TRAINING_DIR / "logs"
FINAL_ADAPTER_DIR = OUTPUTS_DIR / "final_adapter"
EVALUATION_DIR = ROOT / "evaluation"

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
WANDB_PROJECT = "techmentor-llm"
DATASET_VERSION = "v1"
MAX_SEQ_LENGTH = 2048


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    evaluation_strategy: str = "steps"
    fp16: bool = True
    max_seq_length: int = MAX_SEQ_LENGTH
    seed: int = 42


def get_run_name(lora: LoRAConfig, training: TrainingConfig) -> str:
    lr_label = f"{training.learning_rate:.0e}".replace("e-0", "e").replace("e+", "e")
    return f"llama32-r{lora.r}-lr{lr_label}-{DATASET_VERSION}"


def lora_config_dict(lora: LoRAConfig) -> dict:
    return asdict(lora)


def training_config_dict(training: TrainingConfig, lora: LoRAConfig) -> dict:
    return {
        **asdict(training),
        "model_id": MODEL_ID,
        "dataset_version": DATASET_VERSION,
        "lora_rank": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_dropout": lora.lora_dropout,
        "run_name": get_run_name(lora, training),
    }
