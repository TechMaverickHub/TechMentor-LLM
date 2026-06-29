"""Shared training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from training.config import (
    EVAL_PATH,
    FINAL_ADAPTER_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    STATS_PATH,
    TRAIN_PATH,
    LoRAConfig,
    TrainingConfig,
    get_run_name,
    lora_config_dict,
    training_config_dict,
)
from training.prompts import format_training_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_stats() -> dict[str, Any]:
    if not STATS_PATH.exists():
        return {"dataset_version": "v1"}
    with STATS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def prepare_sft_datasets(tokenizer: Any) -> tuple[Dataset, Dataset]:
    train_rows = load_jsonl(TRAIN_PATH)
    eval_rows = load_jsonl(EVAL_PATH)

    def to_text(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "text": format_training_text(
                    tokenizer,
                    row["instruction"],
                    row["input"],
                    row["output"],
                )
            }
            for row in rows
        ]

    return (
        Dataset.from_list(to_text(train_rows)),
        Dataset.from_list(to_text(eval_rows)),
    )


def ensure_output_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)


def save_training_artifacts(
    lora: LoRAConfig,
    training: TrainingConfig,
    wandb_url: str | None = None,
) -> None:
    ensure_output_dirs()

    training_payload = training_config_dict(lora, training)
    lora_payload = lora_config_dict(lora)

    training_config_path = OUTPUTS_DIR / "training_config.json"
    lora_config_path = OUTPUTS_DIR / "lora_config.json"
    requirements_path = OUTPUTS_DIR / "requirements.txt"

    with training_config_path.open("w", encoding="utf-8") as handle:
        json.dump(training_payload, handle, indent=2)

    with lora_config_path.open("w", encoding="utf-8") as handle:
        json.dump(lora_payload, handle, indent=2)

    requirements_path.write_text(
        "\n".join(
            [
                "transformers>=4.43",
                "datasets>=2.20",
                "trl>=0.9",
                "peft>=0.12",
                "bitsandbytes>=0.43",
                "accelerate>=0.33",
                "wandb>=0.17",
                "unsloth",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if wandb_url:
        (OUTPUTS_DIR / "wandb_run_url.txt").write_text(wandb_url + "\n", encoding="utf-8")


def build_wandb_config(lora: LoRAConfig, training: TrainingConfig) -> dict[str, Any]:
    stats = load_dataset_stats()
    return {
        **training_config_dict(lora, training),
        "train_samples": stats.get("train_samples"),
        "eval_samples": stats.get("eval_samples"),
        "total_samples": stats.get("total_samples"),
        "run_name": get_run_name(lora, training),
    }


def save_adapter(model: Any, tokenizer: Any, output_dir: Path | None = None) -> Path:
    target = output_dir or FINAL_ADAPTER_DIR
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return target
