"""Fine-tune Llama 3.2 3B with Unsloth + QLoRA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import wandb
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from training.config import (
    FINAL_ADAPTER_DIR,
    LOGS_DIR,
    MODEL_ID,
    OUTPUTS_DIR,
    WANDB_PROJECT,
    LoRAConfig,
    TrainingConfig,
    get_run_name,
)
from training.utils import (
    build_wandb_config,
    ensure_output_dirs,
    prepare_sft_datasets,
    save_adapter,
    save_training_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TechMentor with Unsloth QLoRA.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and model only; skip training.")
    return parser.parse_args()


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    lora: LoRAConfig,
    training: TrainingConfig,
    use_wandb: bool,
) -> SFTTrainer:
    sft_config = SFTConfig(
        output_dir=str(OUTPUTS_DIR),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.batch_size,
        per_device_eval_batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        eval_steps=training.eval_steps,
        eval_strategy=training.evaluation_strategy,
        fp16=training.fp16 and torch.cuda.is_available(),
        bf16=False,
        logging_dir=str(LOGS_DIR),
        report_to="wandb" if use_wandb else "none",
        seed=training.seed,
        max_seq_length=training.max_seq_length,
        dataset_text_field="text",
        packing=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )


def main() -> None:
    args = parse_args()
    lora = LoRAConfig()
    training = TrainingConfig(
        epochs=args.epochs or TrainingConfig.epochs,
        learning_rate=args.learning_rate or TrainingConfig.learning_rate,
        batch_size=args.batch_size or TrainingConfig.batch_size,
    )
    use_wandb = not args.no_wandb
    run_name = get_run_name(lora, training)

    ensure_output_dirs()
    wandb_url = None

    if use_wandb:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            job_type="fine-tuning",
            config=build_wandb_config(lora, training),
        )
        wandb_url = run.get_url()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=training.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        target_modules=lora.target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=training.seed,
    )

    train_dataset, eval_dataset = prepare_sft_datasets(tokenizer)
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    if args.dry_run:
        print("Dry run complete. Model and datasets loaded successfully.")
        save_training_artifacts(lora, training, wandb_url)
        if use_wandb:
            wandb.finish()
        return

    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        lora,
        training,
        use_wandb,
    )

    trainer.train()
    save_adapter(model, tokenizer, FINAL_ADAPTER_DIR)
    save_training_artifacts(lora, training, wandb_url)

    if use_wandb:
        wandb.finish()

    print(f"Training complete. Adapter saved to {FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
