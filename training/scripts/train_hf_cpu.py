"""Fine-tune Llama 3.2 3B with Hugging Face Transformers + PEFT QLoRA.

Defaults to 4-bit QLoRA (same as train_hf.py) with smaller batch/sequence settings.
Use --full-precision for CPU-only float32 LoRA when no CUDA GPU is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training.config import (
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

FINAL_ADAPTER_CPU_DIR = OUTPUTS_DIR / "final_adapter_cpu"
CPU_DEFAULT_BATCH_SIZE = 1
CPU_DEFAULT_MAX_SEQ_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TechMentor with HF + PEFT QLoRA (4-bit by default)."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=CPU_DEFAULT_MAX_SEQ_LENGTH,
        help="Lower values reduce memory usage.",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Load full float32 weights on CPU (no 4-bit). Use when CUDA is unavailable.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and model only; skip training.")
    return parser.parse_args()


def load_model_and_tokenizer(*, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit QLoRA requires a CUDA GPU (bitsandbytes does not support 4-bit on CPU). "
                "Pass --full-precision to train with float32 LoRA on CPU instead."
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False

    model.gradient_checkpointing_enable()
    return model, tokenizer


def apply_lora(model, lora: LoRAConfig):
    peft_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora.target_modules,
    )
    return get_peft_model(model, peft_config)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training: TrainingConfig,
    use_wandb: bool,
    *,
    use_4bit: bool,
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
        fp16=use_4bit and torch.cuda.is_available(),
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
        use_cpu=not use_4bit,
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
    use_4bit = not args.full_precision
    lora = LoRAConfig()
    training = TrainingConfig(
        epochs=args.epochs or TrainingConfig.epochs,
        learning_rate=args.learning_rate or TrainingConfig.learning_rate,
        batch_size=args.batch_size or CPU_DEFAULT_BATCH_SIZE,
        max_seq_length=args.max_seq_length,
        fp16=use_4bit,
    )
    use_wandb = not args.no_wandb
    run_name = f"{get_run_name(lora, training)}-cpu"
    quantization = "4bit" if use_4bit else "none"
    device_label = "cuda" if use_4bit and torch.cuda.is_available() else "cpu"

    if use_4bit:
        print("Using 4-bit QLoRA (NF4) with bitsandbytes.")
    else:
        print(
            "Using full-precision float32 LoRA on CPU. "
            "This is slow and needs substantial RAM."
        )

    ensure_output_dirs()
    FINAL_ADAPTER_CPU_DIR.mkdir(parents=True, exist_ok=True)
    wandb_url = None

    if use_wandb:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            job_type="fine-tuning-cpu",
            config={
                **build_wandb_config(lora, training),
                "device": device_label,
                "quantization": quantization,
            },
        )
        wandb_url = run.get_url()

    model, tokenizer = load_model_and_tokenizer(use_4bit=use_4bit)
    model = apply_lora(model, lora)

    train_dataset, eval_dataset = prepare_sft_datasets(tokenizer)
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")
    print(
        f"Quantization: {quantization} | Device: {device_label} | "
        f"Max sequence length: {training.max_seq_length}"
    )

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
        training,
        use_wandb,
        use_4bit=use_4bit,
    )

    trainer.train()
    save_adapter(model, tokenizer, FINAL_ADAPTER_CPU_DIR)
    save_training_artifacts(lora, training, wandb_url)

    if use_wandb:
        wandb.finish()

    print(f"Training complete. Adapter saved to {FINAL_ADAPTER_CPU_DIR}")


if __name__ == "__main__":
    main()
