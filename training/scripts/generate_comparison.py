"""Generate base vs fine-tuned comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.config import EVALUATION_DIR, FINAL_ADAPTER_DIR, MODEL_ID
from training.prompts import TEST_PROMPTS, format_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned model outputs.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=FINAL_ADAPTER_DIR,
        help="Path to fine-tuned LoRA adapter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION_DIR / "before_after.md",
        help="Markdown output path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_report(sections: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Base vs Fine-Tuned Model Comparison",
        "",
        "Side-by-side outputs for TechMentor evaluation prompts.",
        "",
    ]
    output_path.write_text("\n".join(header + sections) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for 4-bit inference comparison.")

    if not args.adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {args.adapter_path}. "
            "Run training first: uv run python -m training.scripts.train_unsloth"
        )

    tokenizer = load_tokenizer()
    sections: list[str] = []

    base_model = load_base_model()
    finetuned_base = load_base_model()
    finetuned_model = PeftModel.from_pretrained(finetuned_base, str(args.adapter_path))

    for item in TEST_PROMPTS:
        prompt = format_inference_prompt(tokenizer, item["instruction"], item["input"])
        base_output = generate(base_model, tokenizer, prompt, args.max_new_tokens)
        finetuned_output = generate(finetuned_model, tokenizer, prompt, args.max_new_tokens)

        sections.extend(
            [
                f"## {item['category']}",
                "",
                f"**Instruction:** {item['instruction']}",
                "",
                f"**Input:**",
                "",
                "```text",
                item["input"],
                "```",
                "",
                "### Base Model Output",
                "",
                "```text",
                base_output,
                "```",
                "",
                "### Fine-Tuned Model Output",
                "",
                "```text",
                finetuned_output,
                "```",
                "",
                "---",
                "",
            ]
        )

    write_report(sections, args.output)
    print(f"Saved comparison report -> {args.output}")


if __name__ == "__main__":
    main()
