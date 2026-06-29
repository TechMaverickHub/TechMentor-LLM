"""Dataset validation, EDA, and optional W&B artifact logging."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CLEANED_PATH = ROOT / "data" / "cleaned" / "cleaned_dataset.json"
REPORTS_DIR = ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
EVAL_PATH = PROCESSED_DIR / "eval.jsonl"
STATS_PATH = REPORTS_DIR / "dataset_stats.json"
CATEGORY_CHART_PATH = REPORTS_DIR / "category_distribution.png"
ANSWER_LENGTH_CHART_PATH = REPORTS_DIR / "answer_length_distribution.png"
SAMPLE_RECORDS_PATH = REPORTS_DIR / "sample_records.md"

FOCUS_CATEGORIES = ("backend", "python", "sql", "ml", "system_design")
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
REQUIRED_FIELDS = ("instruction", "input", "output")
MIN_OUTPUT_LENGTH = 10
SAMPLE_COUNT = 20


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return rows


def validate_row(row: dict[str, Any], index: int, split: str) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] is None:
            errors.append(f"{split}[{index}]: missing '{field}'")

    output = row.get("output")
    if output is not None and len(str(output).strip()) <= MIN_OUTPUT_LENGTH:
        errors.append(
            f"{split}[{index}]: output too short "
            f"(<= {MIN_OUTPUT_LENGTH} chars after strip)"
        )
    return errors


def validate_alpaca_dataset(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    errors: list[str] = []
    for index, row in enumerate(train_rows):
        errors.extend(validate_row(row, index, "train"))
    for index, row in enumerate(eval_rows):
        errors.extend(validate_row(row, index, "eval"))

    if errors:
        details = "\n".join(errors[:20])
        extra = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Alpaca validation failed:\n{details}{extra}")

    print(f"Validated {len(train_rows) + len(eval_rows)} Alpaca rows")


def load_cleaned_records() -> list[dict[str, Any]]:
    with CLEANED_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_categories(cleaned: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in FOCUS_CATEGORIES}
    for record in cleaned:
        category = record.get("category")
        if category in counts:
            counts[category] += 1
    return counts


def count_sources(cleaned: list[dict[str, Any]]) -> dict[str, int]:
    sources: dict[str, int] = {}
    for record in cleaned:
        source = record.get("source") or "unknown"
        sources[source] = sources.get(source, 0) + 1
    return sources


def answer_length_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    lengths = (
        pd.Series([str(row["output"]).split() for row in rows])
        .str.len()
        .astype(float)
    )
    return {
        "avg_answer_length": round(float(lengths.mean()), 2),
        "median_answer_length": float(lengths.median()),
        "max_answer_length": int(lengths.max()),
        "min_answer_length": int(lengths.min()),
    }


def detect_duplicates(rows: list[dict[str, Any]]) -> dict[str, int]:
    inputs = [str(row.get("input", "")).lower().strip() for row in rows]
    instructions = [str(row.get("instruction", "")).lower().strip() for row in rows]
    outputs = [str(row.get("output", "")).lower().strip() for row in rows]

    duplicate_inputs = len(inputs) - len(set(inputs))
    duplicate_instructions = len(instructions) - len(set(instructions))
    duplicate_outputs = len(outputs) - len(set(outputs))

    return {
        "duplicate_inputs": duplicate_inputs,
        "duplicate_instructions": duplicate_instructions,
        "duplicate_outputs": duplicate_outputs,
        "duplicates_removed": duplicate_inputs + duplicate_outputs,
    }


def plot_category_distribution(category_counts: dict[str, int], output_path: Path) -> None:
    labels = [label.replace("_", " ").title() for label in FOCUS_CATEGORIES]
    values = [category_counts[category] for category in FOCUS_CATEGORIES]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="steelblue")
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_answer_length_distribution(rows: list[dict[str, Any]], output_path: Path) -> None:
    lengths = [len(str(row["output"]).split()) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30, color="darkorange", edgecolor="white")
    plt.title("Answer Length Distribution")
    plt.xlabel("Answer Length (words)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_sample_records(rows: list[dict[str, Any]], output_path: Path) -> None:
    rng = random.Random(42)
    samples = rng.sample(rows, k=min(SAMPLE_COUNT, len(rows)))

    lines = [
        "# Sample Records",
        "",
        f"Random sample of {len(samples)} instruction-tuning examples for manual review.",
        "",
    ]

    for index, row in enumerate(samples, start=1):
        lines.extend(
            [
                f"## Sample {index}",
                "",
                f"**Instruction:** {row['instruction']}",
                "",
                f"**Input:**",
                "",
                "```text",
                row["input"],
                "```",
                "",
                f"**Output:**",
                "",
                "```text",
                row["output"],
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_stats() -> dict[str, Any]:
    if not STATS_PATH.exists():
        return {}
    with STATS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_stats(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    existing_stats: dict[str, Any],
) -> dict[str, Any]:
    all_rows = train_rows + eval_rows
    category_counts = count_categories(cleaned)
    duplicate_stats = detect_duplicates(all_rows)

    stats = {
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "total_samples": len(all_rows),
        "categories": category_counts,
        "sources": count_sources(cleaned),
        "answer_length": answer_length_stats(all_rows),
        "duplicates": duplicate_stats,
        "cleaning": {
            key: existing_stats[key]
            for key in (
                "total_records",
                "raw_records",
                "removed_records",
                "duplicates_removed",
                "removed_null_question",
                "removed_empty_question",
                "removed_low_quality_answer",
                "backend",
                "sql",
                "ml",
            )
            if key in existing_stats
        },
    }
    return stats


def save_stats(stats: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def log_to_wandb(stats: dict[str, Any], artifact_paths: list[Path]) -> None:
    import wandb

    run = wandb.init(
        project="techmentor-llm",
        job_type="dataset-validation",
        config={
            "dataset_version": stats["dataset_version"],
            "preprocessing_version": stats["preprocessing_version"],
            "num_samples": stats["total_samples"],
            "train_samples": stats["train_samples"],
            "eval_samples": stats["eval_samples"],
            "categories": stats["categories"],
            "sources": stats["sources"],
            "answer_length": stats["answer_length"],
            "duplicates": stats["duplicates"],
        },
    )

    for key, value in stats["answer_length"].items():
        wandb.summary[key] = value

    for category, count in stats["categories"].items():
        wandb.summary[f"category_{category}"] = count

    artifact = wandb.Artifact(
        f"techmentor-dataset-{stats['dataset_version']}",
        type="dataset",
        description="TechMentor instruction-tuning dataset",
        metadata={
            "dataset_version": stats["dataset_version"],
            "preprocessing_version": stats["preprocessing_version"],
        },
    )
    for path in artifact_paths:
        artifact.add_file(str(path), name=path.name)

    run.log_artifact(artifact)
    wandb.finish()
    print("Logged dataset artifact to Weights & Biases")


def run_validation(use_wandb: bool) -> dict[str, Any]:
    train_rows = load_jsonl(TRAIN_PATH)
    eval_rows = load_jsonl(EVAL_PATH)
    cleaned = load_cleaned_records()

    validate_alpaca_dataset(train_rows, eval_rows)

    stats = build_stats(train_rows, eval_rows, cleaned, load_existing_stats())
    all_rows = train_rows + eval_rows

    plot_category_distribution(stats["categories"], CATEGORY_CHART_PATH)
    plot_answer_length_distribution(all_rows, ANSWER_LENGTH_CHART_PATH)
    write_sample_records(all_rows, SAMPLE_RECORDS_PATH)
    save_stats(stats)

    print(f"Train samples: {stats['train_samples']}")
    print(f"Eval samples:  {stats['eval_samples']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Saved stats -> {STATS_PATH}")
    print(f"Saved chart -> {CATEGORY_CHART_PATH}")
    print(f"Saved chart -> {ANSWER_LENGTH_CHART_PATH}")
    print(f"Saved samples -> {SAMPLE_RECORDS_PATH}")

    if use_wandb:
        artifact_paths = [
            TRAIN_PATH,
            EVAL_PATH,
            STATS_PATH,
            CATEGORY_CHART_PATH,
            ANSWER_LENGTH_CHART_PATH,
        ]
        log_to_wandb(stats, artifact_paths)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TechMentor dataset and generate reports.")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log dataset metadata and artifact to Weights & Biases (requires wandb login).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(use_wandb=args.wandb)


if __name__ == "__main__":
    main()
