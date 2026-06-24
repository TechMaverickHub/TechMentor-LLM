"""Convert cleaned dataset into Alpaca-format instruction tuning examples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = ROOT / "data" / "cleaned" / "cleaned_dataset.json"
PROCESSED_DIR = ROOT / "data" / "processed"

MIX_RATIOS = {
    "interview_qa": 0.70,
    "followup": 0.10,
    "code_explanation": 0.10,
    "interview_evaluation": 0.10,
}


def load_cleaned(path: Path = CLEANED_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_interview_qa(record: dict[str, Any]) -> dict[str, str] | None:
    question = record.get("question")
    answer = record.get("answer")
    if not question or not answer:
        return None
    return {
        "instruction": "Answer the technical interview question.",
        "input": question,
        "output": answer,
    }


def build_followup(record: dict[str, Any]) -> dict[str, str] | None:
    question = record.get("question")
    followup = record.get("followup")
    if not question or not followup:
        return None
    return {
        "instruction": "Generate a follow-up interview question.",
        "input": question,
        "output": followup,
    }


def build_code_explanation(record: dict[str, Any]) -> dict[str, str] | None:
    question = record.get("question")
    code = record.get("code")
    answer = record.get("answer")
    if not question or not code or not answer:
        return None
    return {
        "instruction": "Explain the following code.",
        "input": f"Question: {question}\nCode:\n{code}",
        "output": answer,
    }


def build_interview_evaluation(record: dict[str, Any]) -> dict[str, str] | None:
    question = record.get("question")
    answer = record.get("answer")
    if not question or not answer:
        return None

    word_count = len(answer.split())
    if word_count >= 40:
        score = 8
    elif word_count >= 20:
        score = 6
    else:
        score = 5

    candidate_answer = answer.split(".")[0].strip()
    if len(candidate_answer.split()) > 20:
        candidate_answer = " ".join(candidate_answer.split()[:20])

    output = (
        f"Score: {score}/10\n"
        f"Strengths: Addresses the main topic with relevant technical points.\n"
        f"Weaknesses: Could include more examples, trade-offs, and implementation detail.\n"
        f"Improved Answer: {answer}"
    )
    return {
        "instruction": "Evaluate the candidate answer.",
        "input": f"Question: {question}\nAnswer: {candidate_answer}.",
        "output": output,
    }


def sample_records(
    records: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(records) >= count:
        return rng.sample(records, count)
    return [rng.choice(records) for _ in range(count)]


def build_mixed_dataset(
    records: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, str]]:
    qa_pool = [r for r in records if r.get("question") and r.get("answer")]
    followup_pool = [r for r in records if r.get("question") and r.get("followup")]
    code_pool = [r for r in records if r.get("question") and r.get("code") and r.get("answer")]

    if not followup_pool:
        raise ValueError("No follow-up records available for Alpaca conversion.")

    total_size = int(len(followup_pool) / MIX_RATIOS["followup"])
    counts = {
        "interview_qa": int(total_size * MIX_RATIOS["interview_qa"]),
        "followup": int(total_size * MIX_RATIOS["followup"]),
        "code_explanation": int(total_size * MIX_RATIOS["code_explanation"]),
        "interview_evaluation": int(total_size * MIX_RATIOS["interview_evaluation"]),
    }
    counts["followup"] = len(followup_pool)
    total_size = int(counts["followup"] / MIX_RATIOS["followup"])
    counts["interview_qa"] = int(total_size * MIX_RATIOS["interview_qa"])
    counts["code_explanation"] = int(total_size * MIX_RATIOS["code_explanation"])
    counts["interview_evaluation"] = total_size - counts["interview_qa"] - counts["followup"] - counts["code_explanation"]

    builders = {
        "interview_qa": (qa_pool, build_interview_qa),
        "followup": (followup_pool, build_followup),
        "code_explanation": (code_pool, build_code_explanation),
        "interview_evaluation": (qa_pool, build_interview_evaluation),
    }

    examples: list[dict[str, str]] = []
    for task_name, count in counts.items():
        pool, builder = builders[task_name]
        sampled = sample_records(pool, count, rng)
        for record in sampled:
            example = builder(record)
            if example:
                examples.append(example)

    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(42)
    records = load_cleaned()
    examples = build_mixed_dataset(records, rng)

    train_rows, eval_rows = train_test_split(
        examples,
        test_size=0.1,
        random_state=42,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.jsonl"
    eval_path = PROCESSED_DIR / "eval.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"Built {len(examples)} Alpaca examples")
    print(f"  Train: {len(train_rows)} -> {train_path}")
    print(f"  Eval:  {len(eval_rows)} -> {eval_path}")


if __name__ == "__main__":
    main()
