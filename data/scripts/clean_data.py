"""Clean and unify raw datasets into a single schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"
REPORTS_DIR = ROOT / "reports"

UNIFIED_FIELDS = (
    "question",
    "answer",
    "candidate_answer",
    "followup",
    "code",
    "category",
    "difficulty",
    "source",
)

CATEGORY_MAP = {
    "django": "backend",
    "fastapi": "backend",
    "flask": "backend",
    "back-end": "backend",
    "back_end": "backend",
    "backend": "backend",
    "postgres": "sql",
    "mysql": "sql",
    "database_and_sql": "sql",
    "database_systems": "sql",
    "deep learning": "ml",
    "deep_learning": "ml",
    "llm": "ml",
    "machine_learning": "ml",
    "artificial_intelligence": "ml",
}

HF_QUESTION_PREFIXES = (
    "Hello! Let's begin our interview. ",
    "Welcome to your technical interview. ",
    "Let's begin with a technical question: ",
)


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned or None


def normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    normalized = re.sub(r"\s+", " ", str(category)).strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return CATEGORY_MAP.get(normalized, CATEGORY_MAP.get(normalized.replace("_", " "), normalized))


def normalize_difficulty(difficulty: str | None) -> str | None:
    if difficulty is None:
        return None
    return str(difficulty).strip().lower() or None


def empty_record() -> dict[str, Any]:
    return {field: None for field in UNIFIED_FIELDS}


def extract_hf_question(content: str) -> str:
    text = content.strip()
    for prefix in HF_QUESTION_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def load_hf_interview(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        conversations = json.load(handle)

    records: list[dict[str, Any]] = []
    for conversation in conversations:
        messages = conversation.get("messages", [])
        assistants = [m["content"] for m in messages if m.get("role") == "assistant"]
        users = [m["content"] for m in messages if m.get("role") == "user"]
        if not assistants:
            continue

        record = empty_record()
        record["question"] = extract_hf_question(assistants[0])
        record["candidate_answer"] = users[0] if users else None
        record["followup"] = assistants[1] if len(assistants) > 1 else None
        record["source"] = "hf_interview"
        records.append(record)
    return records


def load_kaggle(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        record = empty_record()
        record["question"] = row.get("Question")
        record["answer"] = row.get("Answer")
        record["category"] = row.get("Category")
        record["difficulty"] = row.get("Difficulty")
        record["source"] = "kaggle"
        records.append(record)
    return records


def load_codeqa(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)

    records: list[dict[str, Any]] = []
    for item in items:
        answer = item.get("answer")
        if answer is None:
            continue
        answer_text = str(answer).strip()
        if len(answer_text) < 30 or len(answer_text.split()) < 5:
            continue

        record = empty_record()
        record["question"] = item.get("question")
        record["answer"] = answer
        record["code"] = item.get("code")
        record["category"] = "python"
        record["source"] = "codeqa"
        records.append(record)
    return records


def standardize_record(record: dict[str, Any]) -> dict[str, Any]:
    standardized = empty_record()
    for field in UNIFIED_FIELDS:
        value = record.get(field)
        if field == "category":
            standardized[field] = normalize_category(value)
        elif field == "difficulty":
            standardized[field] = normalize_difficulty(value)
        else:
            standardized[field] = normalize_text(value)
    return standardized


def is_low_quality_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    text = answer.strip()
    if not text:
        return False
    return len(text.split()) < 5


def merge_records(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge complementary fields when the same question appears in multiple sources."""
    for field in UNIFIED_FIELDS:
        if field == "source":
            continue
        incoming_value = incoming.get(field)
        existing_value = existing.get(field)
        if incoming_value is None:
            continue
        if existing_value is None:
            existing[field] = incoming_value
            continue
        if field in {"followup", "candidate_answer"} and incoming.get("source") == "hf_interview":
            existing[field] = incoming_value
        elif field in {"answer", "category", "difficulty"} and incoming.get("source") == "kaggle":
            existing[field] = incoming_value
        elif field == "code" and incoming.get("source") == "codeqa":
            existing[field] = incoming_value


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_count = len(records)
    duplicates_removed = 0
    removed_null_question = 0
    removed_empty_question = 0
    removed_low_quality_answer = 0

    question_index: dict[str, int] = {}
    cleaned: list[dict[str, Any]] = []

    for record in records:
        question = record.get("question")
        if question is None:
            removed_null_question += 1
            continue

        question_key = question.lower().strip()
        if not question_key:
            removed_empty_question += 1
            continue

        if is_low_quality_answer(record.get("answer")):
            removed_low_quality_answer += 1
            continue

        if question_key in question_index:
            duplicates_removed += 1
            merge_records(cleaned[question_index[question_key]], record)
            continue

        question_index[question_key] = len(cleaned)
        cleaned.append(record)

    stats = {
        "total_records": len(cleaned),
        "raw_records": raw_count,
        "removed_records": raw_count - len(cleaned),
        "duplicates_removed": duplicates_removed,
        "removed_null_question": removed_null_question,
        "removed_empty_question": removed_empty_question,
        "removed_low_quality_answer": removed_low_quality_answer,
        "backend": sum(1 for r in cleaned if r.get("category") == "backend"),
        "sql": sum(1 for r in cleaned if r.get("category") == "sql"),
        "ml": sum(1 for r in cleaned if r.get("category") == "ml"),
    }
    return cleaned, stats


def resolve_kaggle_path() -> Path:
    for name in ("kaggle_questions.csv", "kaggle.csv"):
        path = RAW_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError("Expected kaggle_questions.csv or kaggle.csv in data/raw/")


def load_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(load_hf_interview(RAW_DIR / "hf_interview.json"))
    records.extend(load_kaggle(resolve_kaggle_path()))
    records.extend(load_codeqa(RAW_DIR / "codeqa.json"))
    return [standardize_record(record) for record in records]


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_sources()
    cleaned, stats = clean_records(records)

    cleaned_path = CLEANED_DIR / "cleaned_dataset.json"
    stats_path = REPORTS_DIR / "dataset_stats.json"

    with cleaned_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, indent=2, ensure_ascii=False)

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Cleaned {stats['raw_records']} -> {stats['total_records']} records")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Saved -> {cleaned_path}")
    print(f"  Stats -> {stats_path}")


if __name__ == "__main__":
    main()
