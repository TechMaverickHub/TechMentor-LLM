"""Download raw datasets into data/raw/."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

HF_INTERVIEW_PARQUET = (
    "hf://datasets/remomo/software-interview-questions/data/train-00000-of-00001.parquet"
)
CODEQA_PARQUET = (
    "hf://datasets/vm2825/CodeQA-dataset/data/train-00000-of-00001.parquet"
)
KAGGLE_DATASET = "syedmharis/software-engineering-interview-questions-dataset"


def download_hf_interview(output: Path = RAW_DIR / "hf_interview.json") -> Path:
    """Hugging Face: remomo/software-interview-questions."""
    print("Downloading Hugging Face interview questions...")
    df = pd.read_parquet(HF_INTERVIEW_PARQUET)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output, orient="records", indent=2, force_ascii=False)
    print(f"  Saved {len(df)} rows -> {output}")
    return output


def download_kaggle_questions(output: Path = RAW_DIR / "kaggle_questions.csv") -> Path:
    """Kaggle: syedmharis/software-engineering-interview-questions-dataset."""
    print("Downloading Kaggle interview questions...")
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if len(csv_files) == 1:
        df = pd.read_csv(csv_files[0], encoding="latin-1")
        df.to_csv(output, index=False, encoding="utf-8")
    else:
        frames = [pd.read_csv(path, encoding="latin-1") for path in csv_files]
        pd.concat(frames, ignore_index=True).to_csv(output, index=False, encoding="utf-8")

    row_count = len(pd.read_csv(output))
    print(f"  Saved {row_count} rows from {len(csv_files)} file(s) -> {output}")
    return output


def download_codeqa(output: Path = RAW_DIR / "codeqa.json") -> Path:
    """CodeQA dataset from Liu & Wan (EMNLP 2021 Findings). Paper: arXiv:2109.08365."""
    print("Downloading CodeQA dataset...")
    df = pd.read_parquet(CODEQA_PARQUET)
    df = df.rename(
        columns={
            "Instruction": "question",
            "input_code": "code",
            "output_code": "answer",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output, orient="records", indent=2, force_ascii=False)
    print(f"  Saved {len(df)} rows -> {output}")
    return output


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_hf_interview()
    download_kaggle_questions()
    download_codeqa()
    print(f"\nAll datasets saved under {RAW_DIR}")


if __name__ == "__main__":
    main()
