# TechMentor-LLM

Domain-specific instruction-tuning pipeline for a technical interview mentor model. The goal is to fine-tune **Llama 3.2 3B** with **QLoRA**, **PEFT**, **Unsloth**, and **Weights & Biases** so the model can:

- Answer technical interview questions
- Generate interview questions and follow-ups
- Evaluate candidate responses
- Explain code snippets
- Review resumes

This repository currently implements the **dataset engineering and preprocessing** stage. Fine-tuning is planned next — see [`plan.md`](plan.md) for the full roadmap.

---

## Project structure

```text
TechMentor-LLM/
├── data/
│   ├── raw/                  # Downloaded source datasets
│   ├── cleaned/              # Unified, deduplicated records
│   ├── processed/            # Alpaca-format train/eval splits
│   └── scripts/
│       ├── download_data.py  # Fetch raw data from HF, Kaggle, CodeQA
│       ├── clean_data.py     # Standardize, filter, deduplicate
│       └── convert_to_alpaca.py
├── training/
│   └── validate_dataset.py   # Day 3 validation, EDA, W&B artifacts
├── notebooks/
│   └── dataset_report.ipynb  # Dataset exploration and statistics
├── reports/
│   ├── dataset_stats.json
│   ├── category_distribution.png
│   ├── answer_length_distribution.png
│   └── sample_records.md
├── plan.md
│   day3_plan.md
└── pyproject.toml
```

---

## Data sources

| Source | File | Use case |
|--------|------|----------|
| [remomo/software-interview-questions](https://huggingface.co/datasets/remomo/software-interview-questions) | `hf_interview.json` | Interview simulation, follow-up generation |
| [syedmharis/software-engineering-interview-questions-dataset](https://www.kaggle.com/datasets/syedmharis/software-engineering-interview-questions-dataset) | `kaggle_questions.csv` | Technical Q&A, categories, difficulty |
| [CodeQA](https://arxiv.org/abs/2109.08365) | `codeqa.json` | Code explanation, Python reasoning |

All three sources are normalized into a single schema (`question`, `answer`, `candidate_answer`, `followup`, `code`, `category`, `difficulty`, `source`), then converted into Alpaca instruction examples.

---

## Requirements

- Python **3.12.9+**
- [uv](https://docs.astral.sh/uv/) for dependency management

Optional credentials:

- **Kaggle** — `kagglehub` uses your Kaggle API token if configured (`~/.kaggle/kaggle.json`)
- **Hugging Face** — only required for gated datasets; the datasets used here are public

---

## Setup

```bash
git clone <repo-url>
cd TechMentor-LLM
uv sync
```

---

## Data pipeline

Run the three stages in order:

### 1. Download raw data

```bash
uv run python data/scripts/download_data.py
```

Writes:

```text
data/raw/hf_interview.json
data/raw/kaggle_questions.csv
data/raw/codeqa.json
```

### 2. Clean and unify

```bash
uv run python data/scripts/clean_data.py
```

Applies:

- Schema standardization across all sources
- Null and empty question removal
- Low-quality answer filtering (`< 5` words)
- Deduplication by question (merges complementary fields across sources)
- Text and category normalization

Writes:

```text
data/cleaned/cleaned_dataset.json
reports/dataset_stats.json
```

### 3. Convert to Alpaca format

```bash
uv run python data/scripts/convert_to_alpaca.py
```

Builds a mixed instruction dataset:

| Task | Share | Instruction |
|------|-------|-------------|
| Interview QA | 70% | Answer the technical interview question. |
| Follow-up generation | 10% | Generate a follow-up interview question. |
| Code explanation | 10% | Explain the following code. |
| Interview evaluation | 10% | Evaluate the candidate answer. |

Splits **90% train / 10% eval** (`random_state=42`) and writes:

```text
data/processed/train.jsonl
data/processed/eval.jsonl
```

Each line is an Alpaca example:

```json
{
  "instruction": "Answer the technical interview question.",
  "input": "What is dependency injection?",
  "output": "Dependency injection is a design pattern..."
}
```

---

## Dataset validation (Day 3)

After building `train.jsonl` and `eval.jsonl`, run validation and generate reports:

```bash
uv run python training/validate_dataset.py
```

This script:

- Validates every Alpaca row (`instruction`, `input`, `output`; output length > 10)
- Updates `reports/dataset_stats.json` with train/eval counts, categories, and answer-length stats
- Generates `reports/category_distribution.png` and `reports/answer_length_distribution.png`
- Detects duplicate inputs/outputs and writes `reports/sample_records.md` (20 random samples)

### Weights & Biases

Log dataset metadata and version the artifact:

```bash
wandb login
uv run python training/validate_dataset.py --wandb
```

Creates a W&B project `techmentor-llm` with job type `dataset-validation` and uploads:

- `train.jsonl`
- `eval.jsonl`
- `dataset_stats.json`
- distribution charts

---

## Dataset report

Open or run the exploration notebook:

```bash
uv run jupyter notebook notebooks/dataset_report.ipynb
```

The report covers dataset overview, category and answer-length visualizations, quality checks, duplicate detection, and sample records.

---

## Current dataset snapshot

After the latest pipeline run:

| Metric | Count |
|--------|------:|
| Raw records | 24,242 |
| Clean records | 7,928 |
| Removed / merged duplicates | 16,314 |
| Train examples | 504 |
| Eval examples | 56 |

Instruction mix: **392** QA · **56** follow-up · **56** code explanation · **56** evaluation.

---

## Next steps

- Fine-tune Llama 3.2 3B with QLoRA + Unsloth on `train.jsonl`
- Evaluate on `eval.jsonl`
- Log experiments to Weights & Biases

See [`plan.md`](plan.md) for the full design document.

---

## License

See [LICENSE](LICENSE).
