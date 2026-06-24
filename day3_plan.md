# Day 3 - Dataset Validation, EDA & W&B Setup

## Goal

Before fine-tuning, validate that the dataset is:

* Clean
* Balanced
* Diverse
* Properly formatted
* Ready for instruction tuning

Create experiment tracking and dataset versioning using Weights & Biases.

---

# Folder Structure

```text
techmentor-llm/

├── data/
│   ├── cleaned/
│   ├── processed/
│
├── reports/
│   ├── dataset_stats.json
│   ├── category_distribution.png
│   ├── answer_length_distribution.png
│
├── notebooks/
│   ├── dataset_report.ipynb
│
├── wandb/
│
└── training/
```

---

# Step 1: Validate Alpaca Dataset

Load:

```python
train.jsonl
eval.jsonl
```

Verify every row contains:

```python
instruction
input
output
```

Validation checks:

```python
assert instruction is not None
assert output is not None
assert len(output.strip()) > 10
```

---

# Step 2: Dataset Statistics

Generate:

```python
total_samples
train_samples
eval_samples
```

Example output:

```json
{
  "train_samples": 7200,
  "eval_samples": 800,
  "total_samples": 8000
}
```

Save:

```text
reports/dataset_stats.json
```

---

# Step 3: Category Distribution

Calculate:

```python
backend
python
sql
ml
system_design
```

Create bar chart.

Example:

```text
Backend        2500
Python         1800
SQL            1200
ML             1600
System Design   900
```

Save:

```text
reports/category_distribution.png
```

---

# Step 4: Answer Length Analysis

Calculate:

```python
avg_answer_length
median_answer_length
max_answer_length
min_answer_length
```

Plot histogram.

Purpose:

Detect:

* very short answers
* noisy records
* abnormal outliers

Save:

```text
reports/answer_length_distribution.png
```

---

# Step 5: Duplicate Detection

Check:

```python
question duplicates
instruction duplicates
output duplicates
```

Report:

```json
{
  "duplicates_removed": 312
}
```

---

# Step 6: Dataset Quality Sampling

Randomly display:

```python
20 samples
```

Manually inspect:

* formatting
* grammar
* category labels
* answer quality

Create:

```text
reports/sample_records.md
```

---

# Step 7: W&B Setup

Install:

```bash
pip install wandb
```

Login:

```bash
wandb login
```

Create project:

```python
wandb.init(
    project="techmentor-llm",
    job_type="dataset-validation"
)
```

---

# Step 8: Dataset Versioning

Log dataset metadata.

Example:

```python
wandb.config.update({
    "dataset_version": "v1",
    "num_samples": 8000,
    "train_samples": 7200,
    "eval_samples": 800
})
```

Track:

* dataset size
* categories
* preprocessing version

---

# Step 9: Upload Dataset Artifact

Create artifact:

```python
artifact = wandb.Artifact(
    "techmentor-dataset",
    type="dataset"
)
```

Add:

```python
train.jsonl
eval.jsonl
dataset_stats.json
```

Log artifact.

Benefits:

* reproducibility
* version control
* experiment lineage

---

# Step 10: Create Dataset Report

Notebook:

```text
dataset_report.ipynb
```

Include:

## Dataset Overview

* Total Samples
* Categories
* Sources

## Visualizations

* Category Distribution
* Answer Length Distribution

## Sample Records

Random examples.

## Quality Checks

* Duplicate Removal
* Missing Values
* Category Balance

---

# Deliverables

By end of Day 3:

✅ train.jsonl

✅ eval.jsonl

✅ dataset_stats.json

✅ category_distribution.png

✅ answer_length_distribution.png

✅ dataset_report.ipynb

✅ W&B Project

✅ Dataset Artifact Version 1

---

# Resume Talking Point

Designed a data validation and quality assurance pipeline for instruction-tuning datasets, including duplicate detection, category balancing, statistical analysis, and dataset versioning using Weights & Biases artifacts before LLM fine-tuning.
