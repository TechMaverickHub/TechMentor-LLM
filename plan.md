# TechMentor-LLM

## Dataset Engineering & Preprocessing Plan

### Objective

Build a domain-specific instruction tuning dataset for fine-tuning Llama 3.2 3B using:

* QLoRA
* PEFT
* Unsloth
* Weights & Biases

The resulting model should be able to:

1. Answer technical interview questions
2. Generate interview questions
3. Evaluate candidate responses
4. Explain code snippets
5. Review resumes
6. Generate follow-up interview questions

---

# Dataset Sources

## Source A - HF Interview Conversations

Example:

```json
{
  "messages":[
    {"role":"system","content":"You are a technical interviewer"},
    {"role":"assistant","content":"What is the difference between an array and a linked list?"},
    {"role":"user","content":"Candidate answer"},
    {"role":"assistant","content":"Can you give me an example?"}
  ]
}
```

### Purpose

Used for:

* Interview simulation
* Follow-up generation
* Conversational interview flow

### Extracted Schema

```json
{
  "question":"What is the difference between an array and a linked list?",
  "candidate_answer":"Candidate answer",
  "followup":"Can you give me an example?",
  "source":"hf_interview"
}
```

---

## Source B - Kaggle Interview Dataset

Example:

```csv
Question:
Develop a machine learning model to predict stock prices.

Answer:
Consider time series analysis, regression models...

Category:
Machine Learning

Difficulty:
Hard
```

### Purpose

Used for:

* Interview QA
* Question generation

### Extracted Schema

```json
{
  "question":"Develop a machine learning model to predict stock prices.",
  "answer":"Consider time series analysis...",
  "category":"machine_learning",
  "difficulty":"hard",
  "source":"kaggle"
}
```

---

## Source C - CodeQA

Example:

```json
{
  "question":"How does the code run a WSGI function?",
  "code":"def runfcgi(func)...",
  "answer":"with a fastcgi server"
}
```

### Purpose

Used for:

* Code explanation
* Python reasoning
* Backend understanding

### Cleaning Rule

Reject records where:

```python
len(answer.split()) < 5
```

OR

```python
len(answer.strip()) < 30
```

Reason:

Many CodeQA answers are too short and noisy.

---

# Final Unified Schema

All datasets become:

```json
{
  "question":"...",
  "answer":"...",
  "candidate_answer":"...",
  "followup":"...",
  "code":"...",
  "category":"...",
  "difficulty":"...",
  "source":"..."
}
```

Unused fields remain null.

---

# clean_data.py

## Responsibilities

### 1. Load Sources

Load:

```text
data/raw/hf_interview.json
data/raw/kaggle.csv
data/raw/codeqa.json
```

---

### 2. Standardize

Convert all records into Unified Schema.

---

### 3. Remove Null Questions

Remove:

```python
question is None
```

---

### 4. Remove Empty Text

Remove:

```python
question.strip() == ""
```

---

### 5. Remove Low Quality Answers

Remove:

```python
len(answer.split()) < 5
```

---

### 6. Remove Duplicates

Deduplicate by:

```python
question.lower().strip()
```

---

### 7. Normalize Text

Apply:

```python
re.sub(r"\s+"," ", text)
```

---

### 8. Normalize Categories

Map:

```python
{
 "django":"backend",
 "fastapi":"backend",
 "flask":"backend",

 "postgres":"sql",
 "mysql":"sql",

 "deep learning":"ml",
 "llm":"ml"
}
```

---

### 9. Dataset Statistics

Generate:

```json
{
 "total_records":0,
 "duplicates_removed":0,
 "backend":0,
 "sql":0,
 "ml":0
}
```

Save:

```text
reports/dataset_stats.json
```

---

### Output

```text
data/cleaned/cleaned_dataset.json
```

---

# convert_to_alpaca.py

Convert cleaned data into instruction tuning examples.

---

# Dataset Type 1 - Interview QA

70%

Input:

```json
{
 "question":"What is JWT?"
}
```

Output:

```json
{
 "instruction":"Answer the technical interview question.",
 "input":"What is JWT?",
 "output":"JWT stands for JSON Web Token..."
}
```

---

# Dataset Type 2 - Follow-up Generation

10%

Generated from HF Interview dataset.

Input:

```json
{
 "question":"Difference between array and linked list?"
}
```

Output:

```json
{
 "instruction":"Generate a follow-up interview question.",
 "input":"Difference between array and linked list?",
 "output":"Can you give me an example?"
}
```

---

# Dataset Type 3 - Code Explanation

10%

Generated from CodeQA.

Input:

```json
{
 "question":"How does this code run a WSGI function?",
 "code":"..."
}
```

Output:

```json
{
 "instruction":"Explain the following code.",
 "input":"Question: ... Code: ...",
 "output":"The function imports WSGIServer..."
}
```

---

# Dataset Type 4 - Interview Evaluation

10%

Auto-generated.

Input:

```json
{
 "question":"What is JWT?"
}
```

Output:

```json
{
 "instruction":"Evaluate the candidate answer.",
 "input":"Question: What is JWT?\nAnswer: JWT is a token.",
 "output":"Score: 6/10\nStrengths...\nWeaknesses...\nImproved Answer..."
}
```

---

# Dataset Mix

```text
70% Interview QA

10% Follow-up Generation

10% Code Explanation

10% Interview Evaluation
```

---

# Train Eval Split

```python
train_test_split(
    test_size=0.1,
    random_state=42
)
```

Result:

```text
90% Train

10% Eval
```

---

# Output Format

train.jsonl

Example:

```json
{
 "instruction":"Answer the technical interview question.",
 "input":"What is dependency injection?",
 "output":"Dependency injection is..."
}
```

---

eval.jsonl

Same format.

---

# dataset_report.ipynb

Create report containing:

### Dataset Size

```text
Raw Samples
Clean Samples
Removed Samples
```

### Category Distribution

```text
Backend
Python
SQL
ML
System Design
```

### Difficulty Distribution

```text
Easy
Medium
Hard
```

### Answer Length Statistics

```text
Average
Median
Maximum
Minimum
```

### Duplicate Removal Statistics

```text
Duplicates Removed
Remaining Records
```

### Sample Records

Display 10 random examples.

---

# Resume Talking Points

Built a multi-source data engineering pipeline aggregating interview conversations, technical Q&A, and code reasoning datasets into a unified instruction-tuning corpus.

Implemented automated preprocessing, quality filtering, deduplication, category normalization, and Alpaca-format conversion for LLM fine-tuning.

Created a 3-task instruction dataset supporting technical question answering, interview evaluation, code explanation, and follow-up question generation for QLoRA-based domain adaptation.
