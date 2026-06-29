"""Generate TechMentor-LLM PDF documentation for learning and interview prep."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF

DOCS_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = DOCS_DIR / "TechMentor-LLM-Interview-Guide.pdf"

FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITALIC = "C:/Windows/Fonts/ariali.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


class GuidePDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4", unit="mm")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=18)
        self.add_font("Body", "", FONT_REGULAR)
        self.add_font("Body", "B", FONT_BOLD)
        self.add_font("Body", "I", FONT_ITALIC)
        self.add_font("Mono", "", FONT_MONO)
        self.toc_entries: list[tuple[int, str, int]] = []

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Body", "I", 8)
        self.set_text_color(110, 110, 110)
        self.set_x(self.l_margin)
        self.cell(0, 8, "TechMentor-LLM | Complete Project Guide", align="L")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Body", "I", 8)
        self.set_text_color(110, 110, 110)
        self.set_x(self.l_margin)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def usable_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def add_title_page(self) -> None:
        self.add_page()
        self.set_font("Body", "B", 24)
        self.set_text_color(20, 40, 80)
        self.ln(45)
        self.set_x(self.l_margin)
        self.multi_cell(self.usable_width(), 12, "TechMentor-LLM\nComplete Project Guide", align="C")
        self.ln(8)
        self.set_font("Body", "", 13)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin)
        self.multi_cell(
            self.usable_width(),
            8,
            "Dataset Engineering, QLoRA Fine-Tuning,\nand Interview Preparation",
            align="C",
        )
        self.ln(20)
        self.set_font("Body", "", 11)
        self.set_x(self.l_margin)
        self.multi_cell(
            self.usable_width(),
            7,
            f"Generated: {date.today().isoformat()}\n"
            "Purpose: Understand every part of the codebase and explain it confidently in technical interviews.",
            align="C",
        )

    def add_heading(self, level: int, text: str) -> None:
        sizes = {1: 18, 2: 14, 3: 12}
        self.ln(4 if level > 1 else 6)
        self.set_font("Body", "B", sizes.get(level, 12))
        self.set_text_color(*(20, 40, 80) if level == 1 else (30, 60, 90))
        self.set_x(self.l_margin)
        self.multi_cell(self.usable_width(), 8, text)
        if level <= 2:
            self.toc_entries.append((level, text, self.page_no()))
        self.ln(2)

    def add_paragraph(self, text: str) -> None:
        self.set_font("Body", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(self.usable_width(), 5.5, text)
        self.ln(2)

    def add_bullet(self, text: str) -> None:
        self.set_font("Body", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(self.usable_width(), 5.5, f"- {text}")
        self.ln(1)

    def add_code(self, text: str) -> None:
        self.set_fill_color(245, 245, 245)
        self.set_font("Mono", "", 8.5)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.multi_cell(self.usable_width(), 4.8, text, fill=True)
        self.ln(2)

    def add_table(self, headers: list[str], rows: list[list[str]]) -> None:
        col_width = self.usable_width() / len(headers)
        self.set_x(self.l_margin)
        self.set_font("Body", "B", 9)
        self.set_fill_color(230, 236, 245)
        for header in headers:
            self.cell(col_width, 7, header, border=1, fill=True)
        self.ln()
        self.set_font("Body", "", 9)
        for row in rows:
            self.set_x(self.l_margin)
            for value in row:
                self.cell(col_width, 7, value, border=1)
            self.ln()
        self.ln(2)

    def add_toc(self) -> None:
        self.add_page()
        self.add_heading(1, "Table of Contents")
        for level, title, page in self.toc_entries:
            indent = 8 if level == 2 else 0
            self.set_font("Body", "", 10)
            self.set_x(self.l_margin + indent)
            self.cell(120, 6, title)
            self.cell(0, 6, str(page), align="R")
            self.ln()


CONTENT: list[tuple[str, object]] = [
    ("toc", None),
    (
        "heading1",
        "1. Executive Summary (30-Second Interview Pitch)",
    ),
    (
        "paragraph",
        "TechMentor-LLM is an end-to-end GenAI project that builds a domain-specific AI interview mentor. "
        "It ingests three public datasets (interview conversations, technical Q&A, and code reasoning), "
        "cleans and unifies them, converts them into instruction-tuning examples, validates quality, "
        "and fine-tunes Llama 3.2 3B Instruct using QLoRA so the model can answer interview questions, "
        "generate follow-ups, explain code, and evaluate candidate responses.",
    ),
    (
        "paragraph",
        "The project demonstrates practical MLOps and LLM engineering: data pipelines, dataset versioning "
        "with Weights & Biases, parameter-efficient fine-tuning (PEFT/LoRA), 4-bit quantization, experiment "
        "tracking, and structured evaluation.",
    ),
    ("heading2", "Key numbers to remember"),
    (
        "table",
        (
            ["Metric", "Value"],
            [
                ["Raw records ingested", "24,242"],
                ["Clean unified records", "7,928"],
                ["Final instruction examples", "560 (504 train / 56 eval)"],
                ["Base model", "Llama 3.2 3B Instruct"],
                ["Fine-tuning method", "QLoRA (4-bit + LoRA rank 16)"],
                ["Trainable params reduction", ">99% vs full fine-tune"],
            ],
        ),
    ),
    ("heading1", "2. Problem Statement and Solution Design"),
    (
        "paragraph",
        "General-purpose LLMs know broad programming concepts but are not optimized for structured "
        "technical interview mentoring. TechMentor solves this with supervised fine-tuning (SFT) on a "
        "curated multi-task instruction dataset.",
    ),
    ("heading2", "What the model should do"),
    ("bullet", "Answer technical interview questions clearly."),
    ("bullet", "Generate follow-up interview questions."),
    ("bullet", "Explain code snippets."),
    ("bullet", "Evaluate candidate answers with structured feedback."),
    ("bullet", "Support backend, SQL, ML, and system design topics."),
    ("heading2", "Design decisions"),
    ("bullet", "Use instruction tuning (Alpaca format) instead of raw text completion."),
    ("bullet", "Mix multiple task types (70/10/10/10) to avoid single-task overfitting."),
    ("bullet", "Use QLoRA to train on consumer GPUs (e.g., Colab T4)."),
    ("bullet", "Keep LoRA adapter separate from base model for flexibility and smaller artifacts."),
    ("heading1", "3. End-to-End Architecture"),
    (
        "paragraph",
        "Pipeline flow: Download -> Clean/Unify -> Convert to Alpaca -> Validate -> Fine-tune -> Infer/Evaluate.",
    ),
    (
        "code",
        "data/raw/  -->  clean_data.py  -->  data/cleaned/\n"
        "data/cleaned/  -->  convert_to_alpaca.py  -->  data/processed/train.jsonl + eval.jsonl\n"
        "data/processed/  -->  validate_dataset.py  -->  reports/ + W&B artifact\n"
        "data/processed/  -->  training/scripts/train_unsloth.py  -->  training/outputs/final_adapter/\n"
        "final_adapter/  -->  inference.ipynb + generate_comparison.py  -->  evaluation/before_after.md",
    ),
    ("heading2", "Folder responsibilities"),
    (
        "table",
        (
            ["Path", "Role"],
            [
                ["data/raw/", "Original downloaded datasets"],
                ["data/cleaned/", "Unified schema, deduplicated records"],
                ["data/processed/", "Alpaca JSONL ready for SFT"],
                ["reports/", "Stats, charts, sample records"],
                ["training/", "Fine-tuning code, configs, outputs"],
                ["evaluation/", "Base vs fine-tuned comparisons"],
                ["notebooks/", "EDA and dataset reporting"],
            ],
        ),
    ),
    ("heading1", "4. Core ML and LLM Concepts You Must Explain"),
    ("heading2", "4.1 Instruction Tuning"),
    (
        "paragraph",
        "Instruction tuning teaches a model to follow task instructions. Each training example has three fields: "
        "instruction (what to do), input (context/question), and output (desired response). This is more "
        "controllable than continuing raw text.",
    ),
    ("heading2", "4.2 Alpaca Format"),
    (
        "code",
        '{\n  "instruction": "Answer the technical interview question.",\n'
        '  "input": "What is dependency injection?",\n'
        '  "output": "Dependency injection is a design pattern..."\n}',
    ),
    ("heading2", "4.3 LoRA (Low-Rank Adaptation)"),
    (
        "paragraph",
        "Instead of updating all billions of model weights, LoRA inserts small trainable matrices into "
        "attention and MLP layers. Only these adapter weights are updated. This reduces memory, speeds training, "
        "and prevents catastrophic forgetting of general knowledge.",
    ),
    ("heading2", "4.4 QLoRA (Quantized LoRA)"),
    (
        "paragraph",
        "QLoRA combines 4-bit quantized base model weights (via BitsAndBytes) with LoRA adapters trained in "
        "higher precision. You get near full fine-tune quality with much lower GPU memory.",
    ),
    ("heading2", "4.5 PEFT"),
    (
        "paragraph",
        "PEFT (Parameter-Efficient Fine-Tuning) is the Hugging Face library abstraction for methods like LoRA. "
        "It manages adapter injection, saving, and loading.",
    ),
    ("heading2", "4.6 Unsloth"),
    (
        "paragraph",
        "Unsloth is an optimized training stack on top of Transformers/TRL that speeds up LoRA training and "
        "reduces memory via kernel optimizations and efficient gradient checkpointing.",
    ),
    ("heading2", "4.7 SFTTrainer (TRL)"),
    (
        "paragraph",
        "Supervised Fine-Tuning Trainer from TRL handles tokenization, loss computation on assistant tokens, "
        "evaluation loops, checkpointing, and logging integration.",
    ),
    ("heading2", "4.8 Weights & Biases (W&B)"),
    (
        "paragraph",
        "W&B tracks experiments: hyperparameters, metrics (train/eval loss), artifacts (dataset versions), "
        "and run lineage for reproducibility.",
    ),
    ("heading1", "5. Data Sources and Why Each Was Chosen"),
    (
        "table",
        (
            ["Source", "File", "Records", "Purpose"],
            [
                ["HF Interview", "hf_interview.json", "~200", "Conversation flow, follow-ups"],
                ["Kaggle SE Q&A", "kaggle_questions.csv", "~200", "Categorized Q&A with difficulty"],
                ["CodeQA paper", "codeqa.json", "~76k", "Code explanation and Python reasoning"],
            ],
        ),
    ),
    ("heading2", "Unified schema (all sources)"),
    (
        "code",
        "question, answer, candidate_answer, followup, code,\n"
        "category, difficulty, source",
    ),
    (
        "paragraph",
        "Unused fields are null. This schema lets one cleaning pipeline handle heterogeneous sources and "
        "merge complementary fields when questions overlap.",
    ),
    ("heading1", "6. Code Walkthrough: download_data.py"),
    (
        "paragraph",
        "Location: data/scripts/download_data.py. Downloads all raw datasets into data/raw/.",
    ),
    ("bullet", "HF dataset loaded via pandas read_parquet from hf:// URI (Hugging Face + fsspec)."),
    ("bullet", "Kaggle dataset downloaded with kagglehub; CSV re-encoded from Latin-1 to UTF-8."),
    ("bullet", "CodeQA loaded from HF mirror; columns renamed to question/code/answer."),
    ("heading2", "Outputs"),
    (
        "code",
        "data/raw/hf_interview.json\n"
        "data/raw/kaggle_questions.csv\n"
        "data/raw/codeqa.json",
    ),
    ("heading1", "7. Code Walkthrough: clean_data.py"),
    (
        "paragraph",
        "This is the data engineering core. It standardizes, filters, deduplicates, and merges records.",
    ),
    ("heading2", "Loading logic"),
    ("bullet", "HF: extracts question from first assistant turn; follow-up from second assistant turn."),
    ("bullet", "Kaggle: maps Question/Answer/Category/Difficulty columns."),
    ("bullet", "CodeQA: rejects short answers (<5 words or <30 chars) early."),
    ("heading2", "Cleaning rules"),
    ("bullet", "Remove null/empty questions."),
    ("bullet", "Remove answers with fewer than 5 words (when answer exists)."),
    ("bullet", "Deduplicate by question.lower().strip()."),
    ("bullet", "On duplicate: merge fields (HF follow-up + Kaggle answer/category + CodeQA code)."),
    ("bullet", "Normalize whitespace and map categories (e.g., machine_learning -> ml)."),
    ("heading2", "Important interview point"),
    (
        "paragraph",
        "Merging on duplicate questions preserves signal from all sources instead of dropping valuable fields. "
        "Example: same question in HF and Kaggle keeps follow-up from HF and reference answer from Kaggle.",
    ),
    ("heading1", "8. Code Walkthrough: convert_to_alpaca.py"),
    (
        "paragraph",
        "Transforms cleaned records into four instruction task types and splits train/eval.",
    ),
    (
        "table",
        (
            ["Task", "Share", "Instruction"],
            [
                ["Interview QA", "70%", "Answer the technical interview question."],
                ["Follow-up", "10%", "Generate a follow-up interview question."],
                ["Code explanation", "10%", "Explain the following code."],
                ["Interview evaluation", "10%", "Evaluate the candidate answer."],
            ],
        ),
    ),
    ("bullet", "Dataset size is anchored by follow-up pool size (limited HF conversations)."),
    ("bullet", "train_test_split(test_size=0.1, random_state=42) for reproducibility."),
    ("bullet", "Outputs JSONL lines for efficient streaming during training."),
    ("heading1", "9. Code Walkthrough: validate_dataset.py (Day 3)"),
    (
        "paragraph",
        "Pre-training quality gate. Ensures Alpaca rows are valid and generates EDA artifacts.",
    ),
    ("bullet", "Step 1: Load train.jsonl and eval.jsonl."),
    ("bullet", "Step 2: Assert each row has instruction, input, output; output length > 10 chars."),
    ("bullet", "Step 3: Update dataset_stats.json with train/eval/total sample counts."),
    ("bullet", "Step 4: Plot category distribution from cleaned_dataset.json."),
    ("bullet", "Step 5: Plot answer length histogram from output token/word counts."),
    ("bullet", "Step 6: Detect duplicate inputs, instructions, and outputs."),
    ("bullet", "Step 7: Write 20 random samples to sample_records.md for manual QA."),
    ("bullet", "Step 8-9: Optional W&B init + dataset artifact (train, eval, stats, charts)."),
    ("heading2", "Why validation before training?"),
    (
        "paragraph",
        "Training on malformed data wastes GPU hours and produces unstable loss curves. Validation catches "
        "empty outputs, schema drift, and duplicate leakage between train and eval early.",
    ),
    ("heading1", "10. Fine-Tuning Stack (Day 4)"),
    ("heading2", "config.py"),
    (
        "paragraph",
        "Central hyperparameters and paths. Defines LoRAConfig (r=16, alpha=32, dropout=0.05) and "
        "TrainingConfig (epochs=2, lr=2e-4, batch=2, grad_accum=4, max_seq_len=2048).",
    ),
    ("heading2", "prompts.py"),
    (
        "paragraph",
        "Converts Alpaca triplets into Llama 3.2 chat template using tokenizer.apply_chat_template. "
        "System role carries instruction; user role carries input; assistant role carries target output.",
    ),
    ("heading2", "utils.py"),
    ("bullet", "Loads JSONL and builds Hugging Face Dataset with a text field."),
    ("bullet", "Saves training_config.json, lora_config.json, wandb_run_url.txt."),
    ("bullet", "Saves adapter + tokenizer to final_adapter/."),
    ("heading2", "train_unsloth.py (primary)"),
    ("bullet", "Loads 4-bit quantized Llama 3.2 via FastLanguageModel."),
    ("bullet", "Injects LoRA adapters into attention/MLP projections."),
    ("bullet", "Trains with TRL SFTTrainer and logs to W&B."),
    ("bullet", "Supports --dry-run and --no-wandb flags."),
    ("heading2", "train_hf.py (fallback)"),
    (
        "paragraph",
        "Same training flow using native Transformers + PEFT + BitsAndBytesConfig for environments "
        "without Unsloth.",
    ),
    ("heading2", "Training hyperparameter rationale"),
    (
        "table",
        (
            ["Hyperparameter", "Value", "Why"],
            [
                ["LoRA rank (r)", "16", "Balance capacity vs overfitting on small dataset"],
                ["LoRA alpha", "32", "Scaling factor (typically 2x rank)"],
                ["Learning rate", "2e-4", "Common effective LR range for LoRA SFT"],
                ["Epochs", "2", "Enough adaptation without memorization"],
                ["Batch size", "2", "Fits T4 memory with 4-bit model"],
                ["Grad accumulation", "4", "Effective batch size = 8"],
                ["Max seq length", "2048", "Covers long code explanation samples"],
                ["Warmup ratio", "0.03", "Stabilizes early training steps"],
            ],
        ),
    ),
    ("heading2", "Llama 3.2 chat formatting"),
    (
        "paragraph",
        "Before tokenization, each Alpaca example becomes a chat transcript. During training, the model "
        "learns to produce assistant messages given system+user context.",
    ),
    (
        "code",
        "System: Answer the technical interview question.\n"
        "User: What is JWT?\n"
        "Assistant: JWT stands for JSON Web Token...",
    ),
    ("heading2", "generate_comparison.py"),
    (
        "paragraph",
        "Loads base 4-bit model and fine-tuned PeftModel adapter separately (fair comparison). Runs six "
        "fixed test prompts and writes evaluation/before_after.md with side-by-side outputs.",
    ),
    ("heading2", "Why not merge adapter into base model yet?"),
    (
        "paragraph",
        "Keeping adapter separate enables smaller artifacts, easier A/B testing, swapping adapters per task, "
        "and safer deployment experimentation before full merge/export.",
    ),
    ("heading1", "11. Inference and Evaluation"),
    ("bullet", "inference.ipynb: loads base model + LoRA adapter, runs test prompts."),
    ("bullet", "generate_comparison.py: compares base vs fine-tuned outputs."),
    ("bullet", "evaluation/before_after.md: side-by-side results for README and demos."),
    ("heading2", "Six evaluation prompt categories"),
    ("bullet", "QA: Explain dependency injection."),
    ("bullet", "Backend: JWT vs OAuth."),
    ("bullet", "SQL: Database indexing."),
    ("bullet", "ML: Bagging vs boosting."),
    ("bullet", "Code explanation: Python merge function."),
    ("bullet", "Interview evaluation: Score a weak JWT answer."),
    ("heading1", "11b. Day-by-Day Project Timeline"),
    (
        "table",
        (
            ["Day", "Focus", "Key Outputs"],
            [
                ["Day 1-2", "Data engineering", "raw/, cleaned/, processed/"],
                ["Day 3", "Validation + W&B", "reports/, dataset artifact"],
                ["Day 4", "QLoRA fine-tuning", "final_adapter/, W&B run"],
            ],
        ),
    ),
    ("heading1", "11c. Technologies Used"),
    (
        "table",
        (
            ["Layer", "Tools"],
            [
                ["Language", "Python 3.12, uv"],
                ["Data", "pandas, scikit-learn, kagglehub"],
                ["ML/LLM", "Transformers, TRL, PEFT, BitsAndBytes, Unsloth"],
                ["Model", "meta-llama/Llama-3.2-3B-Instruct"],
                ["Tracking", "Weights & Biases, Hugging Face Hub"],
                ["Analysis", "Jupyter, matplotlib"],
            ],
        ),
    ),
    ("heading1", "12. How to Run the Full Pipeline"),
    (
        "code",
        "uv sync\n"
        "uv run python data/scripts/download_data.py\n"
        "uv run python data/scripts/clean_data.py\n"
        "uv run python data/scripts/convert_to_alpaca.py\n"
        "uv run python training/scripts/validate_dataset.py\n"
        "uv sync --extra train\n"
        "huggingface-cli login\n"
        "wandb login\n"
        "uv run python -m training.scripts.train_unsloth\n"
        "uv run python -m training.scripts.generate_comparison",
    ),
    ("heading1", "13. Interview Questions and Model Answers"),
    ("heading2", "Q: Why QLoRA instead of full fine-tuning?"),
    (
        "paragraph",
        "A: Full fine-tuning of a 3B model needs much more GPU memory and risks overfitting on small domain "
        "data. QLoRA trains only adapter weights on a quantized base, making single-GPU training practical "
        "while retaining strong task performance.",
    ),
    ("heading2", "Q: Why combine three datasets?"),
    (
        "paragraph",
        "A: Each source covers different behavior: conversational follow-ups (HF), categorized technical Q&A "
        "(Kaggle), and code reasoning (CodeQA). Combining them creates a multi-skill interview mentor instead "
        "of a single-format QA bot.",
    ),
    ("heading2", "Q: How did you handle data quality?"),
    (
        "paragraph",
        "A: Multi-stage filtering (null checks, short-answer rejection, deduplication), category normalization, "
        "duplicate-field merging, Alpaca validation, duplicate detection in final JSONL, and manual sample review "
        "via sample_records.md.",
    ),
    ("heading2", "Q: How do you ensure reproducibility?"),
    (
        "paragraph",
        "A: Fixed random seeds (42), versioned dataset artifacts in W&B, saved configs (training_config.json, "
        "lora_config.json), and deterministic train/eval split.",
    ),
    ("heading2", "Q: What metrics do you track during training?"),
    (
        "paragraph",
        "A: Train loss, eval loss, learning rate schedule, checkpoint steps, and qualitative before/after "
        "comparisons on fixed prompt sets across QA, backend, SQL, ML, code, and evaluation tasks.",
    ),
    ("heading2", "Q: What would you improve next?"),
    ("bullet", "Increase interview conversation data for better follow-up generation."),
    ("bullet", "Add RAG over company-specific interview rubrics."),
    ("bullet", "Use human or LLM-as-judge evaluation with rubric scoring."),
    ("bullet", "Merge and export GGUF for local deployment."),
    ("bullet", "Add resume-review task data and evaluation set."),
    ("heading1", "14. Resume Bullet Points (Ready to Use)"),
    ("bullet", "Built a multi-source data engineering pipeline aggregating interview, Q&A, and code datasets into a unified instruction-tuning corpus."),
    ("bullet", "Implemented preprocessing, deduplication, category normalization, and Alpaca conversion for LLM fine-tuning."),
    ("bullet", "Designed dataset validation and QA with statistical analysis and W&B dataset artifacts."),
    ("bullet", "Fine-tuned Llama 3.2 3B Instruct using QLoRA, PEFT, and Unsloth on a custom instruction dataset."),
    ("bullet", "Reduced trainable parameters by >99% using LoRA while enabling single-GPU training."),
    ("bullet", "Tracked hyperparameters, checkpoints, and metrics with Weights & Biases."),
    ("bullet", "Evaluated base vs fine-tuned model across QA, code explanation, and interview evaluation tasks."),
    ("heading1", "15. Glossary"),
    (
        "table",
        (
            ["Term", "Meaning"],
            [
                ["SFT", "Supervised Fine-Tuning on input-output pairs"],
                ["LoRA", "Low-rank adapter matrices added to frozen layers"],
                ["QLoRA", "LoRA training on 4-bit quantized base weights"],
                ["PEFT", "Parameter-Efficient Fine-Tuning framework"],
                ["TRL", "Transformer Reinforcement Learning library (includes SFTTrainer)"],
                ["JSONL", "One JSON object per line, stream-friendly format"],
                ["Adapter", "Small fine-tuned weight set loaded over base model"],
            ],
        ),
    ),
]


def build_pdf() -> Path:
    pdf = GuidePDF()
    pdf.add_title_page()

    for item in CONTENT:
        kind = item[0]
        value = item[1]
        if kind == "toc":
            continue
        if kind == "heading1":
            pdf.add_heading(1, str(value))
        elif kind == "heading2":
            pdf.add_heading(2, str(value))
        elif kind == "paragraph":
            pdf.add_paragraph(str(value))
        elif kind == "bullet":
            pdf.add_bullet(str(value))
        elif kind == "code":
            pdf.add_code(str(value))
        elif kind == "table":
            headers, rows = value  # type: ignore[misc]
            pdf.add_table(headers, rows)

    # Insert TOC after title by rebuilding with page references
  # Simpler approach: append TOC at end for navigation
    pdf.add_toc()
    pdf.output(str(OUTPUT_PDF))
    return OUTPUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(f"Generated: {path}")
