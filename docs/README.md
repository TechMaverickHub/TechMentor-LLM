# TechMentor-LLM Documentation

## Interview Guide (PDF)

**File:** [`TechMentor-LLM-Interview-Guide.pdf`](TechMentor-LLM-Interview-Guide.pdf)

A complete project guide covering:

- Executive summary and 30-second interview pitch
- Architecture and pipeline flow
- Core ML/LLM concepts (QLoRA, LoRA, PEFT, SFT, W&B)
- File-by-file code walkthrough
- Day 1–4 timeline
- Interview Q&A with model answers
- Resume bullet points
- Glossary

## Regenerate the PDF

```bash
uv sync --extra docs
uv run python docs/generate_documentation.py
```
