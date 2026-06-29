"""Prompt formatting for Llama 3.2 Instruct chat template."""

from __future__ import annotations

from typing import Any

DEFAULT_INSTRUCTION = "Answer the technical interview question."


def build_messages(
    instruction: str,
    user_input: str,
    output: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages


def format_training_text(
    tokenizer: Any,
    instruction: str,
    user_input: str,
    output: str,
) -> str:
    messages = build_messages(instruction, user_input, output)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_inference_prompt(
    tokenizer: Any,
    instruction: str,
    user_input: str,
) -> str:
    messages = build_messages(instruction, user_input, output=None)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


TEST_PROMPTS: list[dict[str, str]] = [
    {
        "category": "QA",
        "instruction": DEFAULT_INSTRUCTION,
        "input": "Explain dependency injection.",
    },
    {
        "category": "Backend",
        "instruction": DEFAULT_INSTRUCTION,
        "input": "Difference between JWT and OAuth?",
    },
    {
        "category": "SQL",
        "instruction": DEFAULT_INSTRUCTION,
        "input": "What is database indexing?",
    },
    {
        "category": "ML",
        "instruction": DEFAULT_INSTRUCTION,
        "input": "Difference between bagging and boosting.",
    },
    {
        "category": "Code Explanation",
        "instruction": "Explain the following code.",
        "input": (
            "Question: What does this function do?\n"
            "Code:\n"
            "def merge_sorted_lists(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    return result + a[i:] + b[j:]"
        ),
    },
    {
        "category": "Interview Evaluation",
        "instruction": "Evaluate the candidate answer.",
        "input": (
            "Question: What is JWT?\n"
            "Answer: JWT is used for login."
        ),
    },
]
