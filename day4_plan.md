# Day 4 – Fine-Tuning Llama 3.2 with QLoRA

## Goal

Fine-tune **Llama 3.2 3B Instruct** on the TechMentor dataset using:

* Unsloth
* Hugging Face Transformers
* TRL (SFTTrainer)
* PEFT (LoRA)
* BitsAndBytes (4-bit)
* Weights & Biases

By the end of today you should have:

* A trained LoRA adapter
* Training metrics logged to W&B
* Saved checkpoints
* Inference notebook for testing

---

# Folder Structure

```text
training/

├── config.py
├── prompts.py
├── utils.py
│
├── scripts/
│   ├── train_unsloth.py
│   ├── train_hf.py
│   ├── generate_comparison.py
│   └── validate_dataset.py
│
├── outputs/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final_adapter/
│
└── logs/
```

---

# Step 1 - Install Dependencies

```bash
pip install \
transformers \
datasets \
trl \
peft \
bitsandbytes \
accelerate \
wandb \
unsloth
```

---

# Step 2 - Login

```python
from huggingface_hub import login

login()
```

```python
import wandb

wandb.login()
```

---

# Step 3 - Load Dataset

Input

```text
train.jsonl
eval.jsonl
```

Expected schema

```json
{
    "instruction":"",
    "input":"",
    "output":""
}
```

Convert each sample into the Llama chat format before tokenization.

---

# Step 4 - Load Base Model

Model

```
meta-llama/Llama-3.2-3B-Instruct
```

Configuration

* 4-bit quantization
* Flash Attention (if available)
* Gradient checkpointing
* Max sequence length: 2048

Reason:

Using 4-bit QLoRA dramatically reduces memory requirements while preserving strong downstream performance.

---

# Step 5 - Apply LoRA

Target modules

```python
q_proj
k_proj
v_proj
o_proj

gate_proj
up_proj
down_proj
```

Configuration

```python
r = 16

lora_alpha = 32

lora_dropout = 0.05

bias = "none"

task_type = "CAUSAL_LM"
```

Save configuration separately.

---

# Step 6 - Configure Training

Recommended configuration

```python
epochs = 2

learning_rate = 2e-4

batch_size = 2

gradient_accumulation_steps = 4

weight_decay = 0.01

warmup_ratio = 0.03

logging_steps = 10

save_steps = 250

evaluation_strategy = "steps"

eval_steps = 250
```

Mixed precision

```
fp16=True
```

---

# Step 7 - Configure W&B

Create project

```
techmentor-llm
```

Track

* train_loss
* eval_loss
* learning_rate
* gpu_memory
* runtime
* epoch
* checkpoint

Also log:

```python
dataset_version

lora_rank

learning_rate

epochs

batch_size
```

Every run should have a meaningful name.

Example

```
llama32-r16-lr2e4-v1
```

---

# Step 8 - Train

Start training.

During training monitor

* Train loss
* Eval loss
* GPU utilization
* ETA
* Samples/sec

Expected runtime

Approximately 30–90 minutes on a Colab T4 depending on dataset size.

---

# Step 9 - Save Adapter

Save

```text
outputs/final_adapter/
```

Include

* adapter_model.safetensors
* adapter_config.json
* tokenizer files

Do **not** merge with the base model yet.

---

# Step 10 - Test Inference

Create

```
inference.ipynb
```

Run prompts from each capability.

### QA

```
Explain dependency injection.
```

### Backend

```
Difference between JWT and OAuth?
```

### SQL

```
What is database indexing?
```

### ML

```
Difference between bagging and boosting.
```

### Code Explanation

Paste a Python function and ask for an explanation.

### Interview Evaluation

```
Question:
What is JWT?

Candidate Answer:
JWT is used for login.

Evaluate this answer.
```

---

# Step 11 - Compare with Base Model

For every prompt store:

```
Base Model Output

↓

Fine-tuned Model Output
```

Create

```
evaluation/before_after.md
```

This will be used later in the README.

---

# Step 12 - Save Training Artifacts

Save

```
training_config.json

lora_config.json

wandb_run_url.txt

requirements.txt
```

---

# Deliverables

By the end of Day 4 you should have:

✅ Fine-tuned LoRA adapter

✅ W&B dashboard

✅ Saved checkpoints

✅ Training logs

✅ Inference notebook

✅ Before vs After comparison

---

# Resume Talking Points

Fine-tuned Llama 3.2 3B Instruct using QLoRA, PEFT, and Unsloth on a custom 8K+ instruction dataset for software engineering interviews.

Reduced trainable parameters by over 99% using LoRA adapters while training on a single NVIDIA T4 GPU.

Tracked hyperparameters, checkpoints, GPU utilization, and training metrics using Weights & Biases.

Evaluated the fine-tuned model against the base model across technical question answering, interview evaluation, code explanation, and follow-up question generation tasks.
