import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

LABEL2ID = {
    "SAFE": 0,
    "PROMPT_INJECTION": 1,
    "PII": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -------------------------
# Load Dataset
# -------------------------
dataset = load_dataset("json", data_files={
    "train": "train_kaggle.jsonl",
    "validation": "val_kaggle.jsonl"
})

def preprocess(example):
    example["label_id"] = LABEL2ID[example["label"]]
    return example

dataset = dataset.map(preprocess)

# -------------------------
# Load Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    out["labels"] = example["label_id"]
    return out

dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

# -------------------------
# Load Model in 4-bit
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    # quantization_config=bnb_config,
    device_map="auto"
)

# model = prepare_model_for_kbit_training(model)

# -------------------------
# Add LoRA
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# Train Args
# -------------------------
training_args = TrainingArguments(
    output_dir="./qwen3guard_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()

# -------------------------
# Save Model
# -------------------------
trainer.save_model("./qwen3guard_finetuned")
tokenizer.save_pretrained("./qwen3guard_finetuned")

print("✅ Fine-tuning complete. Saved at ./qwen3guard_finetuned")