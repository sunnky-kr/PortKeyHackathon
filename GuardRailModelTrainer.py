import json
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from labels import LABELS

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def load_jsonl(path):
    texts, labels = [], []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(label2id[row["label"]])
    return Dataset.from_dict({"text": texts, "label": labels})


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary" if len(LABELS) == 2 else "macro"
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model():
    train_ds = load_jsonl("train.jsonl")
    val_ds = load_jsonl("val.jsonl")


    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="./guardrail_model_2",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        greater_is_better=True,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./guardrail_model_2")
    tokenizer.save_pretrained("./guardrail_model_2")

    print("✅ Guardrail model trained successfully")
    print("Labels:", LABELS)

if __name__ == "__main__":
    train_model()