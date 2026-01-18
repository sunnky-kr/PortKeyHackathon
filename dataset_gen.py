import pandas as pd
from sklearn.model_selection import train_test_split
import json

# -----------------------------
# Load HF dataset parquet
# -----------------------------
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
}

df = pd.read_parquet("hf://datasets/jayavibhav/prompt-injection/" + splits["train"])
print("Original DF:")
print(df.head())

# -----------------------------
# Map labels (0 = SAFE, 1 = PROMPT_INJECTION)
# -----------------------------
label_map = {
    0: "SAFE",
    1: "PROMPT_INJECTION"
}

df["label"] = df["label"].map(label_map)

# Drop null labels if any unexpected values exist
df = df.dropna(subset=["label"])

# Keep only required columns
df = df[["text", "label"]]

print("\nConverted DF:")
print(df.head())

# -----------------------------
# Train/Val split
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

print("\nTrain size:", len(train_df))
print("Val size:", len(val_df))

# -----------------------------
# Save as JSONL
# -----------------------------
train_path = "./train1.jsonl"
val_path = "./val1.jsonl"

train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)
val_df.to_json(val_path, orient="records", lines=True, force_ascii=False)

print("\n✅ Saved:")
print(train_path)
print(val_path)