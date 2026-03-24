# ============================================================
# TRAINING SCRIPT — English Gender Inclusive mT5
# ============================================================

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

BASE_MODEL = "google/flan-t5-small"
OUTPUT_DIR = "./flan_t5_gender_inclusive"
DATA_PATH = "data/A/Train"


# ------------------------------------------------------------
# NORMALIZE COLUMNS
# ------------------------------------------------------------
def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    biased_col = None
    inclusive_col = None

    for col in df.columns:
        if biased_col is None and any(k in col for k in ["non-inclusive", "non inclusive", "biased"]):
            biased_col = col

        if inclusive_col is None and ("inclusive" in col and "non" not in col):
            inclusive_col = col

    if biased_col is None or inclusive_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) < 2:
            raise ValueError(f"Could not detect sentence columns. Found: {df.columns}")

        biased_col, inclusive_col = text_cols[:2]

    print(f"➡ detected columns → biased='{biased_col}', inclusive='{inclusive_col}'")

    df = df.rename(columns={
        biased_col: "biased",
        inclusive_col: "inclusive"
    })

    return df[["biased", "inclusive"]]


# ------------------------------------------------------------
# LOAD ONLY ENGLISH DATA
# ------------------------------------------------------------
def load_data():
    path = os.path.join(DATA_PATH, "English", "SentencePairs.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("English dataset not found!")

    df = pd.read_csv(path).dropna()
    df = normalize_columns(df)

    print(f"✅ Loaded {len(df)} English samples")
    return df


# ------------------------------------------------------------
# PROMPT (IMPROVED)
# ------------------------------------------------------------
def build_prompt(text):
    return f"""Rewrite the following sentence to be gender-inclusive:
{text}
Answer:"""


# ------------------------------------------------------------
# PREPARE DATASET
# ------------------------------------------------------------
def prepare_dataset(df):
    rows = []

    for _, row in df.iterrows():
        rows.append({
            "input_text": build_prompt(row["biased"]),
            "target_text": row["inclusive"]
        })

    return Dataset.from_list(rows)


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------
def tokenize(batch, tokenizer):
    inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    targets = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in targets["input_ids"]
    ]

    inputs["labels"] = labels
    return inputs


# ------------------------------------------------------------
# MAIN TRAIN
# ------------------------------------------------------------
def main():
    print("🚀 Loading data...")
    df = load_data()

    dataset = prepare_dataset(df)

    # ✅ Train/Test split
    dataset = dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    print("🔄 Tokenizing dataset...")
    tokenized_ds = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=8,
    learning_rate=5e-5,
    save_strategy="no",   # 🔥 VERY IMPORTANT
    logging_steps=20,
    report_to="none"
)
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    print("🏋️ Training started...")
    trainer.train()

    print("💾 Saving model...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Training complete! Model saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()