"""
train_bert_reviews.py
────────────────────────────────────────────────────────────

Train a BERT model on laptop review dataset
for sentiment classification.

Dataset expected:
data/laptop_reviews.csv

Columns required:
- review_text
- review_rating

Output:
- bert_laptop_sentiment_model/
- label_encoder.pkl

Install:
pip install torch transformers datasets scikit-learn pandas tqdm

Run:
python train_bert_reviews.py
"""

import os
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

import numpy as np
import pickle


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_PATH = "data/laptop_reviews.csv"

MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = "bert_laptop_sentiment_model"

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5


# ─────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape: {df.shape}")

# Keep only needed columns

df = df[[
    "review_text",
    "review_rating"
]]

# Remove nulls

df.dropna(inplace=True)

# Convert ratings to sentiment labels

def convert_rating_to_sentiment(rating):

    """
    Convert review ratings into sentiment labels

    1-2 → negative
    3   → neutral
    4-5 → positive
    """

    if rating <= 2:
        return "negative"

    elif rating == 3:
        return "neutral"

    else:
        return "positive"


df["sentiment"] = df["review_rating"].apply(
    convert_rating_to_sentiment
)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())


# ─────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["sentiment"]
)

print("\nLabel Mapping:")

for idx, label in enumerate(label_encoder.classes_):
    print(f"{label} → {idx}")

# Save label encoder

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


# ─────────────────────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────────────────────

train_texts, test_texts, train_labels, test_labels = train_test_split(

    df["review_text"].tolist(),

    df["label"].tolist(),

    test_size=0.2,

    random_state=42,

    stratify=df["label"]
)


# ─────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────

print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize(batch):

    return tokenizer(

        batch["text"],

        padding="max_length",

        truncation=True,

        max_length=MAX_LENGTH
    )


# ─────────────────────────────────────────────────────────────
# CREATE HF DATASETS
# ─────────────────────────────────────────────────────────────

train_dataset = Dataset.from_dict({

    "text": train_texts,

    "label": train_labels
})

test_dataset = Dataset.from_dict({

    "text": test_texts,

    "label": test_labels
})

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# PyTorch format

train_dataset.set_format(

    type="torch",

    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

test_dataset.set_format(

    type="torch",

    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

print("\nLoading BERT model...")

model = BertForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=len(label_encoder.classes_)
)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc
    }


# ─────────────────────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=10,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True
)


# ─────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRAINING BERT MODEL")
print("=" * 60)

trainer.train()


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

predictions = trainer.predict(test_dataset)

preds = np.argmax(
    predictions.predictions,
    axis=-1
)

accuracy = accuracy_score(
    test_labels,
    preds
)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(

    test_labels,

    preds,

    target_names=label_encoder.classes_
))


# ─────────────────────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────────────────────

print("\nSaving model...")

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nSaved model to: {OUTPUT_DIR}")
print("Saved label encoder: label_encoder.pkl")


# ─────────────────────────────────────────────────────────────
# SAMPLE PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_review(review_text):

    inputs = tokenizer(

        review_text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    sentiment = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return sentiment


# Example

sample_review = """
Battery life is amazing and performance is super smooth.
"""

prediction = predict_review(sample_review)

print("\nSample Review Prediction:")
print(sample_review)
print(f"Predicted Sentiment: {prediction}")