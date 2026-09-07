"""
Fine-tune a BERT model on the IMDB dataset for sentiment analysis and report accuracy.

Uses only 50,000 rows from the dataset (IMDB's full train+test split combined),
re-split 80/20 into train/test for fine-tuning and evaluation.

Requirements:
    pip install torch transformers datasets scikit-learn accelerate
"""

import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score

MODEL_NAME = "bert-base-uncased"
NUM_ROWS = 50000
SEED = 42


def main():
    # Load the full IMDB dataset (25,000 train + 25,000 test = 50,000 rows)
    dataset = load_dataset("imdb")

    # Combine train and test splits, shuffle, then take only NUM_ROWS rows
    combined = concatenate_datasets([dataset["train"], dataset["test"]])
    combined = combined.shuffle(seed=SEED).select(range(min(NUM_ROWS, len(combined))))

    # Re-split into train (80%) and test (20%)
    split = combined.train_test_split(test_size=0.2, seed=SEED)
    train_dataset = split["train"]
    test_dataset = split["test"]

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir="./imdb_bert_output",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    print(f"Accuracy: {results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()
