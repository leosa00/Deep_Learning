import os
import random

from datasets import load_dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

# ------------------------------------------------------------------
# 1. Load TED talks dataset (EN–ES)
# ------------------------------------------------------------------
print("Loading dataset TankuVie/ted_talks_multilingual_parallel_corpus...")
raw_datasets = load_dataset("TankuVie/ted_talks_multilingual_parallel_corpus")

# This dataset only has a "train" split; we make our own train/validation
full_train = raw_datasets["train"]

# Filter out empty / bad examples to avoid weird losses
def non_empty_example(example):
    en = example.get("en", "")
    es = example.get("es", "")
    return (
        isinstance(en, str)
        and isinstance(es, str)
        and en.strip() != ""
        and es.strip() != ""
    )

full_train = full_train.filter(non_empty_example)

print("Total usable examples:", len(full_train))

splits = full_train.train_test_split(test_size=0.05, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]

print("Train size:", len(train_dataset))
print("Validation size:", len(eval_dataset))

# ------------------------------------------------------------------
# 2. Load model & tokenizer
# ------------------------------------------------------------------
model_name = "google/byt5-small"

print(f"Loading tokenizer and model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Reduce memory
model.config.use_cache = False         
model.gradient_checkpointing_enable()   # save memory (slower but lighter)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

# ------------------------------------------------------------------
# 3. Preprocessing (EN → ES)
# ------------------------------------------------------------------

# Use shorter lengths to avoid OOM & overflow
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

# Small helper if nullcontext isn't available
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            return False

def preprocess_function(batch):
    # Clean texts
    src_texts = []
    tgt_texts = []

    for en, es in zip(batch["en"], batch["es"]):
        if not isinstance(en, str) or en.strip() == "":
            en = " "
        if not isinstance(es, str) or es.strip() == "":
            es = " "

        # Prefix to indicate task
        src_texts.append("translate English to Spanish: " + en)
        tgt_texts.append(es)

    # Encoder inputs (source)
    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    # Decoder labels (target)
    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else nullcontext():
        labels = tokenizer(
            tgt_texts,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
}).map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print(tokenized_datasets)

# ------------------------------------------------------------------
# 4. Data collator
# ------------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
)

# ------------------------------------------------------------------
# 5. Training arguments
# ------------------------------------------------------------------
BATCH_SIZE = 1
EPOCHS = 5

training_args = TrainingArguments(
    output_dir="/dtu/blackhole/1c/222012/byt5-ted-en-es",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=200,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    learning_rate=5e-5,          
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=4,  
    max_grad_norm=1.0,
    fp16=False,                     
    report_to="none",
    remove_unused_columns=False,
    prediction_loss_only=True,
)

# ------------------------------------------------------------------
# 6. Trainer
# ------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train(resume_from_checkpoint=True)

# ------------------------------------------------------------------
# 7. Save model & tokenizer
# ------------------------------------------------------------------
save_path = "./byt5-ted-en-es"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("Model saved to:", save_path)