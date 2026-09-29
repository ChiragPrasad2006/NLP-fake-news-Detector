from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import Dataset

#model
model_id="sentence-transformers/all-MiniLM-L12-v2"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

#load dataset
full_dataset = pd.read_csv("data/WELFake_Dataset.csv")[["title","text","label"]]
print(len(full_dataset))
print(full_dataset.info())

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(full_dataset)

#split dataset
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(len(split_dataset["train"]))
print(len(split_dataset["test"]))

print(model.config)

#tokenize
max_length = 512
def tokenize_function(examples):
    titles = [t if t is not None else "" for t in examples["title"]]
    texts = [t if t is not None else "" for t in examples["text"]]
    return tokenizer(titles, texts, max_length=max_length, truncation=True, padding="max_length")
# Map tokenization across datasets
tokenized_train_data = split_dataset["train"].map(tokenize_function, batched=True)
tokenized_test_data = split_dataset["test"].map(tokenize_function, batched=True)


#Train/test model
training_args = TrainingArguments(
    output_dir="./models/minilm_finetuned",
    per_device_train_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch"
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data
)

trainer.train()
# 2. Evaluate on the test dataset
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
# 3. Save the final fine-tuned model and tokenizer
model_save_path = "./models/minilm_finetuned/final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved successfully to {model_save_path}")