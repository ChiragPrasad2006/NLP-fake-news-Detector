from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd

#load dataset
full_dataset = pd.read_csv("data/WELFake_Dataset.csv")[["title","text","label"]]
print(len(full_dataset))
print(full_dataset.info())

#split dataset
train_size=int(0.8*len(full_dataset))
test_size=int(len(full_dataset))-train_size
train_data,test_data=torch.utils.data.random_split(full_dataset,[train_size,test_size],generator=torch.Generator().manual_seed(0))
print(len(train_data))
print(len(test_data))
