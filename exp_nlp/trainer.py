from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, XLNetTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import *
import torch

def train(model, tokenizer, dataset, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):

        print("epochs:", epoch + 1)
        for inputs, labels in dataset:
            
            optimizer.zero_grad()
            inputs = tokenizer(list(inputs), return_tensors="pt", padding=True, truncation=True)
            outputs = model(inputs["input_ids"], inputs["attention_mask"])["logits"]

            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()

            optimizer.step()

def test(model, tokenizer, dataset):
    print("Testing")
    model.eval()
    accuracies = []
    for text, labels in dataset:
        
        
        inputs = tokenizer(list(text), return_tensors="pt", padding=True, truncation=True)
        outputs = model(inputs["input_ids"], inputs["attention_mask"])["logits"]
        #print(inputs["input_ids"].shape, inputs["input_ids"][0, :5], labels)
        #print(inputs["attention_mask"].sum())
        #print(inputs["input_ids"].shape, labels.shape)
        accuracies.append((outputs.argmax(dim=-1).reshape(-1) == labels).float())
    
    acc = torch.cat(accuracies, dim=0)
    #print(acc, acc.shape)
    print("acc:", acc.mean())