import csv
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
from typing import Optional
import os
import shlex
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
tokenizer =  AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def dataset(path):
  sentences=[]
  labels=[]
  column_1 = []
  column_2 = []
  print("opening... ",path)
  with open(path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
      row="".join(row)
      e=row.split("\t")
      if e[0]!='':
        column_1.append(e[0])
        if str(e[1])=="i":
          column_2.append(1)
        elif str(e[1])=="c":
          column_2.append(0)
      else:
        sentences.append(" ".join(column_1))
        labels.append(column_2)
        column_1 = []
        column_2 = []
          
  return sentences,labels

def encode_sentence(x):
  sentences=tokenizer(x[0], return_tensors='pt',padding=True)
  max=sentences["input_ids"].shape[-1]
  labels=[]
  for elem in x[-1]:
    tokens=[]
    if len(elem)<max:
      tokens=elem+[2]*(max-len(elem))
      labels.append(tokens)
  print(sentences["input_ids"].shape,len(labels[13]))
  return sentences["input_ids"],sentences["attention_mask"],labels


class LoadDataset(Dataset):
    def __init__(self, sentences,labels_binary,labels):
        self.data = sentences 
        self.encoded_data = None
        self.labels = labels
        self.labels_binary=labels_binary
        self.index_dataset()

    def index_dataset(self):
        self.encoded_data = list()
        for i in range(self.data.shape[0]):
          self.encoded_data.append({"inputs":self.data[i],"mask":self.labels_binary[i], "task2":self.labels[i]})