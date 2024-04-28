
import argparse
import json
import torch
from pprint import pprint
from tqdm import tqdm
import random
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import itertools
import json
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import f1_score
import torch.optim as optim
from typing import Tuple, List, Any, Dict
import huggingface_hub
from GED.Reader import tokenizer,encode_sentence,dataset
from GED.Reader import *
from GED.Model import MLP_baseline
from GED.Trainer import Trainer

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device ="cuda"
# load vocabularies
train_path_EN = '/content/drive/MyDrive/multiged-2023/english/en_fce_train.tsv'
dev_path_EN = '/content/drive/MyDrive/multiged-2023/english/en_fce_dev.tsv'

train_path_GR = '/content/drive/MyDrive/multiged-2023/german/de_falko-merlin_train.tsv'
dev_path_GR = '/content/drive/MyDrive/multiged-2023/german/de_falko-merlin_dev.tsv'

train_path_IT = '/content/drive/MyDrive/multiged-2023/italian/it_merlin_train.tsv'
dev_path_IT = '/content/drive/MyDrive/multiged-2023/italian/it_merlin_dev.tsv'

class Main:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-l",action="store_true", help="")
        self.args = parser.parse_args()
        self.leonide = self.args.l
        
    def run(self):
        # English
        data1_1 = dataset(train_path_EN)
        data1_2 = dataset(dev_path_EN) 
        # Espaniol
        data2_1 = dataset(train_path_GR)
        data2_2 = dataset(dev_path_GR) 
        # Francis
        data3_1 = dataset(train_path_IT)
        data3_2 = dataset(dev_path_IT) 

        tokenized_seq1,mask,labels = encode_sentence(data1_1)
        tokenized_seq_dev1,mask_dev1,labels_dev = encode_sentence(data1_2)

        trainingset = LoadDataset(tokenized_seq1,mask,labels)
        devset = LoadDataset(tokenized_seq_dev1,mask_dev1,labels_dev)

        train_dataset = DataLoader(trainingset.encoded_data, batch_size=8, shuffle=True)
        valid_dataset = DataLoader(devset.encoded_data, batch_size=8, shuffle=True)

        main_model_bin = MLP_baseline(params).to(device)
        trainer = Trainer(model = main_model_bin
                  ,loss_function = nn.CrossEntropyLoss(ignore_index=2),
                  optimizer = optim.Adam(main_model_bin.parameters())
                  )
        avg_epoch_loss,train_loss,f1,_,_,_,_,_,_,_,_,_ = trainer.train(train_dataset, valid_dataset, 10)


if __name__ == '__main__':
    main = Main()
    main.run()