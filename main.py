import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any, Tuple
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.nn.functional as F
import os
from transformers import BartForConditionalGeneration, TrainingArguments, T5ForConditionalGeneration
import torch.nn.functional as F
import json
import torch
import transformers
from torch import nn
#import tensorflow as tf
from torch.utils.data import Dataset
from pprint import pprint
from tqdm import tqdm
import torch.nn.functional as F
import random
import re
import itertools
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import json
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import huggingface_hub
import spacy
from transformers import BartTokenizer, BartModel, T5Tokenizer 
from pytorch_lightning.loggers import WandbLogger
from encoder_sentence import encode_sentence_Batch, tokenizer_it, tokenizer_en, tokenizer_ru
from batches_management import get_batches_dataset_division
from Model import MLP_arg_classification
from data_extractor import ReaderM2
from Trainer import Model_Correction
import argparse
import MyDataLoader
from MyDataLoader import MyDataModule
import os
#from Trainable_flags import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
device ="cuda"
nlp = spacy.load('it_core_news_sm')
max_len=10
classes_ERRANT={}
torch.cuda.empty_cache()
class Main:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-l",action="store_true", help="Use Leonide")
        parser.add_argument("-b",action="store_true", help="per batch division")
        parser.add_argument("-a",default='train',type=str, help="Action between train/test/dev")
        parser.add_argument("--num_tokens",default=128,type=int, help="number of tokens max for each batch")
        parser.add_argument("-v",action="store_true", help="Verbose print of all txt outputs")
        parser.add_argument("-m",action="store_true", help="masking input concatenated")
        parser.add_argument("-t",default=128,type=int, help="Truncation set")
        parser.add_argument("-e",default=10,type=int, help="Epochs to use")
        parser.add_argument("-lang",default="it",type=str, help="Language train")
        parser.add_argument("--title",default='MBart',type=str, help="Title of the run")
        parser.add_argument("-s",default="",type=str, help="Load a pre trained model")
        parser.add_argument("-c",action="store_true", help="CrossLingual GEC")
        parser.add_argument("-f",default='mask',type=str, help="type of feedback GEC")
        self.args = parser.parse_args()
        self.leonide = self.args.l
        self.batch = self.args.b
        self.todo = self.args.a
        self.verbose = self.args.v
        self.num_tokens = self.args.num_tokens
        self.masking = self.args.m
        self.truncation = self.args.t
        self.epochs= self.args.e
        self.run_title = self.args.title
        self.lang = self.args.lang
        self.saved_model = self.args.s
        self.crosslingual = self.args.c
        self.feedback = self.args.f
        my_model=MLP_arg_classification()
        self.model = Model_Correction(False,my_model,self.verbose,0,self.truncation,self.run_title)
          
    def run(self):
        print(f"The value you passed is leonide={self.leonide},batch={self.batch}")

        # MERLIN
        ##### Tokenization Merlin ####
        if self.batch and not self.leonide:
          #########DEV##########
          
          #data1_1fm = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-dev.m2',
          #          '/media/errant_env/errant/FALKO-MERLIN/fm-dev.trg',classes_ERRANT)
          #data1_1ru = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.M2',
          #          '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.txt',classes_ERRANT)
          #data1_1_fmM = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-dev-GEDLIKE.m2',
          #            '/media/errant_env/errant/FALKO-MERLIN/fm-dev-words.txt',classes_ERRANT)
          #data1_1ruM = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.GEDLIKE.M2',
          #          '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.words.M2',classes_ERRANT)
          #data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev.m2',
          #          '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
          #data1_3M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev_masked_GEDLIKE.m2',
          #    '/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev_words_file.m2',classes_ERRANT)
          data1_1_itM = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_train_masked_GEDLIKE.m2',
                    '/media/errant_env/errant/MERLIN/Merlin/dataset/original_train_words_file.txt',classes_ERRANT)
          #data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.m2',
          #    '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
          #data1_1M_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.masked_GEDLIKE.m2',
          #    '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.words_file.m2',classes_ERRANT)
          data1_2 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test.m2',
              '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
          data1_2M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test_masked_GEDLIKE.m2',
              '/media/errant_env/errant/MERLIN/Merlin/dataset/original_test_words_file.m2',classes_ERRANT)
          
          #########TEST##########
          data1_1fm = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-test.m2',
                    '/media/errant_env/errant/FALKO-MERLIN/fm-test.trg',classes_ERRANT)
          data1_1_fmM = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-test-GEDLIKE.m2',
                      '/media/errant_env/errant/FALKO-MERLIN/fm-test.trg',classes_ERRANT)
          data1_1ruM = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.GEDLIKE.M2',
                    '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.txt',classes_ERRANT)
          data1_1ru = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.M2',
                    '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.txt',classes_ERRANT)
          
          #data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.test.gold.bea19.m2',
          #    '/media/errant_env/errant/BEA/ABCN.test.gold.bea19.txt',classes_ERRANT)
          #data1_1M_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.test.gold.bea19.masked_GEDLIKE.m2',
          #    '/media/errant_env/errant/BEA/ABCN.test.gold.bea19.words_file.m2',classes_ERRANT)
          
          data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test.m2',
                    '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
          data1_3M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test_masked_GEDLIKE.m2',
              '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
          
          #########TRAIN##########
          data1_1 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_train.m2',
                    '/media/errant_env/errant/MERLIN/Merlin/dataset/train.txt',classes_ERRANT)
          data1_1B = ReaderM2('/media/errant_env/errant/FCE/fce.train.gold.bea19.m2',
              '/media/errant_env/errant/FCE/fce.train.correct.bea19.txt',classes_ERRANT)
          data1_1M = ReaderM2('/media/errant_env/errant/FCE/fce.train.gold.bea19.masked_GEDLIKE.m2',
              '/media/errant_env/errant/FCE/fce.train.gold.bea19.words_file.m2',classes_ERRANT)
          data1_1fm_train = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-train.m2',
                    '/media/errant_env/errant/FALKO-MERLIN/fm-train.trg',classes_ERRANT)
          data1_1fm_trainM = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-train-GEDLIKE.m2',
                    '/media/errant_env/errant/FALKO-MERLIN/fm-train.trg',classes_ERRANT)
          data1_1ru_train = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.train.M2',
                    '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.train.txt',classes_ERRANT)
          data1_1ru_trainM = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.train.GEDLIKE.M2',
                    '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.train.txt',classes_ERRANT)
          
          ###################
          if self.masking:
              if self.feedback=="mask":              
                ############### train ###############
                tokenized_seq1,tokenized_corr = encode_sentence_Batch([x for x,y in zip(data1_1fm_train.X_list,data1_1fm_trainM.X_list)]
                                                                    ,data1_1fm_train.X_corr
                                                                    ,"fm")
                tokenized_seq2,tokenized_corr2 = encode_sentence_Batch([x for x,y in zip(data1_1ru_train.X_list,data1_1ru_trainM.X_list)]
                                                                    ,data1_1ru_train.X_corr
                                                                    ,"ru")
                tokenized_seq3,tokenized_corr3 = encode_sentence_Batch([x for x,y in zip(data1_1.X_list,data1_1_itM.X_list)]
                                                                    ,data1_1.X_corr
                                                                    ,"it")
                tokenized_seq4,tokenized_corr4 = encode_sentence_Batch([x for x,y in zip(data1_1B.X_list,data1_1M.X_list)]
                                                                    ,data1_1B.X_corr
                                                                    ,"en")
                # DEV
                tokenized_seq1_WI,tokenized_corr_Wi = encode_sentence_Batch([x for x,y in zip(data1_1B_WI.X_list,data1_1M_WI.X_list)],
                                                                          data1_1B_WI.X_corr
                                                                          ,"en")
                
                tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch([x for x,y in zip(data1_3.X_list,data1_3M.X_list)],
                                                                      data1_3.X_corr
                                                                      ,"it")
                
                tokenized_fm_dev1,tokenized_corr_fm_dev = encode_sentence_Batch([x for x,y in zip(data1_1fm.X_list,data1_1_fmM.X_list)],
                                                                      data1_1fm.X_corr
                                                                      ,"fm")
                
                tokenized_ru_dev1,tokenized_corr_ru_dev = encode_sentence_Batch([x for x,y in zip(data1_1ru.X_list,data1_1ruM.X_list)],
                                                                      data1_1ru.X_corr
                                                                      ,"ru")
                
                tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch([x for x,y in zip(data1_2.X_list,data1_2M.X_list)],
                                                                      data1_2.X_corr,"it")
              """
              elif self.feedback=="compact":
                tokenized_seq1,tokenized_corr = encode_sentence_Batch([x+" <ssep> "+y for x,y in zip(data1_1M.X_list,data1_1M.X_corr)],
                                                                    data1_1B.X_corr,"en")
                tokenized_seq1_WI,tokenized_corr_Wi = encode_sentence_Batch([x+" <ssep> "+y for x,y in zip(data1_1M_WI.X_list,data1_1M_WI.X_corr)],
                                                                          data1_1B_WI.X_corr,"en")
                ###############IT###############
                tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch([x+" <ssep> "+y for x,y in zip(data1_3M.X_list,data1_3M.X_corr)],
                                                                      data1_3.X_corr,"it")
                tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch([x+" <ssep> "+y for x,y in zip(data1_2M.X_list,data1_2M.X_corr)],
                                                                      data1_2.X_corr,"it")
              """
              #print(len(tokenized_seq1['input_ids']),len(tokenized_seq2['input_ids']),len(tokenized_seq3['input_ids']),len(tokenized_seq4['input_ids']))
              #print(len(tokenized_corr['input_ids']),len(tokenized_corr2['input_ids']),len(tokenized_corr3['input_ids']),len(tokenized_corr4['input_ids']))
              in_train=tokenized_seq1['input_ids']+tokenized_seq2['input_ids']+tokenized_seq3['input_ids']+tokenized_seq4['input_ids']
              mask_train=tokenized_seq1['attention_mask']+tokenized_seq2['attention_mask']+tokenized_seq3['attention_mask']+tokenized_seq4['attention_mask']
              in_train_corr=tokenized_corr['input_ids']+tokenized_corr2['input_ids']+tokenized_corr3['input_ids']+tokenized_corr4['input_ids']

              in_dev=tokenized_seq1_WI['input_ids']#+tokenized_seq_dev1['input_ids']+tokenized_fm_dev1['input_ids']+tokenized_ru_dev1['input_ids']#+tokenized_cz_dev1['input_ids']
              mask_dev=tokenized_seq1_WI['attention_mask']#+tokenized_seq_dev1['attention_mask']+tokenized_fm_dev1['attention_mask']+tokenized_ru_dev1['attention_mask']#+tokenized_cz_dev1['attention_mask']
              in_dev_corr=tokenized_corr_Wi['input_ids']#+tokenized_corr_dev['input_ids']+tokenized_corr_fm_dev['input_ids']+tokenized_corr_ru_dev['input_ids']#+tokenized_corr_cz_dev['input_ids']

              in_test=tokenized_seq_test1['input_ids']
              mask_test=tokenized_seq_test1['attention_mask']
              in_test_corr=tokenized_corr_test['input_ids']
                          
              print("#"*30," CHECK IT TRAIN")
              for e in range(len(in_dev)-30,len(in_dev)):
                      print(tokenizer_ru.decode(in_dev[e][0]),
                      tokenizer_ru.decode(in_dev_corr[e][0]))
              
              print("TRAINING SET:",len(in_train))
              dataset1,correct1= get_batches_dataset_division(
                    in_train,
                    in_train_corr,
                    mask_train,
                    tokenized_corr['attention_mask']+tokenized_corr2['attention_mask']+tokenized_corr3['attention_mask']+tokenized_corr4['attention_mask'],
                    self.num_tokens)

              dataset2,correct2= get_batches_dataset_division(
                    in_dev,
                    in_dev_corr,
                    mask_dev,
                    tokenized_corr_dev['attention_mask']+tokenized_corr_Wi['attention_mask']+tokenized_corr_fm_dev['attention_mask']+tokenized_corr_ru_dev['attention_mask'],#+tokenized_corr_cz_dev['attention_mask'],
                    self.num_tokens)
          
              dataset3,correct3= get_batches_dataset_division(
                    in_test,
                    in_test_corr,
                    mask_test,
                    tokenized_corr_test['attention_mask'],
                    self.num_tokens)
              
              data = MyDataModule(
                dataset1['input_ids'][:2],
                correct1['input_ids'][:2],
                dataset1['attention_mask'][:2],
                correct1['attention_mask'][:2],

                dataset2['input_ids'],
                correct2['input_ids'],
                dataset2['attention_mask'],
                correct2['attention_mask'],
                
                dataset3['input_ids'],
                correct3['input_ids'],
                dataset3['attention_mask'],
                correct3['attention_mask'],
                True)
              
        torch.cuda.empty_cache()
        SEED = 1234
        transformers.set_seed(SEED)
        if self.saved_model!="":
          my_model=torch.load(self.saved_model)
        else:
          my_model=MLP_arg_classification()
        if self.todo=="train":
          print("\n\n START TRAINING \n")
          print(len(data.setup()[0]),"\n")
          self.model = Model_Correction(False,my_model,self.verbose,len(data.setup()[0])*self.epochs,self.truncation,self.run_title)
          trainer = pl.Trainer(max_epochs = self.epochs,logger=self.model.wandb_logger ,accelerator="cuda", devices=1)
          trainer.fit(self.model,train_dataloaders= data)
          #torch.save(self.model.model,"pretrained_model_fce.pt")
        elif self.todo=="dev":
          print("\n\n DEV CHECK \n\n")
          self.model = Model_Correction(True,my_model,self.verbose,len(data.setup()[1]),self.truncation,self.run_title)
          trainer = pl.Trainer(max_epochs = 2,logger=self.model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=0.5)
          trainer.validate(self.model, dataloaders=data)
        elif self.todo=="test":
          print("\n\n TEST CHECK \n\n")
          self.model = Model_Correction(False,torch.load('/media/models/backup_epoch.pt'),self.verbose,len(data.setup()[2]),self.truncation,self.run_title)
          trainer = pl.Trainer(max_epochs = 1,logger=self.model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=1)
          trainer.test(self.model, dataloaders=data)
        else:
           print(f"No existing action for {self.todo}")

if __name__ == '__main__':
    main = Main()
    main.run()
    wandb.finish()