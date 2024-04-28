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
#import sklearn
#from sklearn.metrics import precision_score as sk_precision
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
#from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import huggingface_hub
import spacy
from transformers import BartTokenizer, BartModel, T5Tokenizer 
from pytorch_lightning.loggers import WandbLogger
from encoder_sentence import encode_sentence_Batch,encode_sentence, tokenizer
from batches_management import get_batches_dataset_division
from Model import MLP_arg_classification
from data_extractor import ReaderM2
from Trainer import Model_Correction
import argparse
import MyDataLoader
from MyDataLoader import MyDataModule
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
device ="cuda"
#***---> READ M2***
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
        parser.add_argument("-t",default=80,type=int, help="Truncation set")
        parser.add_argument("-e",default=10,type=int, help="Epochs to use")
        parser.add_argument("-lang",default="it",type=str, help="Language train")
        parser.add_argument("--title",default='MBart',type=str, help="Title of the run")
        parser.add_argument("-s",default="",type=str, help="Load a pre trained model")
        self.args = parser.parse_args()
        self.leonide = self.args.l
        self.batch = self.args.b
        self.todo = self.args.a
        self.verbose = self.args.v
        self.num_tokens = self.args.num_tokens
        self.masking = self.args.m
        self.sep_token = tokenizer.convert_tokens_to_ids("</s>")
        self.truncation = self.args.t
        self.epochs= self.args.e
        self.run_title = self.args.title
        self.lang = self.args.lang
        self.saved_model = self.args.s
        my_model=MLP_arg_classification()
        self.model = Model_Correction(False,my_model,self.verbose,0,self.truncation,self.run_title)
          

    def run(self):
        print(f"The value you passed is leonide={self.leonide},batch={self.batch}")
        ###### M2 Reader ######
        # MERLIN
        
        #DataLoader used for train/test/dev on Merlin
        #data1_1 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_train.m2',
        #         '/media/errant_env/errant/MERLIN/Merlin/dataset/train.txt',classes_ERRANT)
        #data1_2 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test.m2',
        #           '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
        #data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev.m2',
        #         '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
        #torch.save(data1_1,'/media/data/train.pt')
        #torch.save(data1_2,'/media/data/test.pt')
        #torch.save(data1_3,'/media/data/dev.pt')
        
        """
        ###### LEONIDE ######
        #DataLoader used for train/test/dev on Leonide
        data1_1L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_train.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_train.txt',classes_ERRANT)
        data1_2L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_test.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_test.txt',classes_ERRANT)
        data1_3L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_dev.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_dev.txt',classes_ERRANT)
        torch.save(data1_1L,'/media/data/trainL.pt')
        torch.save(data1_2L,'/media/data/testL.pt')
        torch.save(data1_3L,'/media/data/devL.pt')
        """

        # MERLIN
        ##### Tokenization Merlin ####
        if self.batch and not self.leonide:############################################### MASKING ################################
          
          #data1_1 =torch.load('/media/data/train.pt')
          #data1_1.Y_list=[]
          data1_2 =torch.load('/media/data/test.pt')
          data1_2.Y_list=[]
          data1_3 =torch.load('/media/data/dev.pt')
          data1_3.Y_list=[]
          print(f"1. NOT Leonide and batches division  ")

          if self.masking:
              if self.lang=="en":
                data1_1B = ReaderM2('/media/errant_env/errant/BEA/WILOC_FCENucle.m2',
                 '/media/errant_env/errant/BEA/WILOC_FCENucle.txt',classes_ERRANT)
                data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.m2',
                 '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
                print("ENGLISH ACTIVATED...")
                ###### MASKED ENGLISH FCE/BEA ######
                #data1_1M = ReaderM2('/media/errant_env/errant/Fce_only/fce.train.gold.bea19_mask.m2',
                # '/media/errant_env/errant/Fce_only/fce.train.correct.bea19.txt',classes_ERRANT)
                data1_1M = ReaderM2('/media/errant_env/errant/BEA/WILOC_FCENucle_masked.m2',
                 '/media/errant_env/errant/BEA/WILOC_FCENucle.txt',classes_ERRANT)
                data1_1M_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.masked.m2',
                 '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
                data1_2M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test_masked.m2',
                  '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
                data1_3M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev_masked.m2',
                 '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
                torch.save(data1_1M,'/media/data/trainM.pt')
                torch.save(data1_2M,'/media/data/testM.pt')
                torch.save(data1_3M,'/media/data/devM.pt')
                
              elif self.lang=="it":
                ###### MASKED ######
                data1_1M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_train_masked.m2',
                 '/media/errant_env/errant/MERLIN/Merlin/dataset/train.txt',classes_ERRANT)
                data1_2M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test_masked.m2',
                   '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
                data1_3M = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev_masked.m2',
                 '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
                torch.save(data1_1M,'/media/data/trainM.pt')
                torch.save(data1_2M,'/media/data/testM.pt')
                torch.save(data1_3M,'/media/data/devM.pt')

              """#***---> Loads***"""
              data1_1M =torch.load('/media/data/trainM.pt')
              data1_1M.Y_list=[]
              data1_2M =torch.load('/media/data/testM.pt')
              data1_2M.Y_list=[]
              data1_3M =torch.load('/media/data/devM.pt')
              data1_3M.Y_list=[]

              if self.lang=="en":
                tokenizer.src_lang="en_XX"
                tokenizer.tgt_lang="en_XX"
              ###############EN###############
              tokenizer.src_lang="en_XX"
              tokenizer.tgt_lang="en_XX"
              #TRAIN
              tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1B.X_list,data1_1B.X_corr)
              ts1,_ = encode_sentence_Batch(data1_1M.X_list,data1_1M.X_corr)
              #DEV ENGLISH
              tokenized_seq1_WI,tokenized_corr_Wi = encode_sentence_Batch(data1_1B_WI.X_list,data1_1B_WI.X_corr)
              tokenized_seq1_WI_mask,_ = encode_sentence_Batch(data1_1M_WI.X_list,data1_1M_WI.X_corr)

              ###############IT###############

              tokenizer.src_lang="it_IT"
              tokenizer.tgt_lang="it_IT"
              # DEV/TEST ITALIAN
              tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch(data1_3.X_list,data1_3.X_corr)
              tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch(data1_2.X_list,data1_2.X_corr)
              tsd1,_ = encode_sentence_Batch(data1_3M.X_list,data1_3M.X_corr)
              tst1,_ = encode_sentence_Batch(data1_2M.X_list,data1_2M.X_corr)
            
              #in_train=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq1['input_ids'],ts1['input_ids'])]
              #mask_train=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq1['attention_mask'],ts1['attention_mask'])]
              #in_dev=[torch.cat((x,torch.tensor([[self.sep_token]]),y),
              #                  -1) for x,y in zip(tokenized_seq_dev1['input_ids'],
              #                                     tsd1['input_ids'])]+[torch.cat((x,torch.tensor([[self.sep_token]]),y),
              #                                                                    -1) for x,y in zip(tokenized_seq1_WI['input_ids'],
              #                                                                                       tokenized_seq1_WI_mask['input_ids'])]
              #mask_dev=[torch.cat((x,torch.tensor([[self.sep_token]]),y),
              #                    -1) for x,y in zip(tokenized_seq_dev1['attention_mask'],
              #                                       tsd1['attention_mask'])]+[torch.cat((x,torch.tensor([[self.sep_token]]),y),
              #                                                                           -1) for x,y in zip(tokenized_seq1_WI['attention_mask'],
              #                                                                                              tokenized_seq1_WI_mask['attention_mask'])]

              #in_test=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_test1['input_ids'],tst1['input_ids'])]
              #mask_test=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_test1['attention_mask'],tst1['attention_mask'])]
              #okkk
              print(len(tokenized_seq1['input_ids']),len(ts1['input_ids']),len(tokenized_corr['input_ids']))
              for e in range(len(in_train)-30,len(in_train)):
                print(tokenizer.decode(in_train[e][0]),
                      tokenizer.decode(tokenized_corr['input_ids'][e][0]))
              #ok
              #for e in range(len(tokenized_seq1['input_ids'][0])-50,len(tokenized_seq1['input_ids'][0])):
              #  print(tokenizer.decode(tokenized_seq1['input_ids'][e][0]),
              #        tokenizer.decode(ts1['input_ids'][e][0]),
              #        tokenizer.decode(tokenized_corr['input_ids'][e][0]))
              #  print("\n")
              #print("###############")
              #for e in range(len(tokenized_seq1_WI['input_ids'][0])-40,len(tokenized_seq1_WI_mask['input_ids'][0])-20):
              #  print(tokenizer.decode(tokenized_seq1_WI['input_ids'][e][0]),
              #        tokenizer.decode(tokenized_seq1_WI_mask['input_ids'][e][0]),
              #        tokenizer.decode(tokenized_corr_Wi['input_ids'][e][0]))
              #  print()
              #print("###############")

          else:
            ############################################### NO MASKING ################################
            if self.lang=="en":
              ####### FCE/BEA #######
              #data1_1B = ReaderM2('/media/errant_env/errant/Fce_only/fce.train.gold.bea19.m2',
              #     '/media/errant_env/errant/Fce_only/fce.train.correct.bea19.txt',classes_ERRANT)
              data1_1B = ReaderM2('/media/errant_env/errant/BEA/WILOC_FCENucle.m2',
                 '/media/errant_env/errant/BEA/WILOC_FCENucle.txt',classes_ERRANT)
              data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.m2',
                 '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
              #data1_2B = ReaderM2('/media/errant_env/errant/Fce_only/fce.dev.gold.bea19.m2',
              #       '/media/errant_env/errant/Fce_only/fce.dev.correct.bea19.txt',classes_ERRANT)
              #data1_3B = ReaderM2('/media/errant_env/errant/Fce_only/fce.test.gold.bea19.m2',
              #     '/media/errant_env/errant/Fce_only/fce.test.correct.bea19.txt',classes_ERRANT)
              torch.save(data1_1B,'/media/data/trainB.pt')
              #torch.save(data1_2B,'/media/data/testB.pt')
              #torch.save(data1_3B,'/media/data/devB.pt')

              tokenizer.src_lang="en_XX"
              tokenizer.tgt_lang="en_XX"
              print("ENGLISH ACTIVATED...")
              tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1B.X_list,data1_1B.X_corr)
              tokenized_seq1_WI,tokenized_corr_Wi = encode_sentence_Batch(data1_1B_WI.X_list,data1_1B_WI.X_corr)

            else:
              print("ITALIAN ACTIVATED...")
              tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1.X_list,data1_1.X_corr)

            tokenizer.src_lang="it_IT"
            tokenizer.tgt_lang="it_IT"
            tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch(data1_3.X_list,data1_3.X_corr)
            tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch(data1_2.X_list,data1_2.X_corr)


            in_train=tokenized_seq1['input_ids']
            mask_train=tokenized_seq1['attention_mask']
            in_dev=tokenized_seq_dev1['input_ids']
            mask_dev=tokenized_seq_dev1['attention_mask']
            in_test=tokenized_seq_test1['input_ids']
            mask_test=tokenized_seq_test1['attention_mask']


          ############# Per Batch Division #############
          print(tokenizer.decode(tokenized_seq_dev1['input_ids'][-1][0]),
                tokenizer.decode(tsd1['input_ids'][-1][0]),
                tokenizer.decode(tokenized_corr_dev['input_ids'][-1][0]))
          #print(len(tokenized_seq1_WI['input_ids']),len(tokenized_seq1_WI_mask['input_ids']),len(tokenized_corr_Wi['input_ids']))
          #print(len(tokenized_corr_dev['input_ids']+tokenized_corr_Wi['input_ids']))
          #print(len(tokenized_corr_dev['attention_mask']+tokenized_corr_Wi['attention_mask']))
          #print(len(in_dev))
          #print(len(mask_dev))
          dataset1,correct1= get_batches_dataset_division(
                    in_train,
                    tokenized_corr['input_ids'],
                    mask_train,
                    tokenized_corr['attention_mask'],self.num_tokens)
          dataset2,correct2= get_batches_dataset_division(
                    in_dev,
                    tokenized_corr_dev['input_ids']+tokenized_corr_Wi['input_ids'],
                    mask_dev,
                    tokenized_corr_dev['attention_mask']+tokenized_corr_Wi['attention_mask'],self.num_tokens)
          dataset3,correct3= get_batches_dataset_division(
                    in_test,
                    tokenized_corr_test['input_ids'],
                    mask_test,
                    tokenized_corr_test['attention_mask'],self.num_tokens)
          
          data = MyDataModule(
                dataset1['input_ids'],
                correct1['input_ids'],
                dataset1['attention_mask'],
                correct1['attention_mask'],

                dataset2['input_ids'],
                correct2['input_ids'],
                dataset2['attention_mask'],
                correct2['attention_mask'],
                
                dataset3['input_ids'],
                correct3['input_ids'],
                dataset3['attention_mask'],
                correct3['attention_mask'],
                True)
        
        ###################################################
        """
        elif not self.batch and not self.leonide:
          
        
          data1_1 =torch.load('/media/data/train.pt')
          data1_1.Y_list=[]
          data1_2 =torch.load('/media/data/test.pt')
          data1_2.Y_list=[]
          data1_3 =torch.load('/media/data/dev.pt')
          data1_3.Y_list=[]
          print(f"2. NOT Leonide and NOT batches division  ")
          ##### no batch ####
          tokenized_seq1,tokenized_corr = encode_sentence(data1_1.X_list,data1_1.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence(data1_2.X_list,data1_2.X_corr)
          #############Original Batch#############
          data = MyDataModule(tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],

                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],

                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],
                    False)
        
        # MERLIN + LEONIDE
        ##### Tokenization Leonide ####
        elif self.batch and self.leonide:
          print(f"3. Use Leonide and batches division ")

          
          
          data1_1L =torch.load('/media/data/trainL.pt')
          data1_1L.Y_list=[]
          data1_2L =torch.load('/media/data/testL.pt')
          data1_2L.Y_list=[]
          data1_3L =torch.load('/media/data/devL.pt')
          data1_3L.Y_list=[]
          ##### batch ####
          tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1.X_list+data1_1L.X_list+data1_2L.X_list+data1_3L.X_list,
                                                                data1_1.X_corr+data1_1L.X_corr+data1_2L.X_corr+data1_3L.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch(data1_2.X_list,data1_2.X_corr)
          # MASK LEONIDE DA TERMINARE!!!
          
          if self.masking:
             ts1,_ = encode_sentence_Batch(data1_1M.X_list,data1_1.X_corr)
             tsd1,_ = encode_sentence_Batch(data1_3M.X_list,data1_3.X_corr)
             tst1,_ = encode_sentence_Batch(data1_2M.X_list,data1_2.X_corr)
             print(len(tsd1['input_ids']),len(tokenized_seq_dev1['input_ids']))
             in_train=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq1['input_ids'],ts1['input_ids'])]
             mask_train=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq1['attention_mask'],ts1['attention_mask'])]

             in_dev=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_dev1['input_ids'],tsd1['input_ids'])]
             mask_dev=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_dev1['attention_mask'],tsd1['attention_mask'])]

             in_test=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_test1['input_ids'],tst1['input_ids'])]
             mask_test=[torch.cat((x,torch.tensor([[self.sep_token]]),y),-1) for x,y in zip(tokenized_seq_test1['attention_mask'],tst1['attention_mask'])]
          else:
             in_train=tokenized_seq1['input_ids']
             mask_train=tokenized_seq1['attention_mask']
             in_dev=tokenized_seq_dev1['input_ids']
             mask_dev=tokenized_seq_dev1['attention_mask']
             in_test=tokenized_seq_test1['input_ids']
             mask_test=tokenized_seq_test1['attention_mask']
          
          in_train=tokenized_seq1['input_ids']
          mask_train=tokenized_seq1['attention_mask']
          in_dev=tokenized_seq_dev1['input_ids']
          mask_dev=tokenized_seq_dev1['attention_mask']
          in_test=tokenized_seq_test1['input_ids']
          mask_test=tokenized_seq_test1['attention_mask']
          ############# Per Batch Division #############
          dataset1,correct1= get_batches_dataset_division(
                    in_train,
                    tokenized_corr['input_ids'],
                    mask_train,
                    tokenized_corr['attention_mask'],self.num_tokens)
          dataset2,correct2= get_batches_dataset_division(
                    in_dev,
                    tokenized_corr_dev['input_ids'],
                    mask_dev,
                    tokenized_corr_dev['attention_mask'],self.num_tokens)
          dataset3,correct3= get_batches_dataset_division(
                    in_test,
                    tokenized_corr_test['input_ids'],
                    mask_test,
                    tokenized_corr_test['attention_mask'],self.num_tokens)
          data = MyDataModule(
                dataset1['input_ids'],
                correct1['input_ids'],
                dataset1['attention_mask'],
                correct1['attention_mask'],
                dataset2['input_ids'],
                correct2['input_ids'],
                dataset2['attention_mask'],
                correct2['attention_mask'],
                dataset3['input_ids'],
                correct3['input_ids'],
                dataset3['attention_mask'],
                correct3['attention_mask'],
                True)
          
        elif self.leonide and not self.batch:
          print(f"4. Use Leonide but NOT batches division ")
          ##### no batch ####
          
          data1_1L =torch.load('/media/data/trainL.pt')
          data1_1L.Y_list=[]
          data1_2L =torch.load('/media/data/testL.pt')
          data1_2L.Y_list=[]
          data1_3L =torch.load('/media/data/devL.pt')
          data1_3L.Y_list=[]
          tokenized_seq1,tokenized_corr = encode_sentence(data1_1.X_list+data1_1L.X_list+data1_2L.X_list+data1_3L.X_list,
                                                          data1_1.X_corr+data1_1L.X_corr+data1_2L.X_corr+data1_3L.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence(data1_2.X_list,data1_2.X_corr)
          #############Original Batch#############
          data = MyDataModule(tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],
                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],
                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],
                    False)
        """
        torch.cuda.empty_cache()
        SEED = 1234
        #random.seed(SEED)
        #np.random.seed(SEED)
        transformers.set_seed(SEED)
        torch.manual_seed(SEED)
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
          torch.save(self.model.model,"pretrained_model_fce.pt")
        elif self.todo=="dev":
          print("\n\n DEV CHECK \n\n")
          self.model = Model_Correction(True,torch.load('/media/models/backup_epoch.pt'),self.verbose,len(data.setup()[1]),self.truncation,self.run_title)
          trainer = pl.Trainer(max_epochs = 1,logger=self.model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=1)
          trainer.validate(self.model, dataloaders=data)
        elif self.todo=="test":
          print("\n\n TEST CHECK \n\n")
          self.model = Model_Correction(False,torch.load('/media/models/backup_epoch.pt'),self.verbose,len(data.setup()[2]),self.truncation,self.run_title)
          trainer = pl.Trainer(max_epochs = 1,logger=self.model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=1)
          trainer.test(self.model, dataloaders=data)
        else:
           print(f"No existing action for {self.todo}")
        wandb.finish()

if __name__ == '__main__':
    main = Main()
    main.run()