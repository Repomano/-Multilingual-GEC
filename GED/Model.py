from Reader import *
from main import *
import torch.nn.functional as F
import torch
from torch import nn
class HParams():
    vocab_size = len(tokenizer.vocab)
    hidden_dim = 100
    embedding_dim = 768
    num_classes = 2#len(vocab_predicates_EN) 
    bidirectional = True
    freeze = False
    num_layers = 1
    dropout = 0.0
    embeddings = True
    
params = HParams()


class MLP_baseline(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(MLP_baseline, self).__init__()

         #1. ========================== Word embedding layer========================== 
        
        self.word_embedding =  nn.Embedding(hparams.vocab_size, hparams.embedding_dim)

        # USED FOR PRE-TRAIED WMBEDDINGS
        if hparams.embeddings is True:
            print("initializing embeddings from pretrained")
            self.bert = AutoModel.from_pretrained("microsoft/deberta-v3-base") 
            #for param in self.bert.parameters():
            #  param.requires_grad = False
            #self.bert.resize_token_embeddings(len(vocab_tokenizer))
            #self.bert.eval()
        #2. ========================== LSTM definition========================== 
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,batch_first=True,
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(2*hparams.hidden_dim, hparams.num_classes)

    def forward(self, x):
      embeddings = self.bert(x)
      val, (h, c) = self.lstm(embeddings.last_hidden_state)
      o = self.classifier(val.to(device))
      output =  F.log_softmax(o, dim=2)

      # I return the matrix 200x2 of the last layer as trainable flag
      return output.to(device),self.classifier.weight,val