import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any, Tuple
class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                x_train,x_train_corr,y_train,y_train_corr,
                x_dev,x_dev_corr,y_dev,y_dev_corr,
                x_test,x_test_corr,y_test,y_test_corr,
                #train_flags_batch,
                #test_it_flag_batch,
                #dev_flag_batch,
                per_batch
                 ):
      
        super().__init__()
        self.y_train = y_train
        self.y_test = y_test
        self.y_dev = y_dev
        self.x_train = x_train
        self.x_test = x_test
        self.x_dev = x_dev
        self.x_train_corr=x_train_corr
        self.x_test_corr=x_test_corr
        self.x_dev_corr=x_dev_corr
        self.y_train_corr=y_train_corr
        self.y_test_corr=y_test_corr
        self.y_dev_corr=y_dev_corr
        self.per_batch=per_batch
        #self.train_flags_batch=train_flags_batch
        #self.dev_flag_batch=dev_flag_batch
        #self.test_it_flag_batch=test_it_flag_batch
        self.trainingset=[]
        self.testset=[]
        self.devset=[]
        
    def setup(self, stage: Optional[str] = None):
      #self.trainingset = [(x,y,z,w,u) for x,y,z,w,u in zip(self.x_train,self.x_train_corr,self.y_train,self.y_train_corr,self.train_flags_batch)] 
      #self.testset = [(x,y,z,w,u) for x,y,z,w,u in zip(self.x_test,self.x_test_corr,self.y_test,self.y_test_corr,self.test_it_flag_batch)] 
      #self.devset = [(x,y,z,w,u) for x,y,z,w,u in zip(self.x_dev,self.x_dev_corr,self.y_dev,self.y_dev_corr,self.dev_flag_batch)] 
      
      self.trainingset = [(x,y,z,w) for x,y,z,w in zip(self.x_train,self.x_train_corr,self.y_train,self.y_train_corr)] 
      self.testset = [(x,y,z,w) for x,y,z,w in zip(self.x_test,self.x_test_corr,self.y_test,self.y_test_corr)] 
      self.devset = [(x,y,z,w) for x,y,z,w in zip(self.x_dev,self.x_dev_corr,self.y_dev,self.y_dev_corr)] 
      
      return self.trainingset,self.testset,self.devset
    
    
    def train_dataloader(self):
      if not self.per_batch:
        return DataLoader(self.trainingset, batch_size=8)
      else:
        return DataLoader(self.trainingset, batch_size=None)

    def val_dataloader(self):
      if not self.per_batch:
        return DataLoader(self.devset, batch_size=8)
      else:
        return DataLoader(self.devset, batch_size=None)

    def test_dataloader(self):
      if not self.per_batch:
        return DataLoader(self.testset, batch_size=8)
      else:
        return DataLoader(self.testset, batch_size=None)