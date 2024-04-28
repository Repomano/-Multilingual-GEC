import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import wandb
import spacy
from pytorch_lightning.loggers import WandbLogger
from data_extractor import *
import errant_env.errant.errant as errant
from encoder_sentence import tokenizer_en, tokenizer_it, tokenizer_fm, tokenizer_ru, tokenizer_cz
device="cuda"

class Model_Correction(pl.LightningModule):
    def __init__(self,test,model_main,verbose_m2,num_batches,truncation,run_title, embeddings = None, *args, **kwargs):
        super(Model_Correction, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        #WandB
        #Train parameters
        self.model = model_main
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
        #Parameters
        self.tot_loss = []
        self.my_predictions= []
        self.my_labels= []
        self.my_inputs= []
        self.dataset_written=False
        self.validation_predictions = []
        self.testing_predictions= []
        self.max_length=truncation
        self.test=test
        self.outputs=[]
        self.lr = 5e-6
        self.scheduler=None
        self.verbose_m2=verbose_m2
        self.accumulate_lr1=1
        self.accumulate_lr2=1
        self.f05_monitor=0
        self.accumulate_lr3=1
        self.num_batches=num_batches
        self.myloss=10
        self.myloss_val=10
        self.mystep=0

        """
        self.data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev.m2',
                     '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
        self.data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.m2',
               '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
        self.data1_1fm = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-dev.m2',
                     '/media/errant_env/errant/FALKO-MERLIN/fm-dev.trg',classes_ERRANT)
        self.data1_1ru = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.M2',
                     '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.dev.txt',classes_ERRANT)
        """

        self.data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test.m2',
                    '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
        self.data1_1B_WI = ReaderM2('/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.m2',
              '/media/errant_env/errant/BEA/ABCN.dev.gold.bea19.txt',classes_ERRANT)
        self.data1_1fm = ReaderM2('/media/errant_env/errant/FALKO-MERLIN/fm-test.m2',
                    '/media/errant_env/errant/FALKO-MERLIN/fm-test.trg',classes_ERRANT)
        self.data1_1ru = ReaderM2('/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.M2',
                    '/media/errant_env/errant/RULEC-GEC/RULEC-GEC.test.txt',classes_ERRANT)
        
        wandb.init(project='baseline-bart-base', name=run_title)
        self.wandb_logger = WandbLogger(project='baseline-bart-base', log_every_n_steps=1)
        
    
    def compute_loss(self, logits,mask, batch,mask_lbl): 
      hidden_states,loss = self.model.forward(logits,mask,batch)
      return hidden_states , loss

    def forward(self, inputs,mask,labels,mask_lbl):
      logits, loss=self.compute_loss(inputs,mask,labels,mask_lbl)
      return logits, loss
      
    def training_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]
        #position=batch[4]
        logits  ,loss = self.forward(inputs,mask,labels,mask_lbl)
        #print("train step--->",logits.shape,labels.shape,logits.view(-1, logits.shape[-1]).shape,labels.view(-1).shape)
        self.myloss = self.criterion(logits.view(-1, logits.shape[-1]),labels.view(-1))
        param=self.scheduler.state_dict()
        param["learning_rate"]=self.scheduler.get_last_lr()[0]
        param["step"]=self.global_step
        wandb.log(param)
        wandb.log({'train_loss': self.myloss,'epoch':self.current_epoch,'step':self.global_step})
        #self.log('train_loss', self.myloss, prog_bar=True, on_step=True, on_epoch=True)
        self.mystep+=1
        self.scheduler.step()
        return self.myloss

    @torch.no_grad()
    def evaluation(self, inputs,mask,labels,mask_lbl):
        output = self.model.generate(inputs,mask,self.max_length)
        logits, loss =self.compute_loss(inputs,mask,labels,mask_lbl)
        #myloss=loss
        self.myloss_val = self.criterion(logits.view(-1, logits.shape[-1]),labels.view(-1))
        if self.test:
          self.testing_predictions.append(output)
        else:
          self.validation_predictions.append(output)
        return self.myloss_val
    
    def validation_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]
        #position=batch[4]
        loss = self.evaluation(inputs,mask,labels,mask_lbl)
        wandb.log({"valid_loss": loss,"epoch": self.current_epoch})
        #self.log('valid_loss', loss, prog_bar=True, on_epoch=True)
        self.outputs.append((labels,inputs))
        return loss
 
    def test_step(self, batch, batch_nb):
        inputs = batch[0]
        labels = batch[1]
        mask = batch[2]
        mask_lbl = batch[3]
        #position=batch[4]
        output = self.model.generate(inputs,mask,self.max_length)
        loss = self.evaluation(inputs,mask,labels,mask_lbl)
        return {'log': inputs}

    def configure_optimizers(self):
        optimizer=torch.optim.RAdam(self.parameters(),lr=self.lr, betas=(0.9, 0.998),eps=1e-08,foreach=False)
        print("steps ",self.num_batches)
        #Schedulers
        #RU
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=35000,start_factor=1.0,end_factor= 0.0002)
        #FM
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=200000,start_factor=1.0,end_factor= 0.0002)
        #EN
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=280380,start_factor=1.0,end_factor= 0.0002)
        #IT
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=60000,start_factor=1.0,end_factor= 0.0002)
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=590380,start_factor=1.0,end_factor= 0.0002)
        #self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=300380,start_factor=1.0,end_factor= 0.0002)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'epoch',  # Adjust the lr at the end of each epoch
                'frequency': 1  # How many epochs to wait before adjusting the lr
            }
        }
    
    def on_train_epoch_end(self):
      self.scheduler.step()
      wandb.log({'train_loss': self.myloss,'epoch':self.current_epoch,'step':self.global_step})
      
    def get_predictions_on_end(self,out,pred):
        predictions=[]
        lbl=[]
        inp=[]
        for elem,batchs in zip(out,pred):
          for sent in batchs:
            predictions.append(sent)
          for e in elem[0]:
            lbl.append(e)
          for e in elem[1]:
            inp.append(e)
        return predictions,lbl,inp
    
    def verbose_output(self,prediction_sentences,lbl_sentences,inputs_sentences,word,tokenizer):
        # MODIFY THE DECODE!!! PER TOKEN
        f = open('results_out/russian/predictions'+str(self.current_epoch)+word+'.txt', 'w', encoding="utf-8")
        for elem in prediction_sentences:
          doc = nlp(tokenizer.decode(elem,skip_special_tokens=True))
          #doc = tokenizer.decode(elem,skip_special_tokens=True)
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
          #f.write(tokenizer.decode(elem,skip_special_tokens=True)+"\n")
        f.close()
        dataset2 = wandb.Artifact('predict', type='dataset')
        dataset2.add_file('results_out/russian/predictions'+str(self.current_epoch)+word+'.txt')
        wandb.log_artifact(dataset2)
        
        f = open('results_out/russian/inputs'+word+'.txt', 'w', encoding="utf-8")
        for elem in inputs_sentences:
          #doc = tokenizer.decode(elem)
          doc = tokenizer.decode(elem,skip_special_tokens=True)
          output=nlp(doc)
          #doc=nlp(doc[6:doc.index("<ssep>")])
          tokens = [token.text for token in output]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
        f.close()
        dataset1 = wandb.Artifact('original', type='dataset')
        dataset1.add_file('results_out/russian/inputs'+word+'.txt')
        wandb.log_artifact(dataset1)

        f = open('results_out/russian/labels'+word+'.txt', 'w', encoding="utf-8")
        for elem in lbl_sentences:
          doc = nlp(tokenizer.decode(elem,skip_special_tokens=True))
          #doc = tokenizer.decode(elem,skip_special_tokens=True)
          tokens = [token.text for token in doc]
          output = ' '.join(tokens)
          #output=doc
          f.write(output + '\n')
          #f.write(tokenizer.decode(elem,skip_special_tokens=True)+"\n")
        f.close()
        dataset3 = wandb.Artifact('correct', type='dataset')
        dataset3.add_file('results_out/russian/labels'+word+'.txt')
        wandb.log_artifact(dataset3)

    def on_validation_epoch_end(self):
        #self.scheduler.step()
        wandb.log({'train_loss': self.myloss,'epoch':self.current_epoch,'step':self.global_step})
        wandb.log({"valid_loss": self.myloss_val,"epoch": self.current_epoch})

        print("writing...")
        self.my_predictions =[]
        outpt = self.outputs
        self.outputs=[]
        self.my_predictions,self.my_labels,self.my_inputs=self.get_predictions_on_end(outpt,self.validation_predictions)
        self.validation_predictions=[]
        mypred=[]
        mylbl=[]
        myinp=[]
        mypred_en=[]
        mylbl_en=[]
        myinp_en=[]
        mypred_fm=[]
        mylbl_fm=[]
        myinp_fm=[]
        mypred_ru=[]
        mylbl_ru=[]
        myinp_ru=[]
        mypred_cz=[]
        mylbl_cz=[]
        myinp_cz=[]
        print(len(self.my_predictions),len(self.my_labels),len(self.my_inputs))
        
        for e1,e2,e3 in zip(self.my_predictions,self.my_labels,self.my_inputs):
          if "en_XX" in tokenizer_en.decode(e2):
            mypred_en.append(e1)
            mylbl_en.append(e2)
            myinp_en.append(e3)
          elif "it_IT" in tokenizer_en.decode(e2):
            mypred.append(e1)
            mylbl.append(e2)
            myinp.append(e3)
          elif "de_DE" in tokenizer_en.decode(e2):
            mypred_fm.append(e1)
            mylbl_fm.append(e2)
            myinp_fm.append(e3)
          elif "ru_RU" in tokenizer_en.decode(e2):
            mypred_ru.append(e1)
            mylbl_ru.append(e2)
            myinp_ru.append(e3)
          elif "cs_CZ" in tokenizer_en.decode(e2):
            mypred_cz.append(e1)
            mylbl_cz.append(e2)
            myinp_cz.append(e3)
          else:
             print("err err err")
        inp_app=myinp
        inp_app_en=myinp_en
        myinp=[tokenizer_it(elem, return_tensors='pt')["input_ids"][0] for elem in self.data1_3.X_list if len(tokenizer_it(elem, return_tensors='pt')["input_ids"][0])<256]
        myinp_en=[tokenizer_en(elem, return_tensors='pt')["input_ids"][0] for elem in self.data1_1B_WI.X_list if len(tokenizer_en(elem, return_tensors='pt')["input_ids"][0])<256]
        myinp_fm=[tokenizer_fm(elem, return_tensors='pt')["input_ids"][0] for elem in self.data1_1fm.X_list if len(tokenizer_fm(elem, return_tensors='pt')["input_ids"][0])<256]
        myinp_ru=[tokenizer_ru(elem, return_tensors='pt')["input_ids"][0] for elem in self.data1_1ru.X_list if len(tokenizer_ru(elem, return_tensors='pt')["input_ids"][0])<256]
       

        self.lang='it'
        nlp = spacy.load('it_core_news_sm')
        self.annotator = errant.load(self.lang,nlp)
        print("loaded ",self.lang)
        if myinp:
          print(len(myinp),len(mylbl),len(mypred))
          self.f05_monitor=self.get_F05(myinp,mylbl,mypred,self.current_epoch,"_it",tokenizer_it)
        
        self.lang='en'
        nlp = spacy.load('en_core_web_sm')
        self.annotator = errant.load(self.lang,nlp)
        print("loaded ",self.lang)
        if myinp_en:
          print(len(myinp_en),len(mylbl_en),len(mypred_en))
          self.f05_monitor=self.get_F05(myinp_en,mylbl_en,mypred_en,self.current_epoch,"_en",tokenizer_en)
        
        self.lang='de'
        nlp = spacy.load('de_core_news_sm')
        self.annotator = errant.load(self.lang,nlp)
        print("loaded ",self.lang)
        if myinp_en:
          print(len(myinp_fm),len(mylbl_fm),len(mypred_fm))
          self.f05_monitor=self.get_F05(myinp_fm,mylbl_fm,mypred_fm,self.current_epoch,"_fm",tokenizer_fm)

        self.lang='ru'
        nlp = spacy.load('ru_core_news_sm')
        self.annotator = errant.load(self.lang,nlp)
        print("loaded ",self.lang)
        if myinp_en:
          print(len(myinp_ru),len(mylbl_ru),len(mypred_ru))
          self.f05_monitor=self.get_F05(myinp_ru,mylbl_ru,mypred_ru,self.current_epoch,"_ru",tokenizer_ru)
          
        #self.lang='cz'
        #nlp = spacy.load('cz_core_news_sm')
        #self.annotator = errant.load(self.lang,nlp)
        #print("loaded ",self.lang)
        #if myinp_en:
        #  print(len(myinp_cz),len(mylbl_cz),len(mypred_cz))
        #  self.f05_monitor=self.get_F05(myinp_cz,mylbl_cz,mypred_cz,self.current_epoch,"_cz",tokenizer_cz)
          
        if self.verbose_m2:
          print("writing files...")
          self.verbose_output(mypred_en,mylbl_en,myinp_en,"_en",tokenizer_en)
          self.verbose_output(mypred_fm,mylbl_fm,myinp_fm,"_fm",tokenizer_fm)
          self.verbose_output(mypred,mylbl,myinp,"_it",tokenizer_it)
          self.verbose_output(mypred_ru,mylbl_ru,myinp_ru,"_ru",tokenizer_ru)
        ################################# Table #################################
        wandb_log_data = list(zip([tokenizer_en.decode(elem) for elem in inp_app_en],
                                  [tokenizer_en.decode(elem,skip_special_tokens=True) for elem in myinp_en],
                                  [tokenizer_en.decode(elem,skip_special_tokens=True) for elem in mypred_en],
                                  [tokenizer_en.decode(elem,skip_special_tokens=True) for elem in mylbl_en]))
        wandb.log({"Table1": wandb.Table(data=wandb_log_data, columns=["masked","Original" ,"Prediction", "Correct"]),
                   "F05-table1":self.f05_monitor}, commit=True)
        
        wandb_log_data = list(zip([tokenizer_it.decode(elem) for elem in inp_app],
                                  [tokenizer_it.decode(elem,skip_special_tokens=True) for elem in myinp],
                                  [tokenizer_it.decode(elem,skip_special_tokens=True) for elem in mypred],
                                  [tokenizer_it.decode(elem,skip_special_tokens=True) for elem in mylbl]))
        wandb.log({"Table2": wandb.Table(data=wandb_log_data, columns=["masked","Original" ,"Prediction", "Correct"]),
                   "F05-table2":self.f05_monitor}, commit=True)


        wandb_log_data = list(zip([tokenizer_fm.decode(elem) for elem in myinp_fm],
                                  [tokenizer_fm.decode(elem,skip_special_tokens=True) for elem in myinp_fm],
                                  [tokenizer_fm.decode(elem,skip_special_tokens=True) for elem in mypred_fm],
                                  [tokenizer_fm.decode(elem,skip_special_tokens=True) for elem in mylbl_fm]))
        wandb.log({"Table3": wandb.Table(data=wandb_log_data, columns=["masked","Original" ,"Prediction", "Correct"]),
                   "F05-table3":self.f05_monitor}, commit=True)
        
        wandb_log_data = list(zip([tokenizer_ru.decode(elem) for elem in myinp_ru],
                                  [tokenizer_ru.decode(elem,skip_special_tokens=True) for elem in myinp_ru],
                                  [tokenizer_ru.decode(elem,skip_special_tokens=True) for elem in mypred_ru],
                                  [tokenizer_ru.decode(elem,skip_special_tokens=True) for elem in mylbl_ru]))
        wandb.log({"Table4": wandb.Table(data=wandb_log_data, columns=["masked","Original" ,"Prediction", "Correct"]),
                   "F05-table4":self.f05_monitor}, commit=True)
        
        self.my_inputs=[]
        self.my_labels=[]
        if self.current_epoch>26:
          torch.save(self.model,'/media/ckpt-ru/backup_epoch'+str(self.current_epoch)+'.pt')
        
    def get_F05(self,inputs,labels,predictions,current_epoch,lang,tokenizer):
      TP=1
      FP=1
      FN=1

      tp_detection=1
      fp_detection=1
      fn_detection=1
      print(spacy.__version__)

      ########### TYPES ###########
      # FP/TP/FN
      samples_dict={"U-TP"+lang:1,"R-TP"+lang:1,"M-TP"+lang:1,
                  "U-FP"+lang:1,"R-FP"+lang:1,"M-FP"+lang:1,
                  "U-FN"+lang:1,"R-FN"+lang:1,"M-FN"+lang:1}
        
      metrics_dict={"urecall"+lang:0.0,"uprecision"+lang:0.0,"uf_beta"+lang:0.0,
                  "rrecall"+lang:0.0,"rprecision"+lang:0.0,"rf_beta"+lang:0.0,
                  "mrecall"+lang:0.0,"mprecision"+lang:0.0,"mf_beta"+lang:0.0}
      
      for e1,e2,e3 in zip(inputs,labels,predictions):
        output = tokenizer.decode(e1,skip_special_tokens=True)
        output=nlp(output)
        tokens = [token.text for token in output]
        output = ' '.join(tokens)
        #print(output)
        orig = self.annotator.parse(output, tokenise=True)

        ################################################
        output = tokenizer.decode(e2,skip_special_tokens=True)
        output=nlp(output)
        tokens = [token.text for token in output]
        output = ' '.join(tokens)
        #print(output)
        cor = self.annotator.parse(output, tokenise=True)

        ################################################
        output = tokenizer.decode(e3,skip_special_tokens=True)
        output=nlp(output)
        tokens = [token.text for token in output]
        output = ' '.join(tokens)
        #print(output)
        hyp = self.annotator.parse(output, tokenise=True)

        ################################################
        edit_gold = self.annotator.annotate(orig, cor)
        edit_hyp = self.annotator.annotate(orig, hyp)
        app=""
        ################################ to correct ###################
        for e1 in edit_hyp:
            found=False
            found_det=False
            for e2 in edit_gold:
              if e1.o_start==e2.o_start and e1.o_end==e2.o_end:
                app=str(e2.type)[0]
                if e1.c_str==e2.c_str:
                  found=True
                found_det=True
            if not found:
              samples_dict[str(e1.type)[0]+"-FP"+lang]+=1
              FP+=1
            elif found:
              samples_dict[app+"-TP"+lang]+=1
              found=False
              TP+=1
            if found_det:
              found_det=False
              tp_detection+=1
            else:
              fp_detection+=1

          ################################
        for e1 in edit_gold:
            found=False
            found_det=False
            for e2 in edit_hyp:
              if e1.o_start==e2.o_start and e1.o_end==e2.o_end:
                if e1.c_str==e2.c_str:
                  found=True
                found_det=True
            if not found:
              samples_dict[str(e1.type)[0]+"-FN"+lang]+=1
              FN+=1
            if not found_det:
              fn_detection+=1
      #####################Detection###########################
      # Recall
      det_recall = tp_detection / (tp_detection + fn_detection)
      # Precision
      det_precision = tp_detection / (tp_detection + fp_detection)
      # # F0.5
      det_f_beta = (1 + 0.5 ** 2) * (det_precision * det_recall) / ((0.5 ** 2 * det_precision) + det_recall)
      print("1.  DETECTION: ",det_precision,det_recall,det_f_beta)

      ################################################
      # Recall
      recall = TP / (TP + FN)
      # Precision
      precision = TP / (TP + FP)
      # # F0.5
      f_beta = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall)
      print("2.  CORECTION: ",precision,recall,f_beta)

      # Recall/Prec/F0.5
        
      #Unnecessary
      metrics_dict["urecall"+lang] = samples_dict["U-TP"+lang] / (samples_dict["U-TP"+lang] + samples_dict["U-FN"+lang])
      metrics_dict["uprecision"+lang] = samples_dict["U-TP"+lang] / (samples_dict["U-TP"+lang] + samples_dict["U-FP"+lang])
      metrics_dict["uf_beta"+lang] = (1 + 0.5 ** 2) * (metrics_dict["uprecision"+lang] * metrics_dict["urecall"+lang]) / ((0.5 ** 2 * metrics_dict["uprecision"+lang]) + metrics_dict["urecall"+lang])
      #Replacement
      metrics_dict["rrecall"+lang] = samples_dict["R-TP"+lang] / (samples_dict["R-TP"+lang] + samples_dict["R-FN"+lang])
      metrics_dict["rprecision"+lang] = samples_dict["R-TP"+lang] / (samples_dict["R-TP"+lang] + samples_dict["R-FP"+lang])
      metrics_dict["rf_beta"+lang] = (1 + 0.5 ** 2) * (metrics_dict["rprecision"+lang] * metrics_dict["rrecall"+lang]) / ((0.5 ** 2 * metrics_dict["rprecision"+lang]) + metrics_dict["rrecall"+lang])
      #Missing
      metrics_dict["mrecall"+lang] = samples_dict["M-TP"+lang] / (samples_dict["M-TP"+lang] + samples_dict["M-FN"+lang])
      metrics_dict["mprecision"+lang] = samples_dict["M-TP"+lang] / (samples_dict["M-TP"+lang] + samples_dict["M-FP"+lang])
      metrics_dict["mf_beta"+lang] = (1 + 0.5 ** 2) * (metrics_dict["mprecision"+lang] * metrics_dict["mrecall"+lang]) / ((0.5 ** 2 * metrics_dict["mprecision"+lang]) + metrics_dict["mrecall"+lang])


      result_general={"TP"+lang: TP,"FP"+lang: FP,"FN"+lang: FN,"epoch": current_epoch,
                  "Precision"+lang:precision,"Recall"+lang:recall,"F0.5"+lang:f_beta,
                  "det_recall"+lang:det_recall,"det_precision"+lang:det_precision,"det_f_beta"+lang:det_f_beta}
        
      result_general.update(samples_dict)
      result_general.update(metrics_dict)
      wandb.log(result_general)
      result_general={}
      return f_beta
   