from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from Reader import *
from main import *
from GED.Myeval import My_f1
from Model import params
device = "cuda"

class Trainer():

    def __init__(self,model: nn.Module,loss_function,optimizer ,log_steps:int=10_000,log_level:int=2):
      
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.log_steps = log_steps
        self.log_level = log_level
        self.max_len=103
    def train(self, train_dataset:Dataset,valid_dataset:Dataset, epochs:int=1):
        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0
        f1=[]
        al=[]
        vl=[]
        tl=[]
        f1_valid=[]
        results=[]
        for epoch in range(epochs):
            lstm_output=torch.empty([1,self.max_len+1,2*params.hidden_dim]).to(device)
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            all_labels= list()
            all_predictions= list()
            all_labels_pad= list()
            all_predictions_pad= list()
            
            #================== for each batch ==================
            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs'].to(device)
                labels = sample['mask'].to(device)
                self.optimizer.zero_grad()
                predictions,weight_task1,val = self.model(inputs)


                #================== used to pass for the loss function ==================
                pred = torch.argmax(predictions, -1).view(-1)
                
                
                #================== Save the prediction to save for loss/graphs ==================
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                #================== eliminate nthe presence of padding [PAD] ==================

                to_add_pred=[]
                to_add_true=[]
                #to_add_pred_pad=[]
                #to_add_true_pad=[]
                

                for a in range(pred.shape[0]):
                  if labels[a] != 2:
                    to_add_pred.append(pred[a])
                    to_add_true.append(labels[a])
                  #to_add_pred_pad.append(pred[a])
                  #to_add_true_pad.append(labels[a])
                

                # NO PAD
                valid_predictions = torch.stack(to_add_pred)
                valid_labels = torch.stack(to_add_true)
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())

                #PAD
                #valid_predictions_pad = torch.stack(to_add_pred_pad)
                #valid_labels_pad = torch.stack(to_add_true_pad)
                all_labels_pad.extend(pred.tolist())
                all_predictions_pad.extend(labels.tolist())

                

                #================== compute of cross entropy + backprop. ==================
                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()
                
                
                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
                #================== LSTM representation for task2 ==================
                #print(lstm_output.shape,val.shape)
                #lstm_output = torch.cat((lstm_output,val),dim=0)
            
            #================== saves the f1/losses for final scores ==================
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            # list used for plots
            al.append(avg_epoch_loss)
            tl.append(epoch_loss)
            
            f1.append(f1_score(all_labels, all_predictions, average="macro"))

            #show at each epoch the f1
            
            print("f1 train:",f1_score(all_labels, all_predictions, average="macro"))
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            #================== VALIDATION ==================
            valid_loss,f1_eval,all_labels_eval,all_predictions_eval,val_lstm_out = self.evaluate(valid_dataset)
            f1_valid.append(f1_eval)
            vl.append(valid_loss)
            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))
            
            torch.save(self.model,'backup_epoch'+str(epoch)+'.pt')

        avg_epoch_loss = train_loss / epochs
        return al,tl,f1,f1_valid,vl,torch.tensor(all_labels_pad).reshape(-1,131),torch.tensor(all_predictions_pad).reshape(-1,131),weight_task1,torch.tensor(all_predictions_eval).reshape(-1,131),torch.tensor(all_labels_eval).reshape(-1,131),lstm_output[1:],val_lstm_out[1:]
    

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        all_labels= list()
        all_predictions= list()
        all_labels_pad= list()
        all_predictions_pad= list()
        f1 = []
        with torch.no_grad():
            lstm_output_dev=torch.empty([1,self.max_len+1,2*params.hidden_dim]).to(device)
            for sample in valid_dataset:
                inputs = sample['inputs'].to(device)
                labels = sample['mask'].to(device)
                #================== PREDICT CLASSES #==================

                predictions,_,val = self.model(inputs)#get the predictions



                predictions = predictions.view(-1, predictions.shape[-1]) 


                labels = labels.view(-1)
                pred = torch.argmax(predictions, -1).view(-1)
                
                

                #================== eliminate nthe presence of padding [PAD] ==================
                

                to_add_pred=[]
                to_add_true=[]

                for a in range(pred.shape[0]):
                  if labels[a] != 2:
                    to_add_pred.append(pred[a])
                    to_add_true.append(labels[a])
                

                # NO PAD
                valid_predictions = torch.stack(to_add_pred)
                valid_labels = torch.stack(to_add_true)
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())

                #PAD
                
                all_labels_pad.extend(pred.tolist())
                all_predictions_pad.extend(labels.tolist())


                #================== compute of cross entropy + backprop. ==================
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()
                #================== LSTM representation for task2 ==================

                #lstm_output_dev = torch.cat((lstm_output_dev,val),dim=0)
        My_f1(all_predictions,all_labels)
        print('f1 dev :{}'.format(f1_score(all_labels, all_predictions, average="macro")))
        return valid_loss / len(valid_dataset),f1_score(all_labels, all_predictions, average="macro"),all_labels_pad,all_predictions_pad,lstm_output_dev
