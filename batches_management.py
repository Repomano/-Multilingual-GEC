import torch

def get_batches_dataset_division(train,corr,trainm,corrm,num_tokens):
  dataset = {"input_ids":[],"attention_mask":[]}
  correct = {"input_ids":[],"attention_mask":[]}
  app_batch1=[]
  app_batch2=[]
  app_batch3=[]
  app_batch4=[]
  counter=0
  max_len=0
  tot_sentence=0
  total_sentence=0
  added=False
  # train
  for s1,s2,s3,s4 in zip(train,trainm,corr,corrm):
    #Counter different from zero and also max_len
    if max_len<max(s1.shape[-1],s3.shape[-1]) and max_len!=0:
      max_len=max(s1.shape[-1],s3.shape[-1])
      for i in range(len(app_batch1)):
        len_previous=max(max_len-app_batch1[i].shape[-1],max_len-app_batch3[i].shape[-1])
        counter+=len_previous

    elif max_len!=0 and max_len>max(s1.shape[-1],s3.shape[-1]):
        len_previous=max(max_len-s1.shape[-1],max_len-s3.shape[-1])
        counter+=len_previous

    #Counter different from zero
    if counter+max(s1.shape[-1],s3.shape[-1])<num_tokens  and counter!=0:
      app_batch1.append(s1.squeeze())
      app_batch2.append(s2.squeeze())
      app_batch3.append(s3.squeeze())
      app_batch4.append(s4.squeeze())
      total_sentence+=1

      for i in range(len(app_batch1)):
        app_batch1[i]=torch.cat((app_batch1[i],torch.ones(max_len-app_batch1[i].shape[-1])),-1)
        app_batch2[i]=torch.cat((app_batch2[i],torch.ones(max_len-app_batch2[i].shape[-1])),-1)
        app_batch3[i]=torch.cat((app_batch3[i],torch.ones(max_len-app_batch3[i].shape[-1])),-1)
        app_batch4[i]=torch.cat((app_batch4[i],torch.ones(max_len-app_batch4[i].shape[-1])),-1)
      counter+=max(s1.shape[-1],s3.shape[-1])

    #Counter different from zero
    elif counter+max(s1.shape[-1],s3.shape[-1])>=num_tokens and counter!=0:
      dataset['input_ids'].append(torch.stack(app_batch1).type(torch.int64))
      dataset['attention_mask'].append(torch.stack(app_batch2).type(torch.int64))
      correct['input_ids'].append(torch.stack(app_batch3).type(torch.int64))
      correct['attention_mask'].append(torch.stack(app_batch4).type(torch.int64))
      tot_sentence+=torch.stack(app_batch1).shape[0]
      added=True
      app_batch1=[s1.squeeze()]
      app_batch2=[s2.squeeze()]
      app_batch3=[s3.squeeze()]
      app_batch4=[s4.squeeze()]
      total_sentence+=1
      max_len=max(s1.shape[-1],s3.shape[-1])
      counter=max_len

    if total_sentence>len(train)-1:
      dataset['input_ids'].append(torch.stack(app_batch1).type(torch.int64))
      dataset['attention_mask'].append(torch.stack(app_batch2).type(torch.int64))
      correct['input_ids'].append(torch.stack(app_batch3).type(torch.int64))
      correct['attention_mask'].append(torch.stack(app_batch4).type(torch.int64))
      tot_sentence+=torch.stack(app_batch1).shape[0]
    
    elif counter==0:
      app_batch1.append(s1.squeeze())
      app_batch2.append(s2.squeeze())
      app_batch3.append(s3.squeeze())
      app_batch4.append(s4.squeeze())
      total_sentence+=1
      max_len=max(s1.shape[-1],s3.shape[-1])
      counter=max_len
  

  #print("tot     ",len(dataset['input_ids']),len(correct['input_ids']))
  # RIMOZIONE ELEMENTI TROPPO LUNGHI
  dtid=dataset['input_ids']
  dtmk=dataset['attention_mask']
  crid=correct['input_ids']
  crmk=correct['attention_mask']

  dataset['input_ids']=[]
  dataset['attention_mask']=[]
  correct['input_ids']=[]
  correct['attention_mask']=[]

  for i in range(len(dtid)):
    if dtid[i].shape[-1]<num_tokens or crid[i].shape[-1]<num_tokens:
      dataset['input_ids'].append(dtid[i])
      dataset['attention_mask'].append(dtmk[i])
      correct['input_ids'].append(crid[i])
      correct['attention_mask'].append(crmk[i])
      
  print(len(dataset['input_ids'])
              ,len(correct['input_ids'])
              ,len(dataset['attention_mask'])
              ,len(correct['attention_mask']))  
              
              
  return dataset,correct