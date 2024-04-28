def My_f1(predictions,labels):
  FP=0
  FN=0
  TP=0
  
  for elem1,elem2 in zip(predictions,labels):
    if elem1!=0 and elem2!=2 and elem2==0:
      FP+=1
    if elem2!=0 and elem2!=2 and elem1==0:
      FN+=1
    if elem2!=0 and elem2!=2 and elem1!=0:
      TP+=1

  precision=TP/(TP+FN)
  recall=TP/(FP+TP)
  F1=f_beta = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall)
  print(
      " precision: ",precision,
      " recall: ",recall,
      " f1 train:",F1
      )
  return F1
  