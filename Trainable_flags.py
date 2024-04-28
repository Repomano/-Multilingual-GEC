import torch
from encoder_sentence import *
device="cuda"
def get_flags(input_sent,edits_list):    
    output=[]
    correct=torch.zeros(32, dtype=torch.float32)
    error=torch.ones(32, dtype=torch.float32)
    print(len(input_sent),len(edits_list))
    for sentence,edits in zip(input_sent,edits_list):
        flag_sentence=[]
        for i,token in enumerate(sentence.tolist()):
            whatdo="pass"
            for ed in edits:
                if i-1 in range(int(ed[1]),int(ed[2])):
                    whatdo="add"
            if whatdo=="add":
                flag_sentence.append(error)
            else:
                flag_sentence.append(correct)

        output.append(torch.stack(flag_sentence))
    return output


def get_alignment_batch(input_batched, list_flags):
    batched_flags_list=[]
    counter=0
    #print(input_batched)
    for elem in input_batched:
        single_batch=[]
        for e in range(elem.shape[0]):
            if len(list_flags[counter+e])>elem.shape[-1]:
                print("errore generato da:")
                print(len(list_flags[counter+e]),elem.shape)
                print()
                #print(tokenizer.decode(elem[e]))
            if len(list_flags[counter+e])<elem.shape[-1]:
                #print(list_flags[counter+e].shape,elem.shape[-1],torch.zeros([elem.shape[-1]-len(list_flags[counter+e]),32]).shape)
                single_batch.append(torch.cat((list_flags[counter+e].squeeze(),
                                             torch.zeros([elem.shape[-1]-len(list_flags[counter+e]),32])),dim=0))
            else:
                #print(list_flags[counter+e].shape,elem.shape[-1],torch.zeros([len(list_flags[counter+e]),32]).shape)
                single_batch.append(list_flags[counter+e].squeeze())
        counter+=elem.shape[0]
        batched_flags_list.append(torch.stack(single_batch))
    
    print(input_batched[32].shape,batched_flags_list[32].shape)
    return batched_flags_list
