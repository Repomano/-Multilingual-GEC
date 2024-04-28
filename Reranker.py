from data_extractor import ReaderM2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device="cuda"
datait = ReaderM2('/media/results_out/italian/predictions.m2',
                     '/media/results_out/italian/labels_en.txt',{})
dataen = ReaderM2('/media/results_out/english/predictions.m2',
                     '/media/results_out/english/labels_en.txt',{})
datade = ReaderM2('/media/results_out/german/predictions.m2',
                     '/media/results_out/german/labels_en.txt',{})
dataru = ReaderM2('/media/results_out/russian/predictions.m2',
                     '/media/results_out/russian/labels_en.txt',{})
f1 = open("english.m2","w")

def get_sentence_edited(data):
    result=[]
    edits=[]
    no_act=[]
    noedit=[]
    for sentence,edit in zip(data.X_list,data.Y_list):
        sent_edit=[]
        ed=[]
        act=[]
        noed=[]
        for e in edit:
            if "-NONE-" not in e[4]:
                sent_edit.append(" ".join(sentence.split(" ")[:int(e[1])])+" "+e[4]+" "+" ".join(sentence.split(" ")[int(e[2]):]))
                ed.append((sentence,e[1],e[2],e[4],"A "+str(e[1])+" "+str(e[2])+"|||"+"|||".join(e[3:])))
        edits.append(ed)
        result.append(sent_edit)
        no_act.append(sentence)
        noedit.append((sentence,-1,-1,"-NONE-","A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"))
    return result, edits, no_act, noedit

def get_sentence_scores(sentences,edits):
    model_name = "xlnet-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentence_scores = []
    i=0
    # Initialization
    for batch in sentences:
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        batch_scores = probabilities[:, 1].tolist()
        i+=1
        sentence_scores.append(batch_scores)
    # Actual re-ranking in code
    for scores,edit,sntc in zip(sentence_scores,edits,sentences):
        original_score=scores[-1]
        to_add_edits=[]
        added_ranges=[]
        #Get the ordered list
        data = [[x, y, s] for x, y, s in zip(scores, edit, sntc)]
        sorted_data = sorted(data, key=lambda sublist: sublist[0], reverse=True)
        added=False
        f1.write(edit[0][0]+"\n")
        # Recaluclate the re-ranking
        applied=False
        for scdata in sorted_data:
            if applied:
                ..
                
                applied=False
            if scdata[0]<original_score or scdata[0]==original_score:
                if not added:
                    to_add_edits.append(scdata[1])
                    applied=True
                break
            elif scdata[0]>original_score:
                if (not any(int(scdata[1][1]) in range(r[0],r[1]) for r in added_ranges) and not any(int(scdata[1][2]) in range(r[0],r[1]) for r in added_ranges)):
                    added=True
                    to_add_edits.append(scdata[1])
                    added_ranges.append((int(scdata[1][1]),int(scdata[1][2])))
                    applied=True
        
        
        for i,item in enumerate(to_add_edits):
            if to_add_edits.count(item) != 1:
                to_add_edits.pop(i)

        filtered_data = to_add_edits
        if :
            sentence_scores=get_sentence_scores([[" ".join(y[0].split(" ")[:y[1]])+" "+y[3]+" "+" ".join(y[0].split(" ")[y[2]:]) for y in x] for x in filtered_data])
        for item in filtered_data:
            f1.write(item[-1]+"\n")
        f1.write("\n")

    return sentence_scores

input_sentences_it,ed_it,no_act,edact = get_sentence_edited(datait)
input_sentences_ru,ed_ru,no_act,edact = get_sentence_edited(dataru)
input_sentences_en,ed_en,no_act,edact = get_sentence_edited(dataen)
input_sentences_de,ed_de,no_act,edact = get_sentence_edited(datade)


#print(input_sentences_it[0])
#print(ed_it[0][0][0])
"""
input_sentences = [it+ru+de+en+[ed] for it,ru,de,en,ed in zip(input_sentences_it,
                                                    input_sentences_ru,
                                                    input_sentences_de,
                                                    input_sentences_en,no_act)]

input_edits = [it+ru+de+en+[ed] for it,ru,de,en,ed in zip(ed_it,
                                                    ed_ru,
                                                    ed_de,
                                                    ed_en,edact)]
"""

input_sentences = [it+ru+de+[ed] for it,ru,de,ed in zip(input_sentences_it,
                                                    input_sentences_ru,
                                                    input_sentences_de,
                                                    no_act)]

input_edits = [it+ru+de+[ed] for it,ru,de,ed in zip(ed_it,
                                                    ed_ru,
                                                    ed_de,
                                                    edact)]

sentence_scores = get_sentence_scores(input_sentences,input_edits)