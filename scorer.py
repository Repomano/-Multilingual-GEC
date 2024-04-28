import wandb
import errant_env.errant.errant as errant
import spacy
from encoder_sentence import tokenizer
from main import nlp
# IMPLEMENTARE UN MIX DI ERRANT CHE FUNZIONA SIA IN ITALIANO CHE INGLESE
def get_F05(inputs,labels,predictions,current_epoch):
    TP=1
    FP=1
    FN=1

    tp_detection=1
    fp_detection=1
    fn_detection=1

    ########### TYPES ###########
    # FP/TP/FN
    annotator = errant.load('it',nlp)
    samples_dict={"U-TP":1,"R-TP":1,"M-TP":1,
                  "U-FP":1,"R-FP":1,"M-FP":1,
                  "U-FN":1,"R-FN":1,"M-FN":1}
        
    metrics_dict={"urecall":0.0,"uprecision":0.0,"uf_beta":0.0,
                  "rrecall":0.0,"rprecision":0.0,"rf_beta":0.0,
                  "mrecall":0.0,"mprecision":0.0,"mf_beta":0.0}
    c=0
    for e1,e2,e3 in zip(inputs,labels,predictions):
        
        if c==0 and "en_XX" in e1:
           annotator = errant.load('en',nlp)
           c+=1
          
        doc = nlp(tokenizer.decode(e1,skip_special_tokens=True))
        tokens = [token.text for token in doc]
        output = ' '.join(tokens)
        orig = annotator.parse(output)

        ################################################
        doc = nlp(tokenizer.decode(e2,skip_special_tokens=True))
        tokens = [token.text for token in doc]
        output = ' '.join(tokens)
        cor = annotator.parse(output)

        ################################################
        doc = nlp(tokenizer.decode(e3,skip_special_tokens=True))
        tokens = [token.text for token in doc]
        output = ' '.join(tokens)
        hyp = annotator.parse(output)

        ################################################
        edit_gold = annotator.annotate(orig, cor)
        edit_hyp = annotator.annotate(orig, hyp)
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
              samples_dict[str(e1.type)[0]+"-FP"]+=1
              FP+=1
            elif found:
              samples_dict[app+"-TP"]+=1
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
              samples_dict[str(e1.type)[0]+"-FN"]+=1
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
    metrics_dict["urecall"] = samples_dict["U-TP"] / (samples_dict["U-TP"] + samples_dict["U-FN"])
    metrics_dict["uprecision"] = samples_dict["U-TP"] / (samples_dict["U-TP"] + samples_dict["U-FP"])
    metrics_dict["uf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["uprecision"] * metrics_dict["urecall"]) / ((0.5 ** 2 * metrics_dict["uprecision"]) + metrics_dict["urecall"])
    #Replacement
    metrics_dict["rrecall"] = samples_dict["R-TP"] / (samples_dict["R-TP"] + samples_dict["R-FN"])
    metrics_dict["rprecision"] = samples_dict["R-TP"] / (samples_dict["R-TP"] + samples_dict["R-FP"])
    metrics_dict["rf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["rprecision"] * metrics_dict["rrecall"]) / ((0.5 ** 2 * metrics_dict["rprecision"]) + metrics_dict["rrecall"])
    #Missing
    metrics_dict["mrecall"] = samples_dict["M-TP"] / (samples_dict["M-TP"] + samples_dict["M-FN"])
    metrics_dict["mprecision"] = samples_dict["M-TP"] / (samples_dict["M-TP"] + samples_dict["M-FP"])
    metrics_dict["mf_beta"] = (1 + 0.5 ** 2) * (metrics_dict["mprecision"] * metrics_dict["mrecall"]) / ((0.5 ** 2 * metrics_dict["mprecision"]) + metrics_dict["mrecall"])


    result_general={"TP": TP,"FP": FP,"FN": FN,"epoch": current_epoch,
                  "Precision":precision,"Recall":recall,"F0.5":f_beta,
                  "det_recall":det_recall,"det_precision":det_precision,"det_f_beta":det_f_beta}
        
    result_general.update(samples_dict)
    result_general.update(metrics_dict)
    wandb.log(result_general)
    result_general={}
    return f_beta