from Model import MLP_arg_classification
from data_extractor import ReaderM2
from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from Trainer import Model_Correction
from encoder_sentence import encode_sentence_Batch,encode_sentence, tokenizer
from batches_management import get_batches_dataset_division

# otteniamo solo la corretta italiana
data1_1B = ReaderM2('errant_env/errant/MERLIN/Merlin/dataset/original_dev.m2',
                    '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',{})

tokenizer.src_lang="it_IT"
tokenizer.tgt_lang="en_XX"

_,tokenized_italian_corr = encode_sentence_Batch(data1_1B.X_list,data1_1B.X_corr,False)
automodel =  MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",output_hidden_states=True).to("cuda")
i=0
f=open("errant_env/errant/MERLIN/Merlin/dataset/original_dev_It_en_corr.txt","w")

for e1,e2 in zip(tokenized_italian_corr["input_ids"],tokenized_italian_corr["attention_mask"]):
    if i%1000==0:
        print(i)
    result=automodel.generate(e1.to("cuda"),attention_mask=e2.to("cuda"),max_length=256,num_beams=4,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    f.write(tokenizer.decode(result[0],skip_special_tokens=True)+"\n")
    i+=1