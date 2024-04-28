import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AddedToken    

model_name="facebook/mbart-large-50"
#model_name="facebook/mbart-large-50-many-to-many-mmt"
tokenizer_it = MBart50TokenizerFast.from_pretrained(model_name, src_lang="it_IT", tgt_lang="it_IT")
tokenizer_en = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="en_XX")
tokenizer_fm = MBart50TokenizerFast.from_pretrained(model_name, src_lang="de_DE", tgt_lang="de_DE")
tokenizer_ru = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ru_RU", tgt_lang="ru_RU")
tokenizer_cz = MBart50TokenizerFast.from_pretrained(model_name, src_lang="cs_CZ", tgt_lang="cs_CZ")

special_separator = AddedToken('<ssep>', lstrip=True, rstrip=False) 
error_start = AddedToken('<error>', lstrip=True, rstrip=False) 
error_end= AddedToken('</error>', lstrip=True, rstrip=False) 

additional_special_tokens = {'additional_special_tokens': [special_separator,error_start,error_end]}                    
tokenizer_en.add_special_tokens(additional_special_tokens)  
tokenizer_it.add_special_tokens(additional_special_tokens)  
tokenizer_fm.add_special_tokens(additional_special_tokens)  
tokenizer_ru.add_special_tokens(additional_special_tokens)  
tokenizer_cz.add_special_tokens(additional_special_tokens)  

def encode_sentence_Batch(data1,data2,lang):
  tokenized_seq = {"input_ids":[],"attention_mask":[]}
  tokenized_corr = {"input_ids":[],"attention_mask":[]}
  
  for elem1,elem2 in zip(data1,data2):
    tokenized_sentence = tokenizer_en(elem1,text_target=elem2, return_tensors='pt')
    if elem1=="start":
      lang="it"
    if lang=="it":
      tokenized_sentence = tokenizer_it(elem1,text_target=elem2, return_tensors='pt')
    if lang=="fm":
      tokenized_sentence = tokenizer_fm(elem1,text_target=elem2, return_tensors='pt')
    if lang=="ru":
      tokenized_sentence = tokenizer_ru(elem1,text_target=elem2, return_tensors='pt')
    if lang=="cz":
      tokenized_sentence = tokenizer_cz(elem1,text_target=elem2, return_tensors='pt')
      
    tokenized_seq["input_ids"].append(tokenized_sentence["input_ids"])
    tokenized_seq["attention_mask"].append(tokenized_sentence["attention_mask"])
    tokenized_corr["input_ids"].append(tokenized_sentence["labels"])
    tokenized_corr["attention_mask"].append(tokenized_sentence["attention_mask"])

  return tokenized_seq,tokenized_corr