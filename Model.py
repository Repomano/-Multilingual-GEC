import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from encoder_sentence import tokenizer_en, tokenizer_it, tokenizer_fm
#self.automodel =  MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",output_hidden_states=True)
"""# ***---> BART+BERT:***"""
model_huggingface="google/mt5-base"
model_huggingface="facebook/mbart-large-50"
model_name="facebook/mbart-large-50"
#model_name="facebook/mbart-large-50-many-to-many-mmt"
device="cuda"
class MLP_arg_classification(nn.Module):
    def __init__(self, ):
        super(MLP_arg_classification, self).__init__()
        self.automodel =  MBartForConditionalGeneration.from_pretrained(model_name, output_hidden_states=True)
        self.automodel.resize_token_embeddings(len(tokenizer_en.vocab))
        self.automodel.model.shared.weight.requires_grad=False
        self.it_IT=tokenizer_it.get_vocab()["it_IT"]
        self.en_XX=tokenizer_en.get_vocab()["en_XX"]
        self.de_DE=tokenizer_fm.get_vocab()["de_DE"]
        self.ru_RU=tokenizer_fm.get_vocab()["ru_RU"]
        self.cs_CZ=tokenizer_fm.get_vocab()["cs_CZ"]
        self.len_vocab=len(tokenizer_en.get_vocab())
        print(len(tokenizer_en.get_vocab()),len(tokenizer_it.get_vocab()))
        
    def forward(self, x,mask,correct):
      Hd = self.automodel(input_ids=x,attention_mask=mask,labels=correct)
      return Hd.logits,Hd.loss
    
    def generate(self, x,mask,max_length):
      if x[0][0].item()==self.en_XX:
        hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,
                                              num_beams=4,forced_bos_token_id=tokenizer_en.lang_code_to_id["en_XX"])
      elif x[0][0].item()==self.it_IT:
        hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,
                                              num_beams=4,forced_bos_token_id=tokenizer_it.lang_code_to_id["it_IT"])
      elif x[0][0].item()==self.de_DE:
        hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,
                                              num_beams=4,forced_bos_token_id=tokenizer_it.lang_code_to_id["de_DE"])
      elif x[0][0].item()==self.ru_RU:
        hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,
                                              num_beams=4,forced_bos_token_id=tokenizer_it.lang_code_to_id["ru_RU"])
      elif x[0][0].item()==self.cs_CZ:
        hidden_states = self.automodel.generate(x,attention_mask=mask,max_length=max_length,
                                              num_beams=4,forced_bos_token_id=tokenizer_it.lang_code_to_id["cs_CZ"])
      
      return hidden_states