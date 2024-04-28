from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import transformers
from torch import nn
import numpy as np
from Model import MLP_arg_classification
from transformers import BartForConditionalGeneration, TrainingArguments, T5ForConditionalGeneration
from torch import nn
import huggingface_hub




tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="it_IT", tgt_lang="it_IT")
Bart=MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50",output_hidden_states=True)
model=torch.load("pretrained_model_italian.pt")


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/get_sentence.py', methods=['POST'])
def search():
    query = request.get_json().get('query')
    print(query)
    token_in=tokenizer(query, return_tensors='pt', padding=True, truncation=True,max_length=200)
    token_out=model.generate(token_in["input_ids"],token_in["attention_mask"],200)
    result=tokenizer.decode(token_out[0],skip_special_tokens=True)
    print(result)
    response = {'message': str(result)}
    resp = jsonify(response)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == '__main__':
    app.run(port=15000, debug=True, host="0.0.0.0")
