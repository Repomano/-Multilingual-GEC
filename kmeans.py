import torch
from sklearn.cluster import KMeans
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Load MBart tokenizer and model
model_name="facebook/mbart-large-50"
model = MBartForConditionalGeneration.from_pretrained(model_name, output_hidden_states=True)
tokenizer_it = MBart50TokenizerFast.from_pretrained(model_name, src_lang="it_IT", tgt_lang="it_IT")
tokenizer_en = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="en_XX")
tokenizer_fm = MBart50TokenizerFast.from_pretrained(model_name, src_lang="de_DE", tgt_lang="de_DE")
tokenizer_ru = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ru_RU", tgt_lang="ru_RU")

# Example sentences
sentences_A = ["This is the first sentence.", "Here comes the second sentence."]
sentences_B = ["Another sentence.", "One more sentence."]

# Tokenize sentences in list A and B
tokenized_A = tokenizer(sentences_A, return_tensors="pt", padding=True, truncation=True)
tokenized_B = tokenizer(sentences_B, return_tensors="pt", padding=True, truncation=True)

# Generate embeddings for list A
input_ids_A = tokenized_A["input_ids"]
attention_mask_A = tokenized_A["attention_mask"]
embeddings_A = model(input_ids=input_ids_A, attention_mask=attention_mask_A).last_hidden_state

# Generate embeddings for list B
input_ids_B = tokenized_B["input_ids"]
attention_mask_B = tokenized_B["attention_mask"]
embeddings_B = model(input_ids=input_ids_B, attention_mask=attention_mask_B).last_hidden_state

# Concatenate embeddings for each sentence in list A
sentence_embeddings_A = torch.cat([embeddings_A[i, 1:tokenized_A['attention_mask'][i].sum()-1, :] for i in range(len(sentences_A))])

# Concatenate embeddings for each sentence in list B
sentence_embeddings_B = torch.cat([embeddings_B[i, 1:tokenized_B['attention_mask'][i].sum()-1, :] for i in range(len(sentences_B))])

# Concatenate all sentence embeddings
all_sentence_embeddings = torch.cat([sentence_embeddings_A, sentence_embeddings_B])

# Perform k-means clustering on sentence embeddings
num_clusters = 25
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(all_sentence_embeddings)

# Assign each sentence to a cluster
sentence_clusters = kmeans.labels_

# Mean value representation for each sentence
sentence_values = []
for i in range(len(sentences_A)):
    sentence_indices = torch.where(sentence_clusters == i)[0]
    sentence_embeddings = sentence_embeddings_A[sentence_indices]
    sentence_mean_embedding = torch.mean(sentence_embeddings, dim=0)
    sentence_values.append(sentence_mean_embedding.tolist())

for i in range(len(sentences_B)):
    sentence_indices = torch.where(sentence_clusters == len(sentences_A) + i)[0]
    sentence_embeddings = sentence_embeddings_B[sentence_indices]
    sentence_mean_embedding = torch.mean(sentence_embeddings, dim=0)
    sentence_values.append(sentence_mean_embedding.tolist())

# Print the sentence values
for i, value in enumerate(sentence_values):
    print(f"Sentence {i + 1} value: {value}")
