import pandas as pd
import numpy as np
import csv
import re
from transformers import BertTokenizer,BertModel
import torch
from sklearn.decomposition import PCA
import sys

filename = sys.argv[1]

#Read raw data and generate the DF(dataframe)
raw_data = pd.read_csv(filename,sep="\t",names=['ID','Sentence'],quoting=csv.QUOTE_NONE,header=0)

#Remove Newline character and Multiple double quotes in the sentence
def clean_data(text):
    clean_text = re.sub(r'(\\n|\\")',"",text)
    return clean_text

raw_data['Sentence'] = raw_data['Sentence'].apply(clean_data)

#Use bert model transform sentence to the vector
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

#Use batching to decrease the pressure on memory
batch_size = 32
sentence_vectors = []

for i in range(0,len(raw_data),batch_size):
    batch_sentences = raw_data['Sentence'][i:i+batch_size].to_list()

    encoding_text = tokenizer(
        batch_sentences,
        truncation = True,
        max_length = 64,
        padding = 'max_length',
        return_tensors = 'pt'
    )

    with torch.no_grad():
        batch_sentence_vecs = bert_model(**encoding_text)
        sentence_vectors.append(batch_sentence_vecs.pooler_output)

    print(f"batch {int(i/batch_size)+1}/{int(len(raw_data)/batch_size)} complete")


sentence_vectors = torch.cat(sentence_vectors,dim=0)
sentence_vectors_np = sentence_vectors.numpy()

#PCA
pca = PCA(n_components=5)
sentence_vectors_pca = pca.fit_transform(sentence_vectors_np)

#Vector normalization
vec_norms = np.linalg.norm(sentence_vectors_pca,axis=1,keepdims=True)
sen_vec_pca_norm = sentence_vectors_pca/vec_norms

np.save("sentence_vectors_pca.npy", sen_vec_pca_norm)