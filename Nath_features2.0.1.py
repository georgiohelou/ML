import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from page import pagerank
# from Bert_function import author_embedding
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from Text_Embedding import Embed_Author
import pickle
from MLP import prepare_data, MLP, train_model, evaluate_model, predict
import nltk
print("Imported all libraries.")

'''
infile = open('fullEmbeddings.pkl', 'rb')
AllAuthorEmbeddings = pickle.load(infile)
infile.close()
print("AllAuthorEmbeddings loaded")
random_entry = random.choice(AllAuthorEmbeddings)
print(random_entry)
print(len(random_entry))
## one entry in 'fullEmbeddings.pkl' is of size 768 (corresponding to 1 author)

print("---------")
print("loading graph")
# load the graph
G = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

print("calculating deep walk")
#computes Deep Walk
mapping = {old_label:new_label for new_label, old_label in enumerate(G.nodes())}
with open("mapping_nath.pkl", "wb") as myFile:
    pickle.dump(mapping, myFile)

print("calculating deep walk")
#computes Deep Walk
infile = open('mapping_nath.pkl', 'rb')
mapping = pickle.load(infile)
infile.close()

H = nx.relabel_nodes(G, mapping)
dw=DpWalk(H)

with open("deepWalk128layers_length40.pkl", "wb") as myFile:
    pickle.dump(dw, myFile)
'''
print("Loading & printing sentences.pkl")
infile = open('sentences.pkl', 'rb')
sentences = pickle.load(infile)
infile.close()

print("type(sentences) : ", type(sentences))
print("len(sentences) = ", len(sentences))
# print("sentences[0] : \n", sentences[0])

specialChars = ".,()/"
for i in tqdm(range(len(sentences))):
    print(i)
    if i == 1:
        print(sentences[i])
    author_i_Abstracts = sentences[i]
    for specialChar in specialChars:
        author_i_Abstracts = author_i_Abstracts.replace(specialChar, "")
    author_i_Abstracts = author_i_Abstracts.replace("  ", " ")
    author_i_Abstracts = author_i_Abstracts.strip()
    sentences[i] = author_i_Abstracts
    
    if i==1:
        print(sentences[i])
        break

# specialChars = ".,()"
# for i in tqdm(range(len(sentences))):
#     author_i_Abstracts = sentences[i]
#     for specialChar in specialChars:
#         author_i_Abstracts = author_i_Abstracts.replace(specialChar, "")
#     author_i_Abstracts = author_i_Abstracts.replace("  ", " ")
#     author_i_Abstracts = author_i_Abstracts.strip()
#     sentences[i] = author_i_Abstracts

# with open('sentences2.0.pkl', 'wb') as f:
#         pickle.dump(sentences, f)
# f.close()

# print("Operation completed. Check sentences2.0.pkl file.")
