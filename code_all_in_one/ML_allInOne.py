import os
import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from Text_Embedding import Embed_Author
import pickle
import nltk
# nltk.download('punkt')
#*** use python -m nltk.downloader punkt ****
from nltk.tokenize import word_tokenize

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'up', 'to', 'ours', 'had', 'she', 'when', 'at', 'any', 'before', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

print("reading training data")
# read training data
df_train = pd.read_csv('train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

print("reading test data")
# read test data
df_test = pd.read_csv('test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]

print("loading graph")
# load the graph  
G = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# # computes structural features for each node
print("calculating core_number")
core_number = nx.core_number(G)

# #node centrality
print("calculating centrality")
centrality = nx.eigenvector_centrality(G)

# #Clustering Coefficient
print("calculating Clustering Coefficient")
cc=nx.clustering(G)

# #computes the page rank
print("calculating Page rank")
pr=nx.pagerank(G,0.4)

print("calculating deep walk")
#computes Deep Walk
mapping = {old_label:new_label for new_label, old_label in enumerate(G.nodes())}
H = nx.relabel_nodes(G, mapping)
dw=DpWalk(H)

##Create Dict of paperID to abstract
print("extracting abstracts")
DictOfAbstracts={}
DictOfAbstracts= process_abstracts()


# ##Create Dict of AuthorID to paperID
print("extracting paperID")
DictOfPaperID= process_authorFiles()

# ##Create Dict of AuthorID to allAbstarcts
print("matching Author to abstracts")
DictForAuthor= dictionary_concatenation(DictOfAbstracts,DictOfPaperID)

print("removing stop words")
#remove stop words
DictForAuthor_new={}
for author in DictForAuthor.keys():
    concat=DictForAuthor[author]
    word_tokens = word_tokenize(concat)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    DictForAuthor_new[author]=' '.join(filtered_sentence)


#Get embedding for each author
print("creating Author embeddings")
AuthorEmbedding = {}
for author in DictForAuthor:
    AuthorEmbedding[float(author)] = Embed_Author(DictForAuthor[author])


print("set up train features")
X_train = np.zeros((n_train, 768+5+64))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['author']
    X_train[i,:768] = AuthorEmbedding[node]
    X_train[i,768] = G.degree(node)
    X_train[i,769] = core_number[node]
    X_train[i,770] = pr[node]
    X_train[i,771] = centrality[node]
    X_train[i,772] = cc[node]
    X_train[i,773:837] = dw[mapping[node]]
    y_train[i] = row['hindex']

print("set up test features")
X_test = np.zeros((n_test, 768+5+64))
for i,row in df_test.iterrows():
    node = row['author']
    X_test[i,:768] = AuthorEmbedding[node]
    X_test[i,768] = G.degree(node)
    X_test[i,769] = core_number[node]
    X_test[i,770] = pr[node]
    X_test[i,771] = centrality[node]
    X_test[i,772] = cc[node]
    X_test[i,773:837] = dw[mapping[node]]


print("creating model")

reg = MLPRegressor(
    hidden_layer_sizes=(150, 200, 100, 30),
    learning_rate_init=0.001,
    activation='relu',
    max_iter=200,
    random_state=2,
    verbose=True
);
    
print("fitting")
reg.fit(X_train, y_train)

print("getting predictions")
y_pred = reg.predict(X_test)

print("write the predictions to file")
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))


df_test.loc[:,["author","hindex"]].to_csv('submission.csv', index=False)
