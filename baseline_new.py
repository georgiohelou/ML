import os
import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from page import pagerank
from Bert_function import author_embedding
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from Text_Embedding import Embed_Author
import pickle
from MLP import prepare_data, MLP, train_model, evaluate_model, predict
import nltk
# nltk.download('punkt')
#*** use python -m nltk.downloader punkt ****
from nltk.tokenize import word_tokenize


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

# computes structural features for each node
print("calculating core_number")

infile = open('coreNumber.pkl','rb')
core_number = pickle.load(infile)
infile.close()


#node centrality
print("calculating centrality")
infile = open('centrality.pkl','rb')
centrality = pickle.load(infile)
infile.close()

#Clustering Coefficient
print("calculating Clustering Coefficient")

infile = open('ClusteringCoefficient.pkl','rb')
cc = pickle.load(infile)
infile.close()


#computes the page rank
print("calculating Page rank")
infile = open('PageRank.pkl','rb')
pr = pickle.load(infile)
infile.close()


print("calculating deep walk")
#computes Deep Walk
infile = open('mapping.pkl','rb')
mapping = pickle.load(infile)
infile.close()

infile = open('deepWalk.pkl','rb')
dw = pickle.load(infile)
infile.close()



print("generating word embeddings")
infile = open('fullEmbeddings.pkl','rb')
AllAuthorEmbeddings = pickle.load(infile)
infile.close()

infile = open('DictForAuthor_new.pkl','rb')
DictForAuthor_new = pickle.load(infile)
infile.close()

AuthorEmbedding={}
counter=0
for author in DictForAuthor_new: 
    AuthorEmbedding[float(author)]=AllAuthorEmbeddings[counter]
    counter=counter+1


print("set up train features")
# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number 
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
# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number
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
    # what if we change our layer sizes?

    #hidden_layer_sizes=(5,17), 
    #hidden_layer_sizes=(5,17,5), 
    # hidden_layer_sizes=(10,17,10), 
    hidden_layer_sizes=(100,200), 
    # what if we change our learning rate?
    learning_rate_init=0.01,
    # what if we change our activation function? (relu, tanh, identity)
    activation='relu',
    max_iter=200,
    random_state=2, # if set to None, this is random, to an int, static seed
    # set this to True to see how well we are learning over the iterations
    verbose=True
);
    
print("fitting")
reg.fit(X_train, y_train)

print("getting predictions")
y_pred = reg.predict(X_test)

print("write the predictions to file")
# write the predictions to file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))


df_test.loc[:,["author","hindex"]].to_csv('submission2.csv', index=False)





