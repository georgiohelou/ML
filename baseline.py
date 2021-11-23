import os
import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from page import pagerank
from Bert_function import author_embedding
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from MLP import prepare_data, MLP, train_model, evaluate_model, predict

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
core_number = nx.core_number(G)
#node centrality
centrality = nx.eigenvector_centrality(G)

#Clustering Coefficient
print("calculating Clustering Coefficient")
cc=nx.clustering(G)
#computes the page rank
print("calculating Page rank")
pr=nx.pagerank(G,0.4)

print("calculating deep walk")
#computes Deep Walk
mapping = {old_label:new_label for new_label, old_label in enumerate(G.nodes())}
#
H = nx.relabel_nodes(G, mapping)
dw=DpWalk(H)


#*******Get word embeddings
    #iterate for each author
    #concatenate all abstracts in abstract.txt
    #relating to author id in author_papers

    #call the text on author_embedding()
    #-> will get back torch tensor of  Size([768])
    #convert to dictionary

##Create Dict of paperID to abstract
print("extracting abstracts")
DictOfAbstracts= process_abstracts()
#print(DictOfAbstracts)

##Create Dict of AuthorID to paperID
print("extracting paperID")
DictOfPaperID= process_authorFiles()

#print(DictOfPaperID)

##Create Dict of AuthorID to allAbstarcts
print("matching Author to abstracts")
DictForAuthor= dictionary_concatenation(DictOfAbstracts,DictOfPaperID)
#print(DictForAuthor)

##Get embedding for each author
print("creating Author embeddings")
AuthorEmbedding = {}
for author in DictForAuthor: 
    if len(DictForAuthor[author].split()) < 512:
        AuthorEmbedding[float(author)] = author_embedding(DictForAuthor[author])
    else:
        AuthorEmbedding[float(author)] = author_embedding(' '.join(DictForAuthor[author].split()[:512]))


X = np.zeros((n_train, 768+5+64+1))
#775

#create array of features and y
for i,row in df_train.iterrows():
    node = row['author']
    X[i,:768] = AuthorEmbedding[node].numpy()
    X[i,768] = G.degree(node)
    X[i,769] = core_number[node]
    X[i,770] = pr[node]
    X[i,771] = centrality[node]
    X[i,772] = cc[node]
    X[i,773:837] = dw[mapping[node]]
    X[i,837] = row['hindex']

print(X)
#convert to CSV file
pd.DataFrame(X).to_csv("file.csv", header=None, index=None)



# prepare the data
path = 'file.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(837)
# train the model
train_model(train_dl, model)
# evaluate the model
mse = evaluate_model(test_dl, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))


X_t = np.zeros((n_test, 768+5+64))
y_pred = np.zeros(n_test)
for i,row in df_test.iterrows():
    node = row['author']
    X_t[i,:768] = AuthorEmbedding[node].numpy()
    X_t[i,768] = G.degree(node)
    X_t[i,769] = core_number[node]
    X_t[i,770] = pr[node]
    X_t[i,771] = centrality[node]
    X_t[i,772] = cc[node]
    X_t[i,773:837] = dw[mapping[node]]

    y_pred[i] = predict(X_t[i], model)


# # make a single prediction (expect class=1)
# row = X_t[0]
# yhat = predict(row, model)
# print('Predicted: %.3f' % yhat)

# print(df_test)

# # create the training matrix. each node is represented as a vector of 3 features:
# # (1) its degree, (2) its core number ..

# print("Setting train data")
# X_train = np.zeros((n_train, 768+6))
# y_train = np.zeros(n_train)
# for i,row in df_train.iterrows():
#     node = row['author']
#     X_train[i,:768] = AuthorEmbedding[node].numpy()
#     X_train[i,768] = G.degree(node)
#     X_train[i,769] = core_number[node]
#     X_train[i,770] = pr[node]
#     X_train[i,771] = centrality[node]
#     X_train[i,772] = cc[node]
#     X_train[i,773] = dw[mapping[node]]

#     #*******Add word encoding features
#     print(X_train)
#     y_train[i] = row['hindex']

# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number ..
# print("Setting test data")
# X_test = np.zeros((n_test, 6))
# for i,row in df_test.iterrows():
#     node = row['author']
#     X_test[i,0] = G.degree(node)
#     X_test[i,1] = core_number[node]
#     X_test[i,2] = pr[node]
#     X_test[i,3] = centrality[node]
#     X_test[i,4] = cc[node]
#     X_test[i,5] = dw[mapping[node]]
#     #*******Add word encoding features


# # train a regression model and make predictions
# print("Creating Model")
# reg = Lasso(alpha=0.1)
# print("Started Training")
# reg.fit(X_train, y_train)

# print("Started Testing")
# y_pred = reg.predict(X_test)

# # write the predictions to file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))
print(df_test)

df_test.loc[:,["author","hindex"]].to_csv('submission2.csv', index=False)





