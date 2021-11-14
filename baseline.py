import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso
from page import pagerank
from Bert_function import author_embedding


# read training data
df_train = pd.read_csv('train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]

# load the graph    
G = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)


# computes structural features for each node
core_number = nx.core_number(G)

#computes the page rank
pr=nx.pagerank(G,0.4)

#*******Get word embeddings
    #iterate for each author
    #concatenate all abstracts in abstract.txt
    #relating to author id in author_papers

    #call the text on author_embedding()
    #-> will get back torch tensor of  Size([768])
    #convert to dictionary



# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number ..
X_train = np.zeros((n_train, 3))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['author']
    X_train[i,0] = G.degree(node)
    X_train[i,1] = core_number[node]
    X_train[i,2] = pr[node]

    #*******Add word encoding features

    y_train[i] = row['hindex']

# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number ..
X_test = np.zeros((n_test, 3))
for i,row in df_test.iterrows():
    node = row['author']
    X_test[i,0] = G.degree(node)
    X_test[i,1] = core_number[node]
    X_test[i,2] = pr[node]
    #*******Add word encoding features


# train a regression model and make predictions
reg = Lasso(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# write the predictions to file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))


df_test.loc[:,["author","hindex"]].to_csv('submission.csv', index=False)





