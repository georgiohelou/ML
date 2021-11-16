# import libraries
import networkx as nx 
from karateclub import DeepWalk

def DpWalk(G):
    # load the karate club dataset
    #G = nx.karate_club_graph()
    # load the DeepWalk model and set parameters
    #***** can increase dimension
    dw = DeepWalk(dimensions=1)
    # fit the model
    dw.fit(G)
    # extract embeddings
    embedding = dw.get_embedding()

    embedding = dict(enumerate(embedding.flatten()))

    return embedding 