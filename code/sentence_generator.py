import sys
from Text_Embedding import Embed_Author
import pickle
import os
from typing import Dict


### OLD / ALREADY RAN ###
# print("creating Author embeddings")
# AuthorEmbedding = {}
# sentences = []
# for author in DictForAuthor_new:
#     sentences.append((DictForAuthor_new[author]))

# with open('sentences.pkl', 'wb') as f:
#     pickle.dump(sentences, f)
# f.close()
############

infile = open('sentences.pkl', 'rb')
sentences = pickle.load(infile)
infile.close()

print(type(sentences))

var = input("enter bucket ID:\n")
var = int(var)
print("You entered: " + str(var))

if var < 36:
    bucket = sentences[6000*var:6000*(var+1)]
else:
    bucket = sentences[6000*var:]

bucketAuthorEmbeddings = Embed_Author(bucket)
with open(f"bucket{var}.pkl", 'wb') as f:
    pickle.dump(bucketAuthorEmbeddings, f)

print("pickle file saved.")
print("Done")


# AllAuthorEmbeddings = Embed_Author(bucket)
# print("Finished Embedding")

# counter = 0
# for author in DictForAuthor_new:
#     AuthorEmbedding[float(author)] = AllAuthorEmbeddings[counter]
#     counter = counter+1
