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

'''
    Send resulting 'sentences_cleaned_random_max450words.pkl' file to school computers to compute BERT embeddings.
    This parallelized process is done via 'sentence_generator.py' which divides the authors-abstracts dataset in 37
    buckets and computes BERT embeddings for a given one. The bucket number/ID is specified by the user through the 
    command line interface.
    Once all 37 "bucket embeddings" generated, they are reassembled into a single pickle file by running 'master_pickle.py'.
    This results in the generation of 'fullEmbeddings_random.pkl' which should be placed in the 'features' folder.
'''

infile = open('sentences_cleaned_random_max450words.pkl', 'rb')
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
