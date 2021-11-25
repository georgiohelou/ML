# import sister
# embedder = sister.MeanEmbedding(lang="en")


# def Embed_Author(sentence):
#     vector = embedder(sentence)  # 300-dim vector
#     return vector

from sent2vec.vectorizer import Vectorizer
from essential_generators import DocumentGenerator
gen = DocumentGenerator()


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def Embed_Author(sentences):
    counter=0
    AllVectors=[]
    for group in chunker(sentences, 1000):
        print("Vectorizing batch"+str(counter))
        counter=counter+1
        vectorizer = Vectorizer()
        vectorizer.bert(group)
        vectors = vectorizer.vectors
        AllVectors.extend(vectors)
        print("Finished batch")
    print("Finished All batches")
    return AllVectors