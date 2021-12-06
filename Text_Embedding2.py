# import sister
# embedder = sister.MeanEmbedding(lang="en")


# def Embed_Author(sentence):
#     vector = embedder(sentence)  # 300-dim vector
#     return vector

# from sent2vec.vectorizer import Vectorizer
import numpy as np
import gensim
import torch
import transformers as ppb


class Vectorizer:
    def __init__(self):
        self.vectors = []

    def bert(self, sentences, pretrained_weights='distilbert-base-uncased'):
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True), sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        vectors = last_hidden_states[0][:, 0, :].numpy()
        self.vectors = vectors


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def Embed_Author(sentences):
    counter=0
    AllVectors=[]
    for group in chunker(sentences, 500):
        print("Vectorizing batch"+str(counter))
        counter=counter+1
        vectorizer = Vectorizer()
        vectorizer.bert(group)
        vectors = vectorizer.vectors
        AllVectors.extend(vectors)
        print("Finished batch")
    print("Finished All batches")
    print(AllVectors)
    return AllVectors

