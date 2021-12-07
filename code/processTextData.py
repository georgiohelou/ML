'''
WARNING: successfully running this program requires a LARGE amount of RAM and ideally the BERT embeddings generation
process (see end of code) should be parallelized on several machines. 
'''
import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from page import pagerank
import random
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from final_dico_creation import dictionary_concatenation2  # keeps abstracts separated
from Text_Embedding import Embed_Author
import pickle
from MLP import prepare_data, MLP, train_model, evaluate_model, predict
import nltk
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
# nltk.download('punkt')
#*** use python -m nltk.downloader punkt ****
from nltk.tokenize import word_tokenize

def processTextData():

    process_abstracts()

    #Create Dict of paperID to abstract
    print("extracting abstracts")
    DictOfAbstracts={}
    DictOfAbstracts= process_abstracts() ## creates abstracts_data.pkl
    infile = open('./data/abstracts_data.pkl', 'rb')
    DictOfAbstracts = pickle.load(infile)
    infile.close()

    print("extracting authors-paperIDs [dico]")
    DictOfPaperID = process_authorFiles()

    #in this scenario we will randomly remove words to match BERT input size (so use method "2")
    print("matching authors to all of their abstracts *separated* [dico-list]")
    DictForAuthor = dictionary_concatenation2(DictOfAbstracts, DictOfPaperID)

    print("matching authors to all of their abstracts *separated* [dico-list]")
    DictForAuthor = dictionary_concatenation2(DictOfAbstracts, DictOfPaperID)

    with open('./data/authors_with_all_abstracts_separated.pkl', 'wb') as f:
        pickle.dump(DictForAuthor, f)
    f.close()
    print("Operation completed. Check authors_with_all_abstracts_separated.pkl file.")

    ''' 
    To generate a dictionary matching the authors to their *separated & cleaned* abstracts 
    [dico in which keys=author_IDs & values=a list with the separated abstracts cleaned and w/o stopwords].
    Dictionary name: DictForAuthor_cleaned
    Dictionary saved as: authors_with_all_abstracts_separated_cleaned.pkl
    '''
    DictForAuthor_cleaned = DictForAuthor  # make copy
    specialChars = ".,()"
    specialChars2 = ['\r', '\n', '\t', '\\r', '\\n',
                    '\\t', '\\r\\n\\r\\n', '\\r\\n\\r\\n']
    stop_words = {'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                'above', 'up', 'to', 'ours', 'had', 'she', 'when', 'at', 'any', 'before', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    for authorID in tqdm(DictForAuthor_cleaned):  # iterate over dictionary
        # iterate over the author's separate abstracts
        for i in range(len(DictForAuthor_cleaned[authorID])):
            abstractText = str(DictForAuthor_cleaned[authorID][i])
            ## clean string and remove stopwords ##
            for specialChar in specialChars:
                abstractText = abstractText.replace(specialChar, "")
            for specialChar2 in specialChars2:
                abstractText = abstractText.replace(specialChar2, "")
            abstractText = abstractText.replace("   ", " ")
            abstractText = abstractText.replace("  ", " ")
            abstractText = abstractText.strip()
            abstractText_words = abstractText.split()  # isolate words
            for j in range(len(abstractText_words)):
                if abstractText_words[j] in stop_words:
                    abstractText_words[j] = ""
            # for stop_word in stop_words:
            #     abstractText = abstractText.replace(stop_word, "")

            temp = ' '.join(abstractText_words)
            temp = temp.strip()
            temp = temp.replace("  ", " ")
            temp = temp.replace("   ", " ")
            DictForAuthor_cleaned[authorID][i] = ' '.join(temp.split())

    with open('./data/authors_with_all_abstracts_separated_cleaned.pkl', 'wb') as f:
        pickle.dump(DictForAuthor_cleaned, f)
    f.close()
    print("Operation completed. Check authors_with_all_abstracts_separated_cleaned.pkl file.")

    print("random method used")
    minNumWords = 450
    new_dico = {}
    for authorID in tqdm(DictForAuthor_cleaned):
        abstractConcat = ' '.join(DictForAuthor_cleaned[authorID])
        abstractConcatWord = abstractConcat.split()
        numWords = len(abstractConcatWord)
        #abstract size already inferior to bert input size
        if numWords <= minNumWords:
            new_dico[authorID] = ' '.join(abstractConcatWord)
            continue
        #random method chosen to remove extra words
        else:
            while(numWords > minNumWords):
                rmv_index = random.randint(0, numWords-1)
                del abstractConcatWord[rmv_index]
                numWords -= 1
            # print("' '.join(abstractConcatWord) : \n", ' '.join(abstractConcatWord), "\n")
            new_dico[authorID] = ' '.join(abstractConcatWord)
            if len(new_dico[authorID].split()) > minNumWords:
                print("Error")
                print(authorID)
                break

    with open('./data/authors_with_all_abstracts_random_max450words.pkl', 'wb') as f:
        pickle.dump(new_dico, f)
    f.close()
    print("Operation completed. Check authors_with_all_abstracts_random_max450words.pkl file.")

    sentences_cleaned_450words = []
    for author in new_dico:
        sentences_cleaned_450words.append(new_dico[author])

    with open('./data/sentences_cleaned_random_max450words.pkl', 'wb') as f:
        pickle.dump(sentences_cleaned_450words, f)
    f.close()

    '''
    Send resulting 'sentences_cleaned_random_max450words.pkl' file to school computers to compute BERT embeddings.
    This parallelized process is done via 'sentence_generator.py' which divides the authors-abstracts dataset in 37
    buckets and computes BERT embeddings for a given one. The bucket number/ID is specified by the user through the 
    command line interface.
    Once all 37 "bucket embeddings" generated, they are reassembled into a single pickle file by running 'master_pickle.py'.
    This results in the generation of 'fullEmbeddings_random.pkl' which should be placed in the 'features' folder.
    '''
    return
