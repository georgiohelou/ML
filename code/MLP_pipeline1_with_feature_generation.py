import pandas as pd
import numpy as np
from numpy import sqrt
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from page import pagerank
from deepwalk_generate import DpWalk
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
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
from processTextData import processTextData
import yolo
print("All libraries imported.")

print("processing text data")
processTextData()

print("please relocate 'fullEmbeddings_random.pkl' to the 'features' folder before running the 2nd part of the pipeline")
