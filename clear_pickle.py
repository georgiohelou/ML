import pickle
import os
from typing import Dict
from nltk.tokenize import word_tokenize
from processing_abstract import process_abstracts
from process_authorFile import process_authorFiles
from final_dico_creation import dictionary_concatenation
from Text_Embedding import Embed_Author

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
              'above', 'up', 'to', 'ours', 'had', 'she', 'when', 'at', 'any', 'before', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

infile = open('abstracts_data.pkl', 'rb')
DictOfAbstracts = pickle.load(infile)
infile.close()

print("extracting paperID")
DictOfPaperID = process_authorFiles()
#print(DictOfPaperID)

##Create Dict of AuthorID to allAbstarcts
print("matching Author to abstracts")
DictForAuthor = dictionary_concatenation(DictOfAbstracts, DictOfPaperID)

print("removing stop words")
#remove stop words
DictForAuthor_new = {}
for author in DictForAuthor.keys():
    print(author)
    concat = DictForAuthor[author]
    word_tokens = word_tokenize(concat)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    DictForAuthor_new[author] = ' '.join(filtered_sentence[:510])

a_file = open("DictForAuthor_new.pkl", "wb")
pickle.dump(DictForAuthor_new, a_file)
a_file.close()
