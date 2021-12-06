import pickle
import os
from typing import Dict

# infile = open('DictForAuthor_new.pkl', 'rb')
# DictForAuthor_new = pickle.load(infile)
# # print(DictOfAbstracts)
# infile.close()

# with open("DictForAuthor_new.txt", 'w') as f:
#     for key, value in DictForAuthor_new.items():
#         f.write('%s:%s\n' % (key, value))

# f.close()
# # temp = pickle.dump
# # with open('file_test.txt', 'w') as file:
# #     file.write(pickle.dumps(DictOfAbstracts))

# templist = [1, 2, "bla", "vive le roi"]
# with open('templist.pkl', 'wb') as f:
#     pickle.dump(templist, f)

# print("pickle file saved")
# print("pickle file loading...")

with open('fullEmbeddings.pkl', 'rb') as f1:
    fullEmbeddings = pickle.load(f1)
print("new list: \n")
print(type(fullEmbeddings))
print(len(fullEmbeddings))
print(fullEmbeddings[0])


# print("-----")

# listeX = [1,2,3,4,5,6,7,8,9,10]
# for k in range(4):
#     print(k)
#     if k < 3:
#         bucket = listeX[3*k:3*(k+1)]
#         print(bucket)
#     else:
#         bucket = listeX[3*k:]
#         print(bucket)

# dict = {}
