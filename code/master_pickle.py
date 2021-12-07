#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:35:46 2021

Concatenate all the bucket pickle files into 1

"""

import pickle
import os

masterlist = []
for k in range(37):
    print(k)
    print(f"bucket{k}.pkl", 'rb')
    with open(f"bucket{k}.pkl", 'rb') as f:
        bucketlist = pickle.load(f)
        # print("len(bucketlist) : ", len(bucketlist))
        for i in bucketlist:
            masterlist.append(i)
    f.close()

print("len(masterlist) = ", len(masterlist))
print("saving masterpickle...")
with open('fullEmbeddings_random.pkl', 'wb') as f1:
    pickle.dump(masterlist, f1)
