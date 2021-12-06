#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:35:46 2021

@author: nathan.pollet
"""

import pickle; import os

masterlist = []
root = '/Users/nathanpollet/ML/good_save/'
for k in range(37):
    print(k)
    print(f"good_save/bucket{k}.pkl")
    with open(f"good_save/bucket{k}.pkl", 'rb') as f:
        bucketlist = pickle.load(f)
        for i in bucketlist:
            masterlist.append(i)
    f.close()
    
print("len(masterlist) : ", len(masterlist))
print("saving masterpickle...")
with open('fullEmbeddings.pkl', 'wb') as f1:
    pickle.dump(masterlist, f1)
    
