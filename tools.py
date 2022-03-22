# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 20:42
# @Author  : naptmn
# @File    : tools.py
# @Software: PyCharm
import gensim
import numpy as np
import pandas as pd
import torch
# 传入一个词 返回词向量
def getVecByWord(word):
    # 加载
    weight_numpy = np.load(file="vector/emebed.ckpt.npy")
    embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
    word2idx = pd.read_pickle("vector/word2idx.ckpt")
    # 得到个词的ID
    ids = torch.LongTensor([word2idx[word]])
    wordvector = embedding(ids).squeeze()
    # 通过ID结合ix2word得到向量
    return wordvector
# 传入一个分词后的列表 返回词向量集合
def getSet(sentence):
    set = []
    for word in sentence:
        set.append(getVecByWord(word))
    return set
if __name__ =='__main__':
    # print(getVecByWord('太阳'))
    print(getSet(['苹果','是','红色','水果']))