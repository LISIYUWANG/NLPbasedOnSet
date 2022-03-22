# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 19:03
# @Author  : naptmn
# @File    : pretraineduser.py
# @Software: PyCharm
import gensim
import numpy as np
import pandas as pd
import torch
if __name__ =='__main__':
    embed_path = "vector/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"
    wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')
    weight_numpy = np.load(file="vector/emebed.ckpt.npy")
    embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
    word2idx = pd.read_pickle("vector/word2idx.ckpt")
    idx2word = pd.read_pickle("vector/idx2word.ckpt")
    sentences = ["我", "爱", "北京", "天安门"]
    # 得到这几个词的ID
    ids = torch.LongTensor([word2idx[item] for item in sentences])
    print(ids)
    wordvector = embedding(ids)
    # 通过ID结合ix2word得到向量
    print(wordvector)
    print(wordvector.shape)
    # 使用gensim找最相近的词
    most_similar = wv_from_text.most_similar(["爱"], topn=10)
    print(most_similar)