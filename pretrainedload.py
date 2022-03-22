# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 16:32
# @Author  : naptmn
# @File    : pretrainedload.py
# @Software: PyCharm
import gensim
import numpy as np
import pandas as pd
if __name__ =='__main__':
    # 是否第一次使用 没有bin文件
    FIRST = False
    # tencent 预训练的词向量文件路径
    vec_path = "vector/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
    embed_path = "vector/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"
    wv_from_text = None
    if FIRST:
        # 加载词向量文件
        wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
        # 加载词向量文件
        wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
        # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
        wv_from_text.init_sims(replace=True)
        wv_from_text.save(vec_path.replace(".txt", ".bin"))

    else:
        # 之后可以用下面的方式加载词向量
        wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')


    # 加载完成
    #--------------------------------------------------------------------------------------------------------#
    # 提取index
    # 获取所有词
    vocab = wv_from_text.index_to_key
    # 获取所有向量
    word_embedding = wv_from_text.vectors

    # 将向量和词保存下来
    word_embed_save_path = "vector/emebed.ckpt"
    word_save_path = "vector/word.ckpt"
    np.save(word_embed_save_path, word_embedding)
    pd.to_pickle(vocab, word_save_path)

    # 加载保存的向量和词 并且构建word2idx和idx2word字典
    weight_numpy = np.load(file="vector/emebed.ckpt.npy")
    vocab = pd.read_pickle(word_save_path)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    pd.to_pickle(word2idx,"vector/word2idx.ckpt")
    pd.to_pickle(idx2word,"vector/idx2word.ckpt")

