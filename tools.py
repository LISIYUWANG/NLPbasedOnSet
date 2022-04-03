# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 20:42
# @Author  : naptmn
# @File    : tools.py
# @Software: PyCharm
import gensim
import numpy as np
import pandas as pd
import torch
import json

def getVecByWord(word):
    """
    传入一个词 返回词向量
    :param word: 需要的词
    :return:词向量
    """
    # 加载
    weight_numpy = np.load(file="vector/emebed.ckpt.npy")
    embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
    word2idx = pd.read_pickle("vector/word2idx.ckpt")
    # 得到个词的ID
    ids = torch.LongTensor([word2idx[word]])
    wordvector = embedding(ids).squeeze()
    # 通过ID结合ix2word得到向量
    return wordvector

def getSet(sentence):
    """
    传入一个列表，返回这个列表的词向量集合
    :param sentence: 分词列表
    :return:
    """
    set = []
    for word in sentence:
        set.append(getVecByWord(word))
    return set

def getDic(path='./data/xiaoxueci.json'):
    """
    根据路径读取Json
    :param path: 字典路径
    :return: 读好的字典（List形式）
    """
    with open(path) as dict:
        load_dict = json.load(dict)
        return load_dict


def getList(word, dic):
    """
    通过一个词查找他在词典中的解释 返回对应列表，如果没有找到则返回None
    :param word:查找的词
    :param dic:词典
    :return:分词解释List
    """
    for each in dic:
        if each['ci'] == word:
            return each['explanation']
    return 'None'

def filter(exp):
    """
    用来将得到的解释进行过滤 去掉语气词等
    :param exp:
    :return:
    """
    with open('./data/out.json') as out:
        out_dict = json.load(out)
        out_dict = out_dict['punctuation']+out_dict['nothingword']+out_dict['digit']
        for word in exp:
            for out_type in out_dict:
                for out_word in out_type:
                    if out_word in word :
                        exp.remove(word)
    return exp
if __name__ =='__main__':
    # print(getVecByWord('太阳'))
    print(getSet(['苹果','是','红色','水果']))