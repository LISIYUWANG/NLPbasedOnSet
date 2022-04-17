# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 19:58
# @Author  : naptmn
# @File    : train.py
# @Software: PyCharm
import tools
import tqdm
import torch
import numpy as np
import pandas as pd

def getVecByWord(word):
    """
    传入一个词 返回词向量  这里将读入的npy文件常驻内存了 增加训练速度
    :param word: 需要的词
    :return:词向量
    """

    # 得到个词的ID
    ids = torch.LongTensor([word2idx[word]])
    wordvector = embedding(ids).squeeze()
    # 通过ID结合ix2word得到向量
    if torch.cuda.is_available():
        wordvector = wordvector.cuda()
    return wordvector



def nearest(core, word):
    """
    得到与一个非核心词最近的核心词
    :param core: 核心词词典
    :param word: 非核心词
    :return:最近的核心词
    """
    # 计算离一个词最近的核心词
    max = 99999 # 没查到怎么设置最大值 这里有点问题
    nrCore = None
    for cWord in core:
        dis = tools.getEuclidean(getVecByWord(cWord['ci']),getVecByWord(word['ci']))
        if dis<max:
            max = dis
            nrCore = cWord
    return nrCore


def saveWeight(weight_numpy, savePath):
    """
    保存一个权重文件
    :param weight_numpy:权重文件
    :param savePath: 保存路径
    """
    np.save(savePath, weight_numpy)

def judgeCore(core, word):
    """
    判断一个词是否是核心词
    :param core: 核心词词典
    :param word: 词
    :return: True->核心词  False->非核心词
    """
    # 判断一个word是不是核心词
    for c in core:
        if word == c['ci']:
            return True
    return False

def train(core, dict):
    """
    训练函数
    :param core:核心词词典
    :param dict:待训练词典
    :return:
    """
    for word in tqdm.tqdm(dict):
        # 如果是核心词 则跳过 不调整
        if judgeCore(core, word['ci']):
            continue

        # 获取最近核心词
        nr = nearest(core, word)

        # 获取最近核心词以及
        coreVec = getVecByWord(nr['ci'])
        wordVec = getVecByWord(word['ci'])
        lr = tools.getLr(wordVec, coreVec, Step)

        # 靠近核心词
        if Epoch > Step:
            print('Epoch应当小于Step，训练结束')
            return
        else:
            for i in range(Epoch):
                wordVec = tools.nextStep(wordVec,lr)

        # 得到该词的id
        ids = torch.LongTensor([word2idx[word['ci']]])

        # 保存词向量
        wordVec = np.array(wordVec)
        weight_numpy[ids] = wordVec


if __name__ =='__main__':

    # 加载
    weight_numpy = np.load(file="vector/emebed.ckpt.npy")
    embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
    # word2idx 词与索引的对照
    word2idx = pd.read_pickle("vector/word2idx.ckpt")
    pathCore = './data/coredict.json'
    pathDict = './data/xiaoxueci.json' # 葡萄灰 寝不安席 轻重倒置
    # 保存调整过后的路径
    saveTunedPath = 'vector/emebedTuned.ckpt.npy'

    # test
    # weight_numpytuned = np.load(file="vector/emebedTuned.ckpt.npy")
    # ids = torch.LongTensor([word2idx['舍近求远']])
    # print(weight_numpytuned[ids])
    # print(weight_numpy[ids])
    # test


    # Epoch<Step
    Epoch = 1 # 词向量调整次数
    Step = 10 # lr划分系数

    # 读取核心词与词典
    Core = tools.getDic(pathCore)
    Dict = tools.getDic(pathDict)

    # 训练
    train(Core, Dict)

    # 保存调整后的
    saveWeight(weight_numpy, saveTunedPath)