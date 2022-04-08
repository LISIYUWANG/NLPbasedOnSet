# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 12:50
# @Author  : naptmn
# @File    : corewords.py
# @Software: PyCharm
# 用来得到核心词
# 根据词频进行计算
import tools
from tqdm import tqdm
import json


def getW2CDict(dic):
    """
    根据字典进行词频统计 返回“词-词频”的字典
    :param dic:读入的字典
    :return:按照词频降序排序“词-词频”的字典
    """
    wordList = []
    countList = []
    for i in tqdm(range(len(dic))):
        wordList.append(dic[i]['ci'])
        countList.append(getCount(dic[i]['ci'], dic))
    wordDict = dict(zip(wordList, countList))
    wordDict = sorted(wordDict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return wordDict


def getCount(word, dic):
    """
    得到一个词在字典解释中的词频
    :param word:
    :param dic:
    :return:词频
    """
    count = 0
    for i in range(len(dic)):
        for j in range(len(dic[i]['explanation'])):
            if word == dic[i]['explanation'][j]:
                count += 1
    return count


def getTopN(n, w2c):
    """
    得到“词-词频”的字典的TopN
    :param n: N
    :param w2c:“词-词频”字典
    :return: TopN的List
    """
    topN = []
    for i in range(n):
        topN.append(w2c[i])
    return topN


def saveDict(w2c):
    """
    保存一个“词-词频”字典
    :param w2c: 要保存的“词-词频”字典
    """
    temp = {}
    with open('./data/w2c.json', 'w') as f:
        for _, v in enumerate(w2c):
            print(v)
            # v.strip('\n')
            temp[v[0]] = v[1]
        json_dict = json.dumps(temp, indent=4, ensure_ascii=False)
        f.write(json_dict)


if __name__ == '__main__':
    dic = tools.getDic(path='./data/ci.json')
    w2c = getW2CDict(dic)
    saveDict(w2c)
    print(getTopN(30, w2c))
