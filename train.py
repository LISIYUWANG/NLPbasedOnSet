# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 19:58
# @Author  : naptmn
# @File    : train.py
# @Software: PyCharm
import tools

def nearest(Core, word):
    # 计算离一个词最近的核心词
    max = 99999
    nr = None
    for cWord in Core:
        dis = tools.getEuclidean(tools.getVecByWord(cWord['ci']),tools.getVecByWord(word))
        if dis<max:
            max = dis
            nr = cWord
    return nr

if __name__ =='__main__':
    pathCore = './data/coredict.json'
    pathDict = './data/xiaoxueci.json'
    Core = tools.getDic(pathCore)
    Dict = tools.getDic(pathDict)
