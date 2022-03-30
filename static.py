# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 15:22
# @Author  : naptmn
# @File    : static.py
# @Software: PyCharm
import json
from time import sleep


def getDic(path):
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
if __name__ == '__main__':
    path = './data/xiaoxueci.json'
    dic = getDic(path)
    print('字典条目为：',len(dic))

    exp = getList('实在', dic)
    print('筛选前')
    print(exp)
    print('筛选后')
    print(filter(exp))
    # dic = getDic(path)
    # print(getList('实在', dic))
    # print(dic[0])
