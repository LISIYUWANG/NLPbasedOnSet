# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 23:36
# @Author  : naptmn
# @File    : jieba.py
# @Software: PyCharm
import jieba
import gensim
strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
seg_list=jieba.cut(strs[0])
print("[默认是精确模式]"+"/".join(seg_list))
seg_list=jieba.cut(strs[1],cut_all=False)
print("[精确模式]"+"/".join(seg_list))
seg_list=jieba.cut(strs[1],cut_all=True)
print("[全模式]"+"/".join(seg_list))
seg_list=jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print("[搜索引擎模式]"+"/".join(seg_list))


# /////////////////////////////////////////////////////////////////////////////////////
import jieba
import jieba.posseg as pseg
words=pseg.cut("我爱北京天安门")
print("[默认模式]")
for word ,flag in words:
    print("%s %s" % (word,flag))