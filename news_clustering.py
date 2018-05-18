import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import csv


class ClusterAnalysis(object):

    def create_tfidf_matrix(self, file='data.csv', text_column='news_content', label='keyword', encoding='gbk'):
        """
        打开file，读取text_column和label两个列，并将text_column列的文档数据生成tfidf矩阵
        :param file: csv文件名或者csv文件路径
        :param text_column: csv文件中的存储文本内容的列名，在本研究中是新闻内容
        :param label:  标注数据。在本文研究中是新闻关键词
        :param encoding:  csv文件编码方式
        :return:
        """
        df = pd.read_csv(file, encoding=encoding)
        # 文档数据
        self.documents = df[text_column]
        # 真正的文档标签
        self.true_label = df[label]  

        texts = [' '.join(jieba.lcut(doc)) for doc in self.documents]

        # 初始化tfidf特征器。舍弃信息量不大的词语，
        # 这里max_df=0.5，如果一个词语出现在50%以上的文档中，那么这个词是没有信息量的词语。舍弃这种词
        self.tfidf = TfidfVectorizer(max_df=0.5)

        # 生成tfidf矩阵
        self.matrix = self.tfidf.fit_transform(texts)


# 构建tfidf文档特征矩阵
ca.create_tfidf_matrix()

