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
        #文档数据
        self.documents = df[text_column]
        #真正的文档标签
        self.true_label = df[label] #标签是 关键词。后续可以比较聚类效果


        texts = [' '.join(jieba.lcut(doc)) for doc in self.documents]

        #初始化tfidf特征器。舍弃信息量不大的词语，
        #这里max_df=0.5，如果一个词语出现在50%以上的文档中，那么这个词是没有信息量的词语。舍弃这种词
        self.tfidf = TfidfVectorizer(max_df=0.5)

        #生成tfidf矩阵
        self.matrix = self.tfidf.fit_transform(texts)


    def find_best_k_value(self, min_k=40, max_k=100):
        """
         搜寻最佳的聚类数 k
        :param min_k: k值搜寻范围的下阈
        :param max_k: k值搜寻范围的上阈
        :return:
        """

        #记录聚类k及其得分，最终画图从中找出聚类误差最小的k值
        ks, scores = [], []
        for k in range(min_k, max_k):
            #c初始化Kmeans估计器
            estimator = KMeans(n_clusters=k)
            #学习tfidf矩阵数据中的分布规律
            estimator.fit(self.matrix)
            #聚类数为k时，聚类效果的误差值
            score = - estimator.score(self.matrix)
            print(k, score)
            ks.append(k)
            scores.append(score)

        #制作聚类的误差图
        plt.figure(figsize=(8, 5))
        plt.plot(ks, scores, label='find the best k value', color='red', linewidth=1)
        plt.xlabel('cluster k')
        plt.ylabel('Errors')
        plt.show()


    def classify_docuements(self, best_k):
        """
        根据最佳聚类数k， 使用Kmeans聚类算法对文本数据进行标注
        :param best_k: 聚类准确率最好的 k值
        :return:
        """
        self.best_k = best_k
        #c初始化Kmeans估计器
        estimator = KMeans(n_clusters=best_k)
        # 学习tfidf矩阵数据中的分布规律
        estimator.fit(self.matrix)

        #机器给所有的文档打上的标签
        #此处self.label_pred 为一维列表，列表长度等于 文档的数目
        self.label_pred = estimator.labels_

        #k类就有k个中心点。
        #如果特征有n个，那么此处的self.centroids是 best_k*n 的矩阵
        self.centroids = estimator.cluster_centers_

        #order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    def save2csv(self, outputfile='聚类结果.csv'):
        """
        将分类结果保存到outputfile中
        :param outfile:  此处为csv文件的文件名或者文件路径
        :return:
        """
        #保存聚类算法的分类结果 到output文件夹中。
        with open('output/'+str(self.best_k)+'_'+outputfile, 'a+', encoding='gbk', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(('新闻内容', '真实标签', '聚类'))
            for idx, document in enumerate(self.documents):
                writer.writerow((document, self.true_label[idx], self.label_pred[idx]))





#初始化聚类分析器
ca = ClusterAnalysis()

#构建tfidf文档特征矩阵
ca.create_tfidf_matrix()

#寻找最佳k值
#ca.find_best_k_value()

#学习数据中的类群规律
for k in range(50,57):
    ca.classify_docuements(best_k = k)
    #保存分类数据
    ca.save2csv()
