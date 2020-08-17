import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import DBI as DBI
import time
import os
from angle import max_xielv
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
from tslearn.clustering import KShape
from tslearn.clustering import silhouette_score
from tslearn.clustering import KernelKMeans
from tslearn.barycenters import softdtw_barycenter
from decimal import *
import csv


class Cluster(object,):
    def __init__(self, centroid, seq_length, seq_train, seq_interval, start_point, dataset):
        self.Centroid = centroid   # 聚类的个数
        if dataset == 'Google':
            self.dataset = 'Google'
            self.path = "....\src\\new_google_usage.csv"
        elif dataset == 'Ali':
            self.dataset = 'Ali'
            self.path = "....\src\\2018usage.csv"
        self.train_path = None
        self.test_path = None
        self.seq_length = seq_length    # 子序列的长度
        self.seq_interval = seq_interval    # 子序列间的间隔长度
        self.start_point = start_point  # 样本集的起始点（起始点不同，样本集不同）
        self.seq_train = seq_train  # 样本训练的长度

    def data_partition(self):
        """
        历史时间序列数据划分
        :return:
        """

        data = pd.read_csv(self.path)
        data = pd.DataFrame(data)
        if self.dataset == 'Google':
            cpu = data['cpu']*100
            for i in range(len(cpu)):
                cpu[i] = round(cpu[i], 6)
            length = int(len(cpu) * 0.15)
            cpu = cpu[:length]
        else:
            cpu = data['cpu']
        train_length = int(len(cpu) * 0.8)  # 80%训练，20&测试集
        train = cpu[:train_length]
        test = cpu[train_length:]
        train.to_csv('train.csv')
        self.train_path = 'train.csv'
        test.to_csv('test.csv')
        self.test_path = 'test.csv'

    def sax(self, data):
        n_paa_segments = 10
        n_sax_symbols_avg = 8
        n_sax_symbols_slop = 8
        sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                  alphabet_size_slope=n_sax_symbols_slop)
        Sax_data = sax.inverse_transform(sax.fit_transform(data))
        data_new = np.reshape(Sax_data, (Sax_data.shape[0], Sax_data.shape[1]))
        return data_new

    def make_sample(self):
        """
        制作样本集，用于聚类
        :return:
        """
        path = [self.train_path, self.test_path]
        for filename in path:
            if filename == self.train_path:
                # 制作训练集
                raw_data = pd.read_csv(filename)
                df = pd.DataFrame(raw_data)
                data = []
                # 一段子时间序列的长度
                data_len = self.seq_length
                # 子序列间的间隔，180就是半小时，360就是一小时
                interval = self.seq_interval
                if self.dataset == 'Ali':
                    # 解决阿里数据集中原始时间序列中CPU的0值
                    for i in range(self.start_point, 14161 - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            tmp.append(value)
                        data.append(tmp)
                    for i in range(16038 + self.start_point, len(raw_data) - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            tmp.append(value)
                        data.append(tmp)
                elif self.dataset == 'Google':
                    for i in range(self.start_point, len(raw_data) - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            elif np.isinf(value):
                                print(value)
                                value = float(value)
                            tmp.append(value)
                        data.append(tmp)
                # 进行标准化处理
                mu = np.mean(data, axis=1)
                sigma = np.std(data, axis=1)
                standard = []
                for i in range(len(data)):
                    if sigma[i] == 0:
                        print(data[i])
                        continue
                    temp = (np.array(data[i]) - mu[i]) / sigma[i]
                    standard.append(temp)
                standard = pd.DataFrame(standard)
                directory_path = 'start_point_' + str(self.start_point)  # 不同划分方式的样本集放在不同文件中
                file = filename
                if os.path.exists(directory_path):  # 判断文件夹是否存在
                    standard.to_csv(directory_path + '\\' + file)
                else:
                    os.mkdir(directory_path)
                    standard.to_csv(directory_path + '\\' + file)
            else:
                # 制作测试集
                raw_test = pd.read_csv(filename)
                df = pd.DataFrame(raw_test)
                data = []
                # 一段子时间序列的长度
                data_len = self.seq_length
                interval = self.seq_interval
                for i in range(self.start_point, len(raw_test) - data_len, interval):
                    tmp = []
                    for j in range(0, data_len, 1):
                        value = df['cpu'].values[i + j]
                        if np.isnan(value):
                            value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                        elif np.isinf(value):
                            value = float(value)
                        tmp.append(value)
                    data.append(tmp)
                mu = np.mean(data, axis=1)
                sigma = np.std(data, axis=1)
                standard = []
                for i in range(len(data)):
                    temp = (np.array(data[i]) - mu[i]) / sigma[i]
                    standard.append(temp)
                standard = pd.DataFrame(standard)
                standard.to_csv('....\src\Clustering\\final_test.csv')

    def single_clustering(self, data_raw, data_new, centroid_num, model):
        """
        单次聚类
        :return:
        """
        seed = 0
        np.random.seed(seed)
        labels = []
        inertia = []
        centers = []
        if model == 'K-Means':
            kmeans = KMeans(n_clusters=centroid_num).fit(data_new)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            # 每个点到其簇的质心的距离之和，越小越好
            inertia = kmeans.inertia_
        elif model == 'DTW':
            sdtw_km = TimeSeriesKMeans(n_clusters=centroid_num, metric='softdtw', max_iter=2, max_iter_barycenter=2,
                                       metric_params={"gamma": 1.0}, random_state=0, verbose=True).fit(data_new)
            labels = sdtw_km.labels_
            centers = sdtw_km.cluster_centers_
            inertia = sdtw_km.inertia_
        elif model == "K-Shape":
            ks = KShape(n_clusters=centroid_num, verbose=True, random_state=seed).fit(data_new)
            labels = ks.labels_
            centers = ks.cluster_centers_
            inertia = ks.inertia_
        elif model == "Kernel-KMeans":
            data_new = data_new[:100]
            data_raw = data_raw[:100]
            kk = KernelKMeans(n_clusters=centroid_num, kernel="gak", kernel_params={"sigma": "auto"}, max_iter=2,
                              tol=1e-4, verbose=True).fit(data_new)
            labels = kk.labels_
            inertia = kk.inertia_
        D = {}
        for i in range(centroid_num):
            D[i] = []
        for i in range(len(data_raw)):
            D[labels[i]].append(data_raw[i])
        return inertia, D, centers

    def multi_clustering(self, numbers, model, seq_train, sax_flag):
        """
        首先确定最优的聚类个数，然后多次聚类，取DBI最小的数据进行保存
        numbers:单次聚类循环的次数
        :return:
        """
        out = {}  # 分类好的时间序列
        out_cent = []  # 分类好的质心序列
        min_inertia = float('inf')
        raw_data = pd.read_csv('start_point_'+str(self.start_point)+'\\train.csv')
        df_data = pd.DataFrame(raw_data)
        sil_score = []
        print(len(raw_data))
        # 随机取聚类数据
        df_data = df_data.sample(frac=0.6, replace=True, random_state=0, axis=0)
        data_raw = df_data.iloc[:, 1:].values
        data_new = df_data.iloc[:, 1:self.seq_train+1].values
        if sax_flag:
            data_new = self.sax(data_new)
        # 确定最优的聚类个数
        # # example_data = data_new.sample(frac=0.3)
        # example_data = data_new
        # # min = 100000
        # for i in range(2, 9):
        #     print('第%d个：' % i)
        #     kmeans = KMeans(n_clusters=i).fit(example_data)
        #     lab = kmeans.labels_
        #     cent = kmeans.cluster_centers_
        #     # 初始化一个字典作为DBI的输入
        #     d = {}
        #     for j in range(i):
        #         d[j] = []
        #     for j in range(len(example_data)):
        #         d[lab[j]].append(example_data[j])
        #     # x是聚好类的时间序列的输入矩阵，clusters是簇质心，nc是簇类的数量
        #     dbi = DBI.compute_DB_index(d, cent, i)
        #     sil_score.append(dbi)
        #     # if dbi < min:
        #     #     min = dbi
        #     #     self.labels = lab
        #     #     self.centers = cent
        #
        # file = open('....\DBI_elbow_Google.csv', 'a', newline='')
        # content = csv.writer(file)
        # content.writerow(sil_score)
        # file.close()
        # print(sil_score)
        # print(np.argmax(sil_score))
        # self.Centroid = int(np.argmax(sil_score))+1
        # 以最优聚类个数为参数，进行多次聚类，记录inertia最小的聚类结果
        for i in range(numbers):
            # 记录聚类花费的时间
            start = time.time()
            inertia, output, centers = self.single_clustering(data_raw, data_new, self.Centroid, model)
            end = time.time()
            print('聚类耗时：%.10f' % (end - start))
            print(inertia)
            if inertia < min_inertia:
                min_inertia = inertia
                out = output.copy()
                print("Here:"+str(len(out)))
                out_cent = centers.copy()
        for i in range(len(out)):
            data = pd.DataFrame(out[i])
            data.to_csv('start_point_'+str(self.start_point) + '\\class'+str(i) + '.csv')
        if model == 'DTW' or model == 'K-Shape':
            # out_cent = self.sax(out_cent)
            out_cent = pd.DataFrame(np.reshape(out_cent, (self.Centroid, seq_train)))
        elif model == 'K-Means':
            # out_cent = self.sax(out_cent)
            out_cent = pd.DataFrame(out_cent)
        elif model == 'Kernel-KMeans':
            for i in range(len(out)):
                out_cent.append(softdtw_barycenter(out[i], max_iter=5, tol=1e-3))
            # print(out_cent)
            out_cent = pd.DataFrame(np.reshape(out_cent, (self.Centroid, seq_length)))
            # print(out_cent)
        out_cent.to_csv('start_point_'+str(self.start_point)+'\\centers.csv')


if __name__ == '__main__':
    seq_length = 7400
    seq_train = 5000
    dataset = 'Google'
    # dataset = 'Ali'
    cluster = Cluster(centroid=4, seq_length=seq_length, seq_train=seq_train, seq_interval=1, start_point=0, dataset=dataset)
    cluster.data_partition()
    cluster.make_sample()
    cluster.multi_clustering(numbers=1, model='K-Means', seq_train=seq_train, sax_flag=False)
