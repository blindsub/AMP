import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.models import Sequential
import time
import csv
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import *
from keras import layers
import random
import contrast_LSTM
from numpy import array, zeros, argmin, inf, equal, ndim
from tslearn.metrics import dtw, soft_dtw
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters,extract_features
from keras.models import Sequential
from keras.layers import Dense, Dropout,LeakyReLU
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import tsfresh
from tslearn.barycenters import softdtw_barycenter,euclidean_barycenter
from tsfresh.feature_extraction import MinimalFCParameters


class Prediction(object,):
    def __init__(self, path, sample_length, input_length, out_length, distance):
        self.path = path
        self.sample_length = sample_length
        self.input_length = input_length
        self.out_length = out_length
        self.distance = distance

    def data_pre(self, point, num, centers):
        """
        将聚类得到的各个簇分别制作训练集,验证集，用于预测模型的训练和验证
        :param point: 聚类的起始点（用于定位哪一次聚类）
        :param num: 聚类的个数
        :param centers: 质心时间序列
        :return: 质心序列
        """
        # 读取所属类的时间序列数据
        for i in range(num):
            raw_data = pd.read_csv(self.path+'\\class'+str(i)+'.csv')
            data = pd.DataFrame(raw_data).values[:, 1:]
            train_length = int(len(data)*0.9)
            train = data[:train_length]
            valid = data[train_length:]
            # train1 = self.filter_data(data, centers[i], 0.0, 0.10, False)
            # train2 = self.filter_data(data, centers[i], 0.13, 0.22, False)
            # valid1 = self.filter_data(data, centers[i], 0.10, 0.13, False)
            # train = train1 + train2
            # valid = valid1
            out_train = pd.DataFrame(train)
            out_valid = pd.DataFrame(valid)
            directory = 'start_point_'+str(point)
            filename = 'class' + str(i)
            if os.path.exists(directory):
                out_train.to_csv(directory+'\\'+filename+'_train.csv')
                out_valid.to_csv(directory+'\\'+filename+'_valid.csv')
            else:
                os.mkdir(directory)
                out_train.to_csv(directory+'\\'+filename+'_train.csv')
                out_valid.to_csv(directory+'\\'+filename+'_valid.csv')

    def sax(self, data):
        n_paa_segments = 10
        n_sax_symbols_avg = 8
        n_sax_symbols_slop = 8
        sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                  alphabet_size_slope=n_sax_symbols_slop)
        Sax_data = sax.inverse_transform(sax.fit_transform(data))
        data_new = np.reshape(Sax_data, (Sax_data.shape[0], Sax_data.shape[1]))
        return data_new

    def filter_data(self, data, center, start_per, stop_per, sax=False):
        """
        筛选出每个簇里距离质心最近的前百分比时间序列
        :param data: 原始时间序列数据
        :param center: 质心时间序列
        :param start_per: 开始百分比
        :param stop_per:结束百分比
        :param sax:是否对数据进行SAX变换
        :return: 筛选出的时间序列
        """
        sax_data = self.sax(data)
        distance = []
        out_data = []
        if not sax:
            for i in range(len(data)):
                if self.distance == 'vector':
                    temp = self.vector_distance(data[i][:self.input_length], center)
                    distance.append(temp)
                elif self.distance == 'dtw':
                    temp = self.dtw_distance(data[i][:self.input_length], center)
                    distance.append(temp)
        elif sax:
            for i in range(len(data)):
                if self.distance == 'vector':
                    temp = self.vector_distance(sax_data[i][:self.input_length], center)
                    distance.append(temp)
                elif self.distance == 'dtw':
                    temp = self.dtw_distance(sax_data[i][:self.input_length], center)
                    distance.append(temp)
        distance.sort()
        start = int(len(data)*start_per)
        stop = int(len(data)*stop_per)
        start_threshold = distance[start]
        stop_threshold = distance[stop]
        if not sax:
            if self.distance == 'vector':
                for i in range(len(data)):
                    if start_threshold < self.vector_distance(data[i][:self.input_length], center) <= stop_threshold:
                        out_data.append(data[i])
            elif self.distance == 'dtw':
                for i in range(len(data)):
                    if start_threshold < self.dtw_distance(data[i][:self.input_length], center) <= stop_threshold:
                        out_data.append(data[i])
        if sax:
            if self.distance == 'vector':
                for i in range(len(data)):
                    if start_threshold < self.vector_distance(sax_data[i][:self.input_length], center) <= stop_threshold:
                        out_data.append(data[i])
            elif self.distance == 'dtw':
                for i in range(len(data)):
                    if start_threshold < self.dtw_distance(sax_data[i][:self.input_length], center) <= stop_threshold:
                        out_data.append(data[i])
        return out_data

    def vector_distance(self, v1, v2):
        """
        this function calculates de euclidean distance between two
        vectors.
        """
        sum = 0
        # 维度相同
        for i in range(len(v1)):
            sum += (v1[i] - v2[i]) ** 2
        return sum ** 0.5

    def dtw_distance(self, v1, v2):
        # cost = dtw(v1, v2, global_constraint="itakura", itakura_max_slope=2.)
        cost = soft_dtw(v1, v2, gamma=1.0)
        return cost

    def load_model_data(self, filename):
        """
        加载训练集和验证集，并根据预测模型的输入输出制作模型输入和标签
        :return : 训练样本和验证样本
        """
        raw_data = pd.read_csv(filename)
        data = pd.DataFrame(raw_data).values[:, 1:]
        data = np.array(data)
        x_data = data[:, :self.input_length]
        y_data = data[:, self.input_length:]
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
        return x_data, y_data

    def gru_model(self):
        model = Sequential()
        model.add(GRU(2, input_shape=(self.input_length, 1)))
        model.add(Dense(output_dim=self.out_length))
        model.compile(loss='mse', optimizer='adam')
        return model

    def lstm_model(self):
        """
        预测模型
        :return:
        """
        model = Sequential()
        # model.add(GRU(2, input_shape=(self.input_length, 1)))
        model.add(LSTM(2, input_shape=(self.input_length, 1), kernel_regularizer="l2"))
        # model.add(Dropout(0.2))
        model.add(Dense(output_dim=self.out_length))
        model.compile(loss='mse', optimizer='adam')
        return model

    def find_best_model(self, test_seq, all_centers, sax):
        """
        为每个测试序列寻找最优模型（利用重复模式的质心序列）
        :param test_seq: 测试序列
        :param all_centers: 所有的重复模式的质心序列
        :param sax: 是否进行SAX操作的标志
        :return: 字典：不同模型所分配到的测试样本
        """
        dict = {}
        temp_model = []
        if sax:
            sax_test_seq = self.sax(test_seq)
            for k, seq in enumerate(sax_test_seq):
                min_distance = 999999999
                for key, center in all_centers.items():
                    if self.distance == 'vector':
                        distance = self.vector_distance(seq[:self.input_length], center)
                        if distance < min_distance:
                            min_distance = distance
                            temp_model = key
                    elif self.distance == 'dtw':
                        distance = self.dtw_distance(seq[:self.input_length], center)
                        if distance < min_distance:
                            min_distance = distance
                            temp_model = key
                if temp_model in dict.keys():
                    dict[temp_model].append(test_seq[k])
                else:
                    dict[temp_model] = [test_seq[k]]
        elif not sax:
            for seq in test_seq:
                min_distance = 999999999
                for key, center in all_centers.items():
                    if self.distance == 'vector':
                        distance = self.vector_distance(seq[:self.input_length], center)
                        if distance < min_distance:
                            min_distance = distance
                            temp_model = key
                    elif self.distance == 'dtw':
                        distance = self.dtw_distance(seq[:self.input_length], center)
                        if distance < min_distance:
                            min_distance = distance
                            temp_model = key
                if temp_model in dict.keys():
                    dict[temp_model].append(seq)
                else:
                    dict[temp_model] = [seq]
        # 筛选
        # for key in dict.keys():
        #     new_values = self.filter_data(dict[key], all_centers[key], 0.0, 0.15, True)
        #     dict[key] = new_values

        # 记录筛选后的测试集的欧式距离
        seqs = dict['start_point_0_model_0']
        distance = []
        if self.distance == 'vector':
            for i in range(len(seqs)):
                distance.append(self.vector_distance(seqs[i][:self.input_length], all_centers['start_point_0_model_0']))
        elif self.distance == 'dtw':
            for i in range(len(seqs)):
                distance.append(self.dtw_distance(seqs[i][:self.input_length], all_centers['start_point_0_model_0']))
        return dict

    def load_test_data(self, test):
        """
        加载测试数据
        :param
        :return:
        """
        # out_test = []
        # for k in range(len(test) - self.sample_length):
        #     out_test.append(test[k:k + self.sample_length])
        test = np.array(test)
        x_test = test[:, :self.input_length]
        y_test = test[:, self.input_length:]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return [x_test, y_test]

    def predict_one_to_more(self, model, data):
        """
        预测模型为一对多时使用
        :param model:
        :param data:
        :return:
        """
        predicted = model.predict(data)
        return predicted

    def otm_mse(self, predict, true):
        """
        计算一对多模型的MSE
        :param predict:
        :param true:
        :return:
        """
        test_MSE = []
        sum = 0
        for i in range(len(predict)):
            mse = 0
            for j in range(len(predict[0])):
                mse += (predict[i][j] - true[i][j]) ** 2
                sum += (predict[i][j] - true[i][j]) ** 2
            test_MSE.append(mse)
        average = sum / (len(predict) * len(predict[0]))
        # for i in range(len(predict)):
        #     for j in range(len(predict[0])):
        #         sum += (predict[i][j] - true[j]) ** 2
        # average = sum / len(predict[0])
        return average, test_MSE


def tet_dtw_distance(v1, v2):
    cost = dtw(v1, v2, global_constraint="itakura", itakura_max_slope=2.)
    return cost


def reserve(a):
    ans = []
    for i in range(len(a)):
        tmp = a[i]
        maxx = 0
        maxx_index = 0
        for j in range(len(tmp)):
            if tmp[j] > maxx:
                maxx=tmp[j]
                maxx_index = j
        ans.append(maxx_index)
    return ans


def classify(test_data):
    """

    :return:
    """
    path = '....\src\LSTM\\result\classify_data.csv'
    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)

    # 打乱
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df = df

    y = df['32'].astype('int')
    df.drop(['30', '31', '32'], axis=1, inplace=True)

    # test_data = pd.DataFrame(test_data)
    test_data = test_data.reset_index(drop=True)
    test_data = test_data.reset_index()
    test_data.drop(['30', '31'], axis=1, inplace=True)
    s = ['27__sum_values', '27__minimum', '27__median', '27__mean', '27__maximum', '28__minimum', '28__mean',
         '28__maximum', '28__sum_values', '28__median', '26__sum_values', '26__minimum', '26__median', '26__mean',
         '26__maximum', '25__minimum', '25__sum_values', '25__median', '25__maximum', '25__mean', '29__sum_values',
         '29__minimum', '29__median', '29__mean', '29__maximum', '24__maximum', '24__median', '24__minimum',
         '24__sum_values', '24__mean', '5__maximum', '5__mean', '5__minimum', '5__sum_values', '5__median',
         '7__maximum', '7__mean', '7__median', '7__minimum', '7__sum_values', '6__maximum', '6__mean', '6__median',
         '6__minimum', '6__sum_values', '8__sum_values', '8__mean', '8__median', '8__minimum', '8__maximum', '4__mean',
         '4__median', '4__minimum', '4__sum_values', '4__maximum', '23__minimum', '23__median', '23__mean',
         '23__maximum',
         '23__sum_values', '9__minimum', '9__sum_values', '9__median', '9__mean', '9__maximum', '10__sum_values',
         '10__median', '10__mean', '10__maximum', '10__minimum', '3__sum_values', '3__minimum', '3__maximum', '3__mean',
         '3__median', '2__minimum', '2__median', '2__mean', '2__maximum', '2__sum_values', '1__minimum',
         '1__sum_values', '1__maximum', '1__mean', '1__median', '11__mean', '11__median', '11__minimum', '11__maximum',
         '11__sum_values', '22__sum_values', '22__median', '22__mean', '22__maximum', '22__minimum', '0__mean',
         '0__median', '0__minimum', '0__sum_values', '0__maximum', '21__sum_values', '21__mean', '21__median',
         '21__minimum', '21__maximum', '12__minimum', '12__median', '12__mean', '12__maximum', '12__sum_values',
         '20__sum_values', '20__median', '20__mean', '20__maximum', '20__minimum', '19__maximum', '19__mean',
         '19__median', '19__minimum', '19__sum_values', '13__mean', '13__median', '13__minimum', '13__maximum',
         '13__sum_values', '18__mean', '18__median', '18__minimum', '18__maximum', '18__sum_values', '14__sum_values',
         '14__minimum', '14__median', '14__mean', '14__maximum', '17__maximum', '17__mean', '17__median',
         '17__sum_values', '17__minimum', '15__sum_values', '15__mean', '15__median', '15__minimum', '15__maximum',
         '16__maximum', '16__mean', '16__median', '16__sum_values', '16__minimum']
    dict = tsfresh.feature_extraction.settings.from_columns(columns=s)

    # 训练集提取特征(xgb筛选完)
    # train_features_filter = extract_features(df,
    #                                           column_id='index',
    #                                           kind_to_fc_parameters=dict,
    #                                           impute_function=impute)
    # 训练集提取特征(直接用库函数)
    train_features_filter = extract_features(df,
                                              column_id='index',
                                              default_fc_parameters=MinimalFCParameters(),
                                              impute_function=impute)
    # 测试集提取特征(xgb筛选完)
    # test_features_filter = extract_features(test_data,
    #                                          column_id='index',
    #                                          kind_to_fc_parameters=dict,
    #                                          impute_function=impute)
    # 测试集提取特征(直接用库函数)
    test_features_filter = extract_features(test_data,
                                             column_id='index',
                                             default_fc_parameters=MinimalFCParameters(),
                                             impute_function=impute)

    train_features_filter = select_features(train_features_filter, y)

    input_dim = train_features_filter.shape[1]
    train_length = int(len(train_features_filter) * 0.8)
    X_train_data = train_features_filter[:train_length]
    y_train_data = y[:train_length]
    y_train_data = to_categorical(y_train_data)
    X_valid_data = train_features_filter[train_length:]
    y_valid_data = y[train_length:]
    y_valid_data = to_categorical(y_valid_data)


    X_test_data = test_features_filter[:]

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
    model.fit(X_train_data, y_train_data,
              epochs=300,
              batch_size=128,
              validation_data=(X_valid_data, y_valid_data), callbacks=[model_check])
    pred = model.predict(X_test_data)
    # 归类
    labels = reserve(pred)
    return labels


def classify2(test_data):
    """

    :return:
    """
    path = '....\src\LSTM\\result\classify_data.csv'
    # path1 = '....\src\LSTM\\result\\final_centers.csv'
    path1 = '....\src\Clustering\start_point_0\centers.csv'
    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)

    f1 = open(path1, 'r', encoding='utf-8')
    b = pd.read_csv(f1, index_col=0)
    center = pd.DataFrame(b)
    # center = center.reset_index(drop=True)
    # center = center.reset_index()
    # center.drop(['30','31'],axis=1,inplace=True)

    # 训练集
    df = shuffle(df)
    df = df.reset_index(drop=True)
    y = df['32'].astype('int')
    df.drop(['30', '31', '32'], axis=1, inplace=True)
    res = []
    for i in range(len(df)):
        d = []
        for j in range(len(center)):
            # dis = vectorDistance(df.iloc[i], center.iloc[j])
            dis = soft_dtw(df.iloc[i], center.iloc[j])
            d.append(dis)
        res.append(d)

    new_df = pd.DataFrame(res, columns=["0", "1", "2", "3", "4"])
    df = df.reset_index()

    # 测试集
    test_data = test_data.reset_index(drop=True)
    test_data.drop(['30', '31'], axis=1, inplace=True)
    test_res = []
    for i in range(len(test_data)):
        d = []
        for j in range(len(center)):
            # dis = vectorDistance(test_data.iloc[i][1:], center.iloc[j])
            dis = dtw(test_data.iloc[i][1:], center.iloc[j])
            d.append(dis)
        test_res.append(d)

    new_test = pd.DataFrame(test_res, columns=["0", "1", "2", "3", "4"])
    test_data = test_data.reset_index()

    s = ['27__sum_values', '27__minimum', '27__median', '27__mean', '27__maximum', '28__minimum', '28__mean',
         '28__maximum', '28__sum_values', '28__median', '26__sum_values', '26__minimum', '26__median', '26__mean',
         '26__maximum', '25__minimum', '25__sum_values', '25__median', '25__maximum', '25__mean', '29__sum_values',
         '29__minimum', '29__median', '29__mean', '29__maximum', '24__maximum', '24__median', '24__minimum',
         '24__sum_values', '24__mean', '5__maximum', '5__mean', '5__minimum', '5__sum_values', '5__median',
         '7__maximum', '7__mean', '7__median', '7__minimum', '7__sum_values', '6__maximum', '6__mean', '6__median',
         '6__minimum', '6__sum_values', '8__sum_values', '8__mean', '8__median', '8__minimum', '8__maximum', '4__mean',
         '4__median', '4__minimum', '4__sum_values', '4__maximum', '23__minimum', '23__median', '23__mean',
         '23__maximum',
         '23__sum_values', '9__minimum', '9__sum_values', '9__median', '9__mean', '9__maximum', '10__sum_values',
         '10__median', '10__mean', '10__maximum', '10__minimum', '3__sum_values', '3__minimum', '3__maximum', '3__mean',
         '3__median', '2__minimum', '2__median', '2__mean', '2__maximum', '2__sum_values', '1__minimum',
         '1__sum_values', '1__maximum', '1__mean', '1__median', '11__mean', '11__median', '11__minimum', '11__maximum',
         '11__sum_values', '22__sum_values', '22__median', '22__mean', '22__maximum', '22__minimum', '0__mean',
         '0__median', '0__minimum', '0__sum_values', '0__maximum', '21__sum_values', '21__mean', '21__median',
         '21__minimum', '21__maximum', '12__minimum', '12__median', '12__mean', '12__maximum', '12__sum_values',
         '20__sum_values', '20__median', '20__mean', '20__maximum', '20__minimum', '19__maximum', '19__mean',
         '19__median', '19__minimum', '19__sum_values', '13__mean', '13__median', '13__minimum', '13__maximum',
         '13__sum_values', '18__mean', '18__median', '18__minimum', '18__maximum', '18__sum_values', '14__sum_values',
         '14__minimum', '14__median', '14__mean', '14__maximum', '17__maximum', '17__mean', '17__median',
         '17__sum_values', '17__minimum', '15__sum_values', '15__mean', '15__median', '15__minimum', '15__maximum',
         '16__maximum', '16__mean', '16__median', '16__sum_values', '16__minimum']
    # dict = tsfresh.feature_extraction.settings.from_columns(columns=s)
    dict = MinimalFCParameters()

    # 训练集提取特征
    train_features_filter = extract_features(df,
                                              column_id='index',
                                              kind_to_fc_parameters=dict,
                                              impute_function=impute)
    # 测试集提取特征
    test_features_filter = extract_features(test_data,
                                             column_id='index',
                                             kind_to_fc_parameters=dict,
                                             impute_function=impute)

    # test_features_filter = select_features(test_features, y)

    new_df.index.name = 'id'
    features_filter = pd.merge(train_features_filter, new_df, how='left', on='id')
    train_features_filter = features_filter.reset_index(drop=True)

    new_test.index.name = 'id'
    tests_filter = pd.merge(test_features_filter, new_test, how='left', on='id')
    test_features_filter = tests_filter.reset_index(drop=True)

    input_dim = features_filter.shape[1]
    train_length = int(len(train_features_filter) * 0.8)
    X_train_data = train_features_filter[:train_length]
    y_train_data = y[:train_length]
    y_train_data = to_categorical(y_train_data)
    X_valid_data = train_features_filter[train_length:]
    y_valid_data = y[train_length:]
    y_valid_data = to_categorical(y_valid_data)


    X_test_data = test_features_filter[:]

    model = Sequential()
    # model.add(Dense(256, activation='relu', input_dim=input_dim))
    # model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
    model.fit(X_train_data, y_train_data,
              epochs=300,
              batch_size=128,
              # )
              validation_data=(X_valid_data, y_valid_data), callbacks=[model_check])
    pred = model.predict(X_test_data)
    # 归类
    labels = reserve(pred)
    return labels


def classify3(test_data, input_length, groundtruth, class_num):
    path = '....\LSTM\\result\classify_data.csv'
    # test_length = int(len(test_data)*0.12)
    test_data = test_data.values[:, 1:input_length+1]
    # print(test_data.shape)
    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)
    # 打乱
    df = shuffle(df)
    df = df.reset_index(drop=True)
    y = df['44'].astype('int')
    # 查看是否训练样本不平衡
    for i in range(class_num):
        number = list(y).count(i)
        print('Class_'+str(i)+'的训练样本个数有：', number)

    df = df.iloc[:, :input_length]
    train_length = int(len(df)*0.85)
    X_train_data = df[:train_length]
    y_train_data = y[:train_length]
    y_train_data = to_categorical(y_train_data)
    X_valid_data = df[train_length:]
    y_valid_data = y[train_length:]
    y_valid_data = to_categorical(y_valid_data)
    groundtruth = to_categorical(groundtruth[:])
    input_dim = X_train_data.shape[1]
    print("Training.........\n")

    # 全连接神经网络
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=input_dim))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    # model.add(LeakyReLU(alpha=0.5))
    # model.add(Dropout(0.2))
    model.add(Dense(class_num, activation='softmax'))

    # LSTM做时序分类
    # X_train_data = np.reshape(np.array(X_train_data), (X_train_data.shape[0], X_train_data.shape[1], 1))
    # X_valid_data = np.reshape(np.array(X_valid_data), (X_valid_data.shape[0], X_valid_data.shape[1], 1))
    # test_data = np.reshape(np.array(test_data), (test_data.shape[0], test_data.shape[1], 1))
    # model = Sequential()
    # model.add(LSTM(64, input_shape=(input_dim, 1), kernel_regularizer="l2"))
    # model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=True,
                                  save_best_only=True,
                                  period=1,
                                  save_weights_only=False
                                  )
    model.fit(X_train_data, y_train_data,
              epochs=300,
              batch_size=128,
              validation_data=(X_valid_data, y_valid_data),
              callbacks=[model_check])
    score = model.evaluate(test_data, groundtruth, batch_size=128)
    # 预测准确度
    pred = model.predict(test_data)
    # 归类
    labels = reserve(pred)
    return labels, score


def classify3_plus(test_data, input_length, groundtruth):
    # 加强版，增加了时序与各个类的簇心之间的距离
    path = '....\src\LSTM\\result\classify_data.csv'
    # center_path = "....\src\LSTM\\result\\final_centers.csv"
    center_path = '....\src\Clustering\start_point_0\centers.csv'
    all_centers = pd.read_csv(center_path)
    all_centers = pd.DataFrame(all_centers).reset_index(drop=True)
    test_data = test_data.iloc[:, 1:input_length+1]

    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)
    # 打乱
    df = shuffle(df)
    df = df.reset_index(drop=True)
    y = df['2120'].astype('int')
    # 查看是否训练样本不平衡
    for i in range(4):
        number = list(y).count(i)
        print('Class_'+str(i)+'的训练样本个数有：', number)
    df = df.iloc[:, :input_length]
    # 为每个训练样本追加与各个类的簇心的距离
    all_dis = []
    for i in range(len(df)):
        dis = []
        for j in range(len(all_centers)):
            dis.append(vectorDistance(df.iloc[i], all_centers.iloc[j, :input_length]))
        all_dis.append(dis)
    all_dis = pd.DataFrame(all_dis, columns=['0', '1', '2', '3'])
    df.reset_index()
    all_dis.index.name = 'id'
    df.index.name = 'id'
    df = pd.merge(df, all_dis, how='left', on='id')
    df.reset_index(drop=True)
    # 为每个测试样本追加与各个类的簇心的距离
    all_dis = []
    for i in range(len(test_data)):
        dis = []
        for j in range(len(all_centers)):
            dis.append(vectorDistance(test_data.iloc[i], all_centers.iloc[j, :input_length]))
        all_dis.append(dis)
    all_dis = pd.DataFrame(all_dis, columns=['0', '1', '2', '3'])
    test_data.reset_index()
    test_data.index.name = 'id'
    all_dis.index.name = 'id'
    test_data = pd.merge(test_data, all_dis, how='left', on='id')
    test_data.reset_index(drop=True)

    # 划分训练集和验证集
    train_length = int(len(df)*0.9)
    X_train_data = df[:train_length]
    y_train_data = y[:train_length]
    y_train_data = to_categorical(y_train_data)
    X_valid_data = df[train_length:]
    y_valid_data = y[train_length:]
    y_valid_data = to_categorical(y_valid_data)
    groundtruth = to_categorical(groundtruth[:])
    input_dim = X_train_data.shape[1]
    print("Training.........\n")

    # 全连接神经网络
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    # model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    # model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    # model.add(Dense(32))
    # model.add(LeakyReLU(alpha=0.5))
    # model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))

    # LSTM做时序分类
    # X_train_data = np.reshape(np.array(X_train_data), (X_train_data.shape[0], X_train_data.shape[1], 1))
    # X_valid_data = np.reshape(np.array(X_valid_data), (X_valid_data.shape[0], X_valid_data.shape[1], 1))
    # test_data = np.reshape(np.array(test_data), (test_data.shape[0], test_data.shape[1], 1))
    # model = Sequential()
    # model.add(LSTM(64, input_shape=(input_dim, 1), kernel_regularizer="l2"))
    # model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=True,
                                  save_best_only=True,
                                  period=1,
                                  save_weights_only=False
                                  )
    model.fit(X_train_data, y_train_data,
              epochs=200,
              batch_size=128,
              validation_data=(X_valid_data, y_valid_data),
              callbacks=[model_check])
    score = model.evaluate(test_data, groundtruth, batch_size=128)
    # 预测准确度
    pred = model.predict(test_data)
    # 归类
    labels = reserve(pred)
    return labels, score


def classify_CNN(test_data, input_length):
    path = '....\\src\LSTM\\result\classify_data.csv'
    test_data = test_data.values[:, 1:input_length + 1]
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    # print(test_data.shape)
    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)
    # 打乱
    df = shuffle(df)
    df = df.reset_index(drop=True)
    y = df['32'].astype('int')
    # df.drop(['80', '81', '82', '83', '84', '85', '86'], axis=1, inplace=True)
    df = df.values[:, :input_length]
    # print('df的格式：', df.shape)
    train_length = int(len(df) * 0.85)
    df = np.reshape(df, (df.shape[0], df.shape[1], 1))
    X_train_data = df[:train_length]
    y_train_data = y[:train_length]
    y_train_data = to_categorical(y_train_data)
    X_valid_data = df[train_length:]
    y_valid_data = y[train_length:]
    y_valid_data = to_categorical(y_valid_data)

    input_dim = (X_train_data.shape[1], 1)
    print(input_dim)
    print("Training.........\n")

    model = Sequential()
    model.add(layers.Conv1D(100, 6, activation='relu', input_shape=input_dim))
    # model.add(layers.Conv1D(100, 6, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(100, 6, activation='relu'))
    # model.add(layers.Conv1D(100, 6, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=True,
                                  save_best_only=True,
                                  period=1,
                                  save_weights_only=False
                                  )
    model.fit(X_train_data, y_train_data,
              epochs=180,
              batch_size=128,
              validation_data=(X_valid_data, y_valid_data),
              callbacks=[model_check])
    pred = model.predict(test_data)
    # 归类
    labels = reserve(pred)
    return labels


def xgboost(test_data):
    """

    :return:
    """
    path = '....\src\LSTM\\result\classify_data.csv'
    f = open(path, 'r', encoding='utf-8')
    a = pd.read_csv(f, index_col=0)
    f.close()
    df = pd.DataFrame(a)

    # 打乱
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df = df

    y = df['32'].astype('int')
    df.drop(['30', '31', '32'], axis=1, inplace=True)

    # test_data = pd.DataFrame(test_data)
    test_data = test_data.reset_index(drop=True)
    test_data = test_data.reset_index()
    test_data.drop(['30', '31'], axis=1, inplace=True)

    # train_features = extract_features(df,
    #                                   column_id='index',
    #                                   kind_to_fc_parameters=dict,
    #                                   impute_function=impute)
    extraction_settings = MinimalFCParameters()
    # 训练集提取特征
    train_features_filter = extract_features(df,
                                              column_id='index',
                                              kind_to_fc_parameters=extraction_settings,
                                              impute_function=impute)
    # 测试集提取特征
    test_features_filter = extract_features(test_data,
                                             column_id='index',
                                             kind_to_fc_parameters=extraction_settings,
                                             impute_function=impute)

    encoder = LabelEncoder()
    input_dim = train_features_filter.shape[1]
    train_length = int(len(train_features_filter) * 0.8)
    X_train_data = train_features_filter[:train_length]
    y_train_data = y[:train_length]
    # y_train_data = to_categorical(y_train_data)
    y_train_data = encoder.fit_transform(y_train_data)
    X_valid_data = train_features_filter[train_length:]
    y_valid_data = y[train_length:]
    # y_valid_data = to_categorical(y_valid_data)
    y_valid_data = encoder.fit_transform(y_valid_data)

    X_test_data = test_features_filter[:]

    xgb = XGBClassifier(
        n_estimators=20,  # 迭代次数
        learning_rate=0.1,  # 步长
        max_depth=5,  # 树的最大深度
        min_child_weight=1,  # 决定最小叶子节点样本权重和
        silent=1,  # 输出运行信息
        subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
        colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
        objective='multi:softmax',  # 多分类！！！！！！
        num_class=4,
        nthread=4,
        seed=27)

    # true = reserve(y_test_data)
    xgb.fit(X_train_data, y_train_data, verbose=True)
    labels = xgb.predict(X_test_data)
    return labels

def vectorDistance(a, b):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    v1 = np.array(a)
    v2 = np.array(b)
    dist = np.linalg.norm(v1-v2)
    return dist


def plot_results(history, rows, subnum):
    """
            画出真实序列和预测序列
            :param history:
            :param subnum:
            :return:
            """
    plt.subplot(rows, 2, subnum)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')


def percentage(low_mse, high_mse):
    """
    计算两个模型的MSE的差值百分比
    :param low_mse:
    :param high_mse:
    :return:
    """
    percn = (high_mse - low_mse) / high_mse
    return percn


if __name__ == '__main__':
    pred_path = '....\src\LSTM\start_point_'
    model_path = '....\src\LSTM'
    test_path = '....\src\Clustering'
    class_num = 4   # 聚类的个数
    sample_length = 126  # 样本的长度
    input_length = 120  # 输入的长度
    out_length = 6  # 输出的长度
    distance = 'vector'
    # distance = 'dtw'
    points = [0]
    Centroid = {}  # 存放已经训练好的所有模型对应的质心序列
    all_mse = []
    class_dic = {}  # 字典：用于记录测试集样本
    dictionary = {}  # 字典：用于记录不同预测模型的MSE
    dictionary2 = {}  # 字典：用于记录不同预测模型使用次数
    train_time = 0  # 记录预测模型训练耗时
    classify_time = 0  # 记录分类模型训练耗时
    for point in points:
        cluster_path = '....\src\Clustering\start_point_'+str(point)
        pre = Prediction(cluster_path, sample_length, input_length, out_length, distance)
        # 读取质心序列，用于后期测试
        raw_centers = pd.read_csv(cluster_path+'\\centers.csv')
        # raw_centers = pd.read_csv(model_path + "\\result\\final_centers.csv")
        centers = pd.DataFrame(raw_centers).values[:, 1:]
        # 为每个类划分训练集和验证集（有筛选）
        pre.data_pre(point, class_num, centers)
        for i, center in enumerate(centers):
            Centroid['start_point_'+str(point)+'_model_'+str(i)] = center[:input_length]
        # 训练模型+统计预测模型训练耗时
        start = time.time()
        for i in range(class_num):
            x_train, y_train = pre.load_model_data(pred_path+str(point)+'\\class'+str(i)+'_train.csv')
            x_valid, y_valid = pre.load_model_data(pred_path+str(point)+'\\class'+str(i)+'_valid.csv')
            model = pre.lstm_model()
            model_early_stop = EarlyStopping(monitor="val_loss", patience=100)
            model_check = ModelCheckpoint('./model.h5',
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
            history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=128, epochs=250, callbacks=[model_check, model_early_stop])
            model.save(pred_path+str(point) + '\\increase_LSTM' + str(i) + '.h5')
            plot_results(history, int(class_num/2)+1, i+1)
            plt.figure(1)

        # 构建测试集并迭代训练
        for i in range(3):
            raw_test = pd.read_csv(test_path + '\\start_point_0\\train.csv')
            raw_test = pd.DataFrame(raw_test)
            if i == 2:
                fraction = 0.6
            else:
                fraction = 0.4
            tests = raw_test.sample(frac=fraction, replace=True, random_state=0, axis=0)
            tests = tests.values[:, 1:]
            tests = np.array(tests)
            sax = False  # 测试集是否进行SAX操作
            for j in range(class_num):
                locals()['model_'+str(j)] = load_model(os.path.join(model_path, "start_point_0\increase_LSTM"+str(j)+".h5"))
                locals()['model_'+str(j)+'_sample'] = []
            # 为每个测试时序寻找mse最小的预测模型
            x_tests, y_tests = pre.load_test_data(tests)
            for k in range(len(x_tests)):
                min_mse = float('inf')
                final_model = "model"
                x_test = x_tests[k]
                x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
                y_test = y_tests[k]
                y_test = np.reshape(y_test, (1, y_test.shape[0], 1))
                for m in range(class_num):
                    prediction = pre.predict_one_to_more(locals()['model_' + str(m)], x_test)
                    mse, test_mse = pre.otm_mse(prediction, y_test)
                    if mse < min_mse:
                        final_model = "start_point_0" + '_' + str(m)
                        min_mse = float(mse)
                index = final_model[-1]
                locals()['model_'+str(index)+'_sample'].append(tests[k])

            # 存储迭代中的训练数据
            for j in range(class_num):
                path = "....\LSTM\\result\\number"+str(i)+"_class"+str(j)+"_train.csv"
                new_data = locals()['model_'+str(j)+'_sample']
                final_data = pd.DataFrame(new_data)
                final_data.to_csv(path)

            # 继续训练模型
            for k in range(class_num):
                sample = locals()['model_' + str(k)+'_sample']
                sample = pd.DataFrame(sample).values
                # print(sample.shape)
                model = locals()['model_' + str(k)]
                X, Y = pre.load_test_data(sample)
                train_length = int(len(X)*0.9)
                x_train = X[:train_length]
                y_train = Y[:train_length]
                x_valid = X[train_length:]
                y_valid = Y[train_length:]
                model_early_stop = EarlyStopping(monitor="val_loss", patience=100)
                model_check = ModelCheckpoint('./model.h5',
                                                  monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto',
                                                  period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
                history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=128, epochs=250, callbacks=[model_check, model_early_stop])
                model.save(pred_path + str(point) + '\\increase_LSTM' + str(k) + '.h5')
        end = time.time()
        train_time = end - start
        # 构建最终测试集
        raw_test = pd.read_csv(test_path + '\\final_test.csv')
        sax = False  # 测试集是否进行SAX操作

        # 为每个测试集寻找最好的预测模型
        for i in range(class_num):
            locals()['model_'+str(i)] = load_model(os.path.join(model_path, "start_point_0\increase_LSTM"+str(i)+".h5"))
        # 使用分类方法，先给出测试集的GroundTruth，便于计算分类准确率
        # 计算GroundTruth
        # test_gt = []
        # tests = pd.DataFrame(raw_test).values[:, 1:]
        # x_tests, y_tests = pre.load_test_data(tests)
        # for k in range(class_num):
        #     locals()['min_mse_LSTM_'+str(k)] = []
        # for k in range(len(x_tests)):
        #     min_mse = float('inf')
        #     final_model = "model"
        #     x_test = x_tests[k]
        #     x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
        #     y_test = y_tests[k]
        #     y_test = np.reshape(y_test, (1, y_test.shape[0], 1))
        #     for m in range(class_num):
        #         prediction = pre.predict_one_to_more(locals()['model_' + str(m)], x_test)
        #         mse, test_mse = pre.otm_mse(prediction, y_test)
        #         if mse < min_mse:
        #             final_model = "start_point_0" + '_' + str(m)
        #             min_mse = float(mse)
        #     test_gt.append(int(final_model[-1]))
        #
        # tests = pd.DataFrame(raw_test)
        # class_time_1 = time.time()
        # labels, score = classify3(tests, input_length, test_gt, class_num)
        # print('分类模型准确率：', score[1])
        # # labels = xgboost(tests)
        # tests = tests.values[:, 1:]
        # for k in range(class_num):
        #     locals()['model_test'+str(k)] = []
        # for k in range(len(labels)):
        #     locals()['model_test'+str(labels[k])].append(tests[k])
        # for k in range(class_num):
        #     if len(locals()['model_test'+str(k)]) == 0:
        #         continue
        #     x_tests, y_tests = pre.load_test_data(locals()['model_test'+str(k)])
        #     prediction = pre.predict_one_to_more(locals()['model_' + str(k)], x_tests)
        #     mse, test_mse = pre.otm_mse(prediction, y_tests)
        #     dictionary["start_point_0_" + str(k)] = mse
        #     dictionary2["start_point_0_" + str(k)] = len(x_tests)
        # class_time_2 = time.time()
        # print('分类模型训练耗时：%.10f' % (class_time_2-class_time_1))

        # 通过比较mse，得到每个测试时序最好的LSTM
        # 记录下最小mse的分类结果
        tests = pd.DataFrame(raw_test).values[:, 1:]
        x_tests, y_tests = pre.load_test_data(tests)
        for k in range(class_num):
            locals()['min_mse_LSTM_'+str(k)] = []
        for k in range(len(x_tests)):
            min_mse = float('inf')
            final_model = "model"
            x_test = x_tests[k]
            x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
            y_test = y_tests[k]
            y_test = np.reshape(y_test, (1, y_test.shape[0], 1))
            for m in range(class_num):
                prediction = pre.predict_one_to_more(locals()['model_' + str(m)], x_test)
                mse, test_mse = pre.otm_mse(prediction, y_test)
                if mse < min_mse:
                    final_model = "start_point_0" + '_' + str(m)
                    min_mse = float(mse)
            locals()['min_mse_LSTM_'+str(final_model[-1])].append(tests[k])
            if final_model not in dictionary.keys():
                dictionary[final_model] = min_mse
            else:
                dictionary[final_model] += min_mse
            if final_model not in dictionary2.keys():
                dictionary2[final_model] = 1
            else:
                dictionary2[final_model] += 1

    # 对比LSTM+统计对比模型的训练耗时
    c_start = time.time()
    history, MSE = contrast_LSTM.contrast_model(seq_len=sample_length, input_len=input_length, out_len=out_length, cen_num=class_num)
    c_end = time.time()
    plot_results(history, int(class_num/2)+1, class_num+1)
    plt.figure(1)
    plt.show()
    # 划分的LSTM的结果
    total_mse = 0
    num = 0
    print('聚类的预测模型训练耗时：%.10f' % train_time)
    print('对比模型训练耗时：%.10f' % (c_end - c_start))
    for model in dictionary.keys():
        print(model + ':平均MSE为' + str(dictionary[model]/dictionary2[model]) + ',模型使用次数：' + str(dictionary2[model]))
        total_mse += dictionary[model]
        num += dictionary2[model]
    average_mse = total_mse / num
    print('聚类模型的平均MSE：'+str(average_mse))
    per = percentage(average_mse, MSE)
    print("提升百分比："+str(per))




