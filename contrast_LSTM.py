import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU, SimpleRNN
from keras.models import Sequential
import time
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import *
import random


def otm_mse(predict, true):
    """
    计算一对多模型的MSE
    :param predict:
    :param true:
    :return:
    """
    sum = 0
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            sum += (predict[i][j] - true[i][j]) ** 2
    average = sum / (len(predict) * len(predict[0]))
    return average


def tes_pre(file_path, seq_length, input_length):
    raw_data = pd.read_csv(file_path)
    # start = int(len(raw_data) * 0.8)
    df = pd.DataFrame(raw_data).values[:, 1:]
    test = np.array(df)
    x_test = test[:, :input_length]
    y_test = test[:, input_length:]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # raw_data = pd.read_csv(file_path)
    # df = pd.DataFrame(raw_data)
    # data_len = seq_length
    # data = []
    # for i in range(0, len(raw_data) - data_len, 1):
    #     tmp = []
    #     for j in range(0, data_len, 1):
    #         value = df['cpu'].values[i + j]
    #         if np.isnan(value):
    #             value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
    #         tmp.append(value)
    #     data.append(tmp)
    # mu = np.mean(data, axis=1)
    # sigma = np.std(data, axis=1)
    # test = []
    # for i in range(len(data)):
    #     temp = (np.array(data[i]) - mu[i]) / sigma[i]
    #     test.append(temp)
    # test = np.array(test)
    # x_test = test[:, :input_length]
    # y_test = test[:, input_length:]
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def gru_model(input_len, output_len):
    model = Sequential()
    model.add(GRU(2, input_shape=(input_len, 1)))
    model.add(Dense(output_dim=output_len))
    model.compile(loss='mse', optimizer='adam')
    return model


def lstm_model(input_length, output_length):
    """
    预测模型
    :return:
    """
    model = Sequential()
    model.add(LSTM(2, input_shape=(input_length, 1)))
    model.add(Dense(output_dim=output_length))
    model.compile(loss='mse', optimizer='adam')
    return model


def RNN(input_length, output_length):
    model = Sequential()
    model.add(SimpleRNN(2, input_shape=(input_length, 1)))
    model.add(Dense(output_dim=output_length))
    model.compile(loss='mse', optimizer='adam')
    return model


def train_pre(file_path, seq_length, input_length):
    raw_data = pd.read_csv(file_path)
    df = pd.DataFrame(raw_data)
    data_len = seq_length
    data = []
    for i in range(0, 14161 - data_len, 1):
        tmp = []
        for j in range(0, data_len, 1):
            value = df['cpu'].values[i + j]
            if np.isnan(value):
                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
            tmp.append(value)
        data.append(tmp)
    for i in range(16038, len(raw_data) - data_len, 1):
        tmp = []
        for j in range(0, data_len, 1):
            value = df['cpu'].values[i + j]
            if np.isnan(value):
                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
            tmp.append(value)
        data.append(tmp)
    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    train = []
    for i in range(len(data)):
        temp = (np.array(data[i]) - mu[i]) / sigma[i]
        train.append(temp)
    train = np.array(train)
    start = int(len(train)*0.0)
    length = int(len(train)*0.95)
    x_train = train[start:length, :input_length]
    y_train = train[start:length, input_length:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = train[length:, :input_length]
    y_valid = train[length:, input_length:]
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
    return x_train, y_train, x_valid, y_valid


def contrast_model(seq_len, input_len, out_len, cen_num):
    file_path = '....\src\Clustering\start_point_0\\train.csv'
    test_path = '....\src\Clustering\\final_test.csv'
    # test_path = 'D:\研究生\实验室\云环境下时间预测\代码\时间序列聚类\src\LSTM\start_point_0\\'
    seq_length = seq_len
    input_length = input_len
    output_length = out_len

    raw_data = pd.read_csv(file_path)
    train = pd.DataFrame(raw_data).values[:, 1:]
    length = int(len(train) * 0.90)
    x_train = train[:length, :input_length]
    y_train = train[:length, input_length:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = train[length:, :input_length]
    y_valid = train[length:, input_length:]
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
    # model = lstm_model(input_length, output_length)
    # model = gru_model(input_length, output_length)
    model = RNN(input_length, output_length)
    model_early_stop = EarlyStopping(monitor="val_loss", patience=100)
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=128, epochs=250, callbacks=[model_check, model_early_stop])
    model.save('bad_LSTM.h5')

    model = load_model('....\src\LSTM\\bad_LSTM.h5')
    x_test, y_test = tes_pre(test_path, seq_length, input_length)
    prediction = model.predict(x_test)
    average_MSE = otm_mse(prediction, y_test)
    nums = len(x_test)
    print("对比模型使用次数：" + str(nums))
    print("对比模型的平均MSE：" + str(average_MSE))
    return history, average_MSE


if __name__ == '__main__':
    # start = time.time()
    # seq_length = 32
    # input_length = 30
    # output_length = 2
    # class_num = 4
    # contrast_model(seq_length, input_length, output_length, class_num)
    # end = time.time()
    # print('对比模型耗时：%.10f' % (end-start))
    pass
