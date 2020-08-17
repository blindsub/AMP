import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import *
from tslearn.barycenters import softdtw_barycenter,euclidean_barycenter
import os


def load_data(test_filename):
    raw_data = pd.read_csv(test_filename)
    data = pd.DataFrame(raw_data).values[:, 1:]
    # test = np.array(test)
    # x_test = test[:, :30]
    # y_test = test[:, 30:]
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return data


def predict_one_to_one(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_one_to_more(model, data):
    predicted = model.predict(data)
    return predicted


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def otm_mse(predict, true):
    sum = 0
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            sum += (predict[i][j]-true[i][j])**2
    average = sum/(len(predict)*30)
    return average


def oto_mse(predict, true):
    sum = 0
    for i in range(len(predict)):
        sum += (predict[i]-true[i])**2
    average = sum/len(predict)
    return average


def percentage(low_mse, high_mse):
    percn = (high_mse-low_mse)/high_mse
    return percn


def caculate_mid_centers(class_num, iter_num, input_len):
    for i in range(class_num):
        first_filename = "....\src\LSTM\start_point_0\class"+str(i)+"_train.csv"
        first_train_data = load_data(first_filename)
        # 迭代的数据合并求簇心
        # concat_data = []
        # for j in range(iter_num):
        #     inter_filename = "....\src\LSTM\\result\\number"+str(j)+"_class"+str(i)+"_train.csv"
        #     inter_data = load_data(inter_filename)
        #     if j == 0:
        #         concat_data = np.vstack((first_train_data, inter_data))
        #         # concat_data = inter_data
        #     else:
        #         concat_data = np.vstack((concat_data, inter_data))
        #     # centers = softdtw_barycenter(concat_data, gamma=1.0, max_iter=5)
        #     centers = euclidean_barycenter(concat_data)
        #     centers = pd.DataFrame(np.reshape(centers, (1, len(centers))))
        #     centers.to_csv("D:\研究生\实验室\云环境下时间预测\代码\时间序列聚类\src\LSTM\\result\\number"+str(j)+"_class"+str(i)+"_centers.csv")
        # 每次迭代的数据单独求簇心
        for j in range(iter_num):
            inter_filename = "....\src\LSTM\\result\\number"+str(j)+"_class"+str(i)+"_train.csv"
            inter_data = load_data(inter_filename)[:, :input_len]
            print(inter_data.shape)
            # centers = softdtw_barycenter(inter_data, gamma=1.0, max_iter=5)
            centers = euclidean_barycenter(inter_data)
            centers = pd.DataFrame(np.reshape(centers, (1, len(centers))))
            centers.to_csv("....\src\LSTM\\result\\number"+str(j)+"_class"+str(i)+"_centers.csv")


def caculate_final_centers(class_num, iter_num):
    all_centers = []
    for k in range(class_num):
        filname = "....\src\LSTM\\result\\number"+str(iter_num-1)+"_class" + str(k) + "_centers.csv"
        center = pd.read_csv(filname)
        center = pd.DataFrame(center).values[0, 1:]
        all_centers.append(center)
    all_centers = pd.DataFrame(all_centers)
    all_centers.to_csv("....\src\LSTM\\result\\final_centers.csv")


def prepare_classify(class_num, iter_num):
    root_path = "....\src\LSTM\\result"
    all_data = []
    for i in range(class_num):
        temp_data = []
        # for j in range(1, iter_num):
        #     filename = "number"+str(j)+"_class"+str(i)+"_train.csv"
        #     data = pd.read_csv(os.path.join(root_path, filename))
        #     data = pd.DataFrame(data).values[:, 1:]
        #     if j == 1:
        #         temp_data = data
        #     else:
        #         temp_data = np.vstack((temp_data, data))

        filename = "number" + str(iter_num-1) + "_class" + str(i) + "_train.csv"
        data = pd.read_csv(os.path.join(root_path, filename))
        data = pd.DataFrame(data).values[:, 1:]
        temp_data = data

        class_index = [i for _ in range(len(temp_data))]
        temp_data = np.insert(temp_data, len(temp_data[0]), values=class_index, axis=1)
        if i == 0:
            all_data = temp_data
        else:
            all_data = np.vstack((all_data, temp_data))
    all_data = pd.DataFrame(all_data)
    all_data.to_csv(os.path.join(root_path, 'classify_data.csv'))


def calculate_test_mse_center(class_num):
    root_path = '....\src\LSTM\\test_result'
    for i in range(class_num):
        filename = 'mse_test_'+str(i)+'.csv'
        data = pd.read_csv(os.path.join(root_path, filename))
        data = pd.DataFrame(data).values[:, 1:]
        # center = softdtw_barycenter(data, gamma=1.0, max_iter=5)
        center = euclidean_barycenter(data)
        plt.plot(center)
        plt.title('Cluster'+str(i))
        plt.show()


if __name__ == '__main__':
    class_num = 4
    iter_num = 3
    input_len = 120
    caculate_mid_centers(class_num, iter_num, input_len)
    caculate_final_centers(class_num, iter_num)
    prepare_classify(class_num, iter_num)
    # calculate_test_mse_center(class_num)

