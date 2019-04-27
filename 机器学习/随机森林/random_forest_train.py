import numpy as np
from math import log
from CART import build_tree, predict
import random as rd
import pickle

def random_forest_training(data_train, trees_num):
    trees_result = []
    trees_feature = []
    n = np.shape(data_train)[1]
    if (n > 2):
        k = int(log(n - 1, 2)) + 1
    else:
        k = 1
    for i in range(trees_num):
        data_samples, feature = choose_samples(data_train, k)
        tree = build_tree(data_samples)
        trees_result.append(tree)
        trees_feature.append(feature)
    return trees_result, trees_feature

def choose_samples(data, k):
    m, n = np.shape(data)
    feature = []
    for j in range(k):
        feature.append(rd.randint(0, n - 2))
    index = []
    for i in range(m):
        index.append(rd.randint(0, m - 1))
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[index[i]][fea])
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature

def load_data(file_name):
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split('\t')
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train

def get_predict(trees_result, trees_feature, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]
    result = []
    for i in range(m_tree):
        clf = trees_result[i]
        feature = trees_feature[i]
        data = split_data(data_train, feature)
        result_i = []
        for i in range(m):
            result_i.append(list(predict(data[i][0:-1], clf).keys())[0])
        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict

def cal_correct_rate(data_train, final_predict):
    m = len(final_predict)
    corr = 0.0
    for i in range(m):
        if data_train[i][-1] * final_predict[i] > 0:
            corr += 1
    return corr / m

def save_model(trees_result, trees_feature, result_file, feature_file):
    m = len(trees_feature)
    f_fea = open(feature_file, 'w')
    for i in range(m):
        fea_tmp = []
        for x in trees_feature[i]:
            fea_tmp.append(str(x))
        f_fea.writelines("\t".join(fea_tmp) + "\n")
    f_fea.close()
    with open(result_file, 'wb') as f:
        pickle.dump(trees_result, f)

def split_data(data_train, feature):
    m = np.shape(data_train)[0]
    data = []
    for i in range(m):
        data_x_tmp = []
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data

if __name__ == "__main__":
    print("-----------1. load data-----------")
    data_train = load_data("data.txt")
    print("--------2. random forest training-------")
    trees_result, trees_feature = random_forest_training(data_train, 50)
    print("-----3. get prediction correct rate-------------")
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = cal_correct_rate(data_train, result)
    print("\t-------corect rate: ", corr_rate)
    print("----------4. save model------------")
    save_model(trees_result, trees_feature, "result_file", "feature_file")
    
