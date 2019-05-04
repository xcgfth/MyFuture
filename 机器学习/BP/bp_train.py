import numpy as np
from math import sqrt

def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    m, n = np.shape(feature)
    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((n, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((1, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((n_hidden, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((1, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    i = 0
    while i <= maxCycle:
        hidden_input = hidden_in(feature, w0, b0)
        hidden_output = hidden_out(hidden_input)
        output_in = predict_in(hidden_output, w1, b1)
        output_out = predict_out(output_in)
        delta_output = - np.multiply((label - output_out), partial_sig(output_in))
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(hidden_input))
        w1 = w1 - alpha * (hidden_output.T * delta_output)
        b1 = b1 - alpha * np.sum(delta_output, axis=0) * (1.0 / m)
        w0 = w0 - alpha * (feature.T * delta_hidden)
        b0 = b0 - alpha * np.sum(delta_hidden, axis=0) * (1.0 / m)
        if i % 100:
            print("--------------iter: ", i,\
                  ", cost: ", (1.0 / 2) * get_cost(get_predict(feature, w0, w1, b0, b1) - label))
        i += 1
    return w0, w1, b0, b1


def hidden_in(feature, w0, b0):
    m = np.shape(feature)[0]
    hidden_in = feature * w0
    for i in range(m):
        hidden_in[i, ] += b0
    return hidden_in

def hidden_out(hidden_in):
    hidden_output = sig(hidden_in)
    return hidden_output

def predict_in(hidden_out, w1, b1):
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1
    for i in range(m):
        predict_in[i,] += b1
    return predict_in
def predict_out(predict_in):
    result = sig(predict_in)
    return result

def sig(x):
    return 1.0/ (1 + np.exp(-x))

def partial_sig(x):
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))
    return out
def get_cost(cost):
    m, n = np.shape(cost)
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i,j] * cost[i,j]
    return cost_sum / m


def load_data(file_name):
    f = open(file_name)
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()
    m = len(label_tmp)
    n_class = len(set(label_tmp))
    label_data = np.mat(np.zeros((m, n_class)))
   # print(np.shape(label_data))
    for i in range(m):
        #print(label_tmp[i])
        label_data[i, label_tmp[i]] = 1
    return np.mat(feature_data), label_data, n_class

def save_model(w0, w1, b0, b1):
    def write_file(file_name, source):
        f = open(file_name, 'w')
        m, n = np.shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i,j]))
            f.write("\t".join(tmp) + "\n")
        f.close()
    write_file("weight_w0", w0)
    write_file("weight_w1", w1)
    write_file("weight_b0", b0)
    write_file("weight_b1", b1)

def get_predict(feature, w0, w1, b0, b1):
    return predict_out(predict_in(hidden_out(hidden_in(feature, w0, b0)), w1, b1))
def err_rate(label, pre):
    m = np.shape(label)[0]
    err = 0.0
    for i in range(m):
        if label[i][0] != pre[i][0]:
            err += 1
    rate = err  / m
    return rate


if __name__ == "__main__":
    print("-----------1. load data-----------")
    feature, label, n_class = load_data("data.txt")
    print("-----------2. training--------------")
    w0, w1, b0, b1 = bp_train(feature, label, 20, 1000, 0.1, n_class)
    print("-----------3. save model ------------")
    save_model(w0, w1, b0, b1)
    print("-----------4. get prediction-----------")
    result = get_predict(feature, w0, w1, b0, b1)
    print("acc: ", (1 - err_rate(np.argmax(label, axis=1), np.argmax(result, axis=1))))
