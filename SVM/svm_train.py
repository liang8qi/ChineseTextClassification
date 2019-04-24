from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import os
from datetime import timedelta


num_category = 9


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(label, prediction):
    # 标记为Yes 属于
    a = np.zeros(num_category)
    # 标记为Yes 不属于
    b = np.zeros(num_category)
    # 标记为No 属于
    c = np.zeros(num_category)
    # 标记为No 不属于
    d = np.zeros(num_category)

    label_num = len(label)
    right_num = 0
    for i in range(label_num):
        if label[i] == prediction[i]:
            a[label[i]] += 1
            for j in range(num_category):
                if j == label[i]:
                    continue
                d[j] += 1
            right_num += 1
        else:
            b[prediction[i]] += 1
            c[label[i]] += 1

    recall_score = a / (a + c)
    precision_score = a / (a + b)
    f1_val = 2 * recall_score * precision_score / (recall_score + precision_score)

    print("total:{}, right:{}, accuracy:{}".format(label_num, right_num, right_num/label_num))

    return precision_score, recall_score, f1_val, right_num/label_num


if __name__ == '__main__':
    train_data_src = "data/vectorized_data/train.npy"
    verify_data_src = "data/vectorized_data/valid.npy"
    # 保存模型
    model_src = "data/model"
    if not os.path.exists(model_src):
        os.mkdir(model_src)
    # 保存预测结果
    pre_save_src = "data/result/"
    if not os.path.exists(pre_save_src):
        os.mkdir(pre_save_src)

    # 加载数据
    train_data = np.load(train_data_src)
    valid_data = np.load(verify_data_src)
    print("the size of train_data is {}".format(len(train_data)))
    print("the size of valid_data is {}".format(len(valid_data)))
    # 数据归一化 [0, 1]
    scalar = MinMaxScaler()
    train_x = scalar.fit_transform(train_data[:, :-1])
    valid_x = scalar.fit_transform(valid_data[:, :-1])
    # 标签
    train_y = train_data[:, -1].astype(np.int)
    valid_y = valid_data[:, -1].astype(np.int)

    # 初始化SVM
    svm = SVC(kernel="linear")
    # 训练
    print("Start training")
    start_time = time.time()
    svm.fit(train_x, train_y)

    print("finished, the training time is {}".format(get_time_dif(start_time)))
    
    # 保存模型
    joblib.dump(svm, os.path.join(model_src, "svm.m"))

    # svm = joblib.load("data/old/model/svm_stan_1.m")

    # 预测 并保存结果
    pre_label = svm.predict(valid_x)

    np.save(os.path.join(pre_save_src, "valid_pre.npy"), pre_label)

    # 评估
    precision_score, recall_score, f1_val, accuracy = evaluate(valid_y, pre_label)

    for i in range(num_category):
        print("class{}: {} {} {}".format(i, precision_score[i], recall_score[i], f1_val[i]))

