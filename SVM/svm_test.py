from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import timedelta
import time

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


if __name__ == "__main__":
    # 模型保存路径
    model_src = ""
    # 预测结果保存路径
    pre_src = ""
    # 加载测试集
    test_data_src = ""
    test_data = np.load(test_data_src)
    print("the size of test_data is {}".format(len(test_data)))
    # 数据归一化 [0, 1]
    scalar = MinMaxScaler()
    test_x = scalar.fit_transform(test_data[:, :-1])
    test_y = test_data[:, -1].astype(np.int)

    # 加载模型
    svm = joblib.load(model_src)
    start_time = time.time()
    # 预测
    pre_label = svm.predict(test_x)
    print("Finished, the pre time is {}".format(get_time_dif(start_time)))
    # 保存预测结果
    np.save(pre_src, pre_label)

    # 评估
    precision_score, recall_score, f1_val, accuracy = evaluate(test_y, pre_label)

    for i in range(num_category):
        print("class{}: {} {} {}".format(i, precision_score[i], recall_score[i], f1_val[i]))




