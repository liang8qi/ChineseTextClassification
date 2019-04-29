import tensorflow as tf
import numpy as np
from text_cnn_model import TCNNConfig, TextCNNModel
import os
import time
from datetime import timedelta
from data_process import Data
import tensorflow.contrib.keras as kr

categories = {
            "体育": 0,
            "娱乐": 1,
            "彩票": 2,
            "教育": 3,
            "时尚": 4,
            "社会": 5,
            "科技": 6,
            "股票": 7,
            "财经": 8
        }
name = ["体育", "娱乐", "彩票", "教育", "时尚", "社会", "科技", "股票", "财经"]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def batch_iter(x, y, batch_size=128):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        max_len = config.sequence_length
        # padding="pre", truncating="post"的目的是当sequence length大于max_len时，将大于max_len的部分抛弃
        # 当sequence length小于max_len时，从起始位置开始padding
        # 如果max_len=None，则pad_sequences会默认按照最长序列的长度padding和truncating
        x_padded = kr.preprocessing.sequence.pad_sequences(x[start_id:end_id], maxlen=max_len,
                                                           padding="pre", truncating="post")
        yield x_padded, y[start_id:end_id]


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
        category = (label[i] != 0).argmax(axis=0)  # 返回第一个不为0的元素的索引
        if category == prediction[i]:
            a[category] += 1
            for j in range(num_category):
                if j == category:
                    continue
                d[j] += 1
            right_num += 1
        else:
            b[prediction[i]] += 1
            c[category] += 1

    recall_score = a / (a + c)
    precision_score = a / (a + b)
    f1_val = 2 * recall_score * precision_score / (recall_score + precision_score)

    print("total:{}, right:{}, accuracy:{}".format(label_num, right_num, right_num/label_num))

    return precision_score, recall_score, f1_val, right_num/label_num


def feed_data(x, y):
    feed_dict = {
        model.input_x: x,
        model.input_y: y,
        model.keep_dropout_prob: 1.0
    }
    return feed_dict


def test(x_src, y_src, result_src):
    x = np.load(x_src, allow_pickle=True)
    y = np.load(y_src)
    data_len = len(x)

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver.restore(session, os.path.join(model_save_src, "best_validation"))

        print("Testing")
        total_loss = 0.0
        total_acc = 0.0
        total_y_pre = []
        batch_train = batch_iter(x, y)
        for x_batch, y_batch in batch_train:
            batch_len = len(x_batch)
            feed_dict = feed_data(x_batch, y_batch)
            y_pre, loss, acc = session.run([model.y_pred, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
            total_y_pre.extend(y_pre)

        print("the loss is {}, the accuracy is {}".format(total_loss/data_len, total_acc/data_len))
        np.save(result_src, total_y_pre)
        return y, total_y_pre


if __name__ == "__main__":
    # 模型路径
    model_save_src = "fudandata/text_cnn_model"
    num_category = 9
    x_src = "data/vectorized_data/test/x.npy"
    y_src = "data/vectorized_data/test/y.npy"

    result_src = "data/results/test_cnn_pre.npy"
    vocab_src = "data/middle_result/vocab.npy"
    data = Data()
    vocab, _ = data.load_vocab(vocab_src)

    # 模型
    config = TCNNConfig()
    config.vocab_size = len(vocab)
    model = TextCNNModel(config)
    # 测试
    print("Begin Testing")
    start_time = time.time()
    y, y_pre = test(x_src, y_src, result_src)
    print("the time is {}".format(get_time_dif(start_time)))

    # 评估
    precision_score, recall_score, f1_val, accuracy = evaluate(y, y_pre)

    for i in range(num_category):
        print("class{}: {} {} {}".format(i, precision_score[i], recall_score[i], f1_val[i]))

