# decoding: utf-8
from char_cnn_model import CharCNNConfig, CharCNN
import tensorflow as tf
import numpy as np
import os
import time
from datetime import timedelta
import tensorflow.contrib.keras as kr
from data_process import Data


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def batch_iter(x, y, batch_size=128):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)

        x_shuffle_padded = kr.preprocessing.sequence.pad_sequences(x_shuffle[start_id:end_id], maxlen=config.text_size)

        yield x_shuffle_padded, y_shuffle[start_id:end_id]


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_dropout_prob: keep_prob
    }
    return feed_dict


def learning_rate_decay(step, learning_rate, decay_rate, decay_steps):
    return learning_rate * decay_rate ^ (step / decay_steps)


def evaluate(sess, x_, y_):
    # 评估在某一数据上的准确率和损失
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)

        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(x_train_src, y_train_src, x_val_src, y_val_src, epoch_num, model_save_src):

    tensorboard_dir = 'tensorboard'
    if not os.path.exists(model_save_src):
        os.mkdir(model_save_src)
    if not os.path.exists(os.path.join(model_save_src, tensorboard_dir)):
        os.mkdir(os.path.join(model_save_src, tensorboard_dir))

    # save model
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(model_save_src, tensorboard_dir))

    saver = tf.train.Saver(max_to_keep=1)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train = np.load(x_train_src)
    y_train = np.load(y_train_src)
    x_valid = np.load(x_val_src)
    y_valid = np.load(y_val_src)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(epoch_num):

        print("Epoch: {}".format(epoch))
        batch_train = batch_iter(x_train, y_train)
        for x_batch, y_batch in batch_train:

            feed_dict = feed_data(x_batch, y_batch, 0.5)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % 10 == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                # feed_dict = feed_data(x_valid, x_valid, len_valid, 1.0)
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)

                loss_valid, acc_valid = evaluate(session, x_valid, y_valid)
                if acc_valid > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_valid
                    last_improved = total_batch
                    saver.save(sess=session, save_path=os.path.join(model_save_src, "best_validation"))

                    time_dif = get_time_dif(start_time)

                print("iter: {}, train loss: {}, train accuracy: {}, val loss: {}, val accuracy: {}, time: {}".format(
                    total_batch, loss_train, acc_train, loss_valid, acc_valid, time_dif))

            session.run(model.opt, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:
            break

    print("the best accuracy on validation is {}, at {}".format(best_acc_val, last_improved))


if __name__ == "__main__":
    vocab_src = "data/middle_result/vocab.npy"
    data = Data()
    vocab, _ = data.load_vocab(vocab_src)

    config = CharCNNConfig()
    config.vocab_size = len(vocab)
    print(len(vocab))

    model = CharCNN(config)
    # 训练集
    x_train_src = "data/vectorized_data/train/x.npy"
    y_train_src = "data/vectorized_data/train/y.npy"

    # 验证集
    x_valid_src = "data/vectorized_data/validation/x.npy"
    y_valid_src = "data/vectorized_data/validation/y.npy"

    # 模型保存
    model_save_src = "data/text_cnn_model"

    epochs = 100

    train(x_train_src, y_train_src, x_valid_src, y_valid_src, epochs, model_save_src)


