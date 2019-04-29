from char_cnn_model2 import CharCNN, CharCNNConfig
from data_process import Data
import os
import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
import tensorflow.contrib.keras as kr


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    # model.keep_prob: keep_prob
    return feed_dict


def batch_iter(x, y, batch_size=128):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        x_padded = kr.preprocessing.sequence.pad_sequences(x_shuffle[start_id:end_id], maxlen=config.seq_length,
                                                           padding="post", truncating="post")
        yield x_padded, y_shuffle[start_id:end_id]


# 评估在某一数据上的准确率和损失
def evaluate(sess, x_, y_):

    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard
    tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()

    print("Loading training and validation data...")

    # 载入训练集与验证集
    start_time = time.time()
    x_train = np.load(os.path.join(train_dir, "x.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(train_dir, "y.npy"))
    print("the shape of train_x is {}, train_y is {}".format(np.shape(x_train), np.shape(y_train)))
    x_val = np.load(os.path.join(val_dir, "x.npy"), allow_pickle=True)
    y_val = np.load(os.path.join(val_dir, "y.npy"))
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
    require_improvement = 200  # 如果超过n iter未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch: {}".format(epoch))
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, 0.5)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能

                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)

                loss_val, acc_val = evaluate(session, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {}, Train Loss: {}, Train Acc: {},' \
                      + ' Val Loss: {}, Val Acc: {}, Time: {} {}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.opt, feed_dict=feed_dict)  # 训练
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
    print("the best acc on validation is {}".format(best_acc_val))


if __name__ == '__main__':
    train_dir = "data/vectorized_data/train"
    val_dir = "data/vectorized_data/validation"
    vocab_dir = "data/file_dict/train/vocab.npy"

    save_dir = 'data/model2'

    data_process = Data()

    config = CharCNNConfig()

    if not os.path.exists(vocab_dir):
        data_process.build_vocab(train_dir, vocab_dir)

    words, word_to_id = data_process.load_vocab(vocab_dir)
    config.vocab_size = len(words)

    model = CharCNN(config)
    train()
