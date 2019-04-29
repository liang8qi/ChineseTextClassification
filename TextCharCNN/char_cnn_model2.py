# decoding:utf-8
import tensorflow as tf


class CharCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 9  # 类别数
    vocab_size = None  # 词汇表达小
    learning_rate = 1e-3  # 学习率
    decay_rate = 0.9  # 学习率衰减率
    decay_steps = 500  # 学习率衰减速率
    l2_reg_lambda = 0.01
    batch_size = 128  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    # 卷积层
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
    # 全联接层
    fully_layers_units = [1024, 1024]

    normal_stddev = 0.05


class CharCNN(object):
    def __init__(self, config):
        self.config = config

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, config.seq_length], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, config.num_classes], name="input_y")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)

        # Embedding
        with tf.name_scope("Embedding-Layer"):
            self.embedding = tf.get_variable(name="embedding", shape=[config.vocab_size, config.embedding_dim])
            # batch_size * text_size * word_dim
            self.word_vectors = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.word_vectors = tf.expand_dims(self.word_vectors, axis=-1, name="word_vectors")
        tmp = self.word_vectors
        # 卷积层
        for id, size in enumerate(config.conv_layers):
            with tf.name_scope("ConvLayer-%s"%id):
                filter_width = tmp.get_shape()[2].value
                # print("filter_width {}".format(filter_width))
                filter_shape = [size[1], filter_width, 1, size[0]]
                # [filter_height, filter_width, in_channels, out_channels]
                W = tf.Variable(tf.random_normal(shape=filter_shape, stddev=config.normal_stddev),
                                dtype='float32', name='W')
                b = tf.Variable(tf.random_normal(shape=[size[0]], stddev=config.normal_stddev),
                                dtype=tf.float32, name="b")

                conv = tf.nn.conv2d(tmp, filter=W, strides=[1, 1, 1, 1], padding="VALID", name='Conv')
                relu = tf.nn.relu(tf.nn.bias_add(value=conv, bias=b), name="relu")
                # pooling
                if size[-1] is None:
                    # relu shape: [None, text_size - filter_high + 1, 1, filter_num]
                    # 最后两维应调换位置，以便卷积
                    tmp = tf.transpose(relu, [0, 1, 3, 2])
                else:
                    with tf.name_scope("MaxPoolingLayer"):
                        # shape [None, (text_size - filter_high + 1)/pooling_ksize, 1, filter_num]
                        # non-overlapping
                        pooling = tf.nn.max_pool(value=relu, ksize=[1, size[-1], 1, 1], strides=[1, size[-1], 1, 1],
                                                 padding="VALID", name="pooling")
                        tmp = tf.transpose(pooling, [0, 1, 3, 2])

        # 将池化的结果拼接
        num_dim = tmp.get_shape()[1].value * tmp.get_shape()[2].value
        # shape [None, num_dim]
        fc_input = tf.reshape(tmp, [-1, num_dim], name="fc_input")
        # dropout
        fc_input = tf.nn.dropout(fc_input, keep_prob=self.keep_prob)
        w_width = [num_dim, config.fully_layers_units[1]]
        # 全联接层
        for id, units in enumerate(config.fully_layers_units):
            with tf.name_scope("FullConnectionLayer-%s"%id):
                W = tf.Variable(initial_value=tf.random_normal(shape=[w_width[id], units], stddev=config.normal_stddev),
                                dtype=tf.float32, name="W")
                b = tf.Variable(initial_value=tf.random_normal(shape=[units], stddev=config.normal_stddev,
                                                               dtype=tf.float32, name="b"))
                h = tf.nn.xw_plus_b(fc_input, W, b)

            with tf.name_scope("Dropout-%s"%id):
                fc_input = tf.nn.dropout(h, keep_prob=self.keep_prob)
                fc_input = tf.nn.relu(fc_input, name="fc_output")

        # Softmax
        with tf.name_scope("SoftmaxLayer"):
            W = tf.Variable(initial_value=tf.random_normal(shape=[config.fully_layers_units[-1], config.num_classes],
                                                           stddev=config.normal_stddev), dtype=tf.float32, name="W")
            b = tf.Variable(initial_value=tf.random_normal(shape=[config.num_classes], stddev=config.normal_stddev,
                                                           dtype=tf.float32, name="b"))

            output = tf.nn.xw_plus_b(fc_input, W, b)
            # print(output)
            self.y_pred = tf.argmax(output, 1)

        # Loss
        with tf.name_scope("Loss"):

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=output)
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.loss = tf.reduce_mean(cross_entropy) + config.l2_reg_lambda * self.l2_loss
            learning_rate = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=self.global_step,
                                                       decay_rate=config.decay_rate, decay_steps=config.decay_steps,
                                                       staircase=True)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss,
                                                                                    global_step=self.global_step)
        # Accuracy
        with tf.name_scope("Accuracy"):
            self.correct = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32))



