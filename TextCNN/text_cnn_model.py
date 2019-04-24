import tensorflow as tf


class TCNNConfig(object):
    train_batch_size = 128  # 训练数据的batch大小
    vocab_size = 115082  # 词典大小
    word_dim = 64  # 词向量维度
    text_size = 600  # 一个文本的大小（词）
    num_category = 9  # 类别数
    num_dense_units = 9  # 全联接层神经元
    keep_dropout_prob = 0.5
    learning_rate = 0.001
    filter_sizes = [2, 3, 4]  # 卷积核大小
    num_filter = 128  # 每类卷积核数量
    l2_reg_lambda = 0.01
    save_per_batch = 10  # 每10个batch保存一次结果

    decay_rate = 0.9  # 学习率衰减率
    decay_steps = 600  # 学习率衰减速率


class TextCNNModel(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, config.text_size], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, config.num_category], name="input_y")
        self.keep_dropout_prob = tf.placeholder(tf.float32, name="keep_dropout_prob")
        self.l2_loss = tf.constant(0.0)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Embedding
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable(name="embedding2", shape=[config.vocab_size, config.word_dim],
                                             dtype=tf.float32)
            # batch_size * text_size * word_dim
            self.word_vectors = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # dim: [None, text_size, word_dim, 1]
            # in_channels = 1
            self.word_vectors = tf.expand_dims(self.word_vectors, -1, name="word_vectors")

        # 卷积层
        # filter size 3
        with tf.name_scope("ConvLayer"):
            self.pooling_list = []
            for id, filter_size in enumerate(config.filter_sizes):
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, config.word_dim, 1, config.num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.num_filter]), name="b")
                # output_dim = [None, text_size-filter_size + 1, 1, num_filter]
                conv = tf.nn.conv2d(input=self.word_vectors, filter=W,
                                    strides=[1, 1, 1, 1], padding="VALID", name="conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 池化
                # dim = [None, 1, 1, num_filter]
                pooling = tf.nn.max_pool(value=h, ksize=[1, config.text_size-filter_size+1, 1, 1],
                                         strides=[1, 1, 1, 1], padding="VALID", name="pooling")
                self.pooling_list.append(pooling)

        with tf.name_scope("ConcatLayer"):
            # concat
            total_features = config.num_filter * len(config.filter_sizes)
            # dim [None, 1, 1, total_features]
            total_pooling = tf.concat(self.pooling_list, 3)
            # dim [None, total_features]
            total_pooling = tf.reshape(total_pooling, shape=[-1, total_features], name="total_pooling")

        with tf.name_scope("DropoutLayer"):
            # dim [None, total_features]
            dropout = tf.nn.dropout(total_pooling, self.keep_dropout_prob)

        with tf.name_scope("SoftmaxLayer"):
            W = tf.get_variable(name="fc_W", shape=[total_features, config.num_category],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[config.num_category]), name='fc_b')

            self.logists = tf.nn.xw_plus_b(dropout, W, b, name="logist")
            self.prob = tf.nn.softmax(logits=self.logists, name="prob")
            self.y_pred = tf.argmax(self.prob, 1, name="prediction")

        with tf.name_scope("Loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logists, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.loss = tf.reduce_mean(cross_entropy) + config.l2_reg_lambda * self.l2_loss

        with tf.name_scope('optimizer'):
            learning_rate = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=self.global_step,
                                                       decay_rate=config.decay_rate, decay_steps=config.decay_steps,
                                                       staircase=True)

            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                    global_step=self.global_step)

        with tf.name_scope('accuracy'):
            self.correct = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32))
