import tensorflow as tf


class TRCNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64     # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 9        # 类别数
    vocab_size = None       # 词汇表达小

    hidden_size = 100        # 隐藏层神经元

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128        # 每批训练大小
    num_epochs = 100          # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
    l2_reg_lambda = 0.01

    decay_rate = 0.9  # 学习率衰减率
    decay_steps = 600  # 学习率衰减速率


class TextRCNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name="input_text_len")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss = tf.constant(0.0)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # embedding
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],
                                        dtype=tf.float32)

            # batch_size, max_len, embed_dim]
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("RNN"):
            # 双向LSTM
            lstm_fw_cell = self.dropout()

            lstm_bw_cell = self.dropout()

            # outputs:[fw_output, bw_output] fw_output dim: [batch_size, sequence_len, filter_num]
            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedding_inputs,
                                                              sequence_length=self.sequence_lengths, dtype=tf.float32)
            # 论文中要求送入卷积层的每个词的向量由三部分组成[cl, word_dim, cr]，cl为word的左侧上下文，cr为word的右侧上下文
            # 由于每个句子的第一个词没有cl，最后一个词没有cr，这里用0填充
        with tf.name_scope("WordRepresentation"):
            word_dim = 3*config.embedding_dim

            first_word_padding = tf.pad(self.embedding_inputs[:, 0, :],
                                        paddings=[[0, 0], [self.config.embedding_dim, 0]])
            last_word_padding = tf.pad(self.embedding_inputs[:, -1, :],
                                       paddings=[[0, 0], [0, self.config.embedding_dim]])
            first_words = tf.concat([first_word_padding, outputs[1][:, 0, :]], 1)
            first_words = tf.reshape(first_words, shape=[-1, 1, word_dim])

            last_words = tf.concat([outputs[0][:, -1, :], last_word_padding], 1)
            last_words = tf.reshape(last_words, shape=[-1, 1, word_dim])
            # 将其他的词拼接
            # left_x = tf.concat([outputs[0][:, 1:-1, :], self.embedding_inputs[:, 1:-1, :]], 2, name="left_x")
            # right_x = tf.concat([left_x, outputs[1][:, 1:-1, :]], 2, name="right_x")
            concat_x = tf.concat([outputs[0][:, 1:-1, :], self.embedding_inputs[:, 1:-1, :], outputs[1][:, 1:-1, :]],
                                 2, name="words_concat")
            word_repr = tf.concat([first_words, concat_x, last_words], 1, name="final_outputs")

        with tf.name_scope("TextRepresentation"):
            text_rep = tf.layers.dense(word_repr, self.config.hidden_size, activation=tf.nn.relu)

        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(text_rep, keep_prob=self.keep_prob)

        with tf.name_scope("max_pooling"):
            pooled = tf.reduce_max(input_tensor=dropout, axis=1, keep_dims=True)
            pooled = tf.reshape(pooled, [-1, self.config.hidden_size])

        with tf.name_scope("Softmax"):
            W = tf.get_variable(name="fc_W", shape=[self.config.hidden_size, config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name='fc_b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logists = tf.nn.xw_plus_b(pooled, W, b, name="logist")
            self.prob = tf.nn.softmax(logits=self.logists, name="prob")
            self.y_pred = tf.argmax(self.prob, 1, name="prediction")
        with tf.name_scope("Loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logists, labels=self.input_y)
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

    def dropout(self):  # 为每一个rnn核后面加一个dropout层
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.embedding_dim, state_is_tuple=True)

        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
