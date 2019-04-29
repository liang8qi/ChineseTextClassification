import tensorflow as tf


class TDRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64     # 词向量维度
    seq_length = 600       # 序列长度
    num_classes = 2        # 类别数
    vocab_size = None       # 词汇表达小

    hidden_dim = 128        # 隐藏层神经元

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128        # 每批训练大小
    num_epochs = 100          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    decay_rate = 0.9  # 学习率衰减率
    decay_steps = 600  # 学习率衰减速率

    k_val = 15  # RNN保留步长

    max_grad_norm = 100


class TDRNN(object):
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name="input_text_len")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.in_training_mode = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # embedding
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],
                                        dtype=tf.float32)
            # batch_size, max_len, embed_dim]
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("Embedding_Dropout"):
            self.embedding_dropout = tf.nn.dropout(self.embedding_inputs, keep_prob=self.keep_prob)

        with tf.name_scope("DNN"):
            # dim: [sequence_len, batch_size, num_units]
            hidden_list = []
            # 在每个sequence左部padding，为了保证dynamic最后一个step的输出不受padding的影响
            input_padded = tf.pad(self.embedding_dropout, paddings=[[0, 0], [config.k_val - 1, 0], [0, 0]])
            for start in range(config.seq_length):

                end = start + config.k_val
                # print("start: {}, end: {}".format(start, end))
                input = input_padded[:, start:end, :]

                with tf.name_scope("gru"), tf.variable_scope("rnn") as scope:
                    gru_cell = self.dropout()
                    if start != 0:
                        scope.reuse_variables()
                    # state dim: [batch_size, hidden_dim]
                    _, state = tf.nn.dynamic_rnn(gru_cell, input, dtype=tf.float32)

                with tf.name_scope("state_dropout"):
                    state_dropout = tf.nn.dropout(state, keep_prob=self.keep_prob)

                with tf.name_scope("mlp"), tf.variable_scope("mlp") as scope:
                    if start != 0:
                        scope.reuse_variables()

                    W = tf.get_variable("W", shape=[config.hidden_dim, config.hidden_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[config.hidden_dim]), name="b")
                    mlp_output = tf.nn.relu(tf.nn.xw_plus_b(state_dropout, W, b), "mlp_output")
                    # [batch_size, 1, hidden_dim]
                    mlp_output_expand = tf.expand_dims(mlp_output, 1)
                    # print(mlp_output_expand)
                    hidden_list.append(mlp_output_expand)

            # hidden_list: [sequence_len, batch_size, 1, hidden_dim]
            # dim: [batch_size, sequence_len, hidden_dim]
            hidden_list_concat = tf.concat(hidden_list, 1)

        with tf.name_scope("Max_Pooling"):
            # dim: [batch_size, sequence_len, hidden_dim, -1]
            pooling_input = tf.expand_dims(hidden_list_concat, -1)
            # dim: [batch_size, 1, hidden_dim, 1]
            pooled = tf.nn.max_pool(pooling_input, ksize=[1, config.seq_length, 1, 1],
                                    strides=[1, 1, 1, 1], padding="VALID", name="pooling")

            pooled_reshape = tf.reshape(pooled, [-1, self.config.hidden_dim])

        with tf.name_scope("SoftMax"):
            W = tf.get_variable(name="softmax_W", shape=[config.hidden_dim, config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name='softmax_b')

            self.logists = tf.nn.xw_plus_b(pooled_reshape, W, b, name="logist")
            self.prob = tf.nn.softmax(logits=self.logists, name="prob")
            self.y_pred = tf.argmax(self.prob, 1, name="prediction")

        with tf.name_scope("Loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logists, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("Optimizer"):
            learning_rate = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=self.global_step,
                                                       decay_rate=config.decay_rate, decay_steps=config.decay_steps,
                                                       staircase=True)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = self.opt.compute_gradients(self.loss)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], config.max_grad_norm), gv[1]) for gv in grads_and_vars]

            self.train_op = self.opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        with tf.name_scope('Accuracy'):
            self.correct = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def dropout(self):  # 为每一个rnn核后面加一个dropout层
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_dim)
        return tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=self.keep_prob)
