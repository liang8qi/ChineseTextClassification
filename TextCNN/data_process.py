import re
import os
import numpy as np
import jieba
import random
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
max_len = 600  # 向量化文本最大长度
num_category = 9  # 类别数


class Data(object):
    """清洗训练数据，去除数据中的非中文数字字母字符，并对文件进行分词"""

    @staticmethod
    def load_stop_words(stop_words_src):
        stop_words = []
        with open(stop_words_src, "r", encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        return stop_words

    @staticmethod
    def clean_sentence(sentence):
        # 抽出中文字符及数字
        sentence = re.sub(r"[^\u4E00-\u9FD5]", " ", sentence)

        return sentence.strip()

    # 处理一个文件
    def segmentation(self, file_src, output_src, stop_words):
        sentence_list = []

        with open(file_src, "r", encoding="gbk", errors="ignore") as f:
            content = f.read()
            sentences = content.replace('\t', '').replace('\u3000', '').split("\n")
            for sentence in sentences:
                sentence = self.clean_sentence(sentence)
                if len(sentence):
                    sentence_list.append(sentence)

        if len(sentence_list) == 0:
            return False

        output_file = open(output_src, "a", encoding="utf-8")
        for sentence in sentence_list:

            words = jieba.lcut(sentence)
            sentence_segment = []
            for word in words:
                if not len(word):
                    continue
                # 去停用词
                if word not in stop_words:
                    sentence_segment.append(word)
            output_file.write(" ".join(sentence_segment) + " ")
        output_file.close()
        return True

    # 批量处理
    # 同时保存一个输出文件列表
    def batch_process(self, dirs_src, output_src, stop_words):
        if not os.path.exists(output_src):
            os.mkdir(output_src)
        data_set = {}
        dir_list = os.listdir(dirs_src)
        invalid_files = []

        for dir_name in dir_list:
            if dir_name not in categories:
                continue

            output_dir = os.path.join(output_src, dir_name)
            if os.path.exists(output_dir) is not True:
                os.mkdir(output_dir)

            dir_src = os.path.join(dirs_src, dir_name)
            category = categories[dir_name]

            if str(category) not in data_set:
                data_set[str(category)] = []

            file_list = os.listdir(dir_src)
            for file in file_list:
                if file.endswith(".txt") is not True:
                    continue

                output_file_src = os.path.join(output_dir, file.replace("txt", str(category)))
                input_file_src = os.path.join(dir_src, file)

                result = self.segmentation(input_file_src, output_file_src, stop_words)
                if result:
                    data_set[str(category)].append(output_file_src)
                    print("{} is cut successfully".format(output_file_src))
                else:
                    invalid_files.append(input_file_src)

        if len(invalid_files) > 0:
            print("无效文件:")
            for file in invalid_files:
                print(file)

        np.save(os.path.join(output_src, "file_list.npy"), data_set)
        print("Finish")

    # 划分数据集
    @staticmethod
    def divide_data_set(file_list_src, output_src, proportion):
        if not os.path.exists(output_src):
            os.mkdir(output_src)
        data_set = np.load(file_list_src).item()
        train_data_set = []
        verify_data_set = []
        test_data_set = []

        for category in data_set:
            data_cnt = len(data_set[category])
            index_list = random.sample(range(data_cnt), data_cnt)

            middle = int(data_cnt * proportion[0])
            end = int(data_cnt * proportion[1]) + middle

            train_data_set.extend([data_set[category][index] for index in index_list[:middle]])
            verify_data_set.extend([data_set[category][index] for index in index_list[middle:end]])
            test_data_set.extend([data_set[category][index] for index in index_list[end:]])

        print("the size of train_dataset is {}".format(len(train_data_set)))
        print("the size of verify_dataset is {}".format(len(verify_data_set)))
        print("the size of test_dataset is {}".format(len(test_data_set)))

        np.save(os.path.join(output_src, "train_file_list.npy"), train_data_set)
        np.save(os.path.join(output_src, "valid_file_list.npy"), verify_data_set)
        np.save(os.path.join(output_src, "test_file_list.npy"), test_data_set)

        print("Divide Finish")

    # 建立词典
    @staticmethod
    def build_vocab(train_file_list_src, vocab_src):
        train_files = np.load(train_file_list_src)
        print("the num of train files is {}".format(len(train_files)))

        vocab = set()

        for file in train_files:
            with open(file, "r", encoding="utf-8") as f:
                line = f.readline()
                words = line.strip().split(" ")
                for word in words:
                    if not len(word):
                        continue
                    vocab.add(word)

        # 添加一个 <PAD> 来将所有文本pad为同一长度
        vocab = ['<PAD>'] + list(vocab)

        np.save(vocab_src, vocab)
        print("The size of vocab is {}".format(len(vocab)))

    # 加载词典
    @staticmethod
    def load_vocab(vocab_src):
        vocab = np.load(vocab_src)
        word_to_id = dict(zip(vocab, range(len(vocab))))

        return vocab, word_to_id

    @staticmethod
    def text_vectorization(file_src, vocab, word_to_id):
        text_vector = []
        label = int(file_src.split(".")[-1])  # 类别 最后一位为类别
        text_len = 0
        with open(file_src, "r", encoding="utf-8") as f:
            lines = f.readline().strip()
            for line in lines:
                words = line.strip().split(" ")
                for word in words:
                    if word in vocab:
                        text_vector.append(word_to_id[word])
                        text_len += 1

        print("{} is finished".format(file_src))
        return text_vector, label, text_len

    # 将一个数据集中的文件向量化
    def batch_vectorization(self, data_set_src, vocab, word_to_id, output_src):
        if not os.path.exists(output_src):
            print("There is no {}".format(output_src))
            return
        data_set = np.load(data_set_src)
        x_list = []
        y_list = []
        text_len_list = []
        cnt = 0
        print(len(data_set))
        for data in data_set:
            x, y, text_len = self.text_vectorization(data, vocab, word_to_id)
            x_list.append(x)
            y_list.append(y)
            text_len_list.append(text_len)
            print(cnt)
            cnt += 1

        y_pad = kr.utils.to_categorical(y_list, num_classes=num_category)  # 将标签转换为one-hot表示 9类
        # 保存以便多次使用
        np.save(os.path.join(output_src, "x.npy"), x_list)
        np.save(os.path.join(output_src, "y.npy"), y_pad)
        np.save(os.path.join(output_src, "text_len.npy"), text_len_list)

        print("the dataset {} vectorization is finished".format(data_set_src))

    @staticmethod
    def get_sequence_length(x_batch):

        sequence_lengths = []
        for x in x_batch:
            actual_length = np.sum(np.sign(x))
            sequence_lengths.append(actual_length)
        return sequence_lengths


if __name__ == "__main__":

    data = Data()
    # 源数据集路径
    dirs_src = "data/sougoudataset"
    # 分词后的存储路径
    output_src = "data/jieba"
    # 停用词
    stop_words_src = "data/stopwords.txt"
    # 词典的存储路径
    vocab_src = "data/middle_result/vocab.npy"
    # 加载停用词
    stop_words = data.load_stop_words(stop_words_src)
    # 分词
    data.batch_process(dirs_src, output_src, stop_words)

    # 划分数据集
    file_list_src = os.path.join(output_src, "file_list.npy")
    proportion = [0.7, 0.2]
    divided_files_list_src = "data/file_list"
    data.divide_data_set(file_list_src, divided_files_list_src, proportion)

    # 建立词典
    data.build_vocab(os.path.join(divided_files_list_src, "train_file_list.npy"), vocab_src)

    # 加载词典
    vocab, word_to_id = data.load_vocab(vocab_src)

    # 训练集向量化
    # 向量化后的存储路径
    train_src = "data/vectorized_data/train"
    data.batch_vectorization(os.path.join(divided_files_list_src, "train_file_list.npy"),
                             vocab, word_to_id, train_src)
    
    # 验证集向量化
    validation_src = "data/vectorized_data/validation"
    data.batch_vectorization(os.path.join(divided_files_list_src, "valid_file_list.npy"),
                             vocab, word_to_id, validation_src)

    # 测试集向量化
    test_src = "data/vectorized_data/test"
    data.batch_vectorization(os.path.join(divided_files_list_src, "test_file_list.npy"),
                             vocab, word_to_id, test_src)



