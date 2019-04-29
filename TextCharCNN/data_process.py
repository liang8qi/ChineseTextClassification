import re
import os
import numpy as np
import tensorflow.contrib.keras as kr
import random
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
        # 抽出中文字符
        sentence = re.sub(r"[^\u4e00-\u9fff]", " ", sentence)

        # 删除两个以上连续空白符
        sentence = re.sub(r"\s{2,}", "", sentence)

        return sentence.strip().replace(" ", "")

    def clean_file(self, file_src):
        file = []
        with open(file_src, "r", encoding="gbk", errors="ignore") as f:
            content = f.read()
            sentences = content.replace('\t', '').replace('\u3000', '').split("\n")

            for sentence in sentences:
                sentence = self.clean_sentence(sentence)
                if len(sentence):
                    file.append(sentence)

        return "".join(file)

    @staticmethod
    def divide_data_set(data_set_src, output_src, proportion):
        if not os.path.exists(output_src):
            os.mkdir(output_src)
        dir_list = os.listdir(data_set_src)
        #  获取全部数据集
        data_set = {}

        for dir_name in dir_list:
            if not dir_name in categories:
                continue

            category = str(categories[dir_name])
            if category not in data_set:
                data_set[category] = []
            dir_path = os.path.join(data_set_src, dir_name)
            file_list = os.listdir(dir_path)
            for file in file_list:
                if file.endswith(".txt"):
                    data_set[category].append(os.path.join(dir_path, file))

        train_data_set = {}
        valid_data_set = {}
        test_data_set = {}

        train_cnt = 0
        valid_cnt = 0
        test_cnt = 0
        for category in data_set:
            data_cnt = len(data_set[category])
            index_list = random.sample(range(data_cnt), data_cnt)

            middle = int(data_cnt * proportion[0])
            end = int(data_cnt * proportion[1]) + middle

            train_data_set[category] = [data_set[category][index] for index in index_list[:middle]]
            valid_data_set[category] = [data_set[category][index] for index in index_list[middle:end]]
            test_data_set[category] = [data_set[category][index] for index in index_list[end:]]

            train_cnt += len(train_data_set[category])
            valid_cnt += len(valid_data_set[category])
            test_cnt += len(valid_data_set[category])

        print("the size of train_dataset is {}".format(train_cnt))
        print("the size of val_dataset is {}".format(valid_cnt))
        print("the size of test_dataset is {}".format(test_cnt))

        np.save(os.path.join(output_src, "train_file_dict.npy"), train_data_set)
        np.save(os.path.join(output_src, "valid_file_dict.npy"), valid_data_set)
        np.save(os.path.join(output_src, "test_file_dict.npy"), test_data_set)

        print("Divide Finish")

    # 将数据集中的文件存到一个文件中
    def batch_process(self, data_set_dict_src, output_src):
        if not os.path.exists(output_src):
            os.mkdir(output_src)
        data_set = np.load(data_set_dict_src).item()
        output_file = open(os.path.join(output_src, "batch.txt"), "a", encoding="utf-8")
        total_files = []   # 按序保存所有文件的路径，便于最后分析结果时回溯
        invalid_files = []  # 保存无效文件
        for category in data_set:
            file_list = data_set[category]
            for file in file_list:
                content = self.clean_file(file)
                if not len(content.strip()):
                    invalid_files.append(file)
                    continue
                output_file.write(category + '\t' + content + '\n')
                total_files.append(file)

        output_file.close()
        np.save(os.path.join(output_src, "file_list.npy"), total_files)
        print("the size of dataset is {}".format(len(total_files)))
        if len(invalid_files):
            print("无效文件：")
            for file in invalid_files:
                print(file)

    # 读取文件
    @staticmethod
    def read_file(filename):

        contents, labels = [], []
        with open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                label, content = line.split('\t')
                if content:
                    contents.append(list(content))  # 字符级
                    labels.append(label)

        return contents, labels

    @staticmethod
    def build_vocab(train_src, vocab_src):
        words = set()
        with open(train_src, encoding='utf-8', errors='ignore') as f:
            for line in f:
                label, content = line.split('\t')
                if content:
                    for word in content:
                        if len(word):
                            words.add(word)

        vocab = ['<PAD>'] + list(words)
        np.save(vocab_src, vocab)
        print("the size of vocab is {}".format(len(vocab)))

    # 加载字典
    @staticmethod
    def load_vocab(vocab_src):
        # words = open_file(vocab_dir).read().strip().split('\n')
        vocab = np.load(vocab_src)

        word_to_id = dict(zip(vocab, range(len(vocab))))

        return vocab, word_to_id

    # 将数据向量化
    def text_vectorization(self, filename, word_to_id, output_src):
        data_id, label_id = [], []
        with open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                label, content = line.split('\t')
                data_id.append([word_to_id[x] for x in content if x in word_to_id])
                label_id.append(label)

        y_pad = kr.utils.to_categorical(label_id, num_classes=num_category)  # 将标签转换为one-hot表示 9类

        np.save(os.path.join(output_src, "x.npy"), data_id)
        np.save(os.path.join(output_src, "y.npy"), y_pad)

        return data_id, y_pad

    # 为了方便实验对比，添加一个文件，用于处理分词后的数据和划分好的数据集
    # 只需要将保存的划分好的数据集文件重新处理一遍
    @staticmethod
    def load_divided_data_set(train_file_list_src, val_file_list_src, test_file_list_src, output_src):
        train_file_list = np.load(train_file_list_src)
        val_file_list = np.load(val_file_list_src)
        test_file_list = np.load(test_file_list_src)

        # 处理后的数据集
        train_data_set = {}
        val_data_set = {}
        test_data_set = {}

        for file in train_file_list:
            # 最后一位为类别 共9类
            category = file[-1]
            if category not in train_data_set:
                train_data_set[category] = []
            train_data_set[category].append(file)

        for file in val_file_list:
            category = file[-1]
            if category not in val_data_set:
                val_data_set[category] = []
            val_data_set[category].append(file)

        for file in test_file_list:
            category = file[-1]
            if category not in test_data_set:
                test_data_set[category] = []
            test_data_set[category].append(file)

        print("the size of train_dataset is {}".format(len(train_file_list)))
        print("the size of val_dataset is {}".format(len(val_file_list)))
        print("the size of test_dataset is {}".format(len(test_file_list)))

        np.save(os.path.join(output_src, "train_file_dict.npy"), train_data_set)
        np.save(os.path.join(output_src, "valid_file_dict.npy"), val_data_set)
        np.save(os.path.join(output_src, "test_file_dict.npy"), test_data_set)

        print("Redistrict Finish")


if __name__ == "__main__":

    data = Data()
    # 数据集地址
    data_set_src = ""
    # 存储划分后的数据集文件列表
    file_list_src = "data/file_dict"
    # 划分数据集
    data.divide_data_set(data_set_src, file_list_src, [0.7, 0.1])

    # 处理训练集
    train_src = os.path.join(file_list_src, "train")
    data.batch_process(os.path.join(file_list_src, "train_file_dict.npy"), train_src)

    # 在训练集上建立字典
    vocab_src = os.path.join(train_src, "vocab.npy")
    data.build_vocab(os.path.join(train_src, "batch.txt"), vocab_src)
    vocab, word_to_id = data.load_vocab(vocab_src)

    # 训练数据向量化
    data.text_vectorization(os.path.join(train_src, "batch.txt"), word_to_id, train_src)

    # 验证数据处理
    valid_src = os.path.join(file_list_src, "validation")
    data.batch_process(os.path.join(file_list_src, "valid_file_dict.npy"), valid_src)
    # 验证集向量化
    data.text_vectorization(os.path.join(valid_src, "batch.txt"), word_to_id, valid_src)

    # 测试数据处理
    test_src = os.path.join(file_list_src, "test")
    data.batch_process(os.path.join(file_list_src, "valid_file_dict.npy"), test_src)
    # 测试集向量化
    data.text_vectorization(os.path.join(test_src, "batch.txt"), word_to_id, test_src)












