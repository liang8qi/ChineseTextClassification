# decoding: utf-8
import re
import os
import pynlpir
import numpy as np
import random
import math

"""
categories = {
            "C000008": 0,
            "C000010": 1,
            "C000013": 2,
            "C000014": 3,
            "C000016": 4,
            "C000020": 5,
            "C000022": 6,
            "C000023": 7,
            "C000024": 8
        }
"""
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
proportion = [0.7, 0.1]
frequent_min = 3  # 过滤低频词
num_category = 9
num_feature = 2000  # 每个类别选择特征数

class DataProcess(object):

    # 清洗数据
    @staticmethod
    def clean_sentence(sentence):
        # 抽出中文字符
        sentence = re.sub(r"[^\u4e00-\u9fff]", " ", sentence)

        # 删除两个以上连续空白符
        sentence = re.sub(r"\s{2,}", "", sentence).replace(" ", "")

        return sentence.strip()

    # 加载停用词
    @staticmethod
    def load_stop_words(stop_words_src):
        stop_words = []
        with open(stop_words_src, "r", encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        return stop_words

    # 将一个文件分词
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
            words = pynlpir.segment(sentence, pos_tagging=False)
            sentence_segment = []
            for word in words:
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
        pynlpir.open()
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
                    # print("{} is Null".format(input_file_src))
                    invalid_files.append(input_file_src)

        pynlpir.close()
        if len(invalid_files) > 0:
            print("无效文件:")
            for file in invalid_files:
                print(file)

        np.save(os.path.join(output_src, "file_list.npy"), data_set)
        print("Finish")

    # 按比例划分数据集，并将划分后的结果保存起来
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

    # 获取训练集中的所有词，并统计每个词的分布, 去除低频词
    def get_all_words(self, train_file_list_src):
        train_data_list = np.load(train_file_list_src)
        print("the size of train_data_set is {}".format(len(train_data_list)))

        # 用于记录词的分粗 key为词，value为一个num_categories维的矩阵，每一维i表示该词在i类中出现的次数（在一个文件中出现代表一次）
        words_distribute = {}
        # 统计训练集中每类样本数
        num_files = np.zeros(num_category, dtype=np.float)
        for file in train_data_list:
            category = int(file[-1])
            num_files[category] += 1
            # 记录该文件中出现的词
            file_vocab = set()
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    words = line.strip().split(" ")
                    for word in words:
                        file_vocab.add(word)

            # 合并
            for word in file_vocab:
                if word not in words_distribute:
                    words_distribute[word] = np.zeros(num_category, dtype=np.int32)
                # 该词在category类中出现次数+1
                words_distribute[word][category] += 1

        # 去低频词
        distribution = {}
        for word in words_distribute:
            word_num = np.sum(words_distribute[word])
            if word_num < frequent_min:
                continue
            distribution[word] = words_distribute[word]

        return distribution, num_files

    # 选择特征
    @staticmethod
    def select_features(words_distribution, output_src, num_files):
        all_words = list(words_distribution.keys())
        all_words_size = len(all_words)
        word_to_id = dict(zip(all_words, range(all_words_size)))
        distribute_matrix = list(words_distribution.values())
        print(np.shape(distribute_matrix))

        # print(file_num)
        # 训练样本总数
        total_files = np.sum(num_files)
        print("total_files: {}".format(total_files))
        # 每类的概率值
        class_prob = num_files / total_files
        # 划分矩阵 每个词将训练集划分为两个部分（两项），一部分包含该词，另一部分不包含该词 divide_matrix[i]表示包含词i的文档数
        divide_matrix = np.sum(distribute_matrix, 1)  # 每项值必>0
        # 交集 divide_distribute_prob[i,j]表示j类中包含词i的样本数
        divide_distribute_prob = [distribute_matrix[i] / num_files for i in range(all_words_size)]

        # divide_distribute_prob = words_distribute_matrix / divide_matrix
        # 信息增益
        info_gain = np.zeros(all_words_size, dtype=np.float64)
        # 计算训练集的经验熵
        empirical_entropy = -np.sum([prob * math.log(prob, 2) for prob in class_prob])

        # 对每个词计算信息增益
        for word in word_to_id:
            id = word_to_id[word]
            # 第一项
            first_item = -(divide_matrix[id] / total_files) * np.sum(
                [prob * math.log(prob, 2) for prob in divide_distribute_prob[id] if prob])  # prob可能为0

            # 第二项
            second_prob = \
                [(num_files[i] - distribute_matrix[id][i]) / (total_files - divide_matrix[id])
                 for i in range(num_category)]
            second_item = -(1 - divide_matrix[id] / total_files) * np.sum(
                [prob * math.log(prob, 2) for prob in second_prob if prob])

            # 信息增益 经验熵-条件熵
            info_gain[id] = empirical_entropy - (first_item + second_item)

        # 降序排
        words_index = np.argsort(-info_gain, 0)
        print("the total features is {}".format(len(words_index)))
        # 为每个类选择指定数量的特征数
        selected_num = np.zeros(num_category)
        # 选择的特征词
        vocab = set()
        # 特征词的IDF
        words_idf = {}
        for index in words_index:
            word = all_words[index]
            if word in vocab:
                continue
            # cnt = 0
            for category in range(num_category):
                if distribute_matrix[index][category] > 0:
                    selected_num[category] += 1

            # 条件检查
            if np.min(selected_num) >= num_feature:
                break
            # 添加
            vocab.add(word)
            # IDF
            words_idf[word] = math.log(total_files / divide_matrix[index], 2)

        np.save(os.path.join(output_src, "vocab.npy"), list(vocab))
        np.save(os.path.join(output_src, "word_idf.npy"), words_idf)

        print("the size of word_idf is {}".format(len(words_idf)))
        return list(vocab), words_idf

    @staticmethod
    def load_vocab(vocab_dir):

        words_vocab = np.load(vocab_dir)

        word_to_id = dict(zip(words_vocab, range(len(words_vocab))))

        return words_vocab, word_to_id

    @staticmethod
    def text_vectorization(files_list_src, word_to_id, word_idf, output_src):

        files_list = np.load(files_list_src)
        # 向量长度
        vector_len = len(word_to_id)

        # 保存数据集的特征向量
        vectors = []

        for file in files_list:
            category = int(file[-1])
            words_num = 0
            # 初始化文件的特征向量 最后一位为标签
            file_vector = np.zeros(vector_len + 1, dtype=np.float64)
            with open(file, "r") as f:
                for line in f:
                    words = line.strip().split(" ")
                    words_num += len(words)
                    for word in words:
                        if word in word_idf:
                            # print("{} {}".format(word, word_idf[word]))
                            file_vector[word_to_id[word]] += word_idf[word]  # 1 * idf

            if words_num == 0:
                words_num = 1
            # 计算TF-IDF / words_num
            file_vector = file_vector / words_num
            file_vector[-1] = category

            vectors.append(file_vector)

        # 保存
        np.save(output_src, vectors)


if __name__ == '__main__':

    data_process = DataProcess()
    # 停用词表
    stop_word_src = "data/stopwords.txt"
    # 原数据集地址
    # data_set_dir = ""
    # 分词后的地址
    output_src = "data/pynlpir"
    # 中间结果
    middle_results_src = "data/middle_result"

    # 加载停用词
    # stop_words = data_process.load_stop_words(stop_word_src)
    # 对全部数据集进行清洗，分词
    # data_process.batch_process(data_set_dir, output_src, stop_words)

    # 划分数据集
    file_list_src = os.path.join(output_src, "file_list.npy")

    divided_files_list_src = "data/file_list"
    # data_process.divide_data_set(file_list_src, divided_files_list_src, proportion)

    # 获得训练集上的所有词及其分布
    distribution, num_files = data_process.get_all_words(os.path.join(divided_files_list_src, "train_file_list.npy"))
    print(num_files)
    # 选择特征
    data_process.select_features(distribution, middle_results_src, num_files)
   
    words_vocab, word_to_id = data_process.load_vocab(os.path.join(middle_results_src, "vocab.npy"))
    # 加载word_idf
    word_idf = np.load(os.path.join(middle_results_src, "word_idf.npy")).item()

    vectorized_data_src = "data/vectorized_data"
    if not os.path.exists(vectorized_data_src):
        os.mkdir(vectorized_data_src)
    # 训练集向量化
    data_process.text_vectorization(os.path.join(divided_files_list_src, "train_file_list.npy"),
                                    word_to_id, word_idf, os.path.join(vectorized_data_src, "train.npy"))

    # 验证集向量化
    data_process.text_vectorization(os.path.join(divided_files_list_src, "valid_file_list.npy"),
                                    word_to_id, word_idf, os.path.join(vectorized_data_src, "valid.npy"))

    # 测试集向量化
    data_process.text_vectorization(os.path.join(divided_files_list_src, "test_file_list.npy"),
                                    word_to_id, word_idf, os.path.join(vectorized_data_src, "test.npy"))















