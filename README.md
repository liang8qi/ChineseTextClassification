# ChineseTextClassification
Several methods to solve text classification problems
## 1、前言
刚开始接触NLP，选择从最基础的文本分类入手，完全小白一个，一开始连如何处理数据都不知道，
TensorFlow也不会，在写代码的过程中，也参考了很多别人的代码，尤其感谢[gaussic](https://github.com/gaussic/text-classification-cnn-rnn)
，他的代码给了我很大很大的帮助。

PS.很多细节没补充，上传的文件可能也有问题，有时间会继续补充、改正，欢迎提交issue。

## 2、环境
+ Python3.6
+ TensorFlow 1.3
+ numpy
+ scikit-learn
+ jieba
## 3、数据集
### THUCNews
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，本次实验选取体育, 财经, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐共9个类别，每个类别选取5000个样本。
### 复旦中文文本分类语料
由复旦大学计算机信息与技术系国际数据库中心自然语言处理小组发布，包含Art、Literature 、Economy、Sports等20个类别，分为训练集和测试集两个数据集，分别包含9804和9833篇文档。
## 数据预处理
只提取文档中的中文字符，使用jieba进行分词，然后去停用词。

对THUCNews按照0.7:0.1:0.2的比例将数据集划分成训练集、验证集、测试集。

由于复旦中文文本分类语料只包含训练集和测试集，故从训练集中选取0.1的文档做为验证集，最终训练集包含8088篇文档、验证集包含989篇文档、
测试集包含9825篇文档。分布如下：

测试集 | 验证集 | 训练集
:---:|:---:|:---:
Art	| 666 | 74	| 742
Literature | 29 | 4 | 34
Economy	| 1440 | 160 | 1601
Mine | 29 | 4 | 34
Sports | 1127 | 126	| 1254
History | 419 | 47 | 468
Computer | 1215 | 135 | 1350
Transport | 51 | 6 | 59
Energy | 28 | 4 | 33
Politics | 921 | 103 | 1026
Space | 576 | 64 | 642
Communication | 22 | 3 | 27
Agriculture	| 918 | 103	| 1022
Education | 53 | 6 | 61
Military | 66 | 8 | 76
Electronics	| 24 | 3 | 28
Medical | 45 | 6 | 53

## 4、模型介绍
所有模型都没使用预训练的词向量，参数也没有仔细调过，模型性能未达到最优，结果仅供参考。
### SVM 
去掉低频词（频率低于3），使用信息增益选择特征词，然后用TF-IDF作为权重，使用scikit-learn的SVC模型训练。

这个目前在选择特征词的时候有个小BUG，因为是每个类别提取n个特征词，由于代码实现原因
如果这个类别的特征词的个数n，则将会把所有的词全部选入，这个有时间会更新。
### TextCNN
基于这篇论文提出的模型[Convolutional Neural Networks for Sentence Classification 
](https://arxiv.org/abs/1408.5882)
### CharCNN
做了两个模型

+ char_cnn_model使用的模型参考了[gaussic](https://github.com/gaussic/text-classification-cnn-rnn)使用的
模型，在Embedding层后加了一个卷积层、Max 
Pooling层和Softmax层。

+ char_cnn_model2使用的模型是
[Character-level Convolutional Networks for Text Classification](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica)
所述的模型。

### RNN
包括4个模型

+ text_rnn_model也是参考了[gaussic](https://github.com/gaussic/text-classification-cnn-rnn)使用的RNN模型，
使用了双层的LSTM把最后一个Step的输出直接送入Softmax层进行分类。
+ text_bilstm_model是我在做完CNN和RNN的实验后，想到的，就试了一下，也没去查是否有人这么做过，
使用双向LSTM，将每个Step前向和后向的输出拼接起来，送入一个卷积层，然后使用Max Pooling，最后送入
SoftMax。
+ text_rcnn_model是参考了自动化所NLPR的学生发表在AAAI2015上的一篇文章
[Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9745)。

+ text_dnn_model参考了哈工大讯飞联合实验室发表在ACL2018上的
[Disconnected Recurrent Neural Networks for Text Categorization](https://www.aclweb.org/anthology/papers/P/P18/P18-1215/)
感觉自己复现的有些问题，还在改进中，不过目前发现这个模型在处理长文本时，非常耗时。

## 5、实验结果
+ THUCNews
![THUCNews](https://github.com/DrLiLiang/ChineseTextClassification/blob/master/picture/THUNewsResults.png)

**Error-RCNN是我一开始复现RCNN的时候，错误的将BiLSTM的一层MLP理解成卷积层，使用大小为1的window size跑出的结果，竟然比
使用MLP效果更好，目前不确定是否是偶然因素，会进一步进行实验，寻找原因。**
+ 复旦中文文本分类语料
![复旦中文文本分类语料](https://github.com/DrLiLiang/ChineseTextClassification/blob/master/picture/FuDanResults.png)

**最后两行带\*号的为[Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9745)。
这篇论文中作者使用CNN和RCNN得到结果（作者使用了预训练的词向量）**

