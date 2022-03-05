# 一、项目介绍

## 1. 选择（[经典MNIST数据集](https://aistudio.baidu.com/aistudio/datasetdetail/65)）数据集

## 2. 项目背景/意义：
MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。   
该数据集的收集目的是希望通过算法，实现对手写数字的识别。

## 3. 项目拟使用的方法：
使用经典Lenet模型学习深度学习框架搭建，了解卷积层、激活层、池化层、全连接层的用法和意义，主要采用paddle.nn库完成模型。

# 二、数据处理（代码及运行效果参见[data_process.ipynb](./data_process.ipynb)）

## 1. 数据集清洗与介绍：挂载MNIST数据集，gzip解压数据集，tree/ls命令查看目录结构

## 2. 数据集类的定义：自定义MyMNISTDataset类继承paddle.io.Dataset，并实现__init__, __getitem__，__len__方法，decode_idx3_ubyte和decode_idx1_ubyte方法分别完成图片和标签文件的解码，并返回numpy格式的图片数据和标签数据

## 3. 数据集类的测试：通过MyMNISTDataset类构造训练集和测试集，分别获得样本数据和标签数据，测试get、len方法，并用dataloader封装后测试batch效果

## 4. 图像/文本数据的统计分析：对训练集和测试集的第1个样本进行可视化展示，对训练集和测试集的全量样本，分别进行图片数据整体均值和方差计算

# 三、选择模型并完成训练

使用经典Lenet模型，完成两个版本：

第一个版本使用自定义数据集类（gzip解压 + idx文件解析 + expand_dims维度调整），未对数据进行图像归一化处理，测试集准确率为97.91%（代码及运行效果参见[customed_mnist.ipynb](./customed_mnist.ipynb)）

第二个版本使用paddle内置数据集类，并对数据进行图像归一化处理，测试集准确率为98.93%（代码及运行效果参见[internal_mnist.ipynb](./internal_mnist.ipynb)）

# 四、个人总结

零基础入门MNIST手写数字识别，非常适合作为CV领域的入门项目。搭积木的过程比结果更重要，实践才是将知识转化为技能的重要途径。

aistudio链接：https://aistudio.baidu.com/aistudio/projectdetail/3529271

github链接：

gitee链接：
