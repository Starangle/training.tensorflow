# 我的tensorflow学习过程（基于api1.8）

## 前言
很长时间没有用tensorflow了，api和模型变化都有些大，需要重新学习。

## 数据准备
工欲善其事，必先利其器，没有数据是无法做机器学习的，因此需要先下载一些标准的数据集，在这些数据集上再进行实验，目前用到如下数据集：
1. IRIS数据集，见data/IRIS

## 模型构建
在之前的版本中都是使用低阶api构图，然后使用会话来控制训练和测试。现在tensorflow提供了更高级的api来控制构图和训练。

### 数据读取
tf.data 模块提供一系列类和函数，可用于轻松从各种来源读取数据。

tf.data 包含下面五个类：
1. [Dataset](doc/Dataset.md)
2. FixedLengthRecordDataset
3. Iterator
4. TFRecordDataset
5. [TextLineDataset](doc/TextLineDataset.md)

详细的用法参见每个类的链接。

### 模型编写
tensorflow提供了一些高阶的api来构建网络，tf.layers中有一些可以直接用来构建layer的函数，常使用下列的函数：
1. tf.layers.dense可以用来表示全连接层。

从下面的实例可以看到这些函数的用法
1. dnn_experiment  
    [代码](code/dnn_experiment.py)  
    [说明](doc/dnn_experiment.md)