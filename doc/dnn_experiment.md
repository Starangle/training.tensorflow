# 1.dnn识别IRIS数据集
这个dnn仅有一个128节点、以softmax为激活函数的隐层，使用高阶api构建这个模型并进行训练。

## Estimator API

### 创建Estimator对象
Estimator是tensorflow的高阶api，可以使我们将注意力都集中在模型构建和训练上。

具体地，使用Estimator对象可以自动地进行训练的保存和恢复，在保持网络结构不变的情况下，可以中途修改超参数，然后重新训练。

要初始化一个Estimator对象，仅需要一个模型函数，该模型函数需要具有如下签名`def model(features,labels,mode [,params])`。另外一个Estimator还可以接受一个文件夹作为参数，tensorflow会将模型文件和参数文件都保存在该文件夹下，如果不指定文件夹，tensorflow会自己选择一个安全的文件夹，将模型和参数保存在其中。

```python
classifier=tf.estimator.Estimator(model_fn=model,model_dir="tmp")
```
上面的语句生成了一个Estimator对象，使用铭文model的函数作为模型,使用tmp目录作为存储模型和参数的目录。

### 使用Estimator对象进行训练
Estimator对象的有train可以进行训练，train方法接受一个input_fn来输入数据，train的steps参数可以指定训练多少轮。  
一个具体的示例如下：
~~~python
classifier.train(input_fn=train_input_fn,steps=5000)
~~~

### 使用Estimator对象进行评估
评估与训练类似，都需要输入数据，但评估显然只需要计算一次，因此没有steps参数。  
一个具体的示例如下：
~~~python
result=classifier.evaluate(test_input_fn)
~~~

### 函数原型的约定
为了能够使用Estimator，各个函数都需要满足一定的要求。input_fn指定的函数必须能够返回可迭代的对象，每个item是一个feature、label对。tensorflow.estimator.inputs.numpy_input_fn能够将给定的数据作增殖、排序等操作然后传递给模型。

为了顺利使用Estimator类的train、predict和evaluete，model函数必须返回EstimatorSpec函数。

可以在model通过判断mode的值来判断模型处于哪种状态，train对应的是tensorflow.estimator.ModeKeys.TRAIN、predict对应的是tensorflow.estimator.ModeKeys.PREIDICT以及evaluate对应的是tensorflow.estimator.ModeKeys.EVAL。

predict模型下，EstimatorSpec函数必须有predictions参数；evaluate模式下，必须有loss和eval_metric_ops参数；train模式下必须有loss和train_op参数。

按上面的要求编写的代码理论上就可以正确运行了。本文对应的代码为`code\dnn_experiment.py`。

##  训练结果
运行`code\dnn_experiment.py`得到的训练结果如下：
~~~
{'accuracy': 0.96666664, 'loss': 0.20878907, 'global_step': 5000}
~~~