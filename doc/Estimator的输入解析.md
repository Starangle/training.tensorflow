 `tf.estimator.inputs`包含两个函数`numpy_input_fn`和`pandas_input_fn`。

 ## numpy_input_fn
 函数原型如下：
 ~~~python
 tf.estimator.inputs.numpy_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1
)
~~~
`x`:numpy数组或者字典，字典的内容是numpy数组；
`y`:numpy数组或者字典，字典的内容是numpy数组；
`batch_size`:返回的batch大小
`num_epochs`:在数据集上迭代多少轮，如果是None将一直迭代下去
`shuffle`:是否打乱队列
`queue_capacity`:队列中条目数量的多少
`num_threads`:用多少个线程来读取这个队列
该函数的主要功能是对原来的数组做一些修正，最终返回features,labels

## pandas_input_fn
函数原型如下：
~~~python
tf.estimator.inputs.pandas_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1,
    target_column='target'
)
~~~
`x`:pandas的DataFrame对象；
`y`:pandas的Series对象，缺省为None；
`batch_size`:返回的batch的大小
`num_epochs`:在整个数据集上迭代多少遍，如果不是None，将会在读取超过这个数量的时候发生OutOfRangeError
`shuffle`:是否按照随机的顺序读取变量
`queue_capacity`:队列的容纳能力
`num_threads`:操作这个队列的线程数
`target_column`:给y设定的名字
返回函数的签名为`()->(features,target)`,其中features是dict


