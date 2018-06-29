# 测试哪些输入可以被tensorflow接受
`code\test_input.py`中的代码展示了`input_fn`可以接受返回元组(features,label)的函数。
根据tensorflow的文档，函数也可以返回含有features、labels的Dataset。

需要注意的一点是当直接返回元组的时候，evaluate也需要设置steps参数，否则评估将无法停止，这是因为其取值默认为None，根据官网的解释，当steps取值为None时，会持续计算直到input_fn抛出OutOfRange错误或者StopIteration异常。而我编写的input_fn不会抛出异常和错误，因此会一直运行下去。