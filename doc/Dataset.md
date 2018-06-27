https://tensorflow.google.cn/api_docs/python/tf/data/Dataset

# 方法
1. `__init__`
2. `__iter__`
3. `apply(transformation_func)`在数据集上应用transformation_func
4. `batch(batch_size)`设置batch_size的大小，当数据集的大小不是batch_size的整数倍时，最后一个batch的大小为剩下的元素数量。
5. `cache(filename='')`将data中的元素写到文件中
6. `concatenate(dataset)`将两个Dataset串联
7. `filter(predicate)`根据predicate函数决定保存dataset中的哪些值
8. `flat_map(map_func)`

等