https://tensorflow.google.cn/api_docs/python/tf/data/TextLineDataset

一个包含一个或者多个文本文件的Dataset。

# 方法
1. `__init__(filenames,compression_type=None,buffer_size=None)`  
    filenames:一个包含一个或者多个文件名的tensor；  
    compression_type:""或者"ZLIB"或者"GZIP"；  
    buffer_size:可选参数，默认为0。
2. `__iter__()`