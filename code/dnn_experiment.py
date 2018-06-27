import tensorflow as tf
import pandas as pd

train_path=r'data/IRIS/iris_training.csv'
test_path=r'data/IRIS/iris_test.csv'
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']


def dnn_model():
    pass

def line_parse():
    pass

def get_train_data():
    data=pd.read_csv(train_path,names=COLUMNS,header=0)
    return data,data.pop('label')

def get_test_data():
    data=pd.read_csv(test_path,names=COLUMNS,header=0)
    return data,data.pop('label')

def train_input_fn(features,labels,batch_size):
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    dataset=dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def test_input_fn(features,labels,batch_size):
    dataset=tf.data.Dataset.from_tensor_slices(dict(features))
    dataset=dataset.batch(batch_size)
    return dataset

def main():
    print(get_train_data())

if __name__ == '__main__':
    main()