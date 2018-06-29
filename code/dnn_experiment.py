import tensorflow as tf
import pandas as pd

train_path=r'data/IRIS/iris_training.csv'
test_path=r'data/IRIS/iris_test.csv'
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

# 最简单的模型，features使用矩阵作为输入
def model(features,labels,mode):
    input_layer=features
    dense1=tf.layers.dense(inputs=input_layer,units=128,activation=tf.nn.softmax)
    output_layer=tf.layers.dense(inputs=dense1,units=3)

    predictions=tf.argmax(output_layer,1)
    accuracy=tf.metrics.accuracy(labels=labels,predictions=predictions)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=output_layer)

    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops={"accuracy":accuracy})
    
    if mode==tf.estimator.ModeKeys.TRAIN:
        train_op=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss,tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    
def parse(file_path):
    tmp1=pd.read_csv(file_path,header=0,names=CSV_COLUMN_NAMES)
    features,labels=tmp1,tmp1.pop("Species")
    return features.values,labels.values

def main():
    train_x,train_y=parse(train_path)
    test_x,test_y=parse(test_path)
    
    classifier=tf.estimator.Estimator(model_fn=model,model_dir="tmp")

    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x=train_x,
        y=train_y,
        batch_size=120,
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn,steps=5000)

    test_input_fn=tf.estimator.inputs.numpy_input_fn(
        x=test_x,
        y=test_y,
        num_epochs=1,
        shuffle=False) 
    result=classifier.evaluate(test_input_fn)
    print(result)

if __name__ == '__main__':
    main()