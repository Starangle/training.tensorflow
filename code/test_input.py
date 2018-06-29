import tensorflow as tf
import pandas as pd

train_path = r'data/IRIS/iris_training.csv'
test_path = r'data/IRIS/iris_test.csv'
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

# 最简单的模型，features使用矩阵作为输入


def model(features, labels, mode):
    input_layer = features
    dense1 = tf.layers.dense(
        inputs=input_layer, units=128, activation=tf.nn.softmax)
    output_layer = tf.layers.dense(inputs=dense1, units=3)

    predictions = tf.argmax(output_layer, 1)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=output_layer)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={"accuracy": accuracy})

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=0.1).minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_input_fn():
    tmp1 = pd.read_csv(train_path, header=0, names=CSV_COLUMN_NAMES)
    features, labels = tmp1, tmp1.pop("Species")
    return (tf.constant(features.values, dtype=tf.float32),
               tf.constant(labels.values, dtype=tf.int32))


def test_input_fn():
    tmp1 = pd.read_csv(test_path, header=0, names=CSV_COLUMN_NAMES)
    features, labels = tmp1, tmp1.pop("Species")
    return (tf.constant(features.values, dtype=tf.float32),
               tf.constant(labels.values, dtype=tf.int32))


def main():

    classifier = tf.estimator.Estimator(model_fn=model, model_dir="tmp")
    classifier.train(input_fn=train_input_fn, steps=5000)
    result = classifier.evaluate(input_fn=test_input_fn,steps=1)
    print(result)


if __name__ == '__main__':
    main()
