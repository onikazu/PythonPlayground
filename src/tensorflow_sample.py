"""
http://tensorflow.classcat.com/2016/03/09/tensorflow-cc-mnist-for-ml-beginners/
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("temp/data", one_hot=True)
sess = tf.Session()

# モデルの作成
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解情報
y_ = tf.placeholder(tf.float32, [None, 10])

# 損失関数
cross_entropy = tf.reduce_sum(y_ * tf.log(y))
# training operationの定義
# 損失関数の最小化を勾配降下法、学習率0.01で行うことを定義している
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# セッションの初期化
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer
sess.run(init)

# 訓練
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # feed_dictでplaceholderに代入可能
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 評価
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))