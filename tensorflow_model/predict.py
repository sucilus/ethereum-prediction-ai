import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()

block_time = float(input('block_time: '))
block_size = float(input('block_size: '))
block_count = float(input('block_count: '))
uncle_count = float(input('uncle_count: '))

with tf.Session() as sess:
sess.run(model)
save_path = "./saved.cpkt"
saver.restore(sess, save_path)

data = ((block_time, block_size, block_count, uncle_count), (0, 0, 0, 0))
arr = np.array(data, dtype=np.float32)

x_data = arr[0:4]
dict = sess.run(hypothesis, feed_dict={X: x_data})
print(dict[0])

