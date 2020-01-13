# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

app = Flask(__name__)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

# Call the saved trained model
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# session object created
sess = tf.Session()
sess.run(model)

# apply saved model to session
save_path = "./model/saved.cpkt"
saver.restore(sess, save_path)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # receive parameters
        block_time = float(request.form['block_time'])
        block_size = float(request.form['block_size'])
        block_count = float(request.form['block_count'])
        uncle_count = float(request.form['uncle_count'])

        # declare price variable
        price = 0

        # convert arguments to an array
        data = ((block_time, block_size, block_count, uncle_count), (0, 0, 0, 0))
        arr = np.array(data, dtype=np.float32)

        # get the prediction based on input
        x_data = arr[0:4]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
            
        # save the prediction
        price = dict[0]

        return render_template('index.html', price=price)

if __name__ == '__main__':
   app.run(debug = True)
