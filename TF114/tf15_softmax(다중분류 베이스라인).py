from re import X
import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
[2,1,3,2],
[3,1,3,4],
[4,1,5,5],
[1,7,5,5],                           #8 ,4
[1,2,5,6],
[1,6,6,6],
[1,7,6,7]]
y_data = [[0,0,1],
[0,0,1],
[0,0,1],
[0,1,0],
[0,1,0],                           # 8, 3
[0,1,0],
[1,0,0],
[1,0,0]]

x = tf.placeholder(tf.float32, shape=[None,4]) 
y = tf.placeholder(tf.float32, shape=[None,3])

W = tf.Variable(tf.random.normal([4,3]), name='weight')
b = tf.Variable(tf.random.normal([1,3]), name='bias')

# hypothesis =tf.matmul(x ,W)+b 이렇게 하면 값이 1이 넘을수동 있다. 
# 
hypothesis =tf.nn.softmax(tf.matmul(x ,W)+b)

#categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

optimizers = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val =sess.run([optimizers,loss], feed_dict={x:x_data, y: y_data})
        if step % 200 ==0:
            print(step, cost_val)

    #predict
    results = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(results, sess.run(tf.argmax(results, 1)))