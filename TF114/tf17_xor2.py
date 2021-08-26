#인공지능의 겨울을 극복하자
#preceptron = > mlp

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[0,0],[0,1],[1,0],[1,1]] # (4,2)
y_data = [[0],[1],[1],[0]]         # (4,1)

x = tf.placeholder(tf.float32, shape=[None,2]) 
y = tf.placeholder(tf.float32, shape=[None,1])

#히든 레이어 1
W = tf.Variable(tf.random.normal([2,10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
layer1 = tf.sigmoid(tf.matmul(x, W) + b)

#인풋 (4,2)가 (2,10)을 만나   (4,10)이 된다 

#얘가 다음 아웃풋에 들어가면 되기때문에 아래의 hypothesis을 넣어주는 거다



# 아웃풋레이어
W2 = tf.Variable(tf.random.normal([10,1]), name='weight')
b2 = tf.Variable(tf.random.normal([1]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)


# cost = tf.reduce_mean(tf.square(y-hypothesis))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crossentropy 

'''
reduce mean은 전체 원소의 합 나누기 갯수 
reduce sum 은 그냥 전체 원소들의 더하기

'''



# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)

sess.close()