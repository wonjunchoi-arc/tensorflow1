'''
텐서 1에서는 x에 대해 중간중간 들어가는 weight의 모양도 전부 조져줘야한다!!!
이때 앞의 행렬의 열과, 뒤의 행렬의 행이 같아야 곱할 수 있다.

즉 axb 행렬과 mxn 행렬이 있을때, 이 두 행렬을 곱하려면 b와 m이 같아야 한다.

그리고 이 두 행렬을 곱하면 axn 사이즈의 행렬이 나온다.
'''

import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73,51, 65],
            [92,98,11],
            [89,31,33],           # (5,3)
            [99,33,100],
            [17,66,79]]

y_data = [[152],[185],[180],[205],[142]]  #(5,1)

x= tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y= tf.compat.v1.placeholder(tf.float32, shape=[None,1])

W =tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# hypothesis = x * W +b # 행렬의 연산에는 일케 안한다

hypothesis = tf.matmul(x,W) +b  
#텐서 1에서 연산할때는 x 와 W의 순서도 중요하다 그 이유는 행렬 연산의 방법 때문이다. 

#  하단은 점심때 완성띠 하자!!1

cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y)) #mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001) : 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict={x:x_data, y:y_data})

    if epochs % 20 == 0 :
        print(epochs, "cost:", cost_val, "\n", hy_val)

sess.close()