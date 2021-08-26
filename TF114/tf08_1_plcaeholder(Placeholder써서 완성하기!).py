
# y =wx + b 
# w , b 변수로 두고  x,y는 입력되는 값이므로 placeholder로

import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [1,2,3]

x_train =tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train =tf.compat.v1.placeholder(tf.float32, shape=[None])


# W = tf.Variable(1, dtype=tf.float32) 
# b = tf.Variable(1, dtype=tf.float32) 


W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 

hypothesis = x_train *W +b
#hypothesis 얘랑 f(x) 즉 y이랑 같은 말임! =f(x) = wx +b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
# ==> 평균(제곱(예측값 -실제값))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

###########################여기까지가 연산할 수 있는 그래프 모양만 만든것이다###################3333

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #위에 선언한 변수때문에

for step in range(2001):
    _, loss_val,W_val, b_val = sess.run([train, loss,W,b], 
    feed_dict={x_train:[1,2,3],y_train:[1,2,3]})
    #실행하는 부분에서 feed를 먹이고 그값이 여러가지들이 실행되는 곳에서 적용
    if step % 20 ==0:
        # print(step, sess.run(loss), sess.run(W),sess.run(b))
        print(step, loss_val,W_val, b_val)

print('끗')
