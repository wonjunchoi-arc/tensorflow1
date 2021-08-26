import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothsis = x*W + b

#실습
#tf09 1번 방식 3가지로 hypithisis 출력


sess =tf.Session()
sess.run(tf.global_variables_initializer())
aaa =sess.run(hypothsis)
print("hypothsis:", aaa) #aaa: [2.2086694]
sess.close()
# hypothsis: [1.3       1.6       1.9000001]

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# bbb = W.eval()  #변수쩜 이발
# print("bbb:", bbb) #bbb: [2.2086694]
# sess.close()

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# ccc = W.eval(session=hypothsis)
# print("ccc:",hypothsis)
# sess.close()