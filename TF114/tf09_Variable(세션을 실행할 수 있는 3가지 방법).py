import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
#<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess =tf.Session()
sess.run(tf.global_variables_initializer())
aaa =sess.run(W)
print("aaa:", aaa) #aaa: [2.2086694]
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()  #변수쩜 이발
print("bbb:", bbb) #bbb: [2.2086694]
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc:",ccc)
sess.close()