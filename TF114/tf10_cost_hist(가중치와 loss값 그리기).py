import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.,]
y = [2.,4.,6.,]

W = tf.placeholder(tf.float32)

hypothsis = x*W


cost =  tf.reduce_mean(tf.square(hypothsis - y))

# sess = tf.compat.v1.Session()
w_history = []
cost_history = []

with tf.compat.v1.Session() as sess : 
    for i in range(-30, 50) : 
        curr_w = i
        curr_cost = sess.run(cost, feed_dict={W:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print('========= W history ======================================================')
print(w_history)
print('========= cost_history ======================================================')
print(cost_history)


plt.plot(w_history,cost_history)
plt.xlabel('Weight')
plt.ylabel('loss')
plt.show()