import tensorflow as tf

node1 =tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 =tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

'''
placeholder은 안이 비어있는 친구다!!
그곳에 feed_dict라는 친구를 통해 실제값을 전달해서 넣음

'''

adder_node = a +b 

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))
