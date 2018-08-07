import tensorflow as tf
import numpy as np
from python_dealdata import anchor_target_layer

a=tf.constant([[1,4,2,3],[3,4,2,4],[5,4,4,3]])
a1=tf.constant([[1,4,2,3]])
a3=tf.transpose(a, perm=[1, 0])
b=tf.constant([[4,3,2,1],[1,2,1,3],[4,5,6,3],[1,2,3,4]])
b2= tf.constant(1)
tf.global_variables_initializer()
c=tf.matmul(a,b)
final=tf.matmul(c,a3)
c1=tf.matmul(a1,b)
sess=tf.Session()
print(c)
print(sess.run(final))

# print(sess.run(c1))
temp=tf.py_func(anchor_target_layer,[final],[tf.int32])
temp_2=tf.add(temp,b2)
print("5555555555555555555")
print(sess.run(temp))
print(sess.run(temp_2))
print("55555555555555555555555555")
print(sess.run(final[0][0]))
print(sess.run(final[1][1]))

# print("55555555555555555555555555555555")
# print(sess.run(tf.diag(final)))