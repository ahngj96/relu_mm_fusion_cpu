import tensorflow as tf
from tensorflow.python.framework import load_library

t1 = tf.Variable(tf.random_normal([22,24]), dtype="float")
t2 = tf.Variable(tf.random_normal([24,25]), dtype="float")

sess = tf.Session()
sess.run( tf.global_variables_initializer() )

c = tf.matmul(t1,t2)
c = tf.nn.relu(c)
result = sess.run( c )
print( "[PYTHON Result] tensorflow computed")
print(result)
######################################################################
my_mat_op = load_library.load_op_library('./custom_op/my_matmul.so')
my_mat_op2 = load_library.load_op_library('./custom_op/my_matmul2.so')

c2 = my_mat_op2.tensor_mat_mul2(t1,t2)
result2 = sess.run( c2 )
print( "[PYTHON Result] my matmul" )

print(result)

