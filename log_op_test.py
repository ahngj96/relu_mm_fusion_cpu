import tensorflow as tf
from tensorflow.python.framework import load_library

t1 = tf.Variable(tf.random_normal([1,3]), dtype="float")
t2 = tf.Variable(tf.random_normal([1,3]), dtype="float")
t3 = tf.Variable(tf.random_normal([1,3]), dtype="float")

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
print( "\n\n\n\n" )
print( "[PYTHON Graph] t1", t1 )
print( "[PYTHON Graph] t2", t2 )
print( "[PYTHON Graph] t3", t3 )
print( "\n\n")

t1_result = sess.run( t1 )
t2_result = sess.run( t2 )
t3_result = sess.run( t3 )
print( "[PYTHON Result] t1", t1_result )
print( "[PYTHON Result] t2", t2_result )
print( "[PYTHON Result] t3", t3_result )

print( "\n\n")
print( "[PYTHON Result] t1 + t2 : ",t1_result + t2_result )
print( "[PYTHON Result] out1 -t3 : ",(t1_result + t2_result) - t3_result )
print()

out1 = t1 + t2
print( "[PYTHON Graph] out1", out1 )
t4 = out1 - t3
print( "[PYTHON Graph] t4", t4 )
result = sess.run( t4 )
print( "[PYTHON Result] tensorflow computed",result )
print()

######################################################################
print( "[PYTHON]RUN MY LOG OP\n" )
my_log_op = load_library.load_op_library('./custom_op/log_op.so')
c_out1 = t1 + t2
print( "[PYTHON Graph] c_out1 ", c_out1 )

log_tensor = my_log_op.tensor_log(out1)
print( "[PYTHON Graph] log_tensor ", log_tensor )

c_t4 = log_tensor - t3
print( "[PYTHON Graph] log_tensor - t3 ", c_t4 )

c_result = sess.run( c_t4 )
print( "[PYTHON Result]tensorflow computed", c_result )


