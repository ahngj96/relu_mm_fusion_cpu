import tensorflow as tf
from tensorflow.python.framework import load_library

t1 = tf.Variable(tf.random_normal([1,3]), dtype="float")

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
print( "\n\n\n\n" )
print( "[PYTHON Graph] t1", t1 )

t1_result = sess.run( t1 )
print( "[PYTHON Result] t1", t1_result )



######################################################################
print( "[PYTHON]RUN MY LOG OP\n" )
my_vector_add = load_library.load_op_library('./custom_op/vector_add_gpu.so')
scala = tf.constant( value=[10.0] , dtype="float", shape=[1,1] )


output = my_vector_add.tensor_vector_add( t1, scala )
print( "[PYTHON Graph] output ", output )

c_result = sess.run( output )
print( "[PYTHON Result]tensorflow computed", c_result )


