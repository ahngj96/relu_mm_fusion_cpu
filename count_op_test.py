import tensorflow as tf
from tensorflow.python.framework import load_library

input_datas = [0.5, 0.9, 1.5, 0.7, 2.0, 3.5]
t1 = tf.constant( value=input_datas , dtype="float", shape=[1, len(input_datas) ] )

sess = tf.Session()
print( "\n\n\n\n" )
print( "[PYTHON] INPUT DATAS ", input_datas )
print( "[PYTHON Graph] t1", t1 )

my_log_op = load_library.load_op_library('./custom_op/count_op.so')

size = my_log_op.tensor_count(t1)
c_result = sess.run( size )
print( "[PYTHON Result]tensorflow computed", c_result )


