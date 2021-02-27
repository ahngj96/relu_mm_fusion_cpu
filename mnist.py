import tensorflow as tf
from tensorflow.python.framework import load_library
import time

from tensorflow.examples.tutorials.mnist import input_data
my_mat_op = load_library.load_op_library('./custom_op/my_matmul.so')
my_mat_op2 = load_library.load_op_library('./custom_op/my_matmul2.so')

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

sess = tf.Session()
with tf.device("/cpu:0"):
  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, 10])
  
  W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))

  L1 = tf.nn.relu(tf.matmul(X, W1))
  L1_1 = tf.nn.relu(my_mat_op.tensor_mat_mul(X,W1))
  L1_2 = my_mat_op2.tensor_mat_mul2(X,W1)
  
  
  W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))

  L2 = tf.nn.relu(tf.matmul(L1, W2))
  L2_1 = tf.nn.relu(my_mat_op.tensor_mat_mul(L1_1,W2))
  #L2_2 = my_mat_op2.tensor_mat_mul2(L1_2,W2)
  
  #L2_1 = my_mat_op.tensor_mat_mul(L1_1,W2)
  
  W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

  model = tf.matmul(L2, W3)
  model_1 = tf.matmul(L2_1,W3)
  #model_2 = tf.matmul(L2_2,W3)
  print(model_1.shape)
  
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
  cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_1, labels=Y))
  #cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_2, labels=Y))
  optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
  #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
  
  init = tf.global_variables_initializer()

#graph = tf.get_default_graph()
#for op in graph.get_operations():
#    print ("%s %s %s"%(op.name, op.type, op.device))
#    print (op.name)
#    for i in op.inputs:
#        print('\t',i)



sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

print('learn start')
for epoch in range(20):
    total_cost = 0

    for i in range(total_batch):
    #for i in range(10):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼 학습할 데이터를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        #cost_val = sess.run([cost], feed_dict={X: batch_xs, Y: batch_ys})
        #cost_val = sess.run([cost_1], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    #print('Epoch:', '%04d' % (epoch + 1),
    #      'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('finish!')

is_correct = tf.equal(tf.argmax(model_1, 1), tf.argmax(Y, 1))
#is_correct2 = tf.equal(tf.argmax(model_2, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#accuracy1 = tf.reduce_mean(tf.cast(is_correct2, tf.float32))

#start_time = time.time()
#print('acc:', sess.run(accuracy1,
#                        feed_dict={X: mnist.test.images,
#                                   Y: mnist.test.labels}))
#end_time = time.time() - start_time
#print(end_time)

start_time = time.time()
print('acc:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels}))
end_time = time.time() - start_time
print(end_time)
