import tensorflow as tf
import numpy as np
import copy
#from tensorflow.contrib.compiler import xla
#from tensorflow.python.client import timeline


# Time function #############################################################################################################

def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))
  else:
    return 'images/sec: %.1f' % speed_mean



#tf.config.optimizer.set_jit(True)

#sess = tf.Session()
batch_size = 1024
input_row = 32
input_col = 32
input_channel = 3

x_data_ = np.array([np.ones((input_row, input_col,input_channel), dtype=np.float32) for i in range(batch_size)] * batch_num)
x_data_ = np.array([np.ones((batch_size,1024), dtype=np.float32)])
#x_data_ = np.random.rand(batch_size, 1024)

#X = tf.placeholder("float",[None, 1024])
dataset = tf.data.Dataset.from_tensor_slices(x_data_).repeat().batch(batch_size)
dataset_y = tf.data.Dataset.from_tensor_slices(y_data).repeat().batch(batch_size)

dataset = dataset.prefetch(batch_size)
dataset_y = dataset_y.prefetch(batch_size)

#dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/GPU:0", 1))

iterator = dataset.make_initializable_iterator()
iterator_y = dataset_y.make_initializable_iterator()

#sess.run(iterator_y.initializer)
#X = iterator.get_next()
#Y = iterator_y.get_next()
with tf.name_scope("fc_layer1_"):
  Wfc1 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer2_"):
  Wfc2 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer3_"):
  Wfc3 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer4_"):
  Wfc4 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer5_"):
  Wfc5 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer6_"):
  Wfc6 = tf.Variable(tf.ones([1024, 1024]))
with tf.name_scope("fc_layer7_"):
  Wfc7 = tf.Variable(tf.ones([1024, 10240]))
with tf.name_scope("fc_layer8_"):
  Wfc8 = tf.Variable(tf.ones([10240, 4096]))
with tf.name_scope("fc_layer9_"):
  Wfc9 = tf.Variable(tf.ones([4096, 4096]))
with tf.name_scope("fc_layer10_"):
  Wfc10 = tf.Variable(tf.ones([4096,4096]))
with tf.name_scope("fc_layer11_"):
  Wfc11 = tf.Variable(tf.ones([4096,1000]))

with tf.name_scope('agj_fc_layer1_'):
  fc1 = tf.matmul(X, Wfc1)
  fc1 = tf.nn.relu(fc1)

with tf.name_scope('agj_fc_layer2_'):
  fc2 = tf.matmul(fc1, Wfc2)
  fc2 = tf.nn.relu(fc2)

with tf.name_scope('agj_fc_layer3_'):
  fc3 = tf.matmul(fc2, Wfc3)
  fc3 = tf.nn.relu(fc3)

with tf.name_scope('agj_fc_layer4_'):
  fc4 = tf.matmul(fc3, Wfc4)
  fc4 = tf.nn.relu(fc4)

with tf.name_scope('agj_fc_layer5_'):
  fc5 = tf.matmul(fc4, Wfc5)
  fc5 = tf.nn.relu(fc5)

with tf.name_scope('agj_fc_layer6_'):
  fc6 = tf.matmul(fc5, Wfc6)
  fc6 = tf.nn.relu(fc6)

with tf.name_scope('agj_fc_layer7_'):
  fc7 = tf.matmul(fc6, Wfc7)
  fc7 = tf.nn.relu(fc7)

with tf.name_scope('agj_fc_layer8_'):
  fc8 = tf.matmul(fc7, Wfc8)
  fc8 = tf.nn.relu(fc8)

with tf.name_scope('agj_fc_layer9_'):
  fc9 = tf.matmul(fc8, Wfc9)
  fc9 = tf.nn.relu(fc9)

with tf.name_scope('agj_fc_layer10_'):
  fc10 = tf.matmul(fc9, Wfc10)
  fc10 = tf.nn.relu(fc10)

with tf.name_scope('agj_fc_layer11_'):
  fc11 = tf.matmul(fc10, Wfc11)
  fc11 = tf.nn.relu(fc11)

with tf.name_scope('cost_layer12_'):
    cost = tf.reduce_mean(fc11)


#input_shape = L3.get_shape().as_list()
#flat_input_size = input_shape[1] * input_shape[2] * input_shape[3]
#print(flat_input_size)
#flat_input = tf.reshape(L3, shape =[-1, flat_input_size])
#flat_input1 = tf.reshape(L33, shape =[-1, flat_input_size])
#with tf.name_scope("fc1_layer4"):
#  W5 = tf.Variable(tf.ones([flat_input_size, 128]))
#with tf.name_scope('agj_fc1_layer4'):
#  fc1 = tf.matmul(flat_input, W5)
#  fc11 = tf.matmul(flat_input1, W5)
#  fc1 = tf.nn.relu(fc1)
#  fc11 = tf.nn.relu(fc11)
##costs = [fc1, fc11]

tvs = tf.trainable_variables()

opt=tf.train.GradientDescentOptimizer(0.0001)
#opt=tf.train.RMSPropOptimizer(0.0001)
#gvs = opt.compute_gradients(cost,tvs)
gvs,wg_inputs = opt.compute_gradients(cost,tvs)

#print(wg_inputs)
#for wg in wg_inputs:
#  print(wg)
#
#list_relu_input = []
#for r_input in relu_inputs:
#  print(r_input)
#  ptr = r_input.find("layer")
#  ptr_end = r_input.find("_", ptr)
#  ln = int(r_input[ptr+5: ptr_end])
#  list_relu_input.append((ln, relu_inputs[r_input]))
#list_relu_input = sorted(list_relu_input, reverse=True)
#for i in list_relu_input:
#    print(i)
#
#
#i = 1
#total_layernum = len(relu_inputs)
#make_input_list = []
#for relu_input in list_relu_input:
#  if(i % 5 == 0):
#    make_input_list.append(relu_input)
#  i+=1
#
#
#wgs = []
#for op in graph.get_operations():
#  ptr1 = op.name.find("MatMul_1")
#  if ptr1 != -1 :
#    wgs.append(op)
#
#print(wgs)
#make_wg_list = []
#i = 1
#temp = []
#for wg in wgs:
#  temp.append(wg)
#  if(i % 5 == 0):
#    make_wg_list.append(temp)
#    temp =[]
#  i+=1
#print(make_wg_list)
#
#print(len(make_wg_list), len(make_input_list))
#
#relu_target = []
#for a in range(len(make_wg_list)):
#  relu_targets = "" 
#
#  temp_ = []
#  for ii in make_wg_list[a]:
#    relu_targets += ii.name
#    print(ii.name)
#    temp_.append(wg_inputs[ii.name]['a'])
#    temp_.append(wg_inputs[ii.name]['b'])
#  with tf.name_scope("Fake" + relu_target[:-2]):
#      tt = tf.fake_relu_grad(b[1]['fwd_output'], b[1]['grad']
#                             ii[0][])
#  relu_target.append(tt)
#  print()
#input("Press Enter to continue")


res = opt.apply_gradients(gvs)
res = [res]# + fake_ops
#writer = tf.summary.FileWriter("./sample_logs", sess.graph)
#res = xla.compile(computation=res, inputs=)

graph = tf.compat.v1.get_default_graph()
init = tf.global_variables_initializer()

#gpu_options = tf.GPUOptions()
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.allow_growth = True
#config.gpu_options.experimental.timestamped_allocator = True 
sess = tf.Session(config=config)

#run_metadata = tf.RunMetadata()
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# maximum across all sessions and .run calls so far
sess.run(tf.contrib.memory_stats.MaxBytesInUse())
# current usage
sess.run(tf.contrib.memory_stats.BytesInUse())

#sess.run(iterator.initializer)
#sess.run(init,run_metadata=run_metadata, options = run_options )
sess.run(init)
#input("sess.run(init) done [Enter]")
#_= sess.run([res], run_metadata=run_metadata, options = run_options)
#_= sess.run([res])
import time

#run_metadata = tf.RunMetadata()
#sess.run(res, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadatam)

step_train_times = [] #
step = 0 #
#writer = tf.summary.FileWriter("./matmul_logs", sess.graph)

for i in range(10):
  s = time.time()
  #_= sess.run([res], run_metadata=run_metadata, options = run_options)
  _= sess.run(res,feed_dict={X : x_data_})
  #sess.run(res, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata, feed_dict={X : x_data_})
  s = time.time()-s
  step_train_times.append(s)#
  log_str = '%i\t%s' % (#
        step + 1, get_perf_timing_str(batch_size, step_train_times))#
  step += 1#
  print(log_str)#

#input("sess.run()*10 times  done [Enter]")
#tl = timeline.Timeline(run_metadata.step_stats)
#ctf = tl.generate_chrome_trace_format()

#with open('timeline1.json', 'w') as f:
#    f.write(ctf)


#with open("1_iter_yesstream_temp.txt", "w") as out:
#  out.write(str(run_metadata))
