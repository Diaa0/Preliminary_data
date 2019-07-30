import glob
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
###
#display data
###load the graph


DATA_NORMAL_PATH = os.path.join('.', 'data_dir', 'normal')
DATA_ABNORMAL_PATH = os.path.join('.', 'data_dir', 'abnormal')

#####

normal_files_list = glob.glob(os.path.join(DATA_NORMAL_PATH,'*.dat'))
abnormal_files_list = glob.glob(os.path.join(DATA_ABNORMAL_PATH,'*.dat'))

normal_data = []
for f in normal_files_list:
  f_np = np.loadtxt(f)
  data = f_np[:,1]
  normal_data_val.append(data[np.newaxis,:])
normal_data = np.concatenate(normal_data, axis=0)

abnormal_data_val=[] 
for f in abnormal_files_list:
  f_np = np.loadtxt(f)
  data = f_np[:,1]
  abnormal_data_val.append(data[np.newaxis,:])
abnormal_data = np.concatenate(abnormal_data, axis=0)

plt.plot(normal_data[1,:])

######


graph = tf.Graph()
graph_def = tf.GraphDef()
model_file = 'af_model.pb'
with open(model_file, "rb") as f:
  graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)


input_node = "import/input_placehoder"
output_node = "import/linear/soft_pred"


input_operation = graph.get_operation_by_name(input_node)
output_operation = graph.get_operation_by_name(output_node)


normal_data = normal_data[:, :180, np.newaxis]
abnormal_data = abnormal_data[:,:180, np.newaxis]

with tf.Session(graph=graph) as sess:
   preds_np = sess.run(output_operation.outputs[0], feed_dict={input_operation.outputs[0]: abnormal_data})
   print(preds_np)   




