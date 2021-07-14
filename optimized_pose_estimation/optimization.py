'''
The checkpoint files are taken as the input and optimizations are run on the network to get alightweight network
'''

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
    saver = tf.train.import_meta_graph("./models/tf_model/vnect_tf.meta")
    saver.restore(sess, "./models/tf_model/vnect_tf")
    your_outputs = ["split_2"] # This is the final output layer of the network
    graph_def = tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        output_node_names=your_outputs)
    
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=your_outputs,
        max_batch_size=1,
        max_workspace_size_bytes=3200000000,
        precision_mode="FP32", # precision, can be "FP32" (32 floating point precision), "FP16" or "INT8" for 
        minimum_segment_size=3,
        is_dynamic_op=True)

with gfile.FastGFile("./final_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")