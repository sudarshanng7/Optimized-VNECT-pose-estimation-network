'''
Execute this file to convert the caffe weights to checkpoint files and pickle file.
The original network is placed in the folder './models/caffe_model' and running this file saves checkpoint files in './models/tf_model'
These weights are directly loaded to initialize the network
'''

import os
import tensorflow as tf
from src.caffe_to_pickle import caffe_to_pickle
from src.init_vnect import VNect


def tf_weights(pklfile, tf_path, model):
    if not tf.gfile.Exists(tf_path):
        tf.gfile.MakeDirs(tf_path)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        model.load_weights(sess, pklfile)
        saver.save(sess, os.path.join(tf_path, 'vnect_tf'))

# Caffe weights and configuration
caffe_weights = './models/caffe_model'
caffe_prototxt = 'vnect_net.prototxt'
caffe_caffemodel = 'vnect_model.caffemodel'


pkl_name = 'weights.pkl'
pkl_file = os.path.join(caffe_weights, pkl_name)

# Path where the checkpoint files will be saved
tf_save_path = './models/tf_model'

# Conversion to pickle and checkpopint files
if not os.path.exists(pkl_file):
    caffe_to_pickle(caffe_weights, caffe_prototxt, caffe_caffemodel, pkl_name)

model = VNect()
tf_weights(pkl_file, tf_save_path, model)