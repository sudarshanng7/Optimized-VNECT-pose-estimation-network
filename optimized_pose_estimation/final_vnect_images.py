####################### For a stream of Images ############################

# Import the necessary libraries
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
import numpy as np
import time
import cv2
import math
import argparse
import os
import natsort
from src.image_processing import read_square_image, read_pb_graph

parser = argparse.ArgumentParser()
parser.add_argument('--demo_type', default='image')
parser.add_argument('--device', default='gpu')
parser.add_argument('--test_img', default='test_imgs/test_pic1.jpg')
parser.add_argument('--input_size', default=368)
parser.add_argument('--num_of_joints', default=21)
parser.add_argument('--pool_scale', default=8)
parser.add_argument('--plot_2d', default=True)
parser.add_argument('--plot_3d', default=True)
args = parser.parse_args()



#Global Parameters
hm_factor = 8
limb_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
limbs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17, 18]
scales = [1.0, 0.7]
joints_2d = np.zeros(shape=(21, 2), dtype=np.int32)
gpu_count = {'GPU':1} if args.device == 'gpu' else {'GPU':0}
joints_2d_predicted_all = []
total_fps = []


TENSORRT_MODEL_PATH = './models/optim/TensorRT_model32.pb'


sess_config = tf.ConfigProto(device_count=gpu_count, log_device_placement=False,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=sess_config) as sess:
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)
        tf.import_graph_def(trt_graph, name='')
        
        input_batch = sess.graph.get_tensor_by_name('Placeholder:0')
        heatmap = sess.graph.get_tensor_by_name('split_2:0')
        x_heatmap = sess.graph.get_tensor_by_name('split_2:1')
        y_heatmap = sess.graph.get_tensor_by_name('split_2:2')
        z_heatmap = sess.graph.get_tensor_by_name('split_2:3')

        image_format = '.jpg'
        path = './test_imgs/'
        image_file_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(image_format)]
        image_file_array = np.asarray(image_file_list)
        image_file_array = (natsort.natsorted(image_file_array))        
        
        for i in  image_file_array:
            frame = cv2.imread(i)
            cam_img, scaler, [offset_x, offset_y] = read_square_image(frame, 368)
    
    
            input_batch1 = []   
      
            orig_size_input = cam_img.astype(np.float32)
            box_size = [orig_size_input.shape[0],orig_size_input.shape[1]]
            resize_factor = 1
        
            for scale in scales:
                resized_img = cv2.resize(orig_size_input ,(0,0), fx = scale, fy = scale ,interpolation=cv2.INTER_LINEAR )
                pad_h = (box_size[0] - resized_img.shape[0]) // 2
                pad_w = (box_size[1] - resized_img.shape[1]) // 2
                pad_h_offset = (box_size[0] - resized_img.shape[0]) % 2
                pad_w_offset = (box_size[1] - resized_img.shape[1]) % 2
                resized_pad_img = np.pad(resized_img, ((pad_w, pad_w+pad_w_offset), (pad_h, pad_h+pad_h_offset), (0, 0)),
                                          mode='constant', constant_values=0)
    
                
                input_batch1.append(resized_pad_img)
            input_batch1 = np.asarray(input_batch1, dtype=np.float32)
            input_batch1 /= 255.0
            input_batch1 -= 0.4
            
            t0 = time.time() 
            [hm, x_hm, y_hm, z_hm] = sess.run([heatmap, x_heatmap, y_heatmap, z_heatmap], feed_dict={input_batch: input_batch1}) 
    
            hm_size = args.input_size // args.pool_scale
            hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
            x_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
            y_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
            z_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    
            for i in range(len(scales)):
                rescale = 1.0 / scales[i]
                scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
                scaled_x_hm = cv2.resize(x_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
                scaled_y_hm = cv2.resize(y_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
                scaled_z_hm = cv2.resize(z_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
    
                mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
                hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2, mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
                x_hm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2, mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
                y_hm_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2, mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
                z_hm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2, mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            hm_avg /= len(scales)
            x_hm_avg /= len(scales)
            y_hm_avg /= len(scales)
            z_hm_avg /= len(scales)
    
        	
            joints_2d = np.zeros((hm_avg.shape[2], 2))
            for joint_num in limbs:
            	heatmap_resized = cv2.resize(hm_avg[:, :, joint_num], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
            	joint_coord = np.unravel_index(np.argmax(heatmap_resized), (box_size[0], box_size[1]))
            	joints_2d[joint_num, :] = joint_coord
    
            total_fps.append(1 / (time.time() - t0))
            print('The network can process {:>2.2f} frames per second'.format(1 / (time.time() - t0)))
            joints_2d[:, 0] = ((joints_2d[:, 0] - offset_y) / scaler)
            joints_2d[:, 1] = ((joints_2d[:, 1] - offset_x) / scaler)
            joints_2d_predicted = joints_2d
            joints_2d_predicted_all.append(joints_2d_predicted)
            joint_map_predicted = np.zeros(shape=(frame.shape[0], frame.shape[1],3))
            for joint_num in range(joints_2d.shape[0]):
            	cv2.circle(joint_map_predicted, center=(int(joints_2d_predicted[joint_num][1]), int(joints_2d_predicted[joint_num][0])), radius=3,
                            color=(0, 0, 255), thickness=-1)
            
            
            img_limb_predicted = frame
            for limb_num in limbs:    
                x1 = (joints_2d_predicted[limb_num, 0])
                y1 = (joints_2d_predicted[limb_num, 1])
                x2 = (joints_2d_predicted[limb_parents[limb_num], 0])
                y2 = (joints_2d_predicted[limb_parents[limb_num], 1])
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)), (int(length / 2), 3), int(deg),0, 360, 1)
                cv2.fillConvexPoly(img_limb_predicted, polygon, color=(0,255,0))
            
            
            concat_img = np.concatenate((img_limb_predicted, joint_map_predicted), axis=1)
            cv2.imshow('2D img', concat_img.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
print('Average FPS: ', np.mean(total_fps[1:]))

#
#def img_scale_padding():
#    resized_img = cv2.resize(orig_size_input ,(0,0), fx = scale, fy = scale ,interpolation=cv2.INTER_LINEAR )
#    pad_h = (box_size[0] - resized_img.shape[0]) // 2
#    pad_w = (box_size[1] - resized_img.shape[1]) // 2
#    pad_h_offset = (box_size[0] - resized_img.shape[0]) % 2
#    pad_w_offset = (box_size[1] - resized_img.shape[1]) % 2
#    resized_pad_img = np.pad(resized_img, ((pad_w, pad_w+pad_w_offset), (pad_h, pad_h+pad_h_offset), (0, 0)),
#                              mode='constant', constant_values=0)    
#    return resized_pad_img
#
#def img_padding(img, box_size, color='black'):
#    h, w = img.shape[:2]
#    offset_x, offset_y = 0, 0
#    if color == 'black':
#        pad_color = [0, 0, 0]
#    elif color == 'grey':
#        pad_color = [128, 128, 128]
#    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
#    if h > w:
#        offset_x = box_size // 2 - w // 2
#        img_padded[:, offset_x: box_size // 2 + int(np.ceil(w / 2)), :] = img
#    else:
#        offset_y = box_size // 2 - h // 2
#        img_padded[offset_y: box_size // 2 + int(np.ceil(h / 2)), :, :] = img
#
#    return img_padded, [offset_x, offset_y]
#
#
#def read_square_image(img, box_size):
#    h, w = img.shape[:2]
#    scaler = box_size / max(h, w)
#    img_scaled = cv2.resize(img, (0, 0), fx=scaler, fy=scaler, interpolation=cv2.INTER_LINEAR)
#    img_padded, [offset_x, offset_y] = img_padding(img_scaled, box_size)
#    return img_padded, scaler, [offset_x, offset_y]
#
#
#def read_pb_graph(model):
#    with gfile.FastGFile(model,'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#    return graph_def