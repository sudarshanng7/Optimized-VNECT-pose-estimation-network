'''
This is the original implementation of the model to check the CPU runtime of VNECT, this implementation will act as our baseline model
Pose is predicted on stream of images
'''


import os
import numpy as np
import cv2
import caffe
import math
import time

first_frame = True
first_output = True

#Global Parameters
hm_factor = 8
limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
scales = [1.0, 0.7]
joints_2d = np.zeros(shape=(21, 2), dtype=np.int32)


# Setting up the caffe model.
caffe.set_mode_cpu()
model_prototxt_path = os.path.join("./CAFFE_MODEL",'vnect_net.prototxt')
model_weight_path = os.path.join("./CAFFE_MODEL", 'vnect_model.caffemodel')
model = caffe.Net(model_prototxt_path, model_weight_path, caffe.TEST)

for i in model.blobs.keys():
        print(i, model.blobs[i].data.shape)

# Video/image input
image_format = '.jpg'
path = './test_imgs'
image_file_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(image_format)]
image_file_array = np.asarray(image_file_list)
image_file_array = np.sort(image_file_array)
d=0


#frame by frame input 
for i in  image_file_array:
    cam_img = cv2.imread(i)
    cam_img = cv2.resize(cam_img,(848,448))
    input_batch = []
    
    
    initial_time = time.time()
    if(first_frame == True):
        orig_size_input = cam_img.astype(np.float32)
        box_size= [orig_size_input.shape[0],orig_size_input.shape[1]]
        resize_factor = 1
        first_frame = False
    else:
        min_cord = np.amin(joints_2d, axis = 0)
        max_cord = np.amax(joints_2d, axis = 0)
        ymin_cord = min_cord[0] -20
        ymax_cord = max_cord[0] +20
        xmin_cord = min_cord[1] -20
        xmax_cord = max_cord[1] +20
        CROP_RECT = [ymin_cord,ymax_cord,xmin_cord,xmax_cord ]
        img_proto = cam_img[ ymin_cord : ymax_cord , xmin_cord:xmax_cord]
        
        if(min(img_proto.shape[0],img_proto.shape[1])) == img_proto.shape[0]:
            pixel_diff = (img_proto.shape[1]-img_proto.shape[0]) /(2*1.0)
            new_ymin_cord = ymin_cord - pixel_diff
            new_ymax_cord = ymax_cord + pixel_diff
            img_cropped = cam_img[new_ymin_cord:new_ymax_cord ,xmin_cord : xmax_cord ]
            final_crop  = [new_ymin_cord , new_ymax_cord, xmin_cord , xmax_cord ]
        
        elif(min(img_proto.shape[0],img_proto.shape[1])) == img_proto.shape[1]:
            pixel_diff = (img_proto.shape[0]-img_proto.shape[1]) /(2*1.0)
            new_xmin_cord = xmin_cord - pixel_diff
            new_xmax_cord = xmax_cord + pixel_diff
            img_cropped = cam_img[ymin_cord : ymax_cord,new_xmin_cord:new_xmax_cord ]
            final_crop  = [ymin_cord , ymax_cord , new_xmin_cord, new_xmax_cord ]

        
        # convert the image into 368*368 dimension
        resize_factor = np.float(368)/np.float(img_cropped.shape[0])
        img_box = cv2.resize(img_cropped,(0,0),fx = resize_factor, fy = resize_factor,interpolation=cv2.INTER_LINEAR)
        orig_size_input = img_box.astype(np.float32)
        box_size= [orig_size_input.shape[0],orig_size_input.shape[1]]

    # image padding (because of different scales)
    for scale in scales:
        resized_img = cv2.resize(orig_size_input ,(0,0), fx = scale, fy = scale ,interpolation=cv2.INTER_LINEAR )
        pad_h = (box_size[0] - resized_img.shape[0]) // 2
        pad_w = (box_size[1] - resized_img.shape[1]) // 2
        pad_h_offset = (box_size[0] - resized_img.shape[0]) % 2
        pad_w_offset = (box_size[1] - resized_img.shape[1]) % 2
        resized_pad_img = np.pad(resized_img, ( (pad_h, pad_h+pad_h_offset),(pad_w, pad_w+pad_w_offset), (0, 0)),
                             			mode='constant', constant_values=128)
        input_batch.append(resized_pad_img)
    input_batch = np.asarray(input_batch, dtype=np.float32)
    input_batch = np.transpose(input_batch, (0, 3, 1, 2))
    input_batch /= 255.0
    input_batch -= 0.4
    
    model.blobs['data'].reshape(2,3,input_batch.shape[2],input_batch.shape[3])
    model.blobs['data'].data[...] = input_batch
    
    t1 = time.time()
    model.forward()
    t2 = time.time()
    print("Time taken for a single forward pass",i,"frame",t2-t1,'secs')
    
    
    hm = model.blobs['heatmap'].data
    hm = hm.transpose([0, 2, 3, 1])
    hm_size = [box_size[0]/hm_factor,box_size[1]/hm_factor]
    hm_avg = np.zeros(shape=(hm_size[0], hm_size[1], 21))
    
    for i in range(len(scales)):
        rescale = 1.0 / scales[i]
        scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
        mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
        hm_avg += scaled_hm[mid[0] - hm_size[0] // 2: mid[0] + hm_size[0] // 2,
                          mid[1] - hm_size[1] // 2: mid[1] + hm_size[1] // 2, :]
        hm_avg /= len(scales)
        
    heatmap_resized = cv2.resize(hm_avg, (box_size[1], box_size[0]))
    
    if(first_output == True):
        for joint_num in range(heatmap_resized.shape[2]):
            joint_coord = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_num]), (box_size[0], box_size[1]))
            joints_2d[joint_num, :] = joint_coord
            first_output = False
        
        else:
            heatmap_resized = cv2.resize(heatmap_resized, (0,0),fx = (1/resize_factor), fy = (1/resize_factor),interpolation=cv2.INTER_LINEAR)
        
        for joint_num in range(heatmap_resized.shape[2]):
            joint_coord = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_num]), (img_cropped.shape[0], img_cropped.shape[1]))
            joints_2d[joint_num, :] = joint_coord
            joints_2d[joint_num][0] = joints_2d[joint_num][0] + final_crop[0]
            joints_2d[joint_num][1] = joints_2d[joint_num][1] + final_crop[2]
            
    final_time = time.time()
    duration = final_time - initial_time
    print("Total time taken for the frame " , duration ,"secs")
    
    joint_map = np.zeros(shape=(cam_img.shape[0], cam_img.shape[1],3))
    for joint_num in range(joints_2d.shape[0]):
        cv2.circle(joint_map, center=(joints_2d[joint_num][1], joints_2d[joint_num][0]), radius=3,
                      	   color=(255, 0, 0), thickness=-1)
    
    img_limb = cam_img
    for limb_num in range(len(limb_parents)):
        x1 = joints_2d[limb_num, 0]
        y1 = joints_2d[limb_num, 1]
        x2 = joints_2d[limb_parents[limb_num], 0]
        y2 = joints_2d[limb_parents[limb_num], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                   (int(length / 2), 3),
                                   int(deg),
                                   0, 360, 1)
        cv2.fillConvexPoly(img_limb, polygon, color=(0,255,0))
    
    concat_img = np.concatenate((img_limb, joint_map), axis=1)
    cv2.imshow('2D img', concat_img.astype(np.uint8))
    cv2.waitKey(50)
    cv2.destroyAllWindows()














