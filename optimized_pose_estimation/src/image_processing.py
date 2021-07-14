import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
import cv2


def img_padding(img, box_size, color='black'):
    h, w = img.shape[:2]
    offset_x, offset_y = 0, 0
    if color == 'black':
        pad_color = [0, 0, 0]
    elif color == 'grey':
        pad_color = [128, 128, 128]
    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    if h > w:
        offset_x = box_size // 2 - w // 2
        img_padded[:, offset_x: box_size // 2 + int(np.ceil(w / 2)), :] = img
    else:
        offset_y = box_size // 2 - h // 2
        img_padded[offset_y: box_size // 2 + int(np.ceil(h / 2)), :, :] = img

    return img_padded, [offset_x, offset_y]


def read_square_image(img, box_size):
    h, w = img.shape[:2]
    scaler = box_size / max(h, w)
    img_scaled = cv2.resize(img, (0, 0), fx=scaler, fy=scaler, interpolation=cv2.INTER_LINEAR)
    img_padded, [offset_x, offset_y] = img_padding(img_scaled, box_size)
    return img_padded, scaler, [offset_x, offset_y]


def read_pb_graph(model):
    with gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def