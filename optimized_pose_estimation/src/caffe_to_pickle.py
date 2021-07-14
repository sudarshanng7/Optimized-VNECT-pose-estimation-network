# Reference: https://zhuanlan.zhihu.com/p/27298134
# These are helper functions for converting caffe weights to pickle format and checkpoint files

import os
import caffe
import numpy as np
import pickle


def load_caffe(proto_path, weights_path):
    caffe.set_mode_cpu()
    return caffe.Net(proto_path, weights_path, caffe.TEST)


def parameters_info(net):
    for name, param in net.params.items():
        print('Shape of the parameters are: ', name)
        for i in range(len(param)):
            print(param[i].data.shape)
    print('\n')

'''
Refer to https://blog.csdn.net/hjxu2016/article/details/81813535 and https://blog.csdn.net/zziahgf/article/details/78843350
for more information 
'''
def caffe_params(net, tfstyle=True):
    params = {}
    for name, param in net.params.items():
        if len(param) == 1:
            params[name + '/kernel'] = np.transpose(param[0].data, (2, 3, 1, 0)) if tfstyle else param[0].data
        elif len(param) == 2:
            if name == 'scale5c_branch2a':
                params['bn5c_branch2a' + '/gamma'] = param[0].data
                params['bn5c_branch2a' + '/beta'] = param[1].data
            else:
                params[name + '/weights'] = np.transpose(param[0].data, (2, 3, 1, 0)) if tfstyle else param[0].data
                params[name+'/biases'] = param[1].data
        elif len(param) == 3:
            params[name + '/moving_mean'] = param[0].data / param[2].data
            params[name + '/moving_variance'] = param[1].data / param[2].data
    return params


def save_params(net, save_path, pkl_name, tfstyle=True):
    print('converting the weights to pickle file...')
    params = caffe_params(net, tfstyle=tfstyle)
    with open(os.path.join(save_path, pkl_name), 'wb') as f:
        pickle.dump(params, f)
    print('Pickle File saved')


def caffe_to_pickle(base_path, prototxt_file, caffe_model, pkl_weights, save_path=None):
    SAVEPATH = base_path if save_path is None else save_path

    net = load_caffe(os.path.join(base_path, prototxt_file), os.path.join(base_path, caffe_model))
    parameters_info(net)
    save_params(net, SAVEPATH, pkl_weights, tfstyle=True)