#coding=utf-8
import tensorflow as tf

import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# CNN训练网络
def get_cnn_model(checkpoint_path='cnn_servo_model', width=72, height=48, depth=3, session=None):
    
    #数据输入层
    network = input_data(shape=[None, height, width, depth], name='input')

    # 卷积层1
    network = conv_2d(network, 8, [5, 3], activation='relu')

    # 卷积层2
    network = conv_2d(network, 12, [5, 8], activation='relu')
    
    # 卷积层3
    network = conv_2d(network, 16, [5, 16], activation='relu')

    # 卷积层4
    network = conv_2d(network, 24, [3, 20], activation='relu')

    # 卷积层5
    network = conv_2d(network, 24, [3, 24], activation='relu')

    # 全连接层1
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    # 全连接层2
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.8)

    # 全连接层3
    network = fully_connected(network, 50, activation='relu')
    network = dropout(network, 0.8)

    # 全连接层4
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)
 
    # 全连接层5
    network = fully_connected(network, 1, activation='tanh')

    # 回归层
    network = regression(network, loss='mean_square', metric='accuracy', learning_rate=1e-4,name='target') 


    model = tflearn.DNN(network, tensorboard_verbose=2, checkpoint_path=checkpoint_path, session=session) 
    return model

# 普通神经网络
def get_nn_model(checkpoint_path='nn_motor_model', session=None):
    # 数据输入
    network = input_data(shape=[None, 1], name='input')

    # 隐藏层1
    network = fully_connected(network, 12, activation='linear')
    
    # 输出层
    network = fully_connected(network, 1, activation='tanh')

    # 回归
    network = regression(network, loss='mean_square', metric='accuracy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=checkpoint_path, session=session)
    return model