#coding=utf-8
import sys
import json
import numpy as np

import tensorflow as tf

from file_finder import get_latest_filename
from SuironML import get_cnn_model, get_nn_model
from SuironVZ import visualize_data

#加载图像设置
with open('settings.json') as d:
    SETTINGS = json.load(d)


#加载CNN模型
servo_model = None
servo_model = get_cnn_model(SETTINGS['servo_cnn_name'], SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'])
try:    
    servo_model.load(SETTINGS['servo_cnn_name'] + '.ckpt')
except Exception as e:
    print(e)
    
# 可视化是uj
filename = get_latest_filename() 

# 如果给了参数，就预测参数文件
if len(sys.argv) > 1:
    filename = sys.argv[1]

# 可视化
visualize_data(filename, SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], cnn_model=servo_model)