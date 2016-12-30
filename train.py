#coding=utf-8
import sys, os
import json
import numpy as np

from datasets import get_servo_dataset
from SuironML import get_cnn_model 

#加载图片设置
with open('settings.json') as d:
    SETTINGS = json.load(d)


print('[!] Loading dataset...')
X = []
SERVO = []
DATA_FILES = ['data/output_0.csv', 'data/output_1.csv', 'data/output_2.csv', 'data/output_3.csv', 'data/output_4.csv']
for d in DATA_FILES:
    c_x, c_servo = get_servo_dataset(d)
    X = X + c_x
    SERVO = SERVO + c_servo

X = np.array(X)
SERVO = np.array(SERVO) 
print('[!] Finished loading dataset...')

# servo神经网络, motor神经网络
servo_model = get_cnn_model(SETTINGS['servo_cnn_name'], SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'])

# 如果已经存在模型的话就加载已经训练过的模型
if len(sys.argv) > 1:
    servo_model.load(sys.argv[1])

servo_model.fit({'input': X}, {'target': SERVO}, n_epoch=10000,
                validation_set=0.1, show_metric=True, snapshot_epoch=False,
                snapshot_step=10000, run_id=SETTINGS['servo_cnn_name'])
servo_model.save(SETTINGS['servo_cnn_name'] + '.ckpt')