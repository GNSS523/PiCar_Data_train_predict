#coding=utf-8
#数据转化

import numpy as np
import pandas as pd

from img_serializer import deserialize_image
from functions import raw_to_cnn

#获取servo数据集
def get_servo_dataset(filename, start_index=0, end_index=None):
    data = pd.DataFrame.from_csv(filename)
    x = []
    servo = []

    for i in data.index[start_index:end_index]:
        # 去除噪声数据
        if data['servo'][i] < 40 or data['servo'][i] > 150:
            continue

        x.append(deserialize_image(data['image'][i]))
        servo.append(raw_to_cnn(data['servo'][i]))

    return x, servo

# 获取motor数据集
def get_motor_dataset(filename, start_index=0, end_index=None):
    data = pd.DataFrame.from_csv(filename)
    servo = []
    motor = []

    for i in data.index[start_index:end_index]:
        if data['motor'][i] < 40 or data['motor'][i] > 150:
            continue

        if data['servo'][i] < 40 or data['servo'][i] > 150:
            continue
        servo.append(raw_to_cnn(data['servo'][i]))
        motor.append(raw_to_cnn(data['motor'][i], min_arduino=60.0, max_arduino=90.0))

    return servo, motor