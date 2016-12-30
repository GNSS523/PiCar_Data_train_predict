#coding=utf-8
import numpy as np
import cv2
import pandas as pd

from functions import raw_to_cnn, cnn_to_raw, raw_motor_to_rgb
from img_serializer import deserialize_image

#可视化数据
def visualize_data(filename, width=72, height=48, depth=3, cnn_model=None):
    data = pd.DataFrame.from_csv(filename)     

    for i in data.index:
        cur_img = data['image'][i]
        cur_throttle = int(data['servo'][i])
        cur_motor = int(data['motor'][i])        
        
        cur_img_array = deserialize_image(cur_img)        
        y_input = cur_img_array.copy()

        cur_img_array = cv2.resize(cur_img_array, (480, 320), interpolation=cv2.INTER_CUBIC)


        cv2.putText(cur_img_array, "frame: %s" % str(i), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.line(cur_img_array, (240, 300), (240-(90-cur_throttle), 200), (0, 255, 0), 3)


        cv2.line(cur_img_array, (50, 160), (50, 160-(90-cur_motor)), raw_motor_to_rgb(cur_motor), 3)

        #可视化CNN模型
        if cnn_model:
            y = cnn_model.predict([y_input])
            servo_out = cnn_to_raw(y[0])         
            cv2.line(cur_img_array, (240, 300), (240-(90-int(servo_out)), 200), (0, 0, 255), 3)

            x_ = abs(servo_out - 90)
            motor_out = (7.64*np.e**(-0.096*x_)) - 1
            motor_out = int(80 - motor_out) # Only wanna go forwards
            cv2.line(cur_img_array, (100, 160), (100, 160-(90-motor_out)), raw_motor_to_rgb(motor_out), 3)
            print(motor_out, cur_motor)

        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break